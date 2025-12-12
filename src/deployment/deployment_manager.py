"""
Deployment Manager for Chest X-Ray Pneumonia Detection API
Handles blue-green deployments, model updates, and deployment automation
"""

import os
import time
import logging
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import requests
import docker
from docker.errors import DockerException

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    model_path: str
    model_version: str
    image_tag: str
    port: int
    health_check_url: str
    health_check_timeout: int = 30
    health_check_retries: int = 5
    deployment_timeout: int = 300
    environment: str = "staging"

@dataclass
class DeploymentInfo:
    """Deployment information"""
    deployment_id: str
    model_version: str
    image_tag: str
    status: DeploymentStatus
    port: int
    container_id: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    health_status: str = "unknown"
    error_message: Optional[str] = None

class DeploymentManager:
    """
    Manages blue-green deployments for the chest X-ray API
    """
    
    def __init__(self, 
                 base_port: int = 8000,
                 docker_image: str = "chest-xray-api",
                 deployment_dir: str = "deployments"):
        """
        Initialize deployment manager
        
        Args:
            base_port: Base port for deployments (blue: base_port, green: base_port+1)
            docker_image: Docker image name
            deployment_dir: Directory to store deployment metadata
        """
        self.base_port = base_port
        self.docker_image = docker_image
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(exist_ok=True)
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
        except DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            self.docker_client = None
        
        # Deployment slots (blue-green)
        self.blue_port = base_port
        self.green_port = base_port + 1
        
        # Current deployments
        self.deployments: Dict[str, DeploymentInfo] = {}
        self.active_deployment: Optional[str] = None
        
        # Load existing deployments
        self._load_deployments()
        
        logger.info(f"DeploymentManager initialized")
        logger.info(f"Blue port: {self.blue_port}, Green port: {self.green_port}")
    
    def _load_deployments(self):
        """Load deployment metadata from disk"""
        metadata_file = self.deployment_dir / "deployments.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct deployment objects
                for dep_id, dep_data in data.get('deployments', {}).items():
                    self.deployments[dep_id] = DeploymentInfo(
                        deployment_id=dep_data['deployment_id'],
                        model_version=dep_data['model_version'],
                        image_tag=dep_data['image_tag'],
                        status=DeploymentStatus(dep_data['status']),
                        port=dep_data['port'],
                        container_id=dep_data.get('container_id'),
                        created_at=datetime.fromisoformat(dep_data['created_at']) if dep_data.get('created_at') else None,
                        updated_at=datetime.fromisoformat(dep_data['updated_at']) if dep_data.get('updated_at') else None,
                        health_status=dep_data.get('health_status', 'unknown'),
                        error_message=dep_data.get('error_message')
                    )
                
                self.active_deployment = data.get('active_deployment')
                
                logger.info(f"Loaded {len(self.deployments)} deployments from metadata")
                
            except Exception as e:
                logger.error(f"Failed to load deployment metadata: {e}")
    
    def _save_deployments(self):
        """Save deployment metadata to disk"""
        metadata_file = self.deployment_dir / "deployments.json"
        
        try:
            # Convert to serializable format
            data = {
                'deployments': {},
                'active_deployment': self.active_deployment
            }
            
            for dep_id, dep_info in self.deployments.items():
                data['deployments'][dep_id] = {
                    'deployment_id': dep_info.deployment_id,
                    'model_version': dep_info.model_version,
                    'image_tag': dep_info.image_tag,
                    'status': dep_info.status.value,
                    'port': dep_info.port,
                    'container_id': dep_info.container_id,
                    'created_at': dep_info.created_at.isoformat() if dep_info.created_at else None,
                    'updated_at': dep_info.updated_at.isoformat() if dep_info.updated_at else None,
                    'health_status': dep_info.health_status,
                    'error_message': dep_info.error_message
                }
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save deployment metadata: {e}")
    
    def _get_available_port(self) -> int:
        """Get available port for new deployment"""
        # Check which ports are in use
        used_ports = {dep.port for dep in self.deployments.values() if dep.status == DeploymentStatus.ACTIVE}
        
        if self.blue_port not in used_ports:
            return self.blue_port
        elif self.green_port not in used_ports:
            return self.green_port
        else:
            # Both ports in use, need to stop inactive deployment
            logger.warning("Both deployment slots in use")
            return self.blue_port  # Default to blue
    
    def _health_check(self, port: int, timeout: int = 30, retries: int = 5) -> bool:
        """Perform health check on deployment"""
        health_url = f"http://localhost:{port}/health"
        
        for attempt in range(retries):
            try:
                response = requests.get(health_url, timeout=timeout)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy' and health_data.get('model_loaded'):
                        logger.info(f"Health check passed for port {port}")
                        return True
                    else:
                        logger.warning(f"Health check failed: {health_data}")
                        
            except requests.RequestException as e:
                logger.warning(f"Health check attempt {attempt + 1} failed: {e}")
            
            if attempt < retries - 1:
                time.sleep(5)  # Wait before retry
        
        logger.error(f"Health check failed for port {port} after {retries} attempts")
        return False
    
    def build_image(self, model_path: str, model_version: str) -> str:
        """Build Docker image with model"""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        image_tag = f"{self.docker_image}:{model_version}"
        
        logger.info(f"Building Docker image: {image_tag}")
        
        try:
            # Copy model to deployment directory
            model_dest = self.deployment_dir / "models" / "best_chest_xray_model.pth"
            model_dest.parent.mkdir(exist_ok=True)
            shutil.copy2(model_path, model_dest)
            
            # Build image
            build_context = Path("deployment")
            image, build_logs = self.docker_client.images.build(
                path=str(build_context),
                tag=image_tag,
                rm=True,
                forcerm=True
            )
            
            logger.info(f"Successfully built image: {image_tag}")
            return image_tag
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise
    
    def deploy(self, config: DeploymentConfig) -> str:
        """Deploy new version using blue-green strategy"""
        deployment_id = f"deploy_{int(time.time())}"
        
        logger.info(f"Starting deployment {deployment_id}")
        logger.info(f"Model version: {config.model_version}")
        logger.info(f"Environment: {config.environment}")
        
        # Create deployment info
        deployment_info = DeploymentInfo(
            deployment_id=deployment_id,
            model_version=config.model_version,
            image_tag=config.image_tag,
            status=DeploymentStatus.DEPLOYING,
            port=self._get_available_port(),
            created_at=datetime.now()
        )
        
        self.deployments[deployment_id] = deployment_info
        self._save_deployments()
        
        try:
            # Build image if needed
            if not self._image_exists(config.image_tag):
                logger.info("Image not found, building...")
                self.build_image(config.model_path, config.model_version)
            
            # Stop existing container on this port
            self._stop_container_on_port(deployment_info.port)
            
            # Start new container
            container = self._start_container(
                image_tag=config.image_tag,
                port=deployment_info.port,
                model_path=config.model_path,
                deployment_id=deployment_id
            )
            
            deployment_info.container_id = container.id
            deployment_info.updated_at = datetime.now()
            
            # Wait for container to be ready
            logger.info("Waiting for container to start...")
            time.sleep(10)
            
            # Health check
            if self._health_check(deployment_info.port, config.health_check_timeout, config.health_check_retries):
                deployment_info.status = DeploymentStatus.ACTIVE
                deployment_info.health_status = "healthy"
                logger.info(f"Deployment {deployment_id} successful")
            else:
                deployment_info.status = DeploymentStatus.FAILED
                deployment_info.health_status = "unhealthy"
                deployment_info.error_message = "Health check failed"
                logger.error(f"Deployment {deployment_id} failed health check")
            
        except Exception as e:
            deployment_info.status = DeploymentStatus.FAILED
            deployment_info.error_message = str(e)
            logger.error(f"Deployment {deployment_id} failed: {e}")
        
        deployment_info.updated_at = datetime.now()
        self._save_deployments()
        
        return deployment_id
    
    def _image_exists(self, image_tag: str) -> bool:
        """Check if Docker image exists"""
        if not self.docker_client:
            return False
        
        try:
            self.docker_client.images.get(image_tag)
            return True
        except docker.errors.ImageNotFound:
            return False
    
    def _stop_container_on_port(self, port: int):
        """Stop any container running on the specified port"""
        if not self.docker_client:
            return
        
        try:
            containers = self.docker_client.containers.list()
            for container in containers:
                # Check if container is using the port
                port_bindings = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                for container_port, host_bindings in port_bindings.items():
                    if host_bindings:
                        for binding in host_bindings:
                            if int(binding.get('HostPort', 0)) == port:
                                logger.info(f"Stopping container {container.id} on port {port}")
                                container.stop()
                                container.remove()
                                return
        except Exception as e:
            logger.warning(f"Error stopping container on port {port}: {e}")
    
    def _start_container(self, image_tag: str, port: int, model_path: str, deployment_id: str) -> Any:
        """Start Docker container"""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        try:
            # Container configuration
            environment = {
                'MODEL_PATH': '/app/models/best_chest_xray_model.pth',
                'DEVICE': 'cpu',  # Can be configured
                'DEPLOYMENT_ID': deployment_id
            }
            
            volumes = {
                str(Path(model_path).parent.absolute()): {'bind': '/app/models', 'mode': 'ro'}
            }
            
            ports = {8000: port}
            
            container = self.docker_client.containers.run(
                image_tag,
                detach=True,
                ports=ports,
                volumes=volumes,
                environment=environment,
                name=f"chest-xray-api-{deployment_id}",
                restart_policy={"Name": "unless-stopped"}
            )
            
            logger.info(f"Started container {container.id} on port {port}")
            return container
            
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            raise
    
    def promote_deployment(self, deployment_id: str) -> bool:
        """Promote deployment to active (switch traffic)"""
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        
        if deployment.status != DeploymentStatus.ACTIVE:
            logger.error(f"Deployment {deployment_id} is not active")
            return False
        
        # Health check before promotion
        if not self._health_check(deployment.port):
            logger.error(f"Health check failed for deployment {deployment_id}")
            return False
        
        # Update active deployment
        old_active = self.active_deployment
        self.active_deployment = deployment_id
        
        logger.info(f"Promoted deployment {deployment_id} to active")
        
        # Mark old deployment as inactive
        if old_active and old_active in self.deployments:
            self.deployments[old_active].status = DeploymentStatus.INACTIVE
            logger.info(f"Marked deployment {old_active} as inactive")
        
        self._save_deployments()
        return True
    
    def rollback(self) -> bool:
        """Rollback to previous deployment"""
        if not self.active_deployment:
            logger.error("No active deployment to rollback from")
            return False
        
        # Find previous active deployment
        previous_deployments = [
            dep for dep in self.deployments.values()
            if dep.status == DeploymentStatus.INACTIVE and dep.health_status == "healthy"
        ]
        
        if not previous_deployments:
            logger.error("No previous deployment available for rollback")
            return False
        
        # Get most recent inactive deployment
        previous_deployment = max(previous_deployments, key=lambda d: d.updated_at or datetime.min)
        
        logger.info(f"Rolling back to deployment {previous_deployment.deployment_id}")
        
        # Health check previous deployment
        if not self._health_check(previous_deployment.port):
            logger.error("Previous deployment failed health check")
            return False
        
        # Switch active deployment
        current_active = self.active_deployment
        self.active_deployment = previous_deployment.deployment_id
        
        # Update statuses
        previous_deployment.status = DeploymentStatus.ACTIVE
        if current_active in self.deployments:
            self.deployments[current_active].status = DeploymentStatus.INACTIVE
        
        self._save_deployments()
        
        logger.info(f"Rollback completed to deployment {previous_deployment.deployment_id}")
        return True
    
    def cleanup_old_deployments(self, keep_count: int = 3):
        """Clean up old deployments, keeping only the most recent ones"""
        # Sort deployments by creation time
        sorted_deployments = sorted(
            self.deployments.values(),
            key=lambda d: d.created_at or datetime.min,
            reverse=True
        )
        
        # Keep active deployment and most recent ones
        to_keep = set()
        if self.active_deployment:
            to_keep.add(self.active_deployment)
        
        for dep in sorted_deployments[:keep_count]:
            to_keep.add(dep.deployment_id)
        
        # Remove old deployments
        to_remove = []
        for dep_id, dep in self.deployments.items():
            if dep_id not in to_keep:
                to_remove.append(dep_id)
                
                # Stop and remove container
                if dep.container_id:
                    try:
                        container = self.docker_client.containers.get(dep.container_id)
                        container.stop()
                        container.remove()
                        logger.info(f"Removed container for deployment {dep_id}")
                    except Exception as e:
                        logger.warning(f"Failed to remove container for {dep_id}: {e}")
        
        # Remove from tracking
        for dep_id in to_remove:
            del self.deployments[dep_id]
            logger.info(f"Cleaned up deployment {dep_id}")
        
        if to_remove:
            self._save_deployments()
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """Get deployment status"""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self) -> List[DeploymentInfo]:
        """List all deployments"""
        return list(self.deployments.values())
    
    def get_active_deployment(self) -> Optional[DeploymentInfo]:
        """Get active deployment info"""
        if self.active_deployment:
            return self.deployments.get(self.active_deployment)
        return None
    
    def stop_deployment(self, deployment_id: str) -> bool:
        """Stop a specific deployment"""
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        
        if deployment.container_id:
            try:
                container = self.docker_client.containers.get(deployment.container_id)
                container.stop()
                logger.info(f"Stopped deployment {deployment_id}")
                
                deployment.status = DeploymentStatus.INACTIVE
                deployment.updated_at = datetime.now()
                self._save_deployments()
                
                return True
            except Exception as e:
                logger.error(f"Failed to stop deployment {deployment_id}: {e}")
                return False
        
        return False