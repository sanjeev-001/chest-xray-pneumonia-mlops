#!/usr/bin/env python3
"""
Automated Deployment Pipeline
Handles complete deployment workflow with validation and rollback
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from deployment.deployment_manager import DeploymentManager, DeploymentConfig, DeploymentStatus
from deployment.model_server import app as model_app

logger = logging.getLogger(__name__)

class AutomatedDeployment:
    """
    Automated deployment pipeline with validation and rollback capabilities
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.deployment_manager = DeploymentManager(
            base_port=self.config.get('base_port', 8000),
            docker_image=self.config.get('docker_image', 'chest-xray-api')
        )
        
        # Deployment state
        self.current_deployment_id = None
        self.previous_deployment_id = None
        
        logger.info("AutomatedDeployment initialized")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            'base_port': 8000,
            'docker_image': 'chest-xray-api',
            'health_check_timeout': 30,
            'health_check_retries': 5,
            'deployment_timeout': 300,
            'validation_tests': {
                'enabled': True,
                'test_images': [],
                'expected_accuracy': 0.95,
                'max_response_time_ms': 2000
            },
            'rollback': {
                'enabled': True,
                'auto_rollback_on_failure': True
            },
            'notifications': {
                'enabled': False,
                'webhook_url': None,
                'email': None
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def validate_model(self, model_path: str) -> bool:
        """Validate model file before deployment"""
        logger.info("Validating model file...")
        
        # Check file exists
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Check file size (should be reasonable for EfficientNet-B4)
        file_size = Path(model_path).stat().st_size
        if file_size < 1024 * 1024:  # Less than 1MB
            logger.error(f"Model file too small: {file_size} bytes")
            return False
        
        if file_size > 500 * 1024 * 1024:  # More than 500MB
            logger.error(f"Model file too large: {file_size} bytes")
            return False
        
        # Try to load model (basic validation)
        try:
            import torch
            state_dict = torch.load(model_path, map_location='cpu')
            if not isinstance(state_dict, dict):
                logger.error("Invalid model file format")
                return False
            
            logger.info(f"Model validation passed: {file_size / 1024 / 1024:.1f}MB")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def run_validation_tests(self, deployment_id: str) -> bool:
        """Run validation tests on deployed model"""
        if not self.config['validation_tests']['enabled']:
            logger.info("Validation tests disabled")
            return True
        
        logger.info("Running validation tests...")
        
        deployment = self.deployment_manager.get_deployment_status(deployment_id)
        if not deployment:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        base_url = f"http://localhost:{deployment.port}"
        
        try:
            import requests
            
            # Test health endpoint
            health_response = requests.get(f"{base_url}/health", timeout=10)
            if health_response.status_code != 200:
                logger.error("Health check failed")
                return False
            
            health_data = health_response.json()
            if not health_data.get('model_loaded'):
                logger.error("Model not loaded")
                return False
            
            # Test model info endpoint
            info_response = requests.get(f"{base_url}/model/info", timeout=10)
            if info_response.status_code != 200:
                logger.error("Model info endpoint failed")
                return False
            
            # Test prediction endpoint with dummy data if test images provided
            test_images = self.config['validation_tests'].get('test_images', [])
            if test_images:
                for test_image in test_images[:3]:  # Test first 3 images
                    if Path(test_image).exists():
                        with open(test_image, 'rb') as f:
                            files = {'file': f}
                            start_time = time.time()
                            pred_response = requests.post(f"{base_url}/predict", files=files, timeout=30)
                            response_time = (time.time() - start_time) * 1000
                            
                            if pred_response.status_code != 200:
                                logger.error(f"Prediction failed for {test_image}")
                                return False
                            
                            pred_data = pred_response.json()
                            if 'prediction' not in pred_data or 'confidence' not in pred_data:
                                logger.error(f"Invalid prediction response for {test_image}")
                                return False
                            
                            # Check response time
                            max_response_time = self.config['validation_tests'].get('max_response_time_ms', 2000)
                            if response_time > max_response_time:
                                logger.warning(f"Slow response time: {response_time:.1f}ms > {max_response_time}ms")
                            
                            logger.info(f"Test image {test_image}: {pred_data['prediction']} ({pred_data['confidence']:.3f})")
            
            logger.info("Validation tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation tests failed: {e}")
            return False
    
    def send_notification(self, message: str, status: str = "info"):
        """Send deployment notification"""
        if not self.config['notifications']['enabled']:
            return
        
        logger.info(f"Notification ({status}): {message}")
        
        # Webhook notification
        webhook_url = self.config['notifications'].get('webhook_url')
        if webhook_url:
            try:
                import requests
                payload = {
                    'text': f"🏥 Chest X-Ray API Deployment: {message}",
                    'status': status,
                    'timestamp': time.time()
                }
                requests.post(webhook_url, json=payload, timeout=10)
            except Exception as e:
                logger.warning(f"Failed to send webhook notification: {e}")
    
    def deploy(self, model_path: str, model_version: str, environment: str = "staging") -> bool:
        """
        Execute complete deployment pipeline
        
        Args:
            model_path: Path to model file
            model_version: Version identifier
            environment: Deployment environment
            
        Returns:
            True if deployment successful, False otherwise
        """
        logger.info(f"Starting automated deployment pipeline")
        logger.info(f"Model: {model_path}")
        logger.info(f"Version: {model_version}")
        logger.info(f"Environment: {environment}")
        
        self.send_notification(f"Starting deployment of model version {model_version}")
        
        try:
            # Step 1: Validate model
            if not self.validate_model(model_path):
                self.send_notification(f"Model validation failed for version {model_version}", "error")
                return False
            
            # Step 2: Create deployment config
            config = DeploymentConfig(
                model_path=model_path,
                model_version=model_version,
                image_tag=f"{self.config['docker_image']}:{model_version}",
                port=0,  # Will be assigned
                health_check_url="",  # Will be set
                health_check_timeout=self.config['health_check_timeout'],
                health_check_retries=self.config['health_check_retries'],
                deployment_timeout=self.config['deployment_timeout'],
                environment=environment
            )
            
            # Step 3: Deploy
            self.previous_deployment_id = self.deployment_manager.active_deployment
            self.current_deployment_id = self.deployment_manager.deploy(config)
            
            logger.info(f"Deployment started: {self.current_deployment_id}")
            
            # Step 4: Wait for deployment to complete
            max_wait = config.deployment_timeout
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                deployment = self.deployment_manager.get_deployment_status(self.current_deployment_id)
                
                if deployment.status == DeploymentStatus.ACTIVE:
                    logger.info("Deployment completed successfully")
                    break
                elif deployment.status == DeploymentStatus.FAILED:
                    logger.error("Deployment failed")
                    self.send_notification(f"Deployment failed for version {model_version}", "error")
                    return False
                
                time.sleep(5)
            else:
                logger.error("Deployment timeout")
                self.send_notification(f"Deployment timeout for version {model_version}", "error")
                return False
            
            # Step 5: Run validation tests
            if not self.run_validation_tests(self.current_deployment_id):
                logger.error("Validation tests failed")
                
                if self.config['rollback']['auto_rollback_on_failure']:
                    logger.info("Auto-rollback enabled, rolling back...")
                    if self.rollback():
                        self.send_notification(f"Deployment failed, rolled back to previous version", "warning")
                    else:
                        self.send_notification(f"Deployment and rollback both failed!", "error")
                else:
                    self.send_notification(f"Deployment validation failed for version {model_version}", "error")
                
                return False
            
            # Step 6: Promote to active (if not auto-promoted)
            if not self.deployment_manager.promote_deployment(self.current_deployment_id):
                logger.error("Failed to promote deployment")
                self.send_notification(f"Failed to promote deployment {model_version}", "error")
                return False
            
            logger.info(f"Deployment {self.current_deployment_id} promoted to active")
            self.send_notification(f"Successfully deployed model version {model_version}", "success")
            
            # Step 7: Cleanup old deployments
            self.deployment_manager.cleanup_old_deployments(keep_count=3)
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")
            self.send_notification(f"Deployment pipeline error: {str(e)}", "error")
            
            # Auto-rollback on error
            if self.config['rollback']['auto_rollback_on_failure']:
                logger.info("Attempting auto-rollback...")
                self.rollback()
            
            return False
    
    def rollback(self) -> bool:
        """Rollback to previous deployment"""
        logger.info("Starting rollback...")
        
        if self.deployment_manager.rollback():
            logger.info("Rollback completed successfully")
            self.send_notification("Rollback completed successfully", "info")
            return True
        else:
            logger.error("Rollback failed")
            self.send_notification("Rollback failed", "error")
            return False
    
    def status(self) -> Dict[str, Any]:
        """Get deployment status"""
        active_deployment = self.deployment_manager.get_active_deployment()
        all_deployments = self.deployment_manager.list_deployments()
        
        return {
            'active_deployment': {
                'deployment_id': active_deployment.deployment_id if active_deployment else None,
                'model_version': active_deployment.model_version if active_deployment else None,
                'status': active_deployment.status.value if active_deployment else None,
                'port': active_deployment.port if active_deployment else None,
                'health_status': active_deployment.health_status if active_deployment else None
            },
            'total_deployments': len(all_deployments),
            'deployments': [
                {
                    'deployment_id': dep.deployment_id,
                    'model_version': dep.model_version,
                    'status': dep.status.value,
                    'port': dep.port,
                    'health_status': dep.health_status,
                    'created_at': dep.created_at.isoformat() if dep.created_at else None
                }
                for dep in sorted(all_deployments, key=lambda d: d.created_at or datetime.min, reverse=True)
            ]
        }

def main():
    parser = argparse.ArgumentParser(description="Automated Deployment Pipeline")
    parser.add_argument("command", choices=["deploy", "rollback", "status"], help="Command to execute")
    parser.add_argument("--model-path", help="Path to model file (for deploy)")
    parser.add_argument("--model-version", help="Model version (for deploy)")
    parser.add_argument("--environment", default="staging", choices=["staging", "production"], help="Environment")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create deployment pipeline
    pipeline = AutomatedDeployment(config_file=args.config)
    
    if args.command == "deploy":
        if not args.model_path or not args.model_version:
            print("❌ --model-path and --model-version required for deploy command")
            return 1
        
        print(f"🚀 Starting automated deployment...")
        success = pipeline.deploy(args.model_path, args.model_version, args.environment)
        
        if success:
            print("✅ Deployment completed successfully!")
            return 0
        else:
            print("❌ Deployment failed!")
            return 1
    
    elif args.command == "rollback":
        print("🔄 Starting rollback...")
        success = pipeline.rollback()
        
        if success:
            print("✅ Rollback completed successfully!")
            return 0
        else:
            print("❌ Rollback failed!")
            return 1
    
    elif args.command == "status":
        status = pipeline.status()
        print("📊 Deployment Status:")
        print(json.dumps(status, indent=2))
        return 0

if __name__ == "__main__":
    sys.exit(main())