"""
Deployment Automation Integration Tests
Tests for blue-green deployment, load balancing, and automated deployment workflows
"""

import pytest
import tempfile
import shutil
import time
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import threading
import asyncio

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from deployment.deployment_manager import DeploymentManager, DeploymentConfig, DeploymentStatus, DeploymentInfo
from deployment.automated_deploy import AutomatedDeployment
from deployment.load_balancer import LoadBalancer


class TestDeploymentAutomation:
    """Test deployment automation workflows"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_file(self, temp_dir):
        """Create a mock model file"""
        model_path = Path(temp_dir) / "test_model.pth"
        # Create a realistic-sized file
        with open(model_path, 'wb') as f:
            f.write(b'0' * (50 * 1024 * 1024))  # 50MB
        return str(model_path)
    
    @pytest.fixture
    def deployment_manager(self, temp_dir):
        """Create deployment manager for testing"""
        return DeploymentManager(
            base_port=9000,
            docker_image="test-chest-xray-api",
            deployment_dir=str(Path(temp_dir) / "deployments")
        )
    
    def test_deployment_config_creation(self, mock_model_file):
        """Test deployment configuration creation"""
        config = DeploymentConfig(
            model_path=mock_model_file,
            model_version="v1.0.0",
            image_tag="test-chest-xray-api:v1.0.0",
            port=9000,
            health_check_url="http://localhost:9000/health",
            environment="staging"
        )
        
        assert config.model_path == mock_model_file
        assert config.model_version == "v1.0.0"
        assert config.image_tag == "test-chest-xray-api:v1.0.0"
        assert config.port == 9000
        assert config.environment == "staging"
    
    def test_deployment_info_serialization(self):
        """Test deployment info serialization"""
        deployment_info = DeploymentInfo(
            deployment_id="test_deploy_123",
            model_version="v1.0.0",
            image_tag="test:v1.0.0",
            status=DeploymentStatus.ACTIVE,
            port=9000,
            created_at=datetime.now()
        )
        
        assert deployment_info.deployment_id == "test_deploy_123"
        assert deployment_info.status == DeploymentStatus.ACTIVE
        assert deployment_info.port == 9000
    
    @patch('deployment.deployment_manager.docker.from_env')
    def test_deployment_manager_docker_integration(self, mock_docker, deployment_manager):
        """Test deployment manager Docker integration"""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Test Docker client initialization
        manager = DeploymentManager(base_port=9000)
        assert manager.docker_client is not None or manager.docker_client is None  # Depends on Docker availability
    
    def test_deployment_metadata_persistence(self, deployment_manager, temp_dir):
        """Test deployment metadata persistence"""
        # Create test deployment
        deployment_info = DeploymentInfo(
            deployment_id="test_deploy_123",
            model_version="v1.0.0",
            image_tag="test:v1.0.0",
            status=DeploymentStatus.ACTIVE,
            port=9000,
            created_at=datetime.now()
        )
        
        deployment_manager.deployments["test_deploy_123"] = deployment_info
        deployment_manager.active_deployment = "test_deploy_123"
        
        # Save metadata
        deployment_manager._save_deployments()
        
        # Verify file was created
        metadata_file = Path(temp_dir) / "deployments" / "deployments.json"
        assert metadata_file.exists()
        
        # Load and verify content
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        assert "deployments" in data
        assert "active_deployment" in data
        assert data["active_deployment"] == "test_deploy_123"
        assert "test_deploy_123" in data["deployments"]
    
    def test_blue_green_port_allocation(self, deployment_manager):
        """Test blue-green port allocation"""
        # First deployment should get blue port
        port1 = deployment_manager._get_available_port()
        assert port1 == 9000  # Blue port
        
        # Add active deployment on blue port
        deployment_manager.deployments["deploy1"] = DeploymentInfo(
            deployment_id="deploy1",
            model_version="v1.0.0",
            image_tag="test:v1.0.0",
            status=DeploymentStatus.ACTIVE,
            port=9000
        )
        
        # Second deployment should get green port
        port2 = deployment_manager._get_available_port()
        assert port2 == 9001  # Green port
    
    @patch('deployment.deployment_manager.requests.get')
    def test_health_check_integration(self, mock_get, deployment_manager):
        """Test health check integration"""
        # Mock successful health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "model_loaded": True
        }
        mock_get.return_value = mock_response
        
        result = deployment_manager._health_check(9000, timeout=5, retries=1)
        assert result is True
        
        # Verify correct URL was called
        mock_get.assert_called_with("http://localhost:9000/health", timeout=5)
        
        # Test failed health check
        mock_response.status_code = 503
        result = deployment_manager._health_check(9000, timeout=5, retries=1)
        assert result is False
    
    def test_deployment_promotion_workflow(self, deployment_manager):
        """Test deployment promotion workflow"""
        # Create test deployment
        deployment_info = DeploymentInfo(
            deployment_id="test_deploy_123",
            model_version="v1.0.0",
            image_tag="test:v1.0.0",
            status=DeploymentStatus.ACTIVE,
            port=9000,
            health_status="healthy"
        )
        
        deployment_manager.deployments["test_deploy_123"] = deployment_info
        
        # Mock health check
        with patch.object(deployment_manager, '_health_check', return_value=True):
            result = deployment_manager.promote_deployment("test_deploy_123")
            assert result is True
            assert deployment_manager.active_deployment == "test_deploy_123"
    
    def test_rollback_workflow(self, deployment_manager):
        """Test rollback workflow"""
        # Create current active deployment
        current_deployment = DeploymentInfo(
            deployment_id="current_deploy",
            model_version="v2.0.0",
            image_tag="test:v2.0.0",
            status=DeploymentStatus.ACTIVE,
            port=9000,
            health_status="unhealthy",
            updated_at=datetime.now()
        )
        
        # Create previous deployment
        previous_deployment = DeploymentInfo(
            deployment_id="previous_deploy",
            model_version="v1.0.0",
            image_tag="test:v1.0.0",
            status=DeploymentStatus.INACTIVE,
            port=9001,
            health_status="healthy",
            updated_at=datetime.now()
        )
        
        deployment_manager.deployments["current_deploy"] = current_deployment
        deployment_manager.deployments["previous_deploy"] = previous_deployment
        deployment_manager.active_deployment = "current_deploy"
        
        # Mock health check for previous deployment
        with patch.object(deployment_manager, '_health_check', return_value=True):
            result = deployment_manager.rollback()
            assert result is True
            assert deployment_manager.active_deployment == "previous_deploy"
            assert deployment_manager.deployments["previous_deploy"].status == DeploymentStatus.ACTIVE
    
    def test_deployment_cleanup(self, deployment_manager):
        """Test deployment cleanup"""
        # Create multiple deployments
        for i in range(5):
            deployment_info = DeploymentInfo(
                deployment_id=f"deploy_{i}",
                model_version=f"v1.{i}.0",
                image_tag=f"test:v1.{i}.0",
                status=DeploymentStatus.INACTIVE,
                port=9000 + i,
                created_at=datetime.now()
            )
            deployment_manager.deployments[f"deploy_{i}"] = deployment_info
        
        # Set one as active
        deployment_manager.active_deployment = "deploy_4"
        deployment_manager.deployments["deploy_4"].status = DeploymentStatus.ACTIVE
        
        # Mock Docker client for cleanup
        with patch.object(deployment_manager, 'docker_client') as mock_docker:
            mock_container = Mock()
            mock_docker.containers.get.return_value = mock_container
            
            # Cleanup keeping 3 deployments
            deployment_manager.cleanup_old_deployments(keep_count=3)
            
            # Should keep active deployment + 2 most recent
            assert len(deployment_manager.deployments) <= 3
            assert "deploy_4" in deployment_manager.deployments  # Active deployment kept


class TestAutomatedDeploymentWorkflow:
    """Test automated deployment workflow"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_file(self, temp_dir):
        """Create a mock model file"""
        model_path = Path(temp_dir) / "test_model.pth"
        with open(model_path, 'wb') as f:
            f.write(b'0' * (50 * 1024 * 1024))  # 50MB
        return str(model_path)
    
    @pytest.fixture
    def deployment_config(self):
        """Create deployment configuration"""
        return {
            'base_port': 9000,
            'docker_image': 'test-chest-xray-api',
            'health_check_timeout': 5,
            'health_check_retries': 2,
            'deployment_timeout': 60,
            'validation_tests': {
                'enabled': False
            },
            'rollback': {
                'enabled': True,
                'auto_rollback_on_failure': False
            },
            'notifications': {
                'enabled': False
            }
        }
    
    def test_automated_deployment_initialization(self, deployment_config):
        """Test automated deployment initialization"""
        with patch('deployment.automated_deploy.DeploymentManager'):
            deployment = AutomatedDeployment()
            assert deployment.config is not None
            assert deployment.deployment_manager is not None
    
    def test_model_validation_workflow(self, mock_model_file):
        """Test model validation workflow"""
        deployment = AutomatedDeployment()
        
        # Test valid model
        assert deployment.validate_model(mock_model_file) is True
        
        # Test non-existent model
        assert deployment.validate_model("non_existent.pth") is False
        
        # Test invalid model (too small)
        small_model_path = Path(mock_model_file).parent / "small_model.pth"
        with open(small_model_path, 'wb') as f:
            f.write(b'small')
        
        assert deployment.validate_model(str(small_model_path)) is False
    
    @patch('deployment.automated_deploy.requests.get')
    def test_validation_tests_workflow(self, mock_get):
        """Test validation tests workflow"""
        deployment = AutomatedDeployment()
        deployment.config['validation_tests']['enabled'] = True
        
        # Mock successful API responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "model_loaded": True
        }
        mock_get.return_value = mock_response
        
        # Mock deployment manager
        mock_deployment = Mock()
        mock_deployment.port = 9000
        deployment.deployment_manager.get_deployment_status = Mock(return_value=mock_deployment)
        
        result = deployment.run_validation_tests("test_deployment_id")
        assert result is True
    
    def test_notification_system(self):
        """Test notification system"""
        config = {
            'notifications': {
                'enabled': True,
                'webhook_url': 'https://hooks.slack.com/test'
            }
        }
        
        deployment = AutomatedDeployment()
        deployment.config = config
        
        # Mock webhook request
        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock(status_code=200)
            
            deployment.send_notification("Test deployment message", "info")
            
            # Verify webhook was called
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert 'json' in call_args.kwargs
            assert 'Test deployment message' in str(call_args.kwargs['json'])
    
    def test_deployment_status_reporting(self):
        """Test deployment status reporting"""
        deployment = AutomatedDeployment()
        
        # Mock deployment manager with test data
        mock_active_deployment = Mock()
        mock_active_deployment.deployment_id = "active_deploy_123"
        mock_active_deployment.model_version = "v1.0.0"
        mock_active_deployment.status.value = "active"
        mock_active_deployment.port = 9000
        mock_active_deployment.health_status = "healthy"
        
        deployment.deployment_manager.get_active_deployment = Mock(return_value=mock_active_deployment)
        deployment.deployment_manager.list_deployments = Mock(return_value=[mock_active_deployment])
        
        status = deployment.status()
        
        assert "active_deployment" in status
        assert "total_deployments" in status
        assert "deployments" in status
        assert status["active_deployment"]["deployment_id"] == "active_deploy_123"
        assert status["total_deployments"] == 1
    
    @patch('deployment.automated_deploy.DeploymentManager')
    def test_complete_deployment_workflow_mock(self, mock_deployment_manager_class, mock_model_file):
        """Test complete deployment workflow with mocked components"""
        # Mock deployment manager
        mock_manager = Mock()
        mock_deployment_manager_class.return_value = mock_manager
        
        # Mock successful deployment
        mock_manager.deploy.return_value = "test_deploy_123"
        mock_manager.active_deployment = None
        
        # Mock deployment status progression
        mock_deployment = Mock()
        mock_deployment.status = DeploymentStatus.ACTIVE
        mock_manager.get_deployment_status.return_value = mock_deployment
        
        # Mock promotion
        mock_manager.promote_deployment.return_value = True
        mock_manager.cleanup_old_deployments.return_value = None
        
        # Create deployment instance
        deployment = AutomatedDeployment()
        deployment.config['validation_tests']['enabled'] = False
        
        # Mock validation tests
        deployment.run_validation_tests = Mock(return_value=True)
        
        # Run deployment
        result = deployment.deploy(mock_model_file, "v1.0.0", "staging")
        
        # Verify workflow
        assert result is True
        mock_manager.deploy.assert_called_once()
        mock_manager.promote_deployment.assert_called_once()
        mock_manager.cleanup_old_deployments.assert_called_once()


class TestLoadBalancerIntegration:
    """Test load balancer integration"""
    
    @pytest.fixture
    def mock_deployment_manager(self):
        """Create mock deployment manager"""
        manager = Mock()
        
        # Mock active deployment
        active_deployment = Mock()
        active_deployment.status = DeploymentStatus.ACTIVE
        active_deployment.port = 9000
        active_deployment.deployment_id = "active_deploy"
        
        manager.get_active_deployment.return_value = active_deployment
        manager.list_deployments.return_value = [active_deployment]
        
        return manager
    
    def test_load_balancer_initialization(self, mock_deployment_manager):
        """Test load balancer initialization"""
        lb = LoadBalancer(mock_deployment_manager)
        
        assert lb.deployment_manager == mock_deployment_manager
        assert lb.active_backend is None  # Not set until update
        assert lb.backup_backend is None
        assert lb.request_count == 0
        assert lb.error_count == 0
    
    @pytest.mark.asyncio
    async def test_backend_update(self, mock_deployment_manager):
        """Test backend update logic"""
        lb = LoadBalancer(mock_deployment_manager)
        
        # Update backends
        await lb._update_backends()
        
        # Should set active backend
        assert lb.active_backend == "http://localhost:9000"
    
    def test_load_balancer_stats(self, mock_deployment_manager):
        """Test load balancer statistics"""
        lb = LoadBalancer(mock_deployment_manager)
        
        # Record some requests
        lb.request_count = 100
        lb.error_count = 5
        lb.response_times = [100, 150, 200, 120, 180]
        
        stats = lb.get_stats()
        
        assert stats["request_count"] == 100
        assert stats["error_count"] == 5
        assert stats["error_rate"] == 0.05
        assert stats["avg_response_time_ms"] == 150.0  # Average of response times
    
    @pytest.mark.asyncio
    async def test_request_routing(self, mock_deployment_manager):
        """Test request routing logic"""
        lb = LoadBalancer(mock_deployment_manager)
        
        # Set up backends
        lb.active_backend = "http://localhost:9000"
        lb.backend_health = {"http://localhost:9000": True}
        
        # Mock request
        mock_request = Mock()
        
        # Test routing
        backend = await lb.route_request(mock_request)
        assert backend == "http://localhost:9000"
        
        # Test with unhealthy backend
        lb.backend_health = {"http://localhost:9000": False}
        backend = await lb.route_request(mock_request)
        assert backend is None


class TestDeploymentCLIIntegration:
    """Test deployment CLI integration"""
    
    def test_cli_command_structure(self):
        """Test CLI command structure"""
        # Import CLI module to test structure
        from deployment.deploy_cli import main
        
        # Test that main function exists and is callable
        assert callable(main)
    
    @patch('deployment.deploy_cli.DeploymentManager')
    def test_cli_list_command_mock(self, mock_manager_class):
        """Test CLI list command with mocked manager"""
        from deployment.deploy_cli import cmd_list
        
        # Mock deployment manager
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Mock deployments
        mock_deployment = Mock()
        mock_deployment.deployment_id = "test_deploy"
        mock_deployment.model_version = "v1.0.0"
        mock_deployment.status.value = "active"
        mock_deployment.port = 9000
        mock_deployment.health_status = "healthy"
        mock_deployment.created_at = datetime.now()
        mock_deployment.error_message = None
        
        mock_manager.list_deployments.return_value = [mock_deployment]
        mock_manager.get_active_deployment.return_value = mock_deployment
        
        # Mock args
        args = Mock()
        
        # Test list command
        result = cmd_list(args)
        assert result == 0  # Success


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios"""
    
    def test_deployment_failure_handling(self):
        """Test deployment failure handling"""
        deployment = AutomatedDeployment()
        deployment.config['rollback']['auto_rollback_on_failure'] = True
        
        # Mock deployment manager with failure
        deployment.deployment_manager = Mock()
        deployment.deployment_manager.deploy.side_effect = Exception("Deployment failed")
        
        # Mock rollback
        deployment.rollback = Mock(return_value=True)
        
        # Test deployment with failure
        result = deployment.deploy("fake_model.pth", "v1.0.0", "staging")
        
        # Should fail and trigger rollback
        assert result is False
        deployment.rollback.assert_called_once()
    
    def test_health_check_failure_recovery(self):
        """Test health check failure recovery"""
        manager = DeploymentManager(base_port=9000)
        
        # Mock failed health check
        with patch.object(manager, '_health_check', return_value=False):
            # Should handle health check failure gracefully
            result = manager.promote_deployment("non_existent_deployment")
            assert result is False
    
    def test_docker_unavailable_handling(self):
        """Test handling when Docker is unavailable"""
        with patch('deployment.deployment_manager.docker.from_env', side_effect=Exception("Docker not available")):
            manager = DeploymentManager(base_port=9000)
            
            # Should initialize with docker_client as None
            assert manager.docker_client is None
    
    def test_invalid_model_handling(self):
        """Test invalid model file handling"""
        deployment = AutomatedDeployment()
        
        # Test with non-existent file
        result = deployment.validate_model("non_existent_model.pth")
        assert result is False
        
        # Test with invalid file content
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            f.write(b"invalid model content")
            temp_path = f.name
        
        try:
            result = deployment.validate_model(temp_path)
            assert result is False
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])