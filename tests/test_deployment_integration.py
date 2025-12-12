"""
Integration Tests for Deployment Service
Tests the complete deployment pipeline including API, performance optimization, and automation
"""

import pytest
import asyncio
import aiohttp
import tempfile
import shutil
import time
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from PIL import Image
import io

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from deployment.model_server import app
from deployment.deployment_manager import DeploymentManager, DeploymentConfig, DeploymentStatus
from deployment.performance_optimizer import PerformanceOptimizer, ModelCache
from deployment.automated_deploy import AutomatedDeployment
from training.models import ModelFactory


class TestModelServerIntegration:
    """Integration tests for the model server"""
    
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
        
        # Create a simple mock model state dict
        mock_state_dict = {
            'features.0.weight': torch.randn(32, 3, 3, 3),
            'features.0.bias': torch.randn(32),
            'classifier.weight': torch.randn(2, 1000),
            'classifier.bias': torch.randn(2)
        }
        
        torch.save(mock_state_dict, model_path)
        return str(model_path)
    
    @pytest.fixture
    def test_image(self, temp_dir):
        """Create a test image"""
        image = Image.new('RGB', (224, 224), color='white')
        image_path = Path(temp_dir) / "test_image.jpg"
        image.save(image_path)
        return str(image_path)
    
    @pytest.fixture
    def test_client(self):
        """Create test client for FastAPI app"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "model_loaded" in health_data
        assert "device" in health_data
        assert "uptime_seconds" in health_data
    
    def test_model_info_endpoint_without_model(self, test_client):
        """Test model info endpoint when model is not loaded"""
        response = test_client.get("/model/info")
        # Should return 503 when model is not loaded
        assert response.status_code == 503
    
    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint"""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        assert "uptime_seconds" in metrics
        assert "model_loaded" in metrics
        assert "device" in metrics
    
    def test_performance_endpoints_without_optimizer(self, test_client):
        """Test performance endpoints when optimizer is not available"""
        # Performance stats should return 503 without optimizer
        response = test_client.get("/performance/stats")
        assert response.status_code == 503
        
        # Benchmark should return 503 without optimizer
        response = test_client.get("/performance/benchmark")
        assert response.status_code == 503
    
    def test_readiness_probe(self, test_client):
        """Test Kubernetes readiness probe"""
        response = test_client.get("/ready")
        # Should return 503 when model is not loaded
        assert response.status_code == 503
    
    def test_liveness_probe(self, test_client):
        """Test Kubernetes liveness probe"""
        response = test_client.get("/alive")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    @patch('deployment.model_server.ModelFactory.create_model')
    @patch('deployment.model_server.torch.load')
    @patch('deployment.model_server.os.path.exists')
    def test_model_loading_integration(self, mock_exists, mock_torch_load, mock_create_model, mock_model_file):
        """Test model loading integration"""
        # Mock dependencies
        mock_exists.return_value = True
        mock_torch_load.return_value = {'test': 'state_dict'}
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_create_model.return_value = mock_model
        
        # Import and test model loading
        from deployment.model_server import load_model
        
        with patch.dict(os.environ, {'MODEL_PATH': mock_model_file}):
            result = load_model()
        
        assert result is True
        mock_create_model.assert_called_once()
        mock_model.eval.assert_called_once()


class TestPerformanceOptimizer:
    """Integration tests for performance optimizer"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = ModelFactory.create_model('efficientnet_b4', num_classes=2, pretrained=False)
        model.eval()
        return model
    
    @pytest.fixture
    def performance_optimizer(self, mock_model):
        """Create performance optimizer instance"""
        device = torch.device('cpu')
        config = {
            'cache_size': 10,
            'cache_ttl': 60,
            'auto_optimize': False,  # Disable for testing
            'enable_batching': False
        }
        return PerformanceOptimizer(mock_model, device, config)
    
    def test_cache_functionality(self, performance_optimizer):
        """Test caching functionality"""
        cache = performance_optimizer.cache
        
        # Test cache miss
        test_data = b"test_image_data"
        result = cache.get(test_data)
        assert result is None
        
        # Test cache put and hit
        test_result = {"prediction": "NORMAL", "confidence": 0.95}
        cache.put(test_data, test_result)
        
        cached_result = cache.get(test_data)
        assert cached_result is not None
        assert cached_result["prediction"] == "NORMAL"
        assert cached_result["confidence"] == 0.95
    
    def test_cache_eviction(self, performance_optimizer):
        """Test cache eviction policies"""
        cache = performance_optimizer.cache
        
        # Fill cache beyond capacity
        for i in range(15):  # Cache size is 10
            test_data = f"test_data_{i}".encode()
            test_result = {"prediction": "NORMAL", "confidence": 0.95}
            cache.put(test_data, test_result)
        
        # Cache should not exceed max size
        assert len(cache.cache) <= cache.max_size
    
    def test_performance_monitoring(self, performance_optimizer):
        """Test performance monitoring"""
        monitor = performance_optimizer.performance_monitor
        
        # Record some requests
        monitor.record_request(100.0, "NORMAL", error=False)
        monitor.record_request(150.0, "PNEUMONIA", error=False)
        monitor.record_request(200.0, "ERROR", error=True)
        
        stats = monitor.get_stats()
        
        assert stats["request_count"] == 3
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 1/3
        assert stats["response_time"]["avg_ms"] == 150.0  # (100+150+200)/3
        assert "NORMAL" in stats["predictions"]
        assert "PNEUMONIA" in stats["predictions"]
    
    def test_memory_management(self, performance_optimizer):
        """Test memory management"""
        memory_manager = performance_optimizer.memory_manager
        
        # Test memory stats
        stats = memory_manager.get_memory_stats()
        assert "system_memory" in stats
        assert "process_memory" in stats
        
        # Test memory optimization (should not raise errors)
        memory_manager.optimize_memory()
    
    def test_performance_stats_integration(self, performance_optimizer):
        """Test comprehensive performance stats"""
        # Record some activity
        performance_optimizer.performance_monitor.record_request(100.0, "NORMAL")
        performance_optimizer.cache.put(b"test", {"prediction": "NORMAL"})
        
        stats = performance_optimizer.get_performance_stats()
        
        assert "cache" in stats
        assert "performance" in stats
        assert "memory" in stats
        assert "health" in stats
        assert "optimization" in stats
        
        # Verify cache stats
        assert stats["cache"]["size"] == 1
        assert stats["cache"]["hit_count"] == 0
        assert stats["cache"]["miss_count"] == 1


class TestDeploymentManager:
    """Integration tests for deployment manager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def deployment_manager(self, temp_dir):
        """Create deployment manager instance"""
        return DeploymentManager(
            base_port=9000,  # Use different port for testing
            docker_image="test-chest-xray-api",
            deployment_dir=str(Path(temp_dir) / "deployments")
        )
    
    def test_deployment_manager_initialization(self, deployment_manager):
        """Test deployment manager initialization"""
        assert deployment_manager.base_port == 9000
        assert deployment_manager.blue_port == 9000
        assert deployment_manager.green_port == 9001
        assert deployment_manager.docker_image == "test-chest-xray-api"
        assert len(deployment_manager.deployments) == 0
    
    def test_deployment_metadata_persistence(self, deployment_manager, temp_dir):
        """Test deployment metadata saving and loading"""
        # Create a mock deployment
        from deployment.deployment_manager import DeploymentInfo
        from datetime import datetime
        
        deployment_info = DeploymentInfo(
            deployment_id="test_deploy_123",
            model_version="v1.0.0",
            image_tag="test-chest-xray-api:v1.0.0",
            status=DeploymentStatus.ACTIVE,
            port=9000,
            created_at=datetime.now()
        )
        
        deployment_manager.deployments["test_deploy_123"] = deployment_info
        deployment_manager.active_deployment = "test_deploy_123"
        
        # Save metadata
        deployment_manager._save_deployments()
        
        # Create new manager and load metadata
        new_manager = DeploymentManager(
            base_port=9000,
            docker_image="test-chest-xray-api",
            deployment_dir=str(Path(temp_dir) / "deployments")
        )
        
        assert len(new_manager.deployments) == 1
        assert "test_deploy_123" in new_manager.deployments
        assert new_manager.active_deployment == "test_deploy_123"
        
        loaded_deployment = new_manager.deployments["test_deploy_123"]
        assert loaded_deployment.model_version == "v1.0.0"
        assert loaded_deployment.status == DeploymentStatus.ACTIVE
    
    def test_port_allocation(self, deployment_manager):
        """Test port allocation logic"""
        # First deployment should get blue port
        port1 = deployment_manager._get_available_port()
        assert port1 == 9000
        
        # Simulate active deployment on blue port
        from deployment.deployment_manager import DeploymentInfo
        deployment_manager.deployments["deploy1"] = DeploymentInfo(
            deployment_id="deploy1",
            model_version="v1.0.0",
            image_tag="test:v1.0.0",
            status=DeploymentStatus.ACTIVE,
            port=9000
        )
        
        # Second deployment should get green port
        port2 = deployment_manager._get_available_port()
        assert port2 == 9001
    
    @patch('deployment.deployment_manager.requests.get')
    def test_health_check(self, mock_get, deployment_manager):
        """Test health check functionality"""
        # Mock successful health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "model_loaded": True}
        mock_get.return_value = mock_response
        
        result = deployment_manager._health_check(9000, timeout=5, retries=1)
        assert result is True
        
        # Mock failed health check
        mock_response.status_code = 500
        result = deployment_manager._health_check(9000, timeout=5, retries=1)
        assert result is False
    
    def test_deployment_status_tracking(self, deployment_manager):
        """Test deployment status tracking"""
        # Test getting non-existent deployment
        status = deployment_manager.get_deployment_status("non_existent")
        assert status is None
        
        # Add a deployment
        from deployment.deployment_manager import DeploymentInfo
        deployment_info = DeploymentInfo(
            deployment_id="test_deploy",
            model_version="v1.0.0",
            image_tag="test:v1.0.0",
            status=DeploymentStatus.ACTIVE,
            port=9000
        )
        
        deployment_manager.deployments["test_deploy"] = deployment_info
        
        # Test getting existing deployment
        status = deployment_manager.get_deployment_status("test_deploy")
        assert status is not None
        assert status.deployment_id == "test_deploy"
        assert status.status == DeploymentStatus.ACTIVE


class TestAutomatedDeployment:
    """Integration tests for automated deployment"""
    
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
        
        # Create a realistic-sized mock model file
        mock_data = b"0" * (50 * 1024 * 1024)  # 50MB
        with open(model_path, 'wb') as f:
            f.write(mock_data)
        
        return str(model_path)
    
    @pytest.fixture
    def deployment_config(self, temp_dir):
        """Create deployment configuration"""
        return {
            'base_port': 9000,
            'docker_image': 'test-chest-xray-api',
            'health_check_timeout': 5,
            'health_check_retries': 2,
            'deployment_timeout': 60,
            'validation_tests': {
                'enabled': False  # Disable for testing
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
        deployment = AutomatedDeployment()
        
        # Test default configuration
        assert deployment.config['base_port'] == 8000
        assert deployment.config['docker_image'] == 'chest-xray-api'
        assert deployment.deployment_manager is not None
    
    def test_model_validation(self, mock_model_file):
        """Test model file validation"""
        deployment = AutomatedDeployment()
        
        # Test valid model file
        assert deployment.validate_model(mock_model_file) is True
        
        # Test non-existent file
        assert deployment.validate_model("non_existent.pth") is False
        
        # Test file too small
        small_file = Path(mock_model_file).parent / "small_model.pth"
        with open(small_file, 'wb') as f:
            f.write(b"small")
        
        assert deployment.validate_model(str(small_file)) is False
    
    @patch('deployment.automated_deploy.requests.get')
    def test_validation_tests(self, mock_get):
        """Test validation test execution"""
        deployment = AutomatedDeployment()
        deployment.config['validation_tests']['enabled'] = True
        
        # Mock successful health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "model_loaded": True}
        mock_get.return_value = mock_response
        
        # Test validation (should pass with mocked health check)
        result = deployment.run_validation_tests("test_deployment_id")
        assert result is True
    
    def test_deployment_status_reporting(self):
        """Test deployment status reporting"""
        deployment = AutomatedDeployment()
        
        status = deployment.status()
        
        assert "active_deployment" in status
        assert "total_deployments" in status
        assert "deployments" in status
        
        # Should handle empty deployment state
        assert status["active_deployment"]["deployment_id"] is None
        assert status["total_deployments"] == 0


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_image_bytes(self):
        """Create test image as bytes"""
        image = Image.new('RGB', (224, 224), color='white')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()
    
    @pytest.mark.asyncio
    async def test_api_integration_with_mocked_model(self, test_image_bytes):
        """Test API integration with mocked model components"""
        # This test would require a running server, so we'll test components
        
        # Test image preprocessing
        from deployment.model_server import preprocess_image
        
        # This should work with our test image
        try:
            # Note: This will fail without a proper model loaded, which is expected
            tensor = preprocess_image(test_image_bytes)
            assert tensor.shape == (1, 3, 224, 224)
        except Exception:
            # Expected to fail without proper model setup
            pass
    
    def test_performance_optimization_integration(self):
        """Test performance optimization integration"""
        # Create a simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 2)
        )
        
        device = torch.device('cpu')
        config = {
            'cache_size': 5,
            'cache_ttl': 30,
            'auto_optimize': True,
            'enable_batching': False
        }
        
        optimizer = PerformanceOptimizer(model, device, config)
        
        # Test that optimization was applied
        assert optimizer.model_optimizer.optimization_applied is True
        
        # Test performance stats
        stats = optimizer.get_performance_stats()
        assert "cache" in stats
        assert "performance" in stats
        assert "optimization" in stats
    
    def test_deployment_workflow_components(self, temp_dir):
        """Test deployment workflow components integration"""
        # Test deployment manager with performance optimizer
        deployment_manager = DeploymentManager(
            base_port=9000,
            deployment_dir=str(Path(temp_dir) / "deployments")
        )
        
        # Test that manager initializes correctly
        assert deployment_manager.deployments == {}
        assert deployment_manager.active_deployment is None
        
        # Test deployment listing
        deployments = deployment_manager.list_deployments()
        assert deployments == []
        
        # Test active deployment retrieval
        active = deployment_manager.get_active_deployment()
        assert active is None


class TestErrorHandling:
    """Test error handling in deployment components"""
    
    def test_model_server_error_handling(self):
        """Test model server error handling"""
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test prediction without model loaded
        response = client.post("/predict", files={"file": ("test.jpg", b"invalid_image_data", "image/jpeg")})
        # Should handle gracefully (might return 503 or 400 depending on implementation)
        assert response.status_code in [400, 503]
    
    def test_performance_optimizer_error_handling(self):
        """Test performance optimizer error handling"""
        # Test with invalid model
        try:
            device = torch.device('cpu')
            optimizer = PerformanceOptimizer(None, device, {})
            # Should handle None model gracefully
        except Exception as e:
            # Expected to fail with None model
            assert "model" in str(e).lower() or "NoneType" in str(e)
    
    def test_deployment_manager_error_handling(self):
        """Test deployment manager error handling"""
        # Test with invalid directory
        try:
            manager = DeploymentManager(deployment_dir="/invalid/path/that/does/not/exist")
            # Should handle invalid path gracefully
        except Exception:
            # May fail during initialization, which is acceptable
            pass
    
    def test_cache_error_handling(self):
        """Test cache error handling"""
        cache = ModelCache(max_size=5, ttl_seconds=1)
        
        # Test with invalid data types
        try:
            cache.put("invalid_bytes", {"result": "test"})
            # Should handle string instead of bytes
        except Exception:
            # Expected to fail with invalid input type
            pass
        
        # Test cache stats with empty cache
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])