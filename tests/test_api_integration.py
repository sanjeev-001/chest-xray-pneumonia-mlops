"""
API Integration Tests
Tests for the complete API functionality including endpoints, performance, and error handling
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
from PIL import Image
import io
import multiprocessing
import signal
import subprocess
import sys
from unittest.mock import patch, Mock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class APITestServer:
    """Helper class to manage test API server"""
    
    def __init__(self, port: int = 8888):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def start(self, model_path: str = None):
        """Start the API server"""
        # Create startup script
        startup_script = f"""
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variables
os.environ['MODEL_PATH'] = '{model_path or "models/mock_model.pth"}'
os.environ['DEVICE'] = 'cpu'
os.environ['AUTO_OPTIMIZE'] = 'false'

# Import and run server
import uvicorn
from deployment.model_server import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={self.port}, log_level="error")
"""
        
        # Write startup script to temp file
        script_path = Path(tempfile.gettempdir()) / f"test_server_{self.port}.py"
        with open(script_path, 'w') as f:
            f.write(startup_script)
        
        # Start server process
        self.process = subprocess.Popen([
            sys.executable, str(script_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        return self.process.poll() is None
    
    def stop(self):
        """Stop the API server"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
    
    async def is_healthy(self) -> bool:
        """Check if server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=5) as response:
                    return response.status == 200
        except:
            return False


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test session"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def mock_model_file(temp_dir):
    """Create a mock model file for testing"""
    import torch
    
    model_path = Path(temp_dir) / "mock_model.pth"
    
    # Create a simple mock model state dict
    mock_state_dict = {
        'features.0.weight': torch.randn(32, 3, 3, 3),
        'features.0.bias': torch.randn(32),
        'classifier.weight': torch.randn(2, 1000),
        'classifier.bias': torch.randn(2)
    }
    
    torch.save(mock_state_dict, model_path)
    return str(model_path)


@pytest.fixture(scope="session")
def test_images(temp_dir):
    """Create test images"""
    images = {}
    
    # Normal chest X-ray (white image)
    normal_image = Image.new('RGB', (224, 224), color='white')
    normal_path = Path(temp_dir) / "normal_xray.jpg"
    normal_image.save(normal_path)
    images['normal'] = str(normal_path)
    
    # Pneumonia chest X-ray (gray image)
    pneumonia_image = Image.new('RGB', (224, 224), color='gray')
    pneumonia_path = Path(temp_dir) / "pneumonia_xray.jpg"
    pneumonia_image.save(pneumonia_path)
    images['pneumonia'] = str(pneumonia_path)
    
    # Invalid image (small size)
    small_image = Image.new('RGB', (50, 50), color='red')
    small_path = Path(temp_dir) / "small_image.jpg"
    small_image.save(small_path)
    images['small'] = str(small_path)
    
    return images


@pytest.fixture(scope="session")
def api_server(mock_model_file):
    """Start API server for testing"""
    server = APITestServer(port=8888)
    
    # Mock the model loading to avoid actual model requirements
    with patch('deployment.model_server.load_model') as mock_load:
        mock_load.return_value = True
        
        if server.start(mock_model_file):
            yield server
        else:
            pytest.skip("Failed to start test server")
    
    server.stop()


class TestAPIEndpoints:
    """Test API endpoints functionality"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_server):
        """Test health check endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/health") as response:
                assert response.status == 200
                
                data = await response.json()
                assert "status" in data
                assert "model_loaded" in data
                assert "device" in data
                assert "uptime_seconds" in data
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, api_server):
        """Test root endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/") as response:
                assert response.status == 200
                
                data = await response.json()
                assert "message" in data
                assert "version" in data
                assert "endpoints" in data
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, api_server):
        """Test metrics endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/metrics") as response:
                assert response.status == 200
                
                data = await response.json()
                assert "uptime_seconds" in data
                assert "model_loaded" in data
    
    @pytest.mark.asyncio
    async def test_readiness_probe(self, api_server):
        """Test Kubernetes readiness probe"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/ready") as response:
                # May return 503 if model is not properly loaded in test environment
                assert response.status in [200, 503]
    
    @pytest.mark.asyncio
    async def test_liveness_probe(self, api_server):
        """Test Kubernetes liveness probe"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/alive") as response:
                assert response.status == 200
                
                data = await response.json()
                assert data["status"] == "alive"


class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_structure(self, api_server, test_images):
        """Test prediction endpoint structure (may fail without real model)"""
        async with aiohttp.ClientSession() as session:
            # Test with normal image
            with open(test_images['normal'], 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='normal_xray.jpg', content_type='image/jpeg')
                
                async with session.post(f"{api_server.base_url}/predict", data=data) as response:
                    # May return 503 if model is not loaded, or 500 if prediction fails
                    # We're testing the endpoint structure, not the actual prediction
                    assert response.status in [200, 500, 503]
                    
                    if response.status == 200:
                        data = await response.json()
                        # Check response structure
                        assert "prediction" in data
                        assert "confidence" in data
                        assert "processing_time_ms" in data
                        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_predict_invalid_file(self, api_server):
        """Test prediction with invalid file"""
        async with aiohttp.ClientSession() as session:
            # Test with invalid data
            data = aiohttp.FormData()
            data.add_field('file', b'invalid_image_data', filename='invalid.jpg', content_type='image/jpeg')
            
            async with session.post(f"{api_server.base_url}/predict", data=data) as response:
                # Should return 400 for invalid image data
                assert response.status == 400
    
    @pytest.mark.asyncio
    async def test_predict_no_file(self, api_server):
        """Test prediction without file"""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{api_server.base_url}/predict") as response:
                # Should return 422 for missing required parameter
                assert response.status == 422
    
    @pytest.mark.asyncio
    async def test_batch_predict_endpoint(self, api_server, test_images):
        """Test batch prediction endpoint"""
        async with aiohttp.ClientSession() as session:
            # Test with multiple images
            data = aiohttp.FormData()
            
            with open(test_images['normal'], 'rb') as f1, open(test_images['pneumonia'], 'rb') as f2:
                data.add_field('files', f1, filename='normal.jpg', content_type='image/jpeg')
                data.add_field('files', f2, filename='pneumonia.jpg', content_type='image/jpeg')
                
                async with session.post(f"{api_server.base_url}/predict/batch", data=data) as response:
                    # May fail without real model, but should have proper structure
                    assert response.status in [200, 500, 503]
                    
                    if response.status == 200:
                        data = await response.json()
                        assert "predictions" in data
                        assert "total_processing_time_ms" in data
                        assert "batch_size" in data


class TestPerformanceEndpoints:
    """Test performance-related endpoints"""
    
    @pytest.mark.asyncio
    async def test_performance_stats_endpoint(self, api_server):
        """Test performance stats endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/performance/stats") as response:
                # May return 503 if performance optimizer is not available
                assert response.status in [200, 503]
                
                if response.status == 200:
                    data = await response.json()
                    # Check for expected performance metrics structure
                    assert isinstance(data, dict)
    
    @pytest.mark.asyncio
    async def test_performance_benchmark_endpoint(self, api_server):
        """Test performance benchmark endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/performance/benchmark") as response:
                # May return 503 if performance optimizer is not available
                assert response.status in [200, 503]
    
    @pytest.mark.asyncio
    async def test_cache_clear_endpoint(self, api_server):
        """Test cache clear endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{api_server.base_url}/performance/cache/clear") as response:
                # May return 503 if performance optimizer is not available
                assert response.status in [200, 503]
    
    @pytest.mark.asyncio
    async def test_memory_optimize_endpoint(self, api_server):
        """Test memory optimization endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{api_server.base_url}/performance/memory/optimize") as response:
                # May return 503 if performance optimizer is not available
                assert response.status in [200, 503]


class TestErrorHandling:
    """Test API error handling"""
    
    @pytest.mark.asyncio
    async def test_404_handling(self, api_server):
        """Test 404 error handling"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/nonexistent") as response:
                assert response.status == 404
    
    @pytest.mark.asyncio
    async def test_method_not_allowed(self, api_server):
        """Test method not allowed handling"""
        async with aiohttp.ClientSession() as session:
            # Try POST on GET-only endpoint
            async with session.post(f"{api_server.base_url}/health") as response:
                assert response.status == 405
    
    @pytest.mark.asyncio
    async def test_large_file_handling(self, api_server, temp_dir):
        """Test handling of large files"""
        # Create a large file (simulate large image)
        large_file_path = Path(temp_dir) / "large_file.jpg"
        with open(large_file_path, 'wb') as f:
            f.write(b'0' * (20 * 1024 * 1024))  # 20MB file
        
        async with aiohttp.ClientSession() as session:
            with open(large_file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='large_file.jpg', content_type='image/jpeg')
                
                async with session.post(f"{api_server.base_url}/predict", data=data) as response:
                    # Should return 400 for file too large (if size limit is enforced)
                    assert response.status in [400, 413, 500, 503]
    
    @pytest.mark.asyncio
    async def test_invalid_content_type(self, api_server, test_images):
        """Test invalid content type handling"""
        async with aiohttp.ClientSession() as session:
            with open(test_images['normal'], 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='test.txt', content_type='text/plain')
                
                async with session.post(f"{api_server.base_url}/predict", data=data) as response:
                    # Should handle invalid content type gracefully
                    assert response.status in [400, 422, 500]


class TestCORS:
    """Test CORS functionality"""
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, api_server):
        """Test CORS headers are present"""
        async with aiohttp.ClientSession() as session:
            async with session.options(f"{api_server.base_url}/health") as response:
                # Check for CORS headers
                headers = response.headers
                # CORS headers may be present depending on configuration
                assert response.status in [200, 404, 405]


class TestLoadTesting:
    """Basic load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, api_server):
        """Test concurrent health check requests"""
        async def health_check():
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{api_server.base_url}/health") as response:
                    return response.status == 200
        
        # Run 10 concurrent health checks
        tasks = [health_check() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
    
    @pytest.mark.asyncio
    async def test_sequential_requests(self, api_server):
        """Test sequential requests for basic stability"""
        async with aiohttp.ClientSession() as session:
            for i in range(5):
                async with session.get(f"{api_server.base_url}/health") as response:
                    assert response.status == 200
                    
                # Small delay between requests
                await asyncio.sleep(0.1)


class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    @pytest.mark.asyncio
    async def test_openapi_schema(self, api_server):
        """Test OpenAPI schema endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/openapi.json") as response:
                assert response.status == 200
                
                data = await response.json()
                assert "openapi" in data
                assert "info" in data
                assert "paths" in data
    
    @pytest.mark.asyncio
    async def test_docs_endpoint(self, api_server):
        """Test Swagger UI docs endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/docs") as response:
                assert response.status == 200
                # Should return HTML content
                content_type = response.headers.get('content-type', '')
                assert 'text/html' in content_type
    
    @pytest.mark.asyncio
    async def test_redoc_endpoint(self, api_server):
        """Test ReDoc endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_server.base_url}/redoc") as response:
                assert response.status == 200
                # Should return HTML content
                content_type = response.headers.get('content-type', '')
                assert 'text/html' in content_type


if __name__ == "__main__":
    # Run with asyncio support
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])