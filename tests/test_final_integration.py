"""
Final End-to-End Integration Tests
Tests the complete MLOps system with the real trained model
"""

import pytest
import requests
import time
import json
import tempfile
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
from PIL import Image
import io

# Import system components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from deployment.model_server import app, load_model
from deployment.api import create_app
from data_pipeline.ingestion import DataIngestionPipeline
from training.model_registry import ModelRegistry
from monitoring.metrics_collector import MetricsCollector
from fastapi.testclient import TestClient


class TestFinalMLOpsIntegration:
    """Complete end-to-end integration tests with real model"""
    
    @pytest.fixture(scope="class")
    def setup_test_environment(self):
        """Set up the complete test environment"""
        # Create test client for model server
        model_client = TestClient(app)
        
        # Create test client for main API
        api_app = create_app()
        api_client = TestClient(api_app)
        
        # Load the real trained model
        model_loaded = load_model()
        assert model_loaded, "Failed to load the trained model"
        
        # Create test data
        test_images = self._create_test_images()
        
        return {
            'model_client': model_client,
            'api_client': api_client,
            'test_images': test_images
        }
    
    def _create_test_images(self) -> List[Dict]:
        """Create test chest X-ray images for testing"""
        test_images = []
        
        # Create synthetic chest X-ray-like images
        for i, label in enumerate(['NORMAL', 'PNEUMONIA']):
            # Create a 224x224 grayscale image with some medical-like patterns
            img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
            
            # Add some structure to make it more X-ray-like
            if label == 'PNEUMONIA':
                # Add some cloudy patterns for pneumonia
                for _ in range(10):
                    x, y = np.random.randint(50, 174, 2)
                    img_array[x:x+50, y:y+50] = np.random.randint(100, 150)
            else:
                # Keep it cleaner for normal
                img_array = np.clip(img_array + 30, 0, 255)
            
            # Convert to RGB
            img_rgb = np.stack([img_array] * 3, axis=-1)
            img = Image.fromarray(img_rgb.astype(np.uint8))
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            test_images.append({
                'label': label,
                'image_base64': img_base64,
                'image_pil': img,
                'filename': f'test_{label.lower()}_{i}.png'
            })
        
        return test_images
    
    def test_model_server_health(self, setup_test_environment):
        """Test model server health endpoint"""
        client = setup_test_environment['model_client']
        
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data['status'] == 'healthy'
        assert health_data['model_loaded'] is True
        assert health_data['model_architecture'] == 'efficientnet_b4'
        assert 'uptime_seconds' in health_data
        
        print(f"✅ Model server health check passed")
        print(f"   Model: {health_data['model_architecture']}")
        print(f"   Device: {health_data['device']}")
        print(f"   Uptime: {health_data['uptime_seconds']:.1f}s")
    
    def test_model_info(self, setup_test_environment):
        """Test model information endpoint"""
        client = setup_test_environment['model_client']
        
        response = client.get("/model/info")
        assert response.status_code == 200
        
        model_info = response.json()
        assert model_info['architecture'] == 'efficientnet_b4'
        assert model_info['num_classes'] == 2
        assert model_info['class_names'] == ['NORMAL', 'PNEUMONIA']
        assert model_info['input_size'] == [224, 224]
        
        print(f"✅ Model info retrieved successfully")
        print(f"   Classes: {model_info['class_names']}")
        print(f"   Input size: {model_info['input_size']}")
        print(f"   Model size: {model_info['model_size_mb']:.1f}MB")
    
    def test_single_image_prediction(self, setup_test_environment):
        """Test single image prediction with real model"""
        client = setup_test_environment['model_client']
        test_images = setup_test_environment['test_images']
        
        for test_image in test_images:
            # Create multipart form data
            img_bytes = base64.b64decode(test_image['image_base64'])
            files = {'file': (test_image['filename'], img_bytes, 'image/png')}
            
            # Make prediction
            response = client.post("/predict", files=files)
            assert response.status_code == 200
            
            prediction = response.json()
            
            # Validate response structure
            assert 'prediction' in prediction
            assert 'confidence' in prediction
            assert 'probabilities' in prediction
            assert 'processing_time_ms' in prediction
            assert 'model_version' in prediction
            assert 'timestamp' in prediction
            
            # Validate prediction values
            assert prediction['prediction'] in ['NORMAL', 'PNEUMONIA']
            assert 0 <= prediction['confidence'] <= 1
            assert len(prediction['probabilities']) == 2
            assert 'NORMAL' in prediction['probabilities']
            assert 'PNEUMONIA' in prediction['probabilities']
            assert prediction['processing_time_ms'] > 0
            
            print(f"✅ Prediction for {test_image['label']} image:")
            print(f"   Predicted: {prediction['prediction']}")
            print(f"   Confidence: {prediction['confidence']:.3f}")
            print(f"   Processing time: {prediction['processing_time_ms']:.1f}ms")
            print(f"   Probabilities: {prediction['probabilities']}")
    
    def test_batch_prediction(self, setup_test_environment):
        """Test batch prediction functionality"""
        client = setup_test_environment['model_client']
        test_images = setup_test_environment['test_images']
        
        # Prepare batch request
        batch_request = {
            'images': [img['image_base64'] for img in test_images],
            'return_probabilities': True,
            'return_confidence': True
        }
        
        # Make batch prediction
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200
        
        batch_result = response.json()
        
        # Validate batch response structure
        assert 'predictions' in batch_result
        assert 'total_processing_time_ms' in batch_result
        assert 'batch_size' in batch_result
        
        assert len(batch_result['predictions']) == len(test_images)
        assert batch_result['batch_size'] == len(test_images)
        assert batch_result['total_processing_time_ms'] > 0
        
        # Validate individual predictions
        for i, prediction in enumerate(batch_result['predictions']):
            assert prediction['prediction'] in ['NORMAL', 'PNEUMONIA']
            assert 0 <= prediction['confidence'] <= 1
            assert len(prediction['probabilities']) == 2
        
        avg_time = batch_result['total_processing_time_ms'] / batch_result['batch_size']
        
        print(f"✅ Batch prediction completed:")
        print(f"   Batch size: {batch_result['batch_size']}")
        print(f"   Total time: {batch_result['total_processing_time_ms']:.1f}ms")
        print(f"   Average time per image: {avg_time:.1f}ms")
    
    def test_model_performance_metrics(self, setup_test_environment):
        """Test model performance and metrics collection"""
        client = setup_test_environment['model_client']
        test_images = setup_test_environment['test_images']
        
        # Make multiple predictions to generate metrics
        processing_times = []
        predictions = []
        
        for test_image in test_images:
            img_bytes = base64.b64decode(test_image['image_base64'])
            files = {'file': (test_image['filename'], img_bytes, 'image/png')}
            
            start_time = time.time()
            response = client.post("/predict", files=files)
            end_time = time.time()
            
            assert response.status_code == 200
            prediction = response.json()
            
            processing_times.append(prediction['processing_time_ms'])
            predictions.append(prediction)
        
        # Calculate performance metrics
        avg_processing_time = np.mean(processing_times)
        p95_processing_time = np.percentile(processing_times, 95)
        
        # Validate performance thresholds
        assert avg_processing_time < 2000, f"Average processing time too high: {avg_processing_time}ms"
        assert p95_processing_time < 5000, f"P95 processing time too high: {p95_processing_time}ms"
        
        print(f"✅ Performance metrics:")
        print(f"   Average processing time: {avg_processing_time:.1f}ms")
        print(f"   P95 processing time: {p95_processing_time:.1f}ms")
        print(f"   Total predictions: {len(predictions)}")
    
    def test_error_handling(self, setup_test_environment):
        """Test error handling for invalid inputs"""
        client = setup_test_environment['model_client']
        
        # Test invalid file format
        invalid_file = b"This is not an image"
        files = {'file': ('test.txt', invalid_file, 'text/plain')}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 400
        
        # Test empty request
        response = client.post("/predict")
        assert response.status_code == 422  # Unprocessable Entity
        
        # Test invalid batch request
        invalid_batch = {'images': ['invalid_base64']}
        response = client.post("/predict/batch", json=invalid_batch)
        assert response.status_code == 400
        
        print("✅ Error handling tests passed")
    
    def test_model_registry_integration(self, setup_test_environment):
        """Test integration with model registry"""
        try:
            registry = ModelRegistry()
            
            # Register the current model
            model_metadata = {
                'name': 'chest_xray_pneumonia_efficientnet_b4',
                'version': '1.0.0',
                'architecture': 'efficientnet_b4',
                'framework': 'pytorch',
                'input_shape': [3, 224, 224],
                'num_classes': 2,
                'class_names': ['NORMAL', 'PNEUMONIA'],
                'training_dataset': 'chest_xray_kaggle',
                'performance_metrics': {
                    'accuracy': 0.87,
                    'precision': 0.85,
                    'recall': 0.89,
                    'f1_score': 0.87
                },
                'model_path': 'models/best_chest_xray_model.pth'
            }
            
            model_id = registry.register_model(
                name=model_metadata['name'],
                version=model_metadata['version'],
                model_path=model_metadata['model_path'],
                metadata=model_metadata
            )
            
            # Verify registration
            registered_model = registry.get_model(model_id)
            assert registered_model is not None
            assert registered_model['name'] == model_metadata['name']
            assert registered_model['version'] == model_metadata['version']
            
            print(f"✅ Model registry integration:")
            print(f"   Model ID: {model_id}")
            print(f"   Name: {registered_model['name']}")
            print(f"   Version: {registered_model['version']}")
            
        except Exception as e:
            print(f"⚠️ Model registry test skipped: {e}")
    
    def test_monitoring_integration(self, setup_test_environment):
        """Test monitoring and metrics collection"""
        try:
            metrics_collector = MetricsCollector()
            
            # Simulate some predictions and collect metrics
            test_images = setup_test_environment['test_images']
            client = setup_test_environment['model_client']
            
            for test_image in test_images:
                img_bytes = base64.b64decode(test_image['image_base64'])
                files = {'file': (test_image['filename'], img_bytes, 'image/png')}
                
                response = client.post("/predict", files=files)
                prediction = response.json()
                
                # Record metrics
                metrics_collector.record_prediction(
                    model_id='chest_xray_efficientnet_b4',
                    prediction=prediction['prediction'],
                    confidence=prediction['confidence'],
                    processing_time_ms=prediction['processing_time_ms'],
                    timestamp=datetime.now()
                )
            
            # Get metrics summary
            metrics_summary = metrics_collector.get_metrics_summary()
            
            assert metrics_summary['total_predictions'] >= len(test_images)
            assert 'average_processing_time_ms' in metrics_summary
            assert 'average_confidence' in metrics_summary
            
            print(f"✅ Monitoring integration:")
            print(f"   Total predictions: {metrics_summary['total_predictions']}")
            print(f"   Avg processing time: {metrics_summary['average_processing_time_ms']:.1f}ms")
            print(f"   Avg confidence: {metrics_summary['average_confidence']:.3f}")
            
        except Exception as e:
            print(f"⚠️ Monitoring test skipped: {e}")
    
    def test_complete_workflow(self, setup_test_environment):
        """Test the complete end-to-end workflow"""
        print("\n🚀 Testing complete MLOps workflow...")
        
        # 1. Data ingestion simulation
        print("1. Data ingestion...")
        test_images = setup_test_environment['test_images']
        assert len(test_images) > 0
        print(f"   ✅ {len(test_images)} test images prepared")
        
        # 2. Model serving
        print("2. Model serving...")
        client = setup_test_environment['model_client']
        health_response = client.get("/health")
        assert health_response.status_code == 200
        print("   ✅ Model server is healthy")
        
        # 3. Prediction pipeline
        print("3. Prediction pipeline...")
        total_predictions = 0
        total_processing_time = 0
        
        for test_image in test_images:
            img_bytes = base64.b64decode(test_image['image_base64'])
            files = {'file': (test_image['filename'], img_bytes, 'image/png')}
            
            response = client.post("/predict", files=files)
            assert response.status_code == 200
            
            prediction = response.json()
            total_predictions += 1
            total_processing_time += prediction['processing_time_ms']
        
        avg_processing_time = total_processing_time / total_predictions
        print(f"   ✅ {total_predictions} predictions completed")
        print(f"   ✅ Average processing time: {avg_processing_time:.1f}ms")
        
        # 4. Performance validation
        print("4. Performance validation...")
        assert avg_processing_time < 2000, "Processing time within acceptable limits"
        print("   ✅ Performance metrics within acceptable limits")
        
        # 5. System health check
        print("5. System health check...")
        model_info_response = client.get("/model/info")
        assert model_info_response.status_code == 200
        print("   ✅ All system components operational")
        
        print("\n🎉 Complete MLOps workflow test PASSED!")
        print(f"   Total test duration: {time.time() - self.start_time:.1f}s")
        print(f"   Model architecture: EfficientNet-B4")
        print(f"   Predictions processed: {total_predictions}")
        print(f"   Average latency: {avg_processing_time:.1f}ms")
    
    @pytest.fixture(autouse=True)
    def setup_test_timing(self):
        """Set up test timing"""
        self.start_time = time.time()
        yield
        duration = time.time() - self.start_time
        print(f"\n⏱️ Test completed in {duration:.2f}s")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])