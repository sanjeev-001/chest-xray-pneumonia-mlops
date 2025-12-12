#!/usr/bin/env python3
"""
Test Client for Chest X-Ray Pneumonia Detection API
Simple client to test the deployed model server
"""

import requests
import json
import time
from pathlib import Path
import argparse
import sys

def test_health_check(base_url: str):
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Device: {health_data['device']}")
            print(f"   Architecture: {health_data['model_architecture']}")
            print(f"   Uptime: {health_data['uptime_seconds']:.1f}s")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_model_info(base_url: str):
    """Test model info endpoint"""
    print("\n📊 Testing model info...")
    
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            info_data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Architecture: {info_data['architecture']}")
            print(f"   Classes: {info_data['class_names']}")
            print(f"   Input size: {info_data['input_size']}")
            print(f"   Model size: {info_data['model_size_mb']:.1f}MB")
            print(f"   Device: {info_data['device']}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {str(e)}")
        return False

def test_prediction(base_url: str, image_path: str):
    """Test single image prediction"""
    print(f"\n🔮 Testing prediction with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/predict", files=files)
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                pred_data = response.json()
                print(f"✅ Prediction successful")
                print(f"   Prediction: {pred_data['prediction']}")
                print(f"   Confidence: {pred_data['confidence']:.4f}")
                print(f"   Processing time: {pred_data['processing_time_ms']:.1f}ms")
                print(f"   Total request time: {request_time:.1f}ms")
                
                if pred_data.get('probabilities'):
                    print(f"   Probabilities:")
                    for class_name, prob in pred_data['probabilities'].items():
                        print(f"     {class_name}: {prob:.4f}")
                
                return True
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

def test_batch_prediction(base_url: str, image_paths: list):
    """Test batch prediction"""
    print(f"\n📦 Testing batch prediction with {len(image_paths)} images...")
    
    # Check all files exist
    for image_path in image_paths:
        if not Path(image_path).exists():
            print(f"❌ Image file not found: {image_path}")
            return False
    
    try:
        files = []
        for image_path in image_paths:
            files.append(('files', (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg')))
        
        start_time = time.time()
        response = requests.post(f"{base_url}/predict/batch", files=files)
        request_time = (time.time() - start_time) * 1000
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        if response.status_code == 200:
            batch_data = response.json()
            print(f"✅ Batch prediction successful")
            print(f"   Batch size: {batch_data['batch_size']}")
            print(f"   Total processing time: {batch_data['total_processing_time_ms']:.1f}ms")
            print(f"   Total request time: {request_time:.1f}ms")
            print(f"   Average per image: {batch_data['total_processing_time_ms']/batch_data['batch_size']:.1f}ms")
            
            print(f"   Results:")
            for i, pred in enumerate(batch_data['predictions']):
                print(f"     Image {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.4f})")
            
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        return False

def test_metrics(base_url: str):
    """Test metrics endpoint"""
    print("\n📈 Testing metrics...")
    
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            metrics_data = response.json()
            print(f"✅ Metrics retrieved")
            print(f"   Uptime: {metrics_data['uptime_seconds']:.1f}s")
            print(f"   Model loaded: {metrics_data['model_loaded']}")
            if 'memory_usage_mb' in metrics_data:
                print(f"   Memory usage: {metrics_data['memory_usage_mb']:.1f}MB")
            if 'cpu_percent' in metrics_data:
                print(f"   CPU usage: {metrics_data['cpu_percent']:.1f}%")
            return True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Chest X-Ray API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--batch", nargs="+", help="Paths to multiple test images")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')
    
    print(f"🚀 Testing Chest X-Ray Pneumonia Detection API")
    print(f"   Base URL: {base_url}")
    print("=" * 60)
    
    # Test health check
    if not test_health_check(base_url):
        print("❌ API is not healthy. Stopping tests.")
        sys.exit(1)
    
    # Test model info
    test_model_info(base_url)
    
    # Test metrics
    test_metrics(base_url)
    
    # Test predictions if image provided
    if args.image:
        test_prediction(base_url, args.image)
    
    # Test batch predictions if multiple images provided
    if args.batch:
        test_batch_prediction(base_url, args.batch)
    
    # If no specific tests requested, show usage
    if not args.image and not args.batch and not args.all:
        print("\n💡 Usage examples:")
        print(f"   python {sys.argv[0]} --image path/to/xray.jpg")
        print(f"   python {sys.argv[0]} --batch path/to/xray1.jpg path/to/xray2.jpg")
        print(f"   python {sys.argv[0]} --url http://your-api-server:8000 --image test.jpg")
    
    print("\n✅ API testing completed!")

if __name__ == "__main__":
    main()