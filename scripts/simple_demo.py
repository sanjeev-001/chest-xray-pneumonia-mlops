#!/usr/bin/env python3
"""
Simple Demo Script - Test the system with a sample prediction
"""

import requests
import json
import time
from pathlib import Path

def test_system():
    """Test the system with a simple health check and demo prediction."""
    
    print("🏥 Chest X-Ray Pneumonia Detection - Simple Demo")
    print("=" * 50)
    
    # Check if system is running
    print("🔍 Checking system health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ System is healthy!")
            print(f"   Model loaded: {health_data.get('model_loaded', 'Unknown')}")
            print(f"   Model version: {health_data.get('model_version', 'Unknown')}")
        else:
            print("❌ System health check failed")
            return False
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to system. Please start it first with:")
        print("   python quick_start.py")
        return False
    
    # Get model info
    print("\n📊 Getting model information...")
    try:
        response = requests.get("http://localhost:8000/model/info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print("✅ Model information:")
            print(f"   Architecture: {model_info.get('architecture', 'Unknown')}")
            print(f"   Accuracy: {model_info.get('accuracy', 'Unknown')}")
            print(f"   Classes: {model_info.get('classes', 'Unknown')}")
    except:
        print("⚠️ Could not get model information")
    
    # Demo prediction with sample data
    print("\n🧪 Testing prediction capability...")
    
    # Create a simple test request (without actual image)
    try:
        # Test the prediction endpoint structure
        response = requests.post("http://localhost:8000/predict", timeout=10)
        # We expect this to fail with 422 (missing file), which means the endpoint works
        if response.status_code == 422:
            print("✅ Prediction endpoint is working!")
            print("   Ready to accept image files")
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
    except:
        print("❌ Prediction endpoint test failed")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n💡 How to use:")
    print("1. Go to: http://localhost:8080")
    print("2. Drag and drop a chest X-ray image")
    print("3. Get instant pneumonia detection results!")
    print("\n📋 API Documentation: http://localhost:8000/docs")
    
    return True

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print("\n1. Web Interface (Easiest):")
    print("   - Open: http://localhost:8080")
    print("   - Drag & drop X-ray image")
    
    print("\n2. cURL Command:")
    print('   curl -X POST "http://localhost:8000/predict" \\')
    print('        -H "Content-Type: multipart/form-data" \\')
    print('        -F "file=@your_xray.jpg"')
    
    print("\n3. Python Code:")
    print("""
   import requests
   
   with open('chest_xray.jpg', 'rb') as f:
       files = {'file': f}
       response = requests.post('http://localhost:8000/predict', files=files)
       result = response.json()
       print(f"Prediction: {result['prediction']}")
       print(f"Confidence: {result['confidence']:.3f}")
   """)

if __name__ == "__main__":
    if test_system():
        show_usage_examples()