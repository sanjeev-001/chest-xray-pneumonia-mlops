#!/usr/bin/env python3
"""
Simple test script to test image upload to the pneumonia detection API
"""

import requests
import os
from pathlib import Path

def test_api_upload():
    """Test uploading an image to the API."""
    
    print("🧪 Testing Pneumonia Detection API Upload")
    print("=" * 50)
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Check if API is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API is running!")
        else:
            print("❌ API health check failed")
            return
    except:
        print("❌ Cannot connect to API. Make sure it's running on port 8000")
        return
    
    # Look for any image files in common locations
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
    search_paths = [
        ".",
        "data",
        "images", 
        "test_images",
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Pictures"),
        os.path.expanduser("~/Desktop")
    ]
    
    found_images = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    found_images.append(os.path.join(search_path, file))
                    if len(found_images) >= 3:  # Limit to first 3 found
                        break
    
    if found_images:
        print(f"📁 Found {len(found_images)} image(s) to test:")
        for img in found_images[:3]:
            print(f"   - {img}")
        
        # Test with the first image
        test_image = found_images[0]
        print(f"\n🔍 Testing with: {test_image}")
        
        try:
            with open(test_image, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files, timeout=30)
                
            if response.status_code == 200:
                result = response.json()
                print("🎉 SUCCESS! Prediction received:")
                print(f"   Prediction: {result.get('prediction', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.3f}")
                print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")
                
                if 'probabilities' in result:
                    print("   Probabilities:")
                    for class_name, prob in result['probabilities'].items():
                        print(f"     {class_name}: {prob:.3f}")
                        
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing image: {e}")
    
    else:
        print("❌ No image files found to test")
        print("💡 To test manually:")
        print("   1. Put a chest X-ray image in this folder")
        print("   2. Run this script again")
        print("   3. Or use the web interface at http://localhost:8000/docs")

def create_test_image():
    """Create a simple test image if none exists."""
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image (not a real X-ray, just for testing)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save("test_image.jpg")
        print("✅ Created test_image.jpg for testing")
        return "test_image.jpg"
    except ImportError:
        print("⚠️ PIL not available, cannot create test image")
        return None

if __name__ == "__main__":
    test_api_upload()
    
    # If no images found, offer to create a test image
    if not any(os.path.exists(path) for path in ["test_image.jpg"]):
        print("\n💡 Want to create a test image? (y/n): ", end="")
        if input().lower().startswith('y'):
            test_img = create_test_image()
            if test_img:
                print("Now run the script again to test with the created image!")