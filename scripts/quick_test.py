#!/usr/bin/env python3
"""
Quick test script to bypass browser issues and test the API directly
"""

import requests
import os

def test_prediction():
    """Test the prediction API directly."""
    
    print("🧪 Testing Pneumonia Detection API")
    print("=" * 40)
    
    # Test health first
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code == 200:
            print("✅ Server is running!")
            print(f"Health: {health.json()}")
        else:
            print("❌ Server health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the server is running with: python quick_start.py")
        return
    
    # Look for the image file
    image_file = "person936_virus_1598.jpeg"
    
    if not os.path.exists(image_file):
        print(f"❌ Image file '{image_file}' not found")
        print("💡 Make sure the image is in the current directory")
        return
    
    print(f"📁 Found image: {image_file}")
    
    # Test prediction
    try:
        with open(image_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                "http://localhost:8000/predict",
                files=files,
                params={
                    'return_probabilities': True,
                    'return_confidence': True
                },
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("\n🎉 SUCCESS! Prediction Results:")
            print("=" * 40)
            print(f"📊 Prediction: {result.get('prediction', 'Unknown')}")
            print(f"🎯 Confidence: {result.get('confidence', 0):.3f}")
            print(f"⏱️  Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
            
            if 'probabilities' in result:
                print("\n📈 Detailed Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    print(f"   {class_name}: {prob:.3f} ({prob*100:.1f}%)")
            
            # Interpret results
            print("\n🏥 Medical Interpretation:")
            if result.get('prediction') == 'PNEUMONIA':
                print("   ⚠️  PNEUMONIA detected")
                print(f"   Confidence: {result.get('confidence', 0)*100:.1f}%")
            else:
                print("   ✅ NORMAL chest X-ray")
                print(f"   Confidence: {result.get('confidence', 0)*100:.1f}%")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    test_prediction()