#!/usr/bin/env python3
"""
Simple Model Test - Test if your trained model loads correctly
"""

import torch
import os
from pathlib import Path

def test_model_loading():
    """Test if the model file can be loaded."""
    
    print("🧠 Testing Model Loading...")
    print("=" * 40)
    
    # Check if model file exists
    model_paths = [
        "models/best_chest_xray_model.pth",
        "model/best_chest_xray_model.pth",
        "models/chest_xray_model.pth",
        "model/chest_xray_model.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ Model file not found. Looking for:")
        for path in model_paths:
            print(f"   - {path}")
        print("\n💡 Please ensure your .pth file is in one of these locations")
        return False
    
    print(f"✅ Model file found: {model_path}")
    
    # Check file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📁 Model file size: {file_size:.1f} MB")
    
    # Try to load the model
    try:
        print("🔄 Loading model...")
        
        # Load model on CPU first
        model_data = torch.load(model_path, map_location='cpu')
        
        print("✅ Model loaded successfully!")
        
        # Check what's in the model file
        if isinstance(model_data, dict):
            print("📊 Model contents:")
            for key in model_data.keys():
                print(f"   - {key}")
        
        print("\n🎉 Your model is ready to use!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def show_next_steps():
    """Show what to do next."""
    print("\n" + "=" * 40)
    print("🚀 Next Steps:")
    print("=" * 40)
    print("1. Run: python quick_start.py")
    print("2. Or run: python -m uvicorn deployment.model_server:app --port 8000")
    print("3. Open: http://localhost:8000/docs")
    print("4. Upload a chest X-ray image and get predictions!")

if __name__ == "__main__":
    if test_model_loading():
        show_next_steps()
    else:
        print("\n💡 Fix the model file location first, then try again.")