#!/usr/bin/env python3
"""
Kaggle Deployment Script - Run the MLOps system on Kaggle with GPU
"""

import os
import subprocess
import sys

def setup_kaggle_environment():
    """Set up the environment for Kaggle."""
    print("🚀 Setting up Kaggle Environment for MLOps System")
    print("=" * 50)
    
    # Install required packages
    packages = [
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "pillow",
        "torch",
        "torchvision",
        "numpy",
        "requests"
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      capture_output=True)
    
    print("✅ Packages installed!")

def start_model_server_kaggle():
    """Start the model server on Kaggle."""
    print("\n🧠 Starting Model Server on Kaggle...")
    
    # Check if GPU is available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Set environment variables for GPU
    os.environ["DEVICE"] = device
    
    # Start the server
    print("\n🌐 Starting FastAPI server...")
    print("📋 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    
    # Import and start the server
    try:
        from deployment.model_server import app
        import uvicorn
        
        # Start server
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except ImportError:
        print("❌ Model server not found. Creating simple server...")
        create_simple_server()

def create_simple_server():
    """Create a simple server for Kaggle."""
    
    server_code = '''
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import time

app = FastAPI(title="Chest X-Ray Pneumonia Detection", version="1.0.0")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model loading (you'll need to adjust the path)
try:
    model = torch.load("models/best_chest_xray_model.pth", map_location=device)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"message": "Chest X-Ray Pneumonia Detection API", "status": "running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"}
        )
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        processing_time = (time.time() - start_time) * 1000
        
        # Map prediction to class name
        class_names = ["NORMAL", "PNEUMONIA"]
        prediction = class_names[predicted_class]
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "NORMAL": probabilities[0][0].item(),
                "PNEUMONIA": probabilities[0][1].item()
            },
            "processing_time_ms": processing_time,
            "device": str(device)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    # Write the server code to a file
    with open("kaggle_server.py", "w") as f:
        f.write(server_code)
    
    print("📝 Created kaggle_server.py")
    print("🚀 Run: python kaggle_server.py")

def main():
    """Main function for Kaggle deployment."""
    setup_kaggle_environment()
    
    # Check if we're in Kaggle environment
    if "/kaggle" in os.getcwd():
        print("🎯 Detected Kaggle environment!")
        print("💡 GPU acceleration available!")
    
    start_model_server_kaggle()

if __name__ == "__main__":
    main()