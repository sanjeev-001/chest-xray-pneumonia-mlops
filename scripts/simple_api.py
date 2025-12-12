#!/usr/bin/env python3
"""
Simple Flask API for chest X-ray pneumonia detection
Uses the exact architecture that matches your checkpoint
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model variable
model = None
device = torch.device("cpu")

def load_model():
    """Load the trained model with correct architecture"""
    global model
    
    # Create EfficientNet-B4 with exact checkpoint architecture
    model = efficientnet_b4(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),  # 0
        nn.Linear(1792, 512),             # 1 - matches classifier.1 in checkpoint
        nn.ReLU(inplace=True),            # 2
        nn.BatchNorm1d(512),              # 3 - matches classifier.3 in checkpoint  
        nn.Dropout(p=0.2, inplace=True), # 4
        nn.Linear(512, 2)                 # 5 - matches classifier.5 in checkpoint
    )
    
    # Load checkpoint
    checkpoint = torch.load("models/best_chest_xray_model.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model

def preprocess_image(image_bytes):
    """Preprocess image for model"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((380, 380)),  # EfficientNet-B4 optimal size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict pneumonia from chest X-ray"""
    try:
        # Get image from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        image_bytes = file.read()
        
        # Preprocess
        image_tensor = preprocess_image(image_bytes)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            class_names = ["NORMAL", "PNEUMONIA"]  # Must match training: NORMAL=0, PNEUMONIA=1
            predicted_class = class_names[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()
            
            # Get all probabilities
            class_probs = {
                class_names[i]: probabilities[0][i].item() 
                for i in range(len(class_names))
            }
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': class_probs,
                'raw_outputs': outputs.numpy().tolist()
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Starting API on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)