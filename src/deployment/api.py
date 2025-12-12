"""
Deployment API for Chest X-Ray Pneumonia Detection
Fully corrected to match training architecture and preprocessing.
"""

import os
import io
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
MODEL_PATH = "models/best_chest_xray_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
model_loaded_at = None

# -------------------------------------------------------
# RESPONSE MODELS
# -------------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    timestamp: str
    model_version: str = "1.0.0"

# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------
def get_transform():
    """Matches EXACT training preprocessing."""
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Training size (VERY IMPORTANT)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = get_transform()(img)
        return tensor.unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

# -------------------------------------------------------
# MODEL LOADING (CORRECTED)
# -------------------------------------------------------
def load_model():
    global model, model_loaded_at

    logger.info("Loading EfficientNet-B4 model with CORRECT training architecture...")

    # Base EfficientNet-B4 with ImageNet pretrained backbone
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

    # Get feature count
    num_features = model.classifier[1].in_features  # 1792

    # EXACT classifier used in training
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, 2)
    )

    # Load saved weights
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]  # Training saved full checkpoint

    # STRICT load to ensure perfect match
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    model_loaded_at = datetime.now().isoformat()
    logger.info("Model loaded successfully.")

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------
def predict(image_tensor: torch.Tensor):
    start = time.time()

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    # Convert to normal dictionary
    prob_dict = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    pred_idx = int(torch.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    return {
        "prediction": pred_label,
        "confidence": float(probs[pred_idx]),
        "probabilities": prob_dict,
        "processing_time_ms": (time.time() - start) * 1000
    }

# -------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------
app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    return {
        "status": "OK" if model is not None else "NOT READY",
        "device": str(DEVICE),
        "model_loaded_at": model_loaded_at
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    result = predict(image_tensor)

    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        processing_time_ms=result["processing_time_ms"],
        timestamp=datetime.now().isoformat()
    )

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
