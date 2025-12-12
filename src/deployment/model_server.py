"""
Medical Model Server for Chest X-Ray Pneumonia Detection
FastAPI-based REST API for serving the trained EfficientNet-B4 model
"""

import os
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
from datetime import datetime
import json

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our model architecture and performance optimizer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.models import ModelFactory
from deployment.performance_optimizer import PerformanceOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_chest_xray_model.pth")
MODEL_ARCHITECTURE = os.getenv("MODEL_ARCHITECTURE", "efficientnet_b4")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for batch predictions"""
    images: List[str] = Field(..., description="List of base64 encoded images")
    return_probabilities: bool = Field(default=True, description="Return class probabilities")
    return_confidence: bool = Field(default=True, description="Return confidence scores")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str = Field(..., description="Predicted class (NORMAL or PNEUMONIA)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float
    batch_size: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    model_architecture: str
    uptime_seconds: float
    memory_usage_mb: Optional[float] = None

class ModelInfo(BaseModel):
    """Model information response"""
    architecture: str
    num_classes: int
    class_names: List[str]
    input_size: List[int]
    model_size_mb: float
    device: str
    loaded_at: str

# Global variables
app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="Medical AI API for detecting pneumonia in chest X-ray images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and preprocessing
model = None
device = None
class_names = ["NORMAL", "PNEUMONIA"]
start_time = time.time()
model_loaded_at = None
performance_optimizer = None

# Image preprocessing pipeline
def get_image_transforms():
    """Get image preprocessing transforms - must match training size (380x380 for EfficientNet-B4)"""
    return transforms.Compose([
        transforms.Resize((380, 380)),  # EfficientNet-B4 optimal size - matches training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_efficientnet_model():
    """Create EfficientNet-B4 model matching the Kaggle training architecture"""
    try:
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        import torch.nn as nn
        
        # Load pre-trained EfficientNet-B4
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # Modify classifier for binary classification (matching Kaggle notebook)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 2)  # NORMAL vs PNEUMONIA
        )
        
        return model
    except ImportError:
        logger.warning("EfficientNet not available, falling back to ModelFactory")
        return ModelFactory.create_model(
            architecture=MODEL_ARCHITECTURE,
            num_classes=2,
            pretrained=False
        )

def load_model():
    """Load the trained EfficientNet-B4 model with performance optimization"""
    global model, device, model_loaded_at, performance_optimizer
    
    try:
        # Setup device
        device = torch.device(DEVICE if DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {device}")
        
        # Create model architecture (EfficientNet-B4 from Kaggle training)
        if MODEL_ARCHITECTURE == "efficientnet_b4":
            model = create_efficientnet_model()
            logger.info("Created EfficientNet-B4 model architecture")
        else:
            model = ModelFactory.create_model(
                architecture=MODEL_ARCHITECTURE,
                num_classes=2,
                pretrained=False
            )
        
        # Load trained weights if model file exists
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading trained weights from {MODEL_PATH}")
            
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Loaded model_state_dict from checkpoint")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    logger.info("Loaded state_dict from checkpoint")
                else:
                    model.load_state_dict(checkpoint)
                    logger.info("Loaded checkpoint as state_dict")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded checkpoint directly as state_dict")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}, using pre-trained weights only")
        
        model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Initialize performance optimizer
        perf_config = {
            'cache_size': int(os.getenv('CACHE_SIZE', '1000')),
            'cache_ttl': int(os.getenv('CACHE_TTL', '3600')),
            'auto_optimize': os.getenv('AUTO_OPTIMIZE', 'true').lower() == 'true',
            'enable_batching': os.getenv('ENABLE_BATCHING', 'false').lower() == 'true',
            'batch_size': int(os.getenv('BATCH_SIZE', '8')),
            'batch_wait_time': float(os.getenv('BATCH_WAIT_TIME', '0.1'))
        }
        
        performance_optimizer = PerformanceOptimizer(model, device, perf_config)
        logger.info("Performance optimizer initialized")
        
        model_loaded_at = datetime.now().isoformat()
        logger.info(f"Model loaded successfully!")
        logger.info(f"  Architecture: {MODEL_ARCHITECTURE}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.exception("Full error traceback:")
        return False

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess image for model inference"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply transforms
        transform = get_image_transforms()
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(device)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def validate_image(file: UploadFile) -> None:
    """Validate uploaded image"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    if file.size and file.size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size: {MAX_IMAGE_SIZE / 1024 / 1024:.1f}MB"
        )

def predict_image(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on preprocessed image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = class_names[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()
            
            # Get all class probabilities
            class_probs = {
                class_names[i]: probabilities[0][i].item() 
                for i in range(len(class_names))
            }
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": class_probs,
                "processing_time_ms": processing_time
            }
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Chest X-Ray Pneumonia Detection API...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")
    else:
        logger.info("API ready to serve predictions")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    # Get memory usage if possible
    memory_usage = None
    try:
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        model_architecture=MODEL_ARCHITECTURE,
        uptime_seconds=uptime,
        memory_usage_mb=memory_usage
    )

# Model info endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate model size
    model_size = 0
    if os.path.exists(MODEL_PATH):
        model_size = os.path.getsize(MODEL_PATH) / 1024 / 1024  # MB
    
    return ModelInfo(
        architecture=MODEL_ARCHITECTURE,
        num_classes=len(class_names),
        class_names=class_names,
        input_size=[380, 380, 3],  # Matches EfficientNet-B4 training size
        model_size_mb=model_size,
        device=str(device),
        loaded_at=model_loaded_at or "unknown"
    )

# Single image prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_single_image(
    file: UploadFile = File(..., description="Chest X-ray image file"),
    return_probabilities: bool = True,
    return_confidence: bool = True
):
    """Predict pneumonia from a single chest X-ray image"""
    
    # Validate image
    validate_image(file)
    
    # Read image data
    image_data = await file.read()
    
    # Preprocess image
    image_tensor = preprocess_image(image_data)
    
    # Make prediction
    result = predict_image(image_tensor)
    
    # Prepare response
    response = PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"] if return_probabilities else None,
        processing_time_ms=result["processing_time_ms"],
        model_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
    
    return response

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_images(
    files: List[UploadFile] = File(..., description="List of chest X-ray image files"),
    return_probabilities: bool = True
):
    """Predict pneumonia from multiple chest X-ray images"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum batch size is 10 images")
    
    batch_start_time = time.time()
    predictions = []
    
    for file in files:
        try:
            # Validate and process each image
            validate_image(file)
            image_data = await file.read()
            image_tensor = preprocess_image(image_data)
            result = predict_image(image_tensor)
            
            # Create response for this image
            prediction_response = PredictionResponse(
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"] if return_probabilities else None,
                processing_time_ms=result["processing_time_ms"],
                model_version="1.0.0",
                timestamp=datetime.now().isoformat()
            )
            
            predictions.append(prediction_response)
            
        except Exception as e:
            logger.error(f"Failed to process image {file.filename}: {str(e)}")
            # Add error response for this image
            error_response = PredictionResponse(
                prediction="ERROR",
                confidence=0.0,
                probabilities=None,
                processing_time_ms=0.0,
                model_version="1.0.0",
                timestamp=datetime.now().isoformat()
            )
            predictions.append(error_response)
    
    total_processing_time = (time.time() - batch_start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processing_time_ms=total_processing_time,
        batch_size=len(files)
    )

# Readiness probe
@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}

# Liveness probe
@app.get("/alive")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}

# Performance metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Comprehensive performance metrics endpoint"""
    uptime = time.time() - start_time
    
    base_metrics = {
        "uptime_seconds": uptime,
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "model_architecture": MODEL_ARCHITECTURE
    }
    
    # Add performance optimizer metrics if available
    if performance_optimizer:
        perf_stats = performance_optimizer.get_performance_stats()
        base_metrics.update(perf_stats)
    else:
        # Fallback to basic metrics
        try:
            import psutil
            process = psutil.Process()
            base_metrics["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
            base_metrics["cpu_percent"] = process.cpu_percent()
        except ImportError:
            pass
    
    return base_metrics

# Performance management endpoints
@app.get("/performance/stats")
async def get_performance_stats():
    """Get detailed performance statistics"""
    if not performance_optimizer:
        raise HTTPException(status_code=503, detail="Performance optimizer not available")
    
    return performance_optimizer.get_performance_stats()

@app.get("/performance/benchmark")
async def run_benchmark():
    """Run performance benchmark"""
    if not performance_optimizer:
        raise HTTPException(status_code=503, detail="Performance optimizer not available")
    
    try:
        benchmark_results = performance_optimizer.benchmark()
        return benchmark_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@app.post("/performance/cache/clear")
async def clear_cache():
    """Clear prediction cache"""
    if not performance_optimizer:
        raise HTTPException(status_code=503, detail="Performance optimizer not available")
    
    performance_optimizer.clear_cache()
    return {"message": "Cache cleared successfully"}

@app.post("/performance/memory/optimize")
async def optimize_memory():
    """Optimize memory usage"""
    if not performance_optimizer:
        raise HTTPException(status_code=503, detail="Performance optimizer not available")
    
    performance_optimizer.optimize_memory()
    return {"message": "Memory optimization completed"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chest X-Ray Pneumonia Detection API",
        "version": "1.0.0",
        "status": "healthy" if model is not None else "model not loaded",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )