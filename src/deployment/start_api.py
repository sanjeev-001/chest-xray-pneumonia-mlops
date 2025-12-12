#!/usr/bin/env python3
"""
Startup Script for Chest X-Ray Pneumonia Detection API
Easy way to start the model server with proper configuration
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def setup_logging(log_level: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_model_file(model_path: str) -> bool:
    """Check if model file exists"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please download your trained model from Kaggle and place it at:")
        print(f"   {model_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Start Chest X-Ray Pneumonia Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", default="models/best_chest_xray_model.pth", help="Path to model file")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🚀 Starting Chest X-Ray Pneumonia Detection API")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print("=" * 60)
    
    # Check model file
    if not check_model_file(args.model_path):
        sys.exit(1)
    
    # Set environment variables
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["DEVICE"] = args.device
    
    # Import and start server
    try:
        import uvicorn
        from model_server import app
        
        print("🎯 Starting server...")
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"🔍 Health Check: http://{args.host}:{args.port}/health")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        uvicorn.run(
            "model_server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()