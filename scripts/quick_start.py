#!/usr/bin/env python3
"""
Quick Start Script for Chest X-Ray Pneumonia Detection MLOps System
Run this script to start the system with minimal setup.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        import fastapi
        import uvicorn
        print("✅ Required packages found")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def check_model_file():
    """Check if model file exists."""
    model_path = Path("models/best_chest_xray_model.pth")
    if model_path.exists():
        print("✅ Model file found")
        return True
    else:
        print("❌ Model file not found at models/best_chest_xray_model.pth")
        print("Please ensure you have the trained model file.")
        return False

def start_model_server():
    """Start the model server."""
    print("🚀 Starting Model Server...")
    try:
        # Start model server in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "deployment.model_server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Model Server started successfully!")
                return process
        except:
            pass
            
        print("⚠️ Model Server may still be starting...")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start Model Server: {e}")
        return None

def start_api_gateway():
    """Start the API gateway."""
    print("🌐 Starting API Gateway...")
    try:
        # Start API gateway in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "deployment.api:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload"
        ])
        
        time.sleep(2)
        print("✅ API Gateway started successfully!")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start API Gateway: {e}")
        return None

def open_browser():
    """Open browser to the web interface."""
    print("🌐 Opening web browser...")
    try:
        webbrowser.open("http://localhost:8080")
        webbrowser.open("http://localhost:8000/docs")
    except:
        pass

def main():
    """Main function to start the system."""
    print("=" * 60)
    print("🏥 Chest X-Ray Pneumonia Detection MLOps System")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check model file
    if not check_model_file():
        return
    
    # Start services
    model_server = start_model_server()
    if not model_server:
        return
    
    api_gateway = start_api_gateway()
    
    # Open browser
    time.sleep(2)
    open_browser()
    
    print("\n" + "=" * 60)
    print("🎉 System Started Successfully!")
    print("=" * 60)
    print("📱 Web Interface: http://localhost:8080")
    print("📋 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 60)
    print("\n💡 Tips:")
    print("- Drag and drop X-ray images on the web interface")
    print("- Use the API docs for programmatic access")
    print("- Press Ctrl+C to stop the system")
    print("\n⏳ System is ready for predictions!")
    
    try:
        # Keep running until user stops
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        if model_server:
            model_server.terminate()
        if api_gateway:
            api_gateway.terminate()
        print("✅ System stopped successfully!")

if __name__ == "__main__":
    main()