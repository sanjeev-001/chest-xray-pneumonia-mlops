@echo off
REM Run API locally without Docker (fastest option for testing)

echo ========================================
echo Running API Locally (No Docker Build)
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.9+ or use Docker
    pause
    exit /b 1
)

REM Check if model exists
if not exist models\best_chest_xray_model.pth (
    echo ERROR: Model file not found!
    echo Expected: models\best_chest_xray_model.pth
    pause
    exit /b 1
)

echo [1/3] Installing dependencies (if needed)...
pip install torch torchvision fastapi uvicorn pillow opencv-python-headless python-multipart --quiet
echo.

echo [2/3] Starting API server...
echo.
echo API will be available at: http://localhost:8004
echo Press Ctrl+C to stop
echo.

REM Set environment variables
set MODEL_PATH=models/best_chest_xray_model.pth
set MODEL_ARCHITECTURE=efficientnet_b4
set DEVICE=cpu

echo [3/3] Launching...
python -m uvicorn deployment.api:app --host 0.0.0.0 --port 8004 --reload

pause
