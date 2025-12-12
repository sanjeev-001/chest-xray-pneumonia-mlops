@echo off
REM Quick Start Script for Docker Deployment on Windows
REM Chest X-Ray Pneumonia Detection MLOps System

echo ========================================
echo Chest X-Ray MLOps System - Docker Setup
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [1/6] Docker is running...
echo.

REM Check if .env file exists
if not exist .env (
    echo [2/6] Creating .env file from template...
    copy .env.example .env
    echo Please edit .env file with your configuration
    echo.
) else (
    echo [2/6] .env file found...
    echo.
)

REM Create necessary directories
echo [3/6] Creating necessary directories...
if not exist models mkdir models
if not exist data mkdir data
if not exist mlruns mkdir mlruns

REM Check if model exists
if not exist models\best_chest_xray_model.pth (
    echo.
    echo WARNING: Model file not found!
    echo Expected: models\best_chest_xray_model.pth
    echo.
    echo The API will start but predictions will fail without a model.
    echo Please ensure you have trained a model or copy one to the models directory.
    echo.
    pause
)
echo.

REM Pull base images
echo [4/6] Pulling base images (this may take a while)...
docker-compose pull postgres minio
echo.

REM Build services
echo [5/6] Building MLOps services...
docker-compose build
if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo.

REM Start services
echo [6/6] Starting all services...
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start services!
    pause
    exit /b 1
)
echo.

echo ========================================
echo Services Started Successfully!
echo ========================================
echo.
echo Waiting for services to be ready (30 seconds)...
timeout /t 30 /nobreak >nul
echo.

echo Service URLs:
echo   - API Documentation:  http://localhost:8004/docs
echo   - MLflow UI:          http://localhost:5000
echo   - MinIO Console:      http://localhost:9001
echo   - Data Pipeline:      http://localhost:8001/docs
echo   - Training Service:   http://localhost:8002/docs
echo   - Model Registry:     http://localhost:8003/docs
echo   - Monitoring:         http://localhost:8005/docs
echo.

echo Checking service health...
echo.

curl -s http://localhost:8001/health >nul 2>&1
if errorlevel 1 (
    echo   [!] Data Pipeline: Not Ready
) else (
    echo   [OK] Data Pipeline: Ready
)

curl -s http://localhost:8004/health >nul 2>&1
if errorlevel 1 (
    echo   [!] Deployment API: Not Ready
) else (
    echo   [OK] Deployment API: Ready
)

curl -s http://localhost:8005/health >nul 2>&1
if errorlevel 1 (
    echo   [!] Monitoring: Not Ready
) else (
    echo   [OK] Monitoring: Ready
)

echo.
echo ========================================
echo Quick Commands:
echo   View logs:     docker-compose logs -f
echo   Stop services: docker-compose down
echo   Restart:       docker-compose restart
echo ========================================
echo.

pause
