@echo off
REM Quick Start - Build only essential services for testing
REM This is faster than building everything

echo ========================================
echo Quick Start - Essential Services Only
echo ========================================
echo.

REM Check Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    pause
    exit /b 1
)

echo [1/4] Starting infrastructure services (no build needed)...
docker-compose up -d postgres minio mlflow
echo.

echo [2/4] Waiting for infrastructure to be ready (30 seconds)...
timeout /t 30 /nobreak >nul
echo.

echo [3/4] Building and starting deployment API only...
docker-compose build deployment
docker-compose up -d deployment
echo.

echo [4/4] Checking service status...
timeout /t 10 /nobreak >nul
docker-compose ps
echo.

echo ========================================
echo Quick Start Complete!
echo ========================================
echo.
echo API is starting up...
echo Wait 30 seconds, then access:
echo   http://localhost:8004/docs
echo.
echo To build other services later:
echo   docker-compose build
echo   docker-compose up -d
echo.

pause
