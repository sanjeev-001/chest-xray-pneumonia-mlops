@echo off
REM Continue Docker build using cache (faster than starting over)

echo ========================================
echo Continue Docker Build (Using Cache)
echo ========================================
echo.

echo This will continue building from cached layers.
echo Much faster than starting from scratch!
echo.

REM Check what's already built
echo Checking existing images...
docker images | findstr chest-xray
echo.

echo Starting build with cache...
echo This should be faster since we're using cached layers.
echo.

REM Build with cache (default behavior)
docker-compose build

if errorlevel 1 (
    echo.
    echo Build failed. Try these options:
    echo.
    echo 1. Quick start (build only API):
    echo    docker-quick-start.bat
    echo.
    echo 2. Run locally (no Docker):
    echo    run-api-local.bat
    echo.
    echo 3. Build without cache (slower):
    echo    docker-compose build --no-cache
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.

echo Starting services...
docker-compose up -d

echo.
echo Services starting...
echo Wait 30 seconds, then access:
echo   http://localhost:8004/docs
echo.

pause
