@echo off
REM Test Script for Docker Deployment API

echo ========================================
echo Testing Chest X-Ray MLOps API
echo ========================================
echo.

REM Check if services are running
docker-compose ps | findstr "Up" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Services are not running!
    echo Please start services with docker-start.bat
    pause
    exit /b 1
)

echo [1/5] Services are running...
echo.

REM Test health endpoints
echo [2/5] Testing health endpoints...
echo.

echo Testing Data Pipeline...
curl -s http://localhost:8001/health
echo.
echo.

echo Testing Deployment API...
curl -s http://localhost:8004/health
echo.
echo.

echo Testing Monitoring...
curl -s http://localhost:8005/health
echo.
echo.

REM Test API documentation
echo [3/5] API Documentation available at:
echo   http://localhost:8004/docs
echo.

REM Test with sample image if available
echo [4/5] Testing prediction endpoint...
if exist "data\chest_xray\test\NORMAL\*.jpeg" (
    for %%f in (data\chest_xray\test\NORMAL\*.jpeg) do (
        echo Testing with image: %%f
        curl -X POST "http://localhost:8004/predict" ^
          -H "Content-Type: multipart/form-data" ^
          -F "file=@%%f"
        echo.
        goto :prediction_done
    )
) else (
    echo No test images found in data\chest_xray\test\NORMAL\
    echo Please add test images or use the API docs to test manually
)
:prediction_done
echo.

echo [5/5] Testing complete!
echo.
echo ========================================
echo All Tests Completed
echo ========================================
echo.
echo Next steps:
echo   1. Open API docs: http://localhost:8004/docs
echo   2. Upload a chest X-ray image
echo   3. Get prediction results
echo.

pause
