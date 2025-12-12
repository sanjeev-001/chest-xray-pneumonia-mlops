@echo off
REM Quick fix script - rebuild deployment and monitoring with correct dependencies

echo ========================================
echo Fixing Missing Dependencies
echo ========================================
echo.

echo [1/4] Stopping deployment and monitoring...
docker-compose stop deployment monitoring
echo.

echo [2/4] Rebuilding deployment service...
docker-compose build deployment
echo.

echo [3/4] Rebuilding monitoring service...
docker-compose build monitoring
echo.

echo [4/4] Starting services...
docker-compose up -d deployment monitoring
echo.

echo ========================================
echo Services Restarted!
echo ========================================
echo.

echo Waiting 30 seconds for services to initialize...
timeout /t 30 /nobreak >nul
echo.

echo Checking status...
docker-compose ps
echo.

echo Testing API...
curl http://localhost:8004/health
echo.
echo.

echo ========================================
echo Done!
echo ========================================
echo.
echo If successful, access:
echo   http://localhost:8004/docs
echo.

pause
