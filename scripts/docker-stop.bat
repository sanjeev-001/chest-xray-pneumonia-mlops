@echo off
REM Stop Script for Docker Deployment on Windows

echo ========================================
echo Stopping Chest X-Ray MLOps System
echo ========================================
echo.

docker-compose down

echo.
echo Services stopped successfully!
echo.
echo To remove all data (volumes), run:
echo   docker-compose down -v
echo.

pause
