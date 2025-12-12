@echo off
REM Validation Script for MLOps Deployment Setup
REM Checks if all required files and configurations are in place

echo ========================================
echo MLOps Deployment Setup Validation
echo ========================================
echo.

set ERRORS=0
set WARNINGS=0

REM Check Docker
echo [1] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Docker is not installed!
    set /a ERRORS+=1
) else (
    docker --version
    echo   [OK] Docker is installed
)
echo.

REM Check Docker Compose
echo [2] Checking Docker Compose...
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Docker Compose is not installed!
    set /a ERRORS+=1
) else (
    docker-compose --version
    echo   [OK] Docker Compose is installed
)
echo.

REM Check kubectl (optional for Kubernetes)
echo [3] Checking kubectl (optional)...
kubectl version --client >nul 2>&1
if errorlevel 1 (
    echo   [WARN] kubectl is not installed (required for Kubernetes deployment)
    set /a WARNINGS+=1
) else (
    kubectl version --client --short
    echo   [OK] kubectl is installed
)
echo.

REM Check required files
echo [4] Checking required files...

set FILES=docker-compose.yml requirements.txt pyproject.toml

for %%f in (%FILES%) do (
    if exist %%f (
        echo   [OK] %%f exists
    ) else (
        echo   [ERROR] %%f is missing!
        set /a ERRORS+=1
    )
)
echo.

REM Check Dockerfiles
echo [5] Checking Dockerfiles...

if exist deployment\Dockerfile (
    echo   [OK] deployment/Dockerfile exists
) else (
    echo   [ERROR] deployment/Dockerfile is missing!
    set /a ERRORS+=1
)

if exist training\Dockerfile (
    echo   [OK] training/Dockerfile exists
) else (
    echo   [ERROR] training/Dockerfile is missing!
    set /a ERRORS+=1
)

if exist data_pipeline\Dockerfile (
    echo   [OK] data_pipeline/Dockerfile exists
) else (
    echo   [ERROR] data_pipeline/Dockerfile is missing!
    set /a ERRORS+=1
)

if exist model_registry\Dockerfile (
    echo   [OK] model_registry/Dockerfile exists
) else (
    echo   [ERROR] model_registry/Dockerfile is missing!
    set /a ERRORS+=1
)

if exist monitoring\Dockerfile (
    echo   [OK] monitoring/Dockerfile exists
) else (
    echo   [ERROR] monitoring/Dockerfile is missing!
    set /a ERRORS+=1
)
echo.

REM Check Kubernetes manifests
echo [6] Checking Kubernetes manifests...

if exist k8s\namespace.yaml (
    echo   [OK] k8s/namespace.yaml exists
) else (
    echo   [WARN] k8s/namespace.yaml is missing!
    set /a WARNINGS+=1
)

if exist k8s\deployment.yaml (
    echo   [OK] k8s/deployment.yaml exists
) else (
    echo   [WARN] k8s/deployment.yaml is missing!
    set /a WARNINGS+=1
)
echo.

REM Check service modules
echo [7] Checking service modules...

if exist deployment\api.py (
    echo   [OK] deployment/api.py exists
) else (
    echo   [ERROR] deployment/api.py is missing!
    set /a ERRORS+=1
)

if exist training\main.py (
    echo   [OK] training/main.py exists
) else (
    echo   [ERROR] training/main.py is missing!
    set /a ERRORS+=1
)

if exist data_pipeline\main.py (
    echo   [OK] data_pipeline/main.py exists
) else (
    echo   [ERROR] data_pipeline/main.py is missing!
    set /a ERRORS+=1
)

if exist monitoring\main.py (
    echo   [OK] monitoring/main.py exists
) else (
    echo   [ERROR] monitoring/main.py is missing!
    set /a ERRORS+=1
)
echo.

REM Check model file
echo [8] Checking model file...
if exist models\best_chest_xray_model.pth (
    echo   [OK] Model file exists
    for %%A in (models\best_chest_xray_model.pth) do echo   Size: %%~zA bytes
) else (
    echo   [WARN] Model file not found!
    echo   You need to train a model or download a pre-trained one
    set /a WARNINGS+=1
)
echo.

REM Check directories
echo [9] Checking directories...

if exist data (
    echo   [OK] data/ directory exists
) else (
    echo   [WARN] data/ directory missing (will be created)
    mkdir data
    set /a WARNINGS+=1
)

if exist models (
    echo   [OK] models/ directory exists
) else (
    echo   [WARN] models/ directory missing (will be created)
    mkdir models
    set /a WARNINGS+=1
)
echo.

REM Check environment file
echo [10] Checking environment configuration...
if exist .env (
    echo   [OK] .env file exists
) else (
    if exist .env.example (
        echo   [WARN] .env file missing, but .env.example exists
        echo   Run: copy .env.example .env
        set /a WARNINGS+=1
    ) else (
        echo   [ERROR] Neither .env nor .env.example exists!
        set /a ERRORS+=1
    )
)
echo.

REM Summary
echo ========================================
echo Validation Summary
echo ========================================
echo   Errors:   %ERRORS%
echo   Warnings: %WARNINGS%
echo.

if %ERRORS% GTR 0 (
    echo [FAILED] Please fix the errors above before deploying!
    echo.
    pause
    exit /b 1
) else if %WARNINGS% GTR 0 (
    echo [PASSED WITH WARNINGS] You can proceed, but review warnings
    echo.
) else (
    echo [PASSED] All checks passed! Ready to deploy.
    echo.
    echo Next steps:
    echo   1. For Docker: Run docker-start.bat
    echo   2. For Kubernetes: Run k8s-deploy.bat
    echo.
)

pause
