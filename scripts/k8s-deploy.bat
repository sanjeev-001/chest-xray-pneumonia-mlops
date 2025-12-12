@echo off
REM Kubernetes Deployment Script for Windows
REM Chest X-Ray Pneumonia Detection MLOps System

echo ========================================
echo Chest X-Ray MLOps - Kubernetes Deployment
echo ========================================
echo.

REM Check if kubectl is available
kubectl version --client >nul 2>&1
if errorlevel 1 (
    echo ERROR: kubectl is not installed or not in PATH!
    echo Please install kubectl and try again.
    pause
    exit /b 1
)

echo [1/8] kubectl is available...
echo.

REM Check cluster connection
echo [2/8] Checking cluster connection...
kubectl cluster-info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot connect to Kubernetes cluster!
    echo Please ensure your cluster is running and configured.
    pause
    exit /b 1
)
echo Cluster is accessible!
echo.

REM Create namespace
echo [3/8] Creating namespace...
kubectl apply -f k8s/namespace.yaml
echo.

REM Create secrets
echo [4/8] Creating secrets...
kubectl create secret generic mlops-secrets ^
  --from-literal=DATABASE_USER=mlops ^
  --from-literal=DATABASE_PASSWORD=mlops_password ^
  --from-literal=MINIO_ACCESS_KEY=minioadmin ^
  --from-literal=MINIO_SECRET_KEY=minioadmin ^
  -n chest-xray-mlops ^
  --dry-run=client -o yaml | kubectl apply -f -
echo.

REM Create configmap
echo [5/8] Creating configmap...
kubectl create configmap mlops-config ^
  --from-literal=DATABASE_HOST=postgres ^
  --from-literal=DATABASE_PORT=5432 ^
  --from-literal=DATABASE_NAME=mlops ^
  --from-literal=MINIO_ENDPOINT=minio:9000 ^
  --from-literal=MODEL_REGISTRY_URL=http://model-registry-service:8003 ^
  -n chest-xray-mlops ^
  --dry-run=client -o yaml | kubectl apply -f -
echo.

REM Deploy infrastructure
echo [6/8] Deploying infrastructure services...
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/minio.yaml
echo Waiting for infrastructure to be ready...
timeout /t 30 /nobreak >nul
echo.

REM Deploy MLOps services
echo [7/8] Deploying MLOps services...
kubectl apply -f k8s/data-pipeline.yaml
kubectl apply -f k8s/training.yaml
kubectl apply -f k8s/model-registry.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/monitoring.yaml
echo.

REM Check deployment status
echo [8/8] Checking deployment status...
kubectl get pods -n chest-xray-mlops
echo.

echo ========================================
echo Deployment Complete!
echo ========================================
echo.
echo To check status:
echo   kubectl get pods -n chest-xray-mlops
echo.
echo To view logs:
echo   kubectl logs -f deployment/deployment -n chest-xray-mlops
echo.
echo To access services (port forwarding):
echo   kubectl port-forward svc/deployment-service 8004:8004 -n chest-xray-mlops
echo.
echo To delete deployment:
echo   kubectl delete namespace chest-xray-mlops
echo.

pause
