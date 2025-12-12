# Docker & Kubernetes Deployment Guide

Complete guide for deploying the Chest X-Ray Pneumonia Detection MLOps system using Docker and Kubernetes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start with Docker Compose](#quick-start-with-docker-compose)
- [Building Docker Images](#building-docker-images)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- Docker Desktop (Windows/Mac) or Docker Engine (Linux) - version 20.10+
- Docker Compose - version 2.0+
- kubectl - version 1.24+ (for Kubernetes deployment)
- Kubernetes cluster (minikube, kind, or cloud provider)

### System Requirements

- **Minimum**: 8GB RAM, 4 CPU cores, 50GB disk space
- **Recommended**: 16GB RAM, 8 CPU cores, 100GB disk space
- **For GPU support**: NVIDIA GPU with CUDA 11.7+, nvidia-docker2

## Quick Start with Docker Compose

### 1. Prepare Your Environment

Create a `.env` file in the project root:

```bash
# Database Configuration
POSTGRES_DB=mlops
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops_password

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ENDPOINT=minio:9000

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000

# Model Configuration
MODEL_PATH=models/best_chest_xray_model.pth
MODEL_ARCHITECTURE=efficientnet_b4
DEVICE=cpu

# Service Ports
DATA_PIPELINE_PORT=8001
TRAINING_PORT=8002
MODEL_REGISTRY_PORT=8003
DEPLOYMENT_PORT=8004
MONITORING_PORT=8005
```

### 2. Start All Services

```bash
# Start all services in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 3. Verify Services

```bash
# Check health of all services
curl http://localhost:8001/health  # Data Pipeline
curl http://localhost:8002/health  # Training
curl http://localhost:8003/health  # Model Registry
curl http://localhost:8004/health  # Deployment API
curl http://localhost:8005/health  # Monitoring

# Access web interfaces
# MLflow UI: http://localhost:5000
# MinIO Console: http://localhost:9001
# API Documentation: http://localhost:8004/docs
```

### 4. Test the API

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8004/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/xray_image.jpg"
```

### 5. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Building Docker Images

### Build Individual Services

```bash
# Build data pipeline service
docker build -f data_pipeline/Dockerfile -t chest-xray-mlops/data-pipeline:latest .

# Build training service
docker build -f training/Dockerfile -t chest-xray-mlops/training:latest .

# Build model registry service
docker build -f model_registry/Dockerfile -t chest-xray-mlops/model-registry:latest .

# Build deployment service
docker build -f deployment/Dockerfile -t chest-xray-mlops/deployment:latest .

# Build monitoring service
docker build -f monitoring/Dockerfile -t chest-xray-mlops/monitoring:latest .
```

### Build All Services at Once

```bash
# Using docker-compose
docker-compose build

# Or build with no cache
docker-compose build --no-cache
```

### Tag Images for Registry

```bash
# Tag for Docker Hub
docker tag chest-xray-mlops/deployment:latest yourusername/chest-xray-deployment:latest

# Tag for private registry
docker tag chest-xray-mlops/deployment:latest registry.example.com/chest-xray-deployment:latest
```

### Push to Registry

```bash
# Push to Docker Hub
docker push yourusername/chest-xray-deployment:latest

# Push to private registry
docker push registry.example.com/chest-xray-deployment:latest
```

## Kubernetes Deployment

### 1. Setup Kubernetes Cluster

#### Option A: Local Development with Minikube

```bash
# Start minikube
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# Verify cluster
kubectl cluster-info
```

#### Option B: Local Development with Kind

```bash
# Create cluster
kind create cluster --name mlops-cluster

# Verify cluster
kubectl cluster-info --context kind-mlops-cluster
```

#### Option C: Cloud Provider (AWS EKS, GCP GKE, Azure AKS)

Follow your cloud provider's documentation to create a Kubernetes cluster.

### 2. Create Namespace

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Verify namespace
kubectl get namespaces
```

### 3. Configure Secrets and ConfigMaps

```bash
# Create secrets
kubectl create secret generic mlops-secrets \
  --from-literal=DATABASE_USER=mlops \
  --from-literal=DATABASE_PASSWORD=mlops_password \
  --from-literal=MINIO_ACCESS_KEY=minioadmin \
  --from-literal=MINIO_SECRET_KEY=minioadmin \
  -n chest-xray-mlops

# Create configmap
kubectl create configmap mlops-config \
  --from-literal=DATABASE_HOST=postgres \
  --from-literal=DATABASE_PORT=5432 \
  --from-literal=DATABASE_NAME=mlops \
  --from-literal=MINIO_ENDPOINT=minio:9000 \
  --from-literal=MODEL_REGISTRY_URL=http://model-registry-service:8003 \
  -n chest-xray-mlops

# Verify
kubectl get secrets -n chest-xray-mlops
kubectl get configmaps -n chest-xray-mlops
```

### 4. Deploy Infrastructure Services

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Deploy MinIO
kubectl apply -f k8s/minio.yaml

# Wait for infrastructure to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n chest-xray-mlops --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio -n chest-xray-mlops --timeout=300s
```

### 5. Deploy MLOps Services

```bash
# Deploy all services
kubectl apply -f k8s/data-pipeline.yaml
kubectl apply -f k8s/training.yaml
kubectl apply -f k8s/model-registry.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/monitoring.yaml

# Check deployment status
kubectl get deployments -n chest-xray-mlops
kubectl get pods -n chest-xray-mlops
kubectl get services -n chest-xray-mlops
```

### 6. Verify Deployments

```bash
# Check pod status
kubectl get pods -n chest-xray-mlops -w

# Check logs
kubectl logs -f deployment/deployment -n chest-xray-mlops

# Check service endpoints
kubectl get svc -n chest-xray-mlops
```

### 7. Access Services

#### Option A: Port Forwarding (Development)

```bash
# Forward deployment API
kubectl port-forward svc/deployment-service 8004:8004 -n chest-xray-mlops

# Forward monitoring service
kubectl port-forward svc/monitoring-service 8005:8005 -n chest-xray-mlops

# Forward MLflow UI
kubectl port-forward svc/mlflow 5000:5000 -n chest-xray-mlops

# Access services at:
# API: http://localhost:8004
# Monitoring: http://localhost:8005
# MLflow: http://localhost:5000
```

#### Option B: Ingress (Production)

```bash
# Apply ingress configuration
kubectl apply -f k8s/ingress.yaml

# Get ingress address
kubectl get ingress -n chest-xray-mlops

# Access via ingress hostname
# API: http://chest-xray-api.example.com
```

#### Option C: LoadBalancer (Cloud)

Services are already configured with LoadBalancer type. Get external IPs:

```bash
kubectl get svc -n chest-xray-mlops
```

### 8. Scale Deployments

```bash
# Scale deployment service
kubectl scale deployment deployment --replicas=5 -n chest-xray-mlops

# Enable autoscaling
kubectl autoscale deployment deployment \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n chest-xray-mlops

# Check autoscaler status
kubectl get hpa -n chest-xray-mlops
```

### 9. Update Deployments

```bash
# Update image
kubectl set image deployment/deployment \
  deployment=chest-xray-mlops/deployment:v2.0.0 \
  -n chest-xray-mlops

# Check rollout status
kubectl rollout status deployment/deployment -n chest-xray-mlops

# Rollback if needed
kubectl rollout undo deployment/deployment -n chest-xray-mlops
```

### 10. Cleanup

```bash
# Delete all resources in namespace
kubectl delete namespace chest-xray-mlops

# Or delete individual resources
kubectl delete -f k8s/
```

## Configuration

### Environment Variables

#### Data Pipeline Service

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
STORAGE_BACKEND=minio
STORAGE_BUCKET_NAME=mlops-data
```

#### Training Service

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
MLFLOW_TRACKING_URI=http://mlflow:5000
MINIO_ENDPOINT=minio:9000
DEVICE=cuda  # or cpu
BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_EPOCHS=50
```

#### Deployment Service

```bash
MODEL_PATH=models/best_chest_xray_model.pth
MODEL_ARCHITECTURE=efficientnet_b4
DEVICE=cpu
MAX_IMAGE_SIZE=10485760
MODEL_REGISTRY_URL=http://model-registry:8003
CACHE_SIZE=1000
CACHE_TTL=3600
```

#### Monitoring Service

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
DEPLOYMENT_URL=http://deployment:8004
PROMETHEUS_PORT=8080
AUTO_START_PERFORMANCE_MONITORING=true
AUTO_START_PROMETHEUS=true
```

### Resource Limits

Edit resource limits in Kubernetes manifests:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Persistent Storage

Configure persistent volumes for data retention:

```yaml
volumeMounts:
  - name: model-storage
    mountPath: /app/models
volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-pvc
```

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n chest-xray-mlops

# Check logs
kubectl logs <pod-name> -n chest-xray-mlops

# Check events
kubectl get events -n chest-xray-mlops --sort-by='.lastTimestamp'
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -n chest-xray-mlops -- \
  psql -h postgres -U mlops -d mlops

# Check database pod
kubectl logs postgres-0 -n chest-xray-mlops
```

#### 3. Model Not Loading

```bash
# Check if model file exists
kubectl exec -it deployment-<pod-id> -n chest-xray-mlops -- ls -lh /app/models/

# Copy model to pod
kubectl cp models/best_chest_xray_model.pth \
  chest-xray-mlops/deployment-<pod-id>:/app/models/
```

#### 4. Out of Memory

```bash
# Check resource usage
kubectl top pods -n chest-xray-mlops

# Increase memory limits
kubectl edit deployment deployment -n chest-xray-mlops
```

#### 5. Image Pull Errors

```bash
# Check image pull secrets
kubectl get secrets -n chest-xray-mlops

# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n chest-xray-mlops

# Add to deployment
kubectl patch serviceaccount default \
  -p '{"imagePullSecrets": [{"name": "regcred"}]}' \
  -n chest-xray-mlops
```

### Performance Optimization

#### 1. Enable GPU Support

```yaml
# Add to deployment spec
resources:
  limits:
    nvidia.com/gpu: 1
```

#### 2. Use Node Affinity

```yaml
# Schedule GPU workloads on GPU nodes
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: accelerator
          operator: In
          values:
          - nvidia-tesla-t4
```

#### 3. Enable Caching

```bash
# Set environment variables
CACHE_SIZE=1000
CACHE_TTL=3600
ENABLE_BATCHING=true
```

### Monitoring and Debugging

```bash
# View real-time logs
kubectl logs -f deployment/<pod-name> -n chest-xray-mlops

# Execute commands in pod
kubectl exec -it <pod-name> -n chest-xray-mlops -- /bin/bash

# Port forward for debugging
kubectl port-forward <pod-name> 8004:8004 -n chest-xray-mlops

# Check resource usage
kubectl top nodes
kubectl top pods -n chest-xray-mlops
```

## Production Checklist

- [ ] Use production-grade database (managed PostgreSQL)
- [ ] Configure persistent volumes for data
- [ ] Set up proper secrets management (Vault, AWS Secrets Manager)
- [ ] Configure ingress with TLS/SSL
- [ ] Set up monitoring and alerting (Prometheus, Grafana)
- [ ] Configure log aggregation (ELK, Loki)
- [ ] Set resource limits and requests
- [ ] Enable horizontal pod autoscaling
- [ ] Configure network policies
- [ ] Set up backup and disaster recovery
- [ ] Implement CI/CD pipeline
- [ ] Configure health checks and readiness probes
- [ ] Use image tags (not latest)
- [ ] Set up RBAC and security policies

## Next Steps

1. **Test the deployment**: Run integration tests
2. **Monitor performance**: Set up Grafana dashboards
3. **Scale as needed**: Adjust replicas and resources
4. **Implement CI/CD**: Automate deployments
5. **Security hardening**: Apply security best practices

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Project Architecture](ARCHITECTURE.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
