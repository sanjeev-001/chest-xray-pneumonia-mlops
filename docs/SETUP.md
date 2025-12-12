# Setup Guide - Chest X-Ray Pneumonia Detection MLOps System

This guide provides detailed instructions for setting up the project in different environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Docker Setup](#docker-setup)
- [Kubernetes Setup](#kubernetes-setup)
- [Data Setup](#data-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Python**: 3.9 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Git**: 2.30 or higher

### Optional (for production deployment)

- **Kubernetes**: 1.24 or higher
- **kubectl**: Matching your Kubernetes version
- **Helm**: 3.0 or higher (for some deployments)

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB free space

**Recommended:**
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 50+ GB free space
- GPU: NVIDIA GPU with CUDA support (for training)

## Local Development Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops.git
cd chest-xray-pneumonia-mlops
```

### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

**Install production dependencies:**
```bash
pip install -r requirements.txt
```

**Install development dependencies:**
```bash
pip install -r requirements-dev.txt
```

**Or use make commands:**
```bash
make install        # Production dependencies
make install-dev    # Development dependencies
```

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
# Use your preferred text editor
nano .env  # or vim, code, notepad, etc.
```

### Step 5: Install Pre-commit Hooks (Optional)

```bash
pre-commit install
```

## Docker Setup

### Step 1: Verify Docker Installation

```bash
docker --version
docker-compose --version
```

### Step 2: Build Docker Images

```bash
# Build all images
make build

# Or manually
docker-compose build
```

### Step 3: Start Services

```bash
# Start all services
make docker-up

# Or manually
docker-compose up -d
```

### Step 4: Verify Services

```bash
# Check running containers
docker-compose ps

# View logs
docker-compose logs -f
```

### Service Endpoints

Once running, services are available at:

- **Data Pipeline API**: http://localhost:8001
- **Training Service**: http://localhost:8002
- **Model Registry**: http://localhost:8003
- **Deployment API**: http://localhost:8004
- **Monitoring Service**: http://localhost:8005
- **MLflow UI**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Kubernetes Setup

### Prerequisites

- Kubernetes cluster (local or cloud)
- kubectl configured to access your cluster
- Sufficient cluster resources

### Step 1: Create Namespace

```bash
kubectl create namespace mlops
```

### Step 2: Create Secrets

```bash
# Create database secret
kubectl create secret generic db-credentials \
  --from-literal=username=mlops \
  --from-literal=password=your_password \
  -n mlops

# Create MinIO secret
kubectl create secret generic minio-credentials \
  --from-literal=accesskey=minioadmin \
  --from-literal=secretkey=minioadmin \
  -n mlops
```

### Step 3: Deploy Services

```bash
# Deploy all services
make k8s-deploy

# Or manually
kubectl apply -f k8s/ -n mlops
```

### Step 4: Verify Deployment

```bash
# Check pod status
kubectl get pods -n mlops

# Check services
kubectl get svc -n mlops

# View logs
kubectl logs -f deployment/deployment-service -n mlops
```

### Step 5: Access Services

```bash
# Port forward to access services locally
kubectl port-forward svc/deployment-service 8004:8004 -n mlops
kubectl port-forward svc/mlflow 5000:5000 -n mlops
```

## Data Setup

### Option 1: Download from Kaggle

1. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Configure Kaggle credentials:**
   - Go to https://www.kaggle.com/account
   - Create API token
   - Place `kaggle.json` in `~/.kaggle/`

3. **Download dataset:**
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/raw/
   ```

### Option 2: Use Sample Data

```bash
# Create sample data structure
python scripts/create_sample_data.py
```

### Data Structure

After setup, your data directory should look like:

```
data/
├── raw/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── processed/
└── versions/
```

## Configuration

### Environment Variables

Edit `.env` file with your configuration:

```bash
# Database Configuration
DATABASE_URL=postgresql://mlops:password@localhost:5432/mlops

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Service URLs
MODEL_REGISTRY_URL=http://localhost:8003
DEPLOYMENT_URL=http://localhost:8004
```

### Service Configuration

Each service has its own configuration file in `config/`:

- `config/data_pipeline.yaml` - Data pipeline settings
- `config/training.yaml` - Training hyperparameters
- `config/deployment.yaml` - Deployment settings
- `config/monitoring.yaml` - Monitoring thresholds

## Verification

### Step 1: Health Checks

```bash
# Check all services
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8005/health
```

### Step 2: Run Tests

```bash
# Run all tests
make test

# Run specific test suite
pytest tests/test_data_pipeline.py -v
```

### Step 3: Quick Test

```bash
# Run quick integration test
python quick_test.py
```

### Step 4: Test API

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8004/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/xray.jpg"
```

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port
lsof -i :8004  # On Linux/macOS
netstat -ano | findstr :8004  # On Windows

# Kill process or change port in docker-compose.yml
```

#### Docker Build Fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

#### Permission Denied

```bash
# On Linux, add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Or reduce batch size in config/training.yaml
```

#### Database Connection Error

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# View database logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

#### MinIO Connection Error

```bash
# Check MinIO status
docker-compose ps minio

# Access MinIO console
# http://localhost:9001
# Login: minioadmin / minioadmin

# Create buckets manually if needed
```

### Getting Help

If you encounter issues:

1. Check service logs: `docker-compose logs [service-name]`
2. Review documentation in `docs/` folder
3. Search existing GitHub issues
4. Create a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details
   - Relevant logs

## Next Steps

After successful setup:

1. **Train a model**: See `docs/USER_GUIDE.md`
2. **Deploy model**: See `docs/DEPLOYMENT_GUIDE.md`
3. **Monitor system**: See `docs/OPERATIONS_RUNBOOK.md`
4. **Contribute**: See `CONTRIBUTING.md`

## Quick Reference

### Useful Commands

```bash
# Start services
make docker-up

# Stop services
make docker-down

# View logs
make logs

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Clean up
make clean

# Show all commands
make help
```

### Service URLs Quick Reference

| Service | URL | Credentials |
|---------|-----|-------------|
| Data Pipeline | http://localhost:8001 | - |
| Training | http://localhost:8002 | - |
| Model Registry | http://localhost:8003 | - |
| Deployment API | http://localhost:8004 | - |
| Monitoring | http://localhost:8005 | - |
| MLflow UI | http://localhost:5000 | - |
| MinIO Console | http://localhost:9001 | admin/admin123 |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |

## Additional Resources

- [User Guide](docs/USER_GUIDE.md) - How to use the system
- [API Documentation](docs/API_DOCUMENTATION.md) - API reference
- [System Overview](docs/SYSTEM_OVERVIEW.md) - Architecture details
- [Operations Runbook](docs/OPERATIONS_RUNBOOK.md) - Operations guide
