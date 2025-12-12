# Quick Start Guide

Get the Chest X-Ray Pneumonia Detection MLOps system up and running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- 8GB RAM minimum
- 10GB free disk space

## 5-Minute Setup

### 1. Clone and Configure

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops.git
cd chest-xray-pneumonia-mlops

# Copy environment file
cp .env.example .env
```

### 2. Start Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
docker-compose ps
```

### 3. Verify Installation

```bash
# Check health of all services
curl http://localhost:8001/health  # Data Pipeline
curl http://localhost:8004/health  # Deployment API
curl http://localhost:8005/health  # Monitoring
```

### 4. Access Web Interfaces

Open in your browser:

- **MLflow UI**: http://localhost:5000 - Track experiments
- **MinIO Console**: http://localhost:9001 - Storage (login: minioadmin/minioadmin)
- **Grafana**: http://localhost:3000 - Monitoring dashboards (login: admin/admin)
- **API Docs**: http://localhost:8004/docs - Interactive API documentation

### 5. Test the System

```bash
# Run quick test
python quick_test.py
```

## Your First Prediction

### Option 1: Using Python

```python
import requests

# Upload and predict
with open("path/to/xray.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8004/predict",
        files={"file": f}
    )

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Option 2: Using cURL

```bash
curl -X POST "http://localhost:8004/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/xray.jpg"
```

### Option 3: Using the Web UI

Visit http://localhost:8004/docs and use the interactive Swagger UI:
1. Click on `/predict` endpoint
2. Click "Try it out"
3. Upload an X-ray image
4. Click "Execute"

## Train Your First Model

### Using Sample Data

```bash
# Create sample dataset
python scripts/create_sample_data.py

# Start training
python training/train_model.py --epochs 5 --batch-size 32
```

### Using Real Data (Kaggle)

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API token)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw/

# Start training
python training/train_model.py --epochs 10 --batch-size 32
```

## Monitor Training

While training is running:

1. **View real-time metrics**: http://localhost:5000 (MLflow UI)
2. **Check system metrics**: http://localhost:3000 (Grafana)
3. **View logs**: `docker-compose logs -f training`

## Common Commands

```bash
# View all services
docker-compose ps

# View logs
docker-compose logs -f [service-name]

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Clean up everything
docker-compose down -v
```

## What's Running?

After `docker-compose up`, you have:

| Service | Port | Purpose |
|---------|------|---------|
| Data Pipeline | 8001 | Data ingestion & preprocessing |
| Training Service | 8002 | Model training |
| Model Registry | 8003 | Model versioning |
| Deployment API | 8004 | Inference endpoint |
| Monitoring | 8005 | System monitoring |
| MLflow | 5000 | Experiment tracking |
| MinIO | 9000/9001 | Object storage |
| PostgreSQL | 5432 | Metadata database |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Visualization |

## Next Steps

### Learn More

- **Full Setup Guide**: See [SETUP.md](SETUP.md) for detailed installation
- **User Guide**: See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for complete usage
- **API Reference**: See [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)
- **Architecture**: See [docs/SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md)

### Try Advanced Features

1. **Hyperparameter Optimization**:
   ```bash
   python training/optimize_hyperparameters.py --trials 20
   ```

2. **Batch Predictions**:
   ```bash
   python deployment/batch_predict.py --input-dir data/test/
   ```

3. **Model Comparison**:
   ```bash
   python scripts/compare_models.py --model1 v1.0 --model2 v2.0
   ```

4. **Deploy to Production**:
   ```bash
   python deployment/deploy_cli.py --environment production --strategy blue-green
   ```

## Troubleshooting

### Services Won't Start

```bash
# Check Docker is running
docker info

# Check ports are available
netstat -an | grep 8004

# View error logs
docker-compose logs
```

### Out of Memory

```bash
# Increase Docker memory limit in Docker Desktop settings
# Or reduce batch size in config/training.yaml
```

### Can't Access Services

```bash
# Check firewall settings
# Ensure ports 8001-8005, 5000, 9000-9001, 3000 are open

# Check service health
docker-compose ps
```

### Need Help?

- Check [SETUP.md](SETUP.md) for detailed troubleshooting
- Review logs: `docker-compose logs [service-name]`
- Open an issue on GitHub

## Clean Up

When you're done:

```bash
# Stop services (keeps data)
docker-compose down

# Stop and remove all data
docker-compose down -v

# Remove Docker images
docker-compose down --rmi all
```

## Production Deployment

For production deployment:

1. See [docs/PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md)
2. Configure Kubernetes: [docs/PRODUCTION_INFRASTRUCTURE.md](docs/PRODUCTION_INFRASTRUCTURE.md)
3. Set up CI/CD: [docs/CICD_SETUP.md](docs/CICD_SETUP.md)

## Resources

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Tests**: [tests/](tests/)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

Happy coding! 🚀
