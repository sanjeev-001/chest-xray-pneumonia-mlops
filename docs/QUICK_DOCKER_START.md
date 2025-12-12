# 🚀 Quick Docker Start Guide

Get your MLOps system running in 5 minutes!

## Prerequisites

- ✅ Docker Desktop installed and running
- ✅ 8GB+ RAM available
- ✅ 20GB+ disk space

## Step 1: Validate Setup (30 seconds)

```bash
validate-deployment-setup.bat
```

This checks if all required files are in place.

## Step 2: Start Services (3-5 minutes)

```bash
docker-start.bat
```

This will:
1. Create necessary directories
2. Build Docker images
3. Start all services (PostgreSQL, MinIO, MLflow, API, Monitoring)
4. Wait for services to be ready

## Step 3: Test the API (1 minute)

```bash
test-docker-api.bat
```

Or manually test:

```bash
# Health check
curl http://localhost:8004/health

# API documentation
# Open in browser: http://localhost:8004/docs
```

## Step 4: Make a Prediction

### Option A: Using the Web Interface

1. Open http://localhost:8004/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Upload a chest X-ray image
5. Click "Execute"
6. See the prediction results!

### Option B: Using curl

```bash
curl -X POST "http://localhost:8004/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/xray_image.jpg"
```

### Option C: Using Python

```python
import requests

url = "http://localhost:8004/predict"
files = {"file": open("xray_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Available Services

Once started, you can access:

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8004/docs | Interactive API docs |
| **Deployment API** | http://localhost:8004 | Prediction endpoint |
| **MLflow UI** | http://localhost:5000 | Experiment tracking |
| **MinIO Console** | http://localhost:9001 | Object storage (admin/minioadmin) |
| **Data Pipeline** | http://localhost:8001/docs | Data management |
| **Training Service** | http://localhost:8002/docs | Model training |
| **Model Registry** | http://localhost:8003/docs | Model versioning |
| **Monitoring** | http://localhost:8005/docs | System monitoring |

## Common Commands

```bash
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f deployment

# Check service status
docker-compose ps

# Restart a service
docker-compose restart deployment

# Stop all services
docker-stop.bat

# Stop and remove all data
docker-compose down -v
```

## Troubleshooting

### Services not starting?

```bash
# Check Docker is running
docker info

# Check logs
docker-compose logs

# Restart services
docker-compose restart
```

### Port already in use?

Edit `docker-compose.yml` and change the port mappings:

```yaml
ports:
  - "8004:8004"  # Change first number to different port
```

### Model not loading?

Make sure the model file exists:

```bash
dir models\best_chest_xray_model.pth
```

If missing, you need to train a model first or download a pre-trained one.

### Out of memory?

Increase Docker Desktop memory:
1. Open Docker Desktop
2. Settings → Resources
3. Increase Memory to 8GB+
4. Apply & Restart

## Next Steps

1. **Test with your own images**: Upload chest X-ray images via the API
2. **Monitor performance**: Check http://localhost:8005/docs
3. **View experiments**: Check MLflow at http://localhost:5000
4. **Train new models**: Use the training service at http://localhost:8002/docs

## Production Deployment

For production deployment to Kubernetes:

```bash
k8s-deploy.bat
```

See `DOCKER_DEPLOYMENT_GUIDE.md` for detailed instructions.

## Support

- Full documentation: `DOCKER_DEPLOYMENT_GUIDE.md`
- Architecture details: `ARCHITECTURE.md`
- API reference: `docs/API_DOCUMENTATION.md`
- Deployment status: `DEPLOYMENT_STATUS.md`

## That's It! 🎉

Your MLOps system is now running. Start making predictions!
