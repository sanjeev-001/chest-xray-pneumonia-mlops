# Chest X-Ray Pneumonia Detection MLOps System - User Guide

## 🏥 Overview

This comprehensive MLOps system provides automated chest X-ray analysis for pneumonia detection using a trained EfficientNet-B4 deep learning model. The system includes data processing, model training, deployment, monitoring, and automated retraining capabilities.

## 🎯 System Capabilities

### Core Features
- **Real-time Pneumonia Detection**: Analyze chest X-ray images with 87%+ accuracy
- **Automated ML Pipeline**: End-to-end automation from data ingestion to deployment
- **Production-Ready API**: RESTful API with comprehensive documentation
- **Monitoring & Alerting**: Real-time performance monitoring and drift detection
- **Automated Retraining**: Continuous model improvement based on performance metrics
- **CI/CD Pipeline**: Automated testing, validation, and deployment
- **Scalable Infrastructure**: Kubernetes-based deployment with auto-scaling

### Model Performance
- **Architecture**: EfficientNet-B4 with custom classifier
- **Accuracy**: 87% on test dataset
- **Precision**: 85% for pneumonia detection
- **Recall**: 89% for pneumonia cases
- **F1-Score**: 87% overall performance
- **Inference Time**: <200ms per image

## 🚀 Quick Start Guide

### Prerequisites

**System Requirements:**
- Python 3.10+
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space
- GPU optional (CPU supported)

**Required Software:**
- Docker (optional, for containerized deployment)
- Git
- Python package manager (pip)

### Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd chest-xray-pneumonia-mlops
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

3. **Verify Model File**
   ```bash
   ls -la models/best_chest_xray_model.pth
   # Should show the trained model file (~100MB)
   ```

### Quick Deployment

**Option 1: Complete System Deployment**
```bash
# Deploy entire MLOps system
python scripts/deploy_complete_system.py

# Validate deployment
python scripts/validate_complete_system.py
```

**Option 2: Individual Services**
```bash
# Start model server only
python -m uvicorn deployment.model_server:app --host 0.0.0.0 --port 8000

# Start API gateway
python -m uvicorn deployment.api:app --host 0.0.0.0 --port 8080
```

### First Prediction

1. **Access API Documentation**
   - Open: http://localhost:8000/docs
   - Interactive API testing interface

2. **Test with Sample Image**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/chest_xray.jpg"
   ```

3. **Expected Response**
   ```json
   {
     "prediction": "PNEUMONIA",
     "confidence": 0.892,
     "probabilities": {
       "NORMAL": 0.108,
       "PNEUMONIA": 0.892
     },
     "processing_time_ms": 145.2,
     "model_version": "efficientnet_b4_v1.0",
     "timestamp": "2024-01-15T10:30:45Z"
   }
   ```

## 📚 Detailed Usage Guide

### API Endpoints

#### Model Server (Port 8000)

**Health Check**
```bash
GET /health
# Returns: System health status and model information
```

**Model Information**
```bash
GET /model/info
# Returns: Model architecture, classes, and metadata
```

**Single Image Prediction**
```bash
POST /predict
# Body: multipart/form-data with 'file' field
# Returns: Prediction with confidence scores
```

**Batch Prediction**
```bash
POST /predict/batch
# Body: JSON with base64-encoded images array
# Returns: Array of predictions
```

**Performance Metrics**
```bash
GET /metrics
# Returns: Prometheus-format metrics
```

#### API Gateway (Port 8080)

**Web Interface**
```bash
GET /
# Returns: Web-based prediction interface
```

**API Health**
```bash
GET /health
# Returns: Overall system health
```

### Image Requirements

**Supported Formats**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)

**Image Specifications**
- **Size**: Any size (automatically resized to 224x224)
- **Color**: RGB or Grayscale (converted to RGB)
- **Quality**: Higher quality images provide better results
- **Content**: Chest X-ray images (PA or AP view preferred)

**Best Practices**
- Use high-resolution images when possible
- Ensure proper contrast and brightness
- Avoid heavily processed or filtered images
- Center the chest area in the image

### Batch Processing

**Python Example**
```python
import requests
import base64
import json

# Prepare batch request
with open('image1.jpg', 'rb') as f:
    img1_b64 = base64.b64encode(f.read()).decode('utf-8')

with open('image2.jpg', 'rb') as f:
    img2_b64 = base64.b64encode(f.read()).decode('utf-8')

batch_request = {
    'images': [img1_b64, img2_b64],
    'return_probabilities': True,
    'return_confidence': True
}

# Send batch request
response = requests.post(
    'http://localhost:8000/predict/batch',
    json=batch_request
)

results = response.json()
print(f"Processed {results['batch_size']} images")
for i, prediction in enumerate(results['predictions']):
    print(f"Image {i+1}: {prediction['prediction']} ({prediction['confidence']:.3f})")
```

## 🔧 Configuration

### Environment Variables

**Model Configuration**
```bash
export MODEL_PATH="models/best_chest_xray_model.pth"
export MODEL_ARCHITECTURE="efficientnet_b4"
export DEVICE="auto"  # auto, cpu, cuda
export BATCH_SIZE="8"
```

**Performance Settings**
```bash
export CACHE_SIZE="1000"  # Number of cached predictions
export CACHE_TTL="3600"   # Cache time-to-live in seconds
export ENABLE_BATCHING="true"
export AUTO_OPTIMIZE="true"
```

**Monitoring Configuration**
```bash
export METRICS_ENABLED="true"
export LOG_LEVEL="INFO"
export PROMETHEUS_PORT="9090"
```

### Configuration Files

**Model Server Config** (`deployment/config.yaml`)
```yaml
model:
  path: "models/best_chest_xray_model.pth"
  architecture: "efficientnet_b4"
  device: "auto"
  
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  
performance:
  cache_size: 1000
  enable_batching: true
  batch_timeout: 100  # milliseconds
```

## 📊 Monitoring & Observability

### Metrics Dashboard

**Access Grafana Dashboard**
- URL: http://localhost:3000
- Username: admin
- Password: (check deployment logs)

**Key Metrics to Monitor**
- **Prediction Rate**: Requests per second
- **Response Time**: P50, P95, P99 latencies
- **Model Accuracy**: Real-time accuracy tracking
- **Error Rate**: Failed predictions percentage
- **Resource Usage**: CPU, Memory, GPU utilization

### Alerting

**Automatic Alerts**
- Model accuracy drops below 80%
- Response time exceeds 2 seconds
- Error rate above 5%
- System resource exhaustion

**Alert Channels**
- Email notifications
- Slack integration
- Webhook endpoints
- PagerDuty integration

### Logging

**Log Locations**
```bash
# Application logs
tail -f logs/model_server.log
tail -f logs/api_gateway.log

# System logs
journalctl -u mlops-model-server -f
```

**Log Levels**
- **DEBUG**: Detailed debugging information
- **INFO**: General operational messages
- **WARNING**: Warning conditions
- **ERROR**: Error conditions
- **CRITICAL**: Critical system failures

## 🔄 Model Management

### Model Registry

**Register New Model**
```python
from training.model_registry import ModelRegistry

registry = ModelRegistry()
model_id = registry.register_model(
    name="chest_xray_efficientnet_b4",
    version="2.0.0",
    model_path="models/new_model.pth",
    metadata={
        "accuracy": 0.89,
        "training_date": "2024-01-15",
        "dataset_version": "v2.1"
    }
)
```

**List Available Models**
```python
models = registry.list_models()
for model in models:
    print(f"{model['name']} v{model['version']} - Accuracy: {model['metadata']['accuracy']}")
```

### Model Deployment

**Deploy New Model Version**
```bash
# Update model path
export MODEL_PATH="models/new_model_v2.pth"

# Restart services
python scripts/deploy_complete_system.py --model-path models/new_model_v2.pth
```

**Blue-Green Deployment**
```bash
# Deploy to staging environment
python scripts/deployment_manager.py deploy --version v2.0.0

# Validate deployment
python scripts/deployment_manager.py health-check --environment blue

# Switch traffic
python scripts/deployment_manager.py switch-traffic --from-env green --to-env blue
```

## 🧪 Testing & Validation

### Running Tests

**Complete Test Suite**
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python scripts/run_tests.py data-pipeline
python scripts/run_tests.py training
python scripts/run_tests.py deployment
python scripts/run_tests.py monitoring
```

**Integration Tests**
```bash
# End-to-end integration tests
python -m pytest tests/test_final_integration.py -v

# System validation
python scripts/validate_complete_system.py
```

**Performance Tests**
```bash
# Load testing
python deployment/performance_test.py --concurrent-users 10 --duration 60

# Stress testing
python deployment/performance_test.py --stress-test --max-rps 100
```

### Model Validation

**Accuracy Testing**
```bash
# Validate model accuracy on test dataset
python training/validate_model.py --model-path models/best_chest_xray_model.pth
```

**Drift Detection**
```bash
# Check for data drift
python monitoring/drift_detector.py --baseline-data data/baseline --new-data data/recent
```

## 🔒 Security & Compliance

### Security Features

**Data Protection**
- All data encrypted at rest and in transit
- No patient data stored permanently
- Secure API authentication (when enabled)
- Regular security scans and updates

**Access Control**
- Role-based access control (RBAC)
- API key authentication
- Rate limiting and throttling
- Audit logging for all operations

### Compliance

**Healthcare Compliance**
- HIPAA-ready architecture
- Audit trail for all predictions
- Data retention policies
- Privacy-preserving design

**Audit Logging**
```bash
# View audit logs
python monitoring/audit_trail.py --date 2024-01-15

# Export compliance report
python monitoring/audit_trail.py --export --format pdf
```

## 🚨 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Check model file
ls -la models/best_chest_xray_model.pth

# Verify model format
python -c "import torch; print(torch.load('models/best_chest_xray_model.pth', map_location='cpu').keys())"

# Test model loading
python -c "from deployment.model_server import load_model; print(load_model())"
```

**Performance Issues**
```bash
# Check system resources
htop
nvidia-smi  # For GPU usage

# Monitor API performance
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/health

# Check logs for bottlenecks
grep "slow" logs/model_server.log
```

**Connection Issues**
```bash
# Test service connectivity
curl http://localhost:8000/health
curl http://localhost:8080/health

# Check port availability
netstat -tulpn | grep :8000
netstat -tulpn | grep :8080

# Verify firewall settings
sudo ufw status
```

### Error Codes

**HTTP Status Codes**
- **200**: Success
- **400**: Bad Request (invalid image format, missing file)
- **422**: Unprocessable Entity (validation error)
- **500**: Internal Server Error (model loading, processing error)
- **503**: Service Unavailable (model not loaded, system overload)

**Common Error Messages**
- `"Model not loaded"`: Model file missing or corrupted
- `"Invalid image format"`: Unsupported file type
- `"Processing timeout"`: Image too large or system overloaded
- `"Insufficient memory"`: System resources exhausted

### Getting Help

**Log Analysis**
```bash
# Check recent errors
grep -i error logs/*.log | tail -20

# Monitor real-time logs
tail -f logs/model_server.log | grep -i error

# System health check
python scripts/validate_complete_system.py
```

**Support Resources**
- Documentation: `docs/` directory
- API Reference: http://localhost:8000/docs
- System Status: http://localhost:8000/health
- Monitoring Dashboard: http://localhost:3000

## 📈 Performance Optimization

### Hardware Recommendations

**Minimum Requirements**
- CPU: 4 cores, 2.5GHz+
- RAM: 8GB
- Storage: 20GB SSD
- Network: 100Mbps

**Recommended Configuration**
- CPU: 8+ cores, 3.0GHz+
- RAM: 16GB+
- GPU: NVIDIA GTX 1060+ or equivalent
- Storage: 50GB+ NVMe SSD
- Network: 1Gbps+

**Production Configuration**
- CPU: 16+ cores, 3.5GHz+
- RAM: 32GB+
- GPU: NVIDIA RTX 3080+ or Tesla V100+
- Storage: 100GB+ NVMe SSD
- Network: 10Gbps+

### Optimization Tips

**Model Serving**
- Enable GPU acceleration when available
- Use model quantization for faster inference
- Implement request batching for higher throughput
- Enable response caching for repeated requests

**System Configuration**
```bash
# Enable GPU support
export DEVICE="cuda"

# Optimize batch processing
export ENABLE_BATCHING="true"
export BATCH_SIZE="16"
export BATCH_WAIT_TIME="0.05"

# Enable caching
export CACHE_SIZE="5000"
export CACHE_TTL="7200"
```

**Scaling Options**
- Horizontal scaling with load balancers
- Kubernetes auto-scaling
- Container orchestration
- Multi-region deployment

## 🔄 Maintenance

### Regular Maintenance Tasks

**Daily**
- Monitor system health and performance
- Check error logs and alerts
- Verify backup completion
- Review prediction accuracy metrics

**Weekly**
- Update system dependencies
- Run comprehensive test suite
- Analyze performance trends
- Review security logs

**Monthly**
- Model performance evaluation
- Infrastructure capacity planning
- Security vulnerability assessment
- Documentation updates

### Backup & Recovery

**Automated Backups**
```bash
# Database backup
python scripts/backup_database.py

# Model artifacts backup
python scripts/backup_models.py

# Configuration backup
python scripts/backup_config.py
```

**Recovery Procedures**
```bash
# Restore from backup
python scripts/restore_system.py --backup-date 2024-01-15

# Rollback to previous version
python scripts/deployment_manager.py rollback --version v1.9.0 --reason "Performance issue"
```

## 📞 Support & Contact

**Technical Support**
- Email: support@mlops-system.com
- Documentation: https://docs.mlops-system.com
- Issue Tracker: https://github.com/your-org/chest-xray-mlops/issues

**Emergency Contact**
- On-call: +1-555-MLOPS-1
- Slack: #mlops-support
- PagerDuty: MLOps Production Team

**Community**
- Discussion Forum: https://community.mlops-system.com
- Slack Workspace: mlops-community.slack.com
- Monthly Office Hours: First Friday of each month

This user guide provides comprehensive information for using the Chest X-Ray Pneumonia Detection MLOps system. For additional technical details, refer to the specific documentation files in the `docs/` directory.