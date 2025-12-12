# Chest X-Ray Pneumonia Detection API Documentation

## 📋 Overview

This document provides comprehensive API documentation for the Chest X-Ray Pneumonia Detection MLOps system. The system exposes RESTful APIs for real-time pneumonia detection, batch processing, model management, and system monitoring.

## 🌐 Base URLs

- **Model Server**: `http://localhost:8000`
- **API Gateway**: `http://localhost:8080`
- **Monitoring**: `http://localhost:9090` (Prometheus)
- **Dashboard**: `http://localhost:3000` (Grafana)

## 🔐 Authentication

Currently, the API supports optional authentication. When enabled:

- **Method**: JWT Bearer Token
- **Header**: `Authorization: Bearer <token>`
- **Token Endpoint**: `POST /auth/token`

## 📊 Model Server API (Port 8000)

### Health Check

**Endpoint**: `GET /health`

**Description**: Returns system health status and model information.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "efficientnet_b4_v1.0",
  "uptime_seconds": 3600,
  "memory_usage_mb": 2048,
  "gpu_available": true,
  "last_prediction": "2024-01-15T10:30:45Z"
}
```

**Status Codes**:
- `200`: System healthy
- `503`: System unhealthy or model not loaded

---

### Model Information

**Endpoint**: `GET /model/info`

**Description**: Returns detailed model architecture and metadata.

**Response**:
```json
{
  "model_name": "chest_xray_efficientnet_b4",
  "version": "1.0.0",
  "architecture": "EfficientNet-B4",
  "classes": ["NORMAL", "PNEUMONIA"],
  "input_size": [224, 224, 3],
  "parameters": 19000000,
  "accuracy": 0.87,
  "precision": 0.85,
  "recall": 0.89,
  "f1_score": 0.87,
  "training_date": "2024-01-10",
  "dataset_version": "v1.0"
}
```

**Status Codes**:
- `200`: Success
- `503`: Model not loaded

---

### Single Image Prediction

**Endpoint**: `POST /predict`

**Description**: Analyze a single chest X-ray image for pneumonia detection.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: Form data with `file` field containing the image

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"
```

**Python Example**:
```python
import requests

with open('chest_xray.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    print(f"Prediction: {result['prediction']} ({result['confidence']:.3f})")
```

**Response**:
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
  "timestamp": "2024-01-15T10:30:45Z",
  "image_metadata": {
    "width": 1024,
    "height": 1024,
    "format": "JPEG",
    "size_bytes": 245760
  }
}
```

**Status Codes**:
- `200`: Successful prediction
- `400`: Invalid file format or missing file
- `413`: File too large (>10MB)
- `422`: Unprocessable image
- `500`: Internal server error
- `503`: Model not available

**Error Response**:
```json
{
  "error": "Invalid image format",
  "message": "Supported formats: JPEG, PNG, TIFF",
  "code": "INVALID_FORMAT",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

---

### Batch Prediction

**Endpoint**: `POST /predict/batch`

**Description**: Analyze multiple chest X-ray images in a single request.

**Request**:
- **Content-Type**: `application/json`
- **Body**: JSON with base64-encoded images

**Request Body**:
```json
{
  "images": [
    "base64_encoded_image_1",
    "base64_encoded_image_2"
  ],
  "return_probabilities": true,
  "return_confidence": true,
  "batch_id": "optional_batch_identifier"
}
```

**Python Example**:
```python
import requests
import base64

# Prepare images
images = []
for image_path in ['image1.jpg', 'image2.jpg']:
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
        images.append(img_b64)

# Send batch request
batch_request = {
    'images': images,
    'return_probabilities': True,
    'return_confidence': True
}

response = requests.post(
    'http://localhost:8000/predict/batch',
    json=batch_request
)
results = response.json()
```

**Response**:
```json
{
  "batch_id": "batch_20240115_103045",
  "batch_size": 2,
  "processing_time_ms": 290.5,
  "predictions": [
    {
      "image_index": 0,
      "prediction": "PNEUMONIA",
      "confidence": 0.892,
      "probabilities": {
        "NORMAL": 0.108,
        "PNEUMONIA": 0.892
      }
    },
    {
      "image_index": 1,
      "prediction": "NORMAL",
      "confidence": 0.934,
      "probabilities": {
        "NORMAL": 0.934,
        "PNEUMONIA": 0.066
      }
    }
  ],
  "model_version": "efficientnet_b4_v1.0",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

**Status Codes**:
- `200`: Successful batch prediction
- `400`: Invalid request format
- `413`: Batch too large (>50 images)
- `422`: One or more images unprocessable
- `500`: Internal server error
- `503`: Model not available

---

### Performance Metrics

**Endpoint**: `GET /metrics`

**Description**: Returns Prometheus-format metrics for monitoring.

**Response** (Prometheus format):
```
# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total{model="efficientnet_b4",class="NORMAL"} 1250
predictions_total{model="efficientnet_b4",class="PNEUMONIA"} 890

# HELP prediction_duration_seconds Time spent on predictions
# TYPE prediction_duration_seconds histogram
prediction_duration_seconds_bucket{le="0.1"} 450
prediction_duration_seconds_bucket{le="0.2"} 1890
prediction_duration_seconds_bucket{le="0.5"} 2140
prediction_duration_seconds_bucket{le="+Inf"} 2140

# HELP model_accuracy Current model accuracy
# TYPE model_accuracy gauge
model_accuracy 0.87

# HELP system_memory_usage_bytes Current memory usage
# TYPE system_memory_usage_bytes gauge
system_memory_usage_bytes 2147483648
```

**Status Codes**:
- `200`: Metrics available
- `503`: Metrics collection disabled

---

### Model Management

#### Load Model

**Endpoint**: `POST /model/load`

**Description**: Load a specific model version.

**Request**:
```json
{
  "model_path": "models/new_model_v2.pth",
  "model_version": "2.0.0"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model loaded successfully",
  "model_version": "2.0.0",
  "load_time_ms": 2500
}
```

#### Unload Model

**Endpoint**: `POST /model/unload`

**Description**: Unload the current model to free memory.

**Response**:
```json
{
  "status": "success",
  "message": "Model unloaded successfully"
}
```

---

## 🌐 API Gateway (Port 8080)

### Web Interface

**Endpoint**: `GET /`

**Description**: Returns web-based prediction interface.

**Response**: HTML page with file upload interface

---

### Gateway Health

**Endpoint**: `GET /health`

**Description**: Returns overall system health including all services.

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "model_server": {
      "status": "healthy",
      "url": "http://localhost:8000",
      "response_time_ms": 15
    },
    "monitoring": {
      "status": "healthy",
      "url": "http://localhost:9090",
      "response_time_ms": 8
    },
    "database": {
      "status": "healthy",
      "connection_pool": "5/10"
    }
  },
  "uptime_seconds": 7200,
  "version": "1.0.0"
}
```

---

### Proxy Endpoints

The API Gateway proxies requests to the model server with additional features:

#### Proxied Prediction

**Endpoint**: `POST /api/predict`

**Description**: Proxied prediction with load balancing and caching.

**Features**:
- Load balancing across multiple model servers
- Response caching for identical images
- Request rate limiting
- Enhanced error handling

**Request/Response**: Same as model server `/predict` endpoint

---

## 📈 Monitoring API (Port 9090)

### Prometheus Metrics

**Endpoint**: `GET /metrics`

**Description**: Prometheus metrics endpoint for system monitoring.

### Query API

**Endpoint**: `GET /api/v1/query`

**Description**: Query Prometheus metrics.

**Parameters**:
- `query`: PromQL query string
- `time`: Evaluation timestamp (optional)

**Example**:
```bash
curl "http://localhost:9090/api/v1/query?query=predictions_total"
```

---

## 🔧 Configuration API

### Get Configuration

**Endpoint**: `GET /config`

**Description**: Returns current system configuration.

**Response**:
```json
{
  "model": {
    "path": "models/best_chest_xray_model.pth",
    "architecture": "efficientnet_b4",
    "device": "cuda"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1
  },
  "performance": {
    "cache_size": 1000,
    "enable_batching": true,
    "batch_timeout": 100
  }
}
```

### Update Configuration

**Endpoint**: `PUT /config`

**Description**: Update system configuration (requires restart).

**Request**:
```json
{
  "performance": {
    "cache_size": 2000,
    "batch_timeout": 50
  }
}
```

---

## 📊 Rate Limits

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/predict` | 100 requests | 1 minute |
| `/predict/batch` | 10 requests | 1 minute |
| `/health` | 1000 requests | 1 minute |
| `/metrics` | 100 requests | 1 minute |

**Rate Limit Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

**Rate Limit Exceeded Response**:
```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Try again in 60 seconds.",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60
}
```

---

## 🚨 Error Handling

### Standard Error Response

All API endpoints return errors in a consistent format:

```json
{
  "error": "Error type",
  "message": "Human-readable error message",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:45Z",
  "request_id": "req_123456789",
  "details": {
    "additional": "error details"
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_FORMAT` | Unsupported image format | 400 |
| `FILE_TOO_LARGE` | Image exceeds size limit | 413 |
| `MODEL_NOT_LOADED` | Model not available | 503 |
| `PROCESSING_ERROR` | Image processing failed | 422 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily down | 503 |

---

## 🔍 Request/Response Examples

### Complete Prediction Workflow

```python
import requests
import json

# 1. Check system health
health_response = requests.get('http://localhost:8000/health')
print(f"System status: {health_response.json()['status']}")

# 2. Get model information
model_info = requests.get('http://localhost:8000/model/info')
print(f"Model accuracy: {model_info.json()['accuracy']}")

# 3. Make prediction
with open('chest_xray.jpg', 'rb') as f:
    files = {'file': f}
    prediction = requests.post('http://localhost:8000/predict', files=files)
    result = prediction.json()
    
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Processing time: {result['processing_time_ms']}ms")

# 4. Check metrics
metrics = requests.get('http://localhost:8000/metrics')
print("Metrics updated successfully")
```

### Batch Processing Example

```python
import requests
import base64
import json

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Prepare batch
image_paths = ['xray1.jpg', 'xray2.jpg', 'xray3.jpg']
encoded_images = [encode_image(path) for path in image_paths]

batch_request = {
    'images': encoded_images,
    'return_probabilities': True,
    'return_confidence': True,
    'batch_id': 'medical_screening_batch_001'
}

# Send batch request
response = requests.post(
    'http://localhost:8000/predict/batch',
    json=batch_request,
    headers={'Content-Type': 'application/json'}
)

if response.status_code == 200:
    results = response.json()
    print(f"Processed {results['batch_size']} images in {results['processing_time_ms']}ms")
    
    for i, prediction in enumerate(results['predictions']):
        print(f"Image {i+1}: {prediction['prediction']} "
              f"(confidence: {prediction['confidence']:.3f})")
else:
    print(f"Error: {response.status_code} - {response.json()}")
```

---

## 🔧 SDK and Client Libraries

### Python SDK

```python
from mlops_client import ChestXrayClient

# Initialize client
client = ChestXrayClient(base_url='http://localhost:8000')

# Single prediction
result = client.predict('chest_xray.jpg')
print(f"Prediction: {result.prediction} ({result.confidence:.3f})")

# Batch prediction
results = client.predict_batch(['xray1.jpg', 'xray2.jpg'])
for result in results:
    print(f"Prediction: {result.prediction}")

# Health check
health = client.health_check()
print(f"System healthy: {health.is_healthy}")
```

### JavaScript SDK

```javascript
import { ChestXrayClient } from 'mlops-client-js';

const client = new ChestXrayClient('http://localhost:8000');

// Single prediction
const file = document.getElementById('fileInput').files[0];
const result = await client.predict(file);
console.log(`Prediction: ${result.prediction} (${result.confidence})`);

// Health check
const health = await client.healthCheck();
console.log(`System status: ${health.status}`);
```

---

## 📚 Additional Resources

- **Interactive API Documentation**: http://localhost:8000/docs
- **OpenAPI Specification**: http://localhost:8000/openapi.json
- **Postman Collection**: Available in `docs/postman/`
- **SDK Documentation**: Available in `docs/sdk/`
- **Monitoring Dashboard**: http://localhost:3000

For more detailed examples and advanced usage, refer to the complete documentation in the `docs/` directory.