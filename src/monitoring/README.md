# Chest X-Ray Pneumonia Detection - Monitoring System

This monitoring system provides comprehensive performance monitoring, drift detection, and alerting for the chest X-ray pneumonia detection MLOps pipeline.

## Features

### 1. Performance Monitoring
- **Model Accuracy Tracking**: Monitors model accuracy over time with 80% threshold alerting
- **Latency Monitoring**: Tracks response times with 2-second threshold (per requirements)
- **Throughput Monitoring**: Measures requests per second
- **Error Rate Monitoring**: Tracks prediction errors and failures
- **System Resource Monitoring**: CPU, memory, and GPU usage tracking

### 2. Drift Detection
- **Data Drift Detection**: Monitors changes in input image statistics
- **Concept Drift Detection**: Detects model performance degradation
- **Prediction Drift Detection**: Tracks changes in prediction distributions
- **Baseline Management**: Establishes and updates baseline statistics

### 3. Alerting System
- **Multi-Channel Notifications**: Email, Slack, and webhook support
- **Severity-Based Routing**: Critical, warning, and info alert levels
- **Alert Suppression**: Prevents alert spam with cooldown periods
- **Manual Resolution**: Allows manual alert acknowledgment

### 4. Real-Time Dashboards
- **Grafana Integration**: Pre-configured dashboards for visualization
- **Prometheus Metrics**: Standardized metrics export
- **Custom Visualizations**: Model-specific performance charts

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  Training       │    │  Deployment     │
│                 │    │  Pipeline       │    │  Service        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Metrics Collector      │
                    │  - Prediction metrics     │
                    │  - System metrics         │
                    │  - Model performance      │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│ Drift Detector    │  │ Performance       │  │ Prometheus        │
│ - Data drift      │  │ Monitor           │  │ Exporter          │
│ - Concept drift   │  │ - Alerting        │  │ - Metrics export  │
│ - Baseline mgmt   │  │ - Notifications   │  │ - Grafana feed    │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

## Quick Start

### 1. Using Docker Compose (Recommended)

```bash
# Start the complete monitoring stack
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Access services
# - Grafana: http://localhost:3000 (admin/admin123)
# - Prometheus: http://localhost:9090
# - Monitoring API: http://localhost:8005
```

### 2. Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://postgres:password@localhost:5432/chest_xray_metrics"
export AUTO_START_PERFORMANCE_MONITORING=true
export AUTO_START_PROMETHEUS=true

# Start monitoring service
python -m monitoring.main
```

## API Endpoints

### Core Monitoring
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /predictions` - Record prediction metric
- `GET /predictions` - Get recent predictions
- `GET /predictions/summary` - Get prediction summary
- `GET /predictions/accuracy` - Get accuracy metrics

### Drift Detection
- `POST /drift/baseline` - Establish baseline statistics
- `POST /drift/detect` - Run drift detection
- `GET /drift/history` - Get drift detection history

### Performance Monitoring
- `GET /performance/summary` - Get performance summary
- `GET /performance/alerts` - Get active alerts
- `GET /performance/alerts/history` - Get alert history
- `POST /performance/alerts/{alert_key}/resolve` - Resolve alert
- `POST /performance/monitoring/start` - Start monitoring
- `POST /performance/monitoring/stop` - Stop monitoring

### Configuration
- `POST /alerts/configure` - Configure notification channels
- `POST /metrics/prometheus/start` - Start Prometheus exporter

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/chest_xray_metrics

# Monitoring
AUTO_START_PERFORMANCE_MONITORING=true
AUTO_START_PROMETHEUS=true
PROMETHEUS_PORT=8080

# Alerting
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_FROM=alerts@your-domain.com
ALERT_EMAIL_TO=admin@your-domain.com

# Slack (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### Alert Configuration

Configure email notifications:
```bash
curl -X POST "http://localhost:8005/alerts/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "email_config": {
      "smtp_host": "smtp.gmail.com",
      "smtp_port": 587,
      "username": "your-email@gmail.com",
      "password": "your-app-password",
      "from_email": "alerts@your-domain.com",
      "to_emails": ["admin@your-domain.com"]
    }
  }'
```

Configure Slack notifications:
```bash
curl -X POST "http://localhost:8005/alerts/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "slack_webhook": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  }'
```

## Drift Detection

### 1. Establish Baseline

Before running drift detection, establish baseline statistics:

```bash
curl -X POST "http://localhost:8005/drift/baseline" \
  -H "Content-Type: application/json" \
  -d '{"hours": 168}'  # Use 1 week of data
```

### 2. Run Drift Detection

```bash
curl -X POST "http://localhost:8005/drift/detect" \
  -H "Content-Type: application/json" \
  -d '{"hours": 24}'  # Check last 24 hours
```

### 3. Interpret Results

The drift detection returns:
- **Data Drift Score**: Changes in input image characteristics
- **Concept Drift Score**: Changes in model accuracy/performance
- **Prediction Drift Score**: Changes in prediction distributions
- **Overall Drift Score**: Maximum of the above scores
- **Recommendations**: Actionable suggestions based on detected drift

## Performance Thresholds

The system monitors the following thresholds (configurable):

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Model Accuracy | < 80% | Critical |
| Response Time | > 2 seconds | Warning |
| Error Rate | > 5% | Critical |
| CPU Usage | > 90% | Warning |
| Memory Usage | > 90% | Warning |
| Throughput | < 1 RPS | Warning |

## Grafana Dashboards

The system includes pre-configured Grafana dashboards:

1. **Model Performance Dashboard**
   - Accuracy over time
   - Response time distribution
   - Prediction distribution
   - Error rates

2. **System Resources Dashboard**
   - CPU and memory usage
   - GPU memory usage
   - Network and disk I/O

3. **Drift Detection Dashboard**
   - Drift scores over time
   - Alert timeline
   - Baseline comparisons

4. **Alerts Dashboard**
   - Active alerts
   - Alert history
   - Alert resolution times

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check PostgreSQL is running
   docker ps | grep postgres
   
   # Check connection string
   echo $DATABASE_URL
   ```

2. **No Metrics Available**
   ```bash
   # Check if predictions are being recorded
   curl "http://localhost:8005/predictions?limit=10"
   
   # Record a test prediction
   curl -X POST "http://localhost:8005/predictions" \
     -H "Content-Type: application/json" \
     -d '{
       "prediction": "PNEUMONIA",
       "confidence": 0.95,
       "processing_time_ms": 45.0,
       "model_version": "v1.0.0"
     }'
   ```

3. **Drift Detection Fails**
   ```bash
   # Establish baseline first
   curl -X POST "http://localhost:8005/drift/baseline" \
     -H "Content-Type: application/json" \
     -d '{"hours": 24}'
   ```

4. **Alerts Not Sending**
   ```bash
   # Check alert configuration
   curl "http://localhost:8005/performance/alerts"
   
   # Test email configuration
   python -c "
   import smtplib
   server = smtplib.SMTP('smtp.gmail.com', 587)
   server.starttls()
   server.login('your-email', 'your-password')
   print('Email config OK')
   "
   ```

### Logs

Check service logs:
```bash
# Docker Compose
docker-compose -f docker-compose.monitoring.yml logs monitoring-service

# Manual setup
tail -f /var/log/chest-xray-monitoring.log
```

## Development

### Running Tests

```bash
# Run all monitoring tests
pytest tests/test_monitoring_metrics.py -v

# Run specific test categories
pytest tests/test_monitoring_metrics.py::TestDriftDetector -v
pytest tests/test_monitoring_metrics.py::TestPerformanceMonitor -v
```

### Adding Custom Metrics

1. **Add to MetricsCollector**:
   ```python
   def record_custom_metric(self, metric_name: str, value: float):
       # Implementation
   ```

2. **Add to Prometheus Exporter**:
   ```python
   custom_gauge = Gauge('chest_xray_custom_metric', 'Description')
   custom_gauge.set(value)
   ```

3. **Add to Grafana Dashboard**:
   - Edit `grafana_dashboard.json`
   - Add new panel with custom metric query

### Contributing

1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Test with Docker Compose setup

## Security Considerations

1. **Database Security**:
   - Use strong passwords
   - Enable SSL connections
   - Restrict network access

2. **API Security**:
   - Add authentication middleware
   - Rate limiting
   - Input validation

3. **Alert Security**:
   - Secure webhook endpoints
   - Encrypt email credentials
   - Use environment variables for secrets

## Performance Optimization

1. **Database Optimization**:
   - Regular cleanup of old metrics
   - Proper indexing
   - Connection pooling

2. **Memory Management**:
   - Limit in-memory metric storage
   - Efficient data structures
   - Garbage collection tuning

3. **Network Optimization**:
   - Batch metric updates
   - Compress large payloads
   - Connection reuse

## License

This monitoring system is part of the Chest X-Ray Pneumonia Detection MLOps pipeline.