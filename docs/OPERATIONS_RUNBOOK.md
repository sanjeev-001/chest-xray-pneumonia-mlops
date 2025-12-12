# Chest X-Ray Pneumonia Detection MLOps - Operations Runbook

## 🚀 Deployment Procedures

### Quick Deployment

**Complete System Deployment**
```bash
# Deploy entire MLOps system
python scripts/deploy_complete_system.py

# Validate deployment
python scripts/validate_complete_system.py
```

**Individual Service Deployment**
```bash
# Model server only
python -m uvicorn deployment.model_server:app --host 0.0.0.0 --port 8000

# API gateway
python -m uvicorn deployment.api:app --host 0.0.0.0 --port 8080

# Monitoring stack
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

### Production Deployment

**Blue-Green Deployment**
```bash
# Deploy to blue environment
python scripts/deployment_manager.py deploy --environment blue --version v2.0.0

# Health check
python scripts/deployment_manager.py health-check --environment blue

# Switch traffic
python scripts/deployment_manager.py switch-traffic --from green --to blue

# Rollback if needed
python scripts/deployment_manager.py rollback --to-version v1.9.0
```

**Kubernetes Deployment**
```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n mlops

# Scale deployment
kubectl scale deployment model-server --replicas=3 -n mlops
```

## 🔧 Configuration Management

### Environment Variables

**Production Configuration**
```bash
export MODEL_PATH="models/best_chest_xray_model.pth"
export DEVICE="cuda"
export LOG_LEVEL="INFO"
export METRICS_ENABLED="true"
export CACHE_SIZE="5000"
export BATCH_SIZE="16"
```

**Development Configuration**
```bash
export MODEL_PATH="models/best_chest_xray_model.pth"
export DEVICE="cpu"
export LOG_LEVEL="DEBUG"
export METRICS_ENABLED="false"
export CACHE_SIZE="100"
export BATCH_SIZE="4"
```

### Configuration Files

**Update Model Server Config**
```yaml
# deployment/config.yaml
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
  batch_timeout: 100
```

## 📊 Monitoring & Alerting

### Health Checks

**System Health Check**
```bash
# Complete system validation
python scripts/validate_complete_system.py

# Individual service checks
curl http://localhost:8000/health
curl http://localhost:8080/health
curl http://localhost:9090/-/healthy
```

**Automated Health Monitoring**
```bash
# Set up health check cron job
echo "*/5 * * * * /usr/bin/python3 /path/to/scripts/validate_complete_system.py" | crontab -
```

### Metrics Collection

**View Current Metrics**
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# System metrics
python monitoring/metrics_collector.py --show-current

# Performance metrics
python deployment/performance_test.py --quick-check
```

**Grafana Dashboard Access**
- URL: http://localhost:3000
- Username: admin
- Password: Check deployment logs or use `admin`

### Alert Configuration

**Critical Alerts**
- Model accuracy < 80%
- Response time > 2 seconds
- Error rate > 5%
- System memory > 90%
- Disk space < 10%

**Alert Channels**
```yaml
# monitoring/alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@company.com'
    subject: 'MLOps Alert: {{ .GroupLabels.alertname }}'
```

## 🔄 Model Management

### Model Deployment

**Deploy New Model Version**
```bash
# Register new model
python training/model_registry.py register \
  --name "chest_xray_efficientnet_b4" \
  --version "2.0.0" \
  --path "models/new_model_v2.pth" \
  --accuracy 0.89

# Deploy model
python scripts/deployment_manager.py deploy-model \
  --model-version "2.0.0" \
  --environment production
```

**Model Rollback**
```bash
# Rollback to previous version
python scripts/deployment_manager.py rollback-model \
  --to-version "1.9.0" \
  --reason "Performance degradation"
```

### Model Registry Operations

**List Available Models**
```bash
python training/model_registry.py list
```

**Model Metadata**
```bash
python training/model_registry.py info --version "2.0.0"
```

**Archive Old Models**
```bash
python training/model_registry.py archive --older-than 30d
```

## 🔧 Maintenance Procedures

### Daily Maintenance

**Daily Health Check**
```bash
#!/bin/bash
# daily_maintenance.sh

echo "=== Daily MLOps System Maintenance ==="
date

# System health check
python scripts/validate_complete_system.py

# Check disk space
df -h

# Check memory usage
free -h

# Check recent errors
grep -i error logs/*.log | tail -20

# Backup database
python scripts/backup_database.py

echo "=== Maintenance Complete ==="
```

### Weekly Maintenance

**Weekly Tasks**
```bash
#!/bin/bash
# weekly_maintenance.sh

# Update dependencies
pip install --upgrade -r requirements.txt

# Run comprehensive tests
python scripts/run_tests.py all

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete

# Model performance review
python monitoring/performance_monitor.py --weekly-report

# Security scan
python scripts/security_scan.py
```

### Monthly Maintenance

**Monthly Tasks**
```bash
#!/bin/bash
# monthly_maintenance.sh

# Full system backup
python scripts/backup_complete_system.py

# Performance optimization
python deployment/performance_optimizer.py --analyze

# Security audit
python scripts/security_audit.py --full

# Capacity planning
python monitoring/capacity_planner.py --monthly-report

# Update documentation
python scripts/update_documentation.py
```

## 🚨 Incident Response

### Common Issues

#### Model Server Not Responding

**Symptoms:**
- HTTP 503 errors
- Connection timeouts
- Health check failures

**Diagnosis:**
```bash
# Check process status
ps aux | grep model_server

# Check logs
tail -f logs/model_server.log

# Check system resources
htop
nvidia-smi  # For GPU systems
```

**Resolution:**
```bash
# Restart model server
python scripts/deployment_manager.py restart --service model-server

# If restart fails, redeploy
python scripts/deploy_complete_system.py --force
```

#### High Memory Usage

**Symptoms:**
- Slow response times
- Out of memory errors
- System instability

**Diagnosis:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check model memory usage
python -c "
import torch
print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
print(f'GPU Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB')
"
```

**Resolution:**
```bash
# Clear cache
python -c "
import torch
torch.cuda.empty_cache()
"

# Reduce batch size
export BATCH_SIZE="4"

# Restart services
python scripts/deployment_manager.py restart --all
```

#### Model Accuracy Degradation

**Symptoms:**
- Accuracy alerts
- Increased false positives/negatives
- User complaints

**Diagnosis:**
```bash
# Check recent predictions
python monitoring/audit_trail.py --recent --accuracy-check

# Data drift analysis
python monitoring/drift_detector.py --analyze-recent

# Model performance metrics
python monitoring/performance_monitor.py --detailed-report
```

**Resolution:**
```bash
# Trigger retraining
python training/retraining_orchestrator.py --trigger-immediate

# Rollback to previous model if needed
python scripts/deployment_manager.py rollback-model --to-version "1.9.0"

# Investigate data quality
python data_pipeline/validation.py --check-recent-data
```

### Emergency Procedures

#### Complete System Failure

**Immediate Actions:**
1. Check system status: `python scripts/validate_complete_system.py`
2. Check infrastructure: `kubectl get pods -n mlops`
3. Review recent changes: `git log --oneline -10`
4. Check system resources: `htop`, `df -h`

**Recovery Steps:**
```bash
# Emergency deployment from backup
python scripts/emergency_recovery.py --from-backup latest

# Validate recovery
python scripts/validate_complete_system.py --comprehensive

# Notify stakeholders
python scripts/notify_stakeholders.py --incident "System Recovery Complete"
```

#### Data Breach Response

**Immediate Actions:**
1. Isolate affected systems
2. Preserve evidence
3. Notify security team
4. Document incident

**Technical Steps:**
```bash
# Stop all services
python scripts/deployment_manager.py stop --all

# Secure audit logs
cp -r logs/ /secure/backup/incident-$(date +%Y%m%d)

# Run security scan
python scripts/security_scan.py --comprehensive --output incident-report.json

# Generate incident report
python scripts/incident_report.py --type security --output incident-$(date +%Y%m%d).pdf
```

## 📋 Operational Checklists

### Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Backup completed
- [ ] Rollback plan prepared
- [ ] Stakeholders notified
- [ ] Monitoring configured

### Post-Deployment Checklist

- [ ] Health checks passing
- [ ] Metrics collecting properly
- [ ] Alerts configured
- [ ] Performance within SLA
- [ ] No error spikes
- [ ] User acceptance confirmed
- [ ] Documentation updated
- [ ] Team notified

### Incident Response Checklist

- [ ] Incident detected and logged
- [ ] Severity assessed
- [ ] Response team notified
- [ ] Initial diagnosis completed
- [ ] Mitigation steps taken
- [ ] System stability confirmed
- [ ] Root cause identified
- [ ] Permanent fix implemented
- [ ] Post-incident review scheduled

## 📞 Contact Information

### On-Call Rotation

**Primary On-Call:** +1-555-MLOPS-1
**Secondary On-Call:** +1-555-MLOPS-2
**Escalation:** +1-555-MLOPS-ESC

### Team Contacts

- **MLOps Team Lead:** mlops-lead@company.com
- **DevOps Engineer:** devops@company.com
- **Data Scientist:** data-science@company.com
- **Security Team:** security@company.com

### External Vendors

- **Cloud Provider:** AWS Support
- **Monitoring:** Grafana Support
- **Security:** Security Vendor Support

## 📚 Additional Resources

- **Runbook Repository:** https://github.com/company/mlops-runbooks
- **Documentation:** https://docs.company.com/mlops
- **Monitoring Dashboard:** http://monitoring.company.com
- **Incident Management:** https://company.pagerduty.com

This operations runbook provides comprehensive procedures for deploying, monitoring, and maintaining the Chest X-Ray Pneumonia Detection MLOps system.