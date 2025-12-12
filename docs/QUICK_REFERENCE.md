# Quick Reference Card

One-page reference for the Chest X-Ray Pneumonia Detection MLOps System.

## 🚀 Quick Start

```bash
# Clone and start
git clone https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops.git
cd chest-xray-pneumonia-mlops
docker-compose up -d

# Test prediction
curl -X POST "http://localhost:8004/predict" -F "file=@xray.jpg"
```

## 📚 Documentation Quick Links

| Need | Document |
|------|----------|
| Get started in 5 min | [QUICKSTART.md](QUICKSTART.md) |
| Detailed setup | [SETUP.md](SETUP.md) |
| Common questions | [FAQ.md](FAQ.md) |
| System architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Reproduce results | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |
| API reference | [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) |
| Deploy to production | [docs/PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md) |

## 🔗 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Deployment API | http://localhost:8004 | Predictions |
| MLflow UI | http://localhost:5000 | Experiments |
| MinIO Console | http://localhost:9001 | Storage |
| Grafana | http://localhost:3000 | Monitoring |
| API Docs | http://localhost:8004/docs | Interactive API |

## 💻 Common Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Run tests
make test

# Train model
python training/train_model.py

# Deploy
python deployment/deploy_cli.py
```

## 🐍 Python API

```python
import requests

# Single prediction
with open("xray.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8004/predict",
        files={"file": f}
    )
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
files = [("files", open(f"xray{i}.jpg", "rb")) for i in range(10)]
response = requests.post(
    "http://localhost:8004/predict/batch",
    files=files
)
```

## 📊 Performance Metrics

- **Accuracy**: 87.0%
- **Precision**: 85.0%
- **Recall**: 89.0%
- **F1-Score**: 87.0%
- **Inference Time**: <200ms
- **Throughput**: 50-100 images/sec

## 🏗️ Architecture

```
Client → API Gateway → Deployment Service → Model
                    ↓
         Monitoring Service → Alerts
                    ↓
         Data Pipeline → Storage
                    ↓
         Training Service → Model Registry
```

## 🔧 Configuration

```bash
# Environment variables (.env)
DATABASE_URL=postgresql://user:pass@localhost:5432/mlops
MINIO_ENDPOINT=localhost:9000
MLFLOW_TRACKING_URI=http://localhost:5000
```

## 🧪 Testing

```bash
# All tests
make test

# Specific test
pytest tests/test_deployment.py -v

# With coverage
pytest --cov=. --cov-report=html
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | Change port in docker-compose.yml |
| Out of memory | Reduce batch_size in config |
| Services won't start | Check `docker-compose logs` |
| Can't connect | Check firewall, ensure ports open |

## 📦 Project Structure

```
├── data_pipeline/      # Data processing
├── training/          # Model training
├── model_registry/    # Model versioning
├── deployment/        # Inference API
├── monitoring/        # System monitoring
├── tests/            # Test suite
├── docs/             # Documentation
├── k8s/              # Kubernetes configs
└── docker-compose.yml # Local setup
```

## 🔐 Security

- Use strong passwords
- Enable TLS in production
- Keep dependencies updated
- Follow [SECURITY.md](SECURITY.md)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit PR

See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: mlops@example.com
- **Docs**: [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

## 📝 License

MIT License - see [LICENSE](LICENSE)

---

**For complete documentation, see [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)**
