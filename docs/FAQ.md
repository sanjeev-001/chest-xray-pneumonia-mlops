# Frequently Asked Questions (FAQ)

Common questions and answers about the Chest X-Ray Pneumonia Detection MLOps System.

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Training](#training)
- [Deployment](#deployment)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## General Questions

### What is this project?

This is a complete, production-ready MLOps system for detecting pneumonia from chest X-ray images using deep learning. It includes everything from data ingestion to model deployment, monitoring, and continuous improvement.

### Who is this project for?

- **ML Engineers**: Learn MLOps best practices
- **Data Scientists**: Deploy models to production
- **DevOps Engineers**: Understand ML infrastructure
- **Healthcare Professionals**: Understand AI-assisted diagnosis
- **Students**: Learn end-to-end ML systems
- **Researchers**: Reproduce and build upon results

### What makes this different from other ML projects?

- **Complete MLOps Pipeline**: Not just a model, but a full production system
- **Production-Ready**: Includes monitoring, CI/CD, and deployment automation
- **Well-Documented**: Comprehensive documentation for all components
- **Reproducible**: Detailed instructions to reproduce results
- **Best Practices**: Follows industry standards and best practices

### Is this ready for production use?

Yes, with proper configuration and compliance review. The system includes:
- Scalable architecture
- Monitoring and alerting
- Security features
- Audit logging
- Disaster recovery

However, for medical use, ensure compliance with local regulations (HIPAA, GDPR, etc.) and conduct thorough validation.

### What license is this under?

MIT License - you're free to use, modify, and distribute this project.

## Installation & Setup

### What are the system requirements?

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB
- OS: Linux, macOS, or Windows

**Recommended:**
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 50+ GB
- GPU: NVIDIA GPU with 8GB+ VRAM
- OS: Linux (Ubuntu 20.04+)

### Do I need a GPU?

Not required, but highly recommended:
- **Training**: GPU reduces training time from hours to minutes
- **Inference**: CPU is sufficient for real-time predictions
- **Development**: CPU is fine for testing

### Can I run this on Windows?

Yes! The project supports:
- **Docker**: Recommended for Windows users
- **WSL2**: Windows Subsystem for Linux
- **Native**: With some path adjustments

### How long does setup take?

- **Docker setup**: 5-10 minutes
- **Local setup**: 15-30 minutes
- **Kubernetes setup**: 30-60 minutes
- **Full production setup**: 2-4 hours

### I'm getting port conflicts. What should I do?

```bash
# Check what's using the port
netstat -ano | findstr :8004  # Windows
lsof -i :8004  # Linux/macOS

# Option 1: Stop the conflicting service
# Option 2: Change port in docker-compose.yml
```

### Can I use a different database?

Yes, the system is designed to be flexible:
- **PostgreSQL**: Default and recommended
- **MySQL**: Supported with minor changes
- **SQLite**: For development only

## Usage

### How do I make a prediction?

**Using Python:**
```python
import requests

with open("xray.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8004/predict",
        files={"file": f}
    )
print(response.json())
```

**Using cURL:**
```bash
curl -X POST "http://localhost:8004/predict" \
  -F "file=@xray.jpg"
```

**Using the Web UI:**
Visit http://localhost:8004/docs

### What image formats are supported?

- JPEG (.jpg, .jpeg)
- PNG (.png)
- DICOM (.dcm) - with additional setup

### What's the maximum image size?

- **Default**: 10 MB per image
- **Configurable**: Can be increased in configuration
- **Recommended**: 1-5 MB for optimal performance

### Can I process multiple images at once?

Yes! Use the batch prediction endpoint:

```python
files = [
    ("files", open("xray1.jpg", "rb")),
    ("files", open("xray2.jpg", "rb")),
]
response = requests.post(
    "http://localhost:8004/predict/batch",
    files=files
)
```

### How accurate is the model?

Current performance on test set:
- **Accuracy**: 87.0%
- **Precision**: 85.0%
- **Recall**: 89.0%
- **F1-Score**: 87.0%

Note: This is for research/educational purposes. Clinical use requires validation.

## Training

### How do I train a new model?

```bash
# Basic training
python training/train_model.py

# With custom parameters
python training/train_model.py \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0001
```

### How long does training take?

- **CPU**: 6-8 hours
- **GPU (RTX 3070)**: 45-60 minutes
- **GPU (V100)**: 30-40 minutes
- **GPU (A100)**: 20-30 minutes

### Where do I get the training data?

From Kaggle:
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

See [SETUP.md](SETUP.md) for detailed instructions.

### Can I use my own dataset?

Yes! Structure your data like this:
```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### How do I tune hyperparameters?

```bash
# Automated hyperparameter optimization
python training/optimize_hyperparameters.py --trials 50
```

This uses Optuna to find optimal hyperparameters.

### Can I use a different model architecture?

Yes! Modify `training/models.py`:
```python
# Current: EfficientNet-B4
# Alternatives: ResNet, DenseNet, Vision Transformer
```

### How do I track experiments?

Use MLflow UI:
```bash
python launch_mlflow_ui.py
# Open http://localhost:5000
```

## Deployment

### How do I deploy to production?

See [docs/PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md) for detailed instructions.

Quick version:
```bash
# Build images
make build

# Deploy to Kubernetes
make k8s-deploy
```

### Can I deploy to AWS/GCP/Azure?

Yes! The system is cloud-agnostic:
- **AWS**: EKS, ECS, SageMaker
- **GCP**: GKE, Cloud Run
- **Azure**: AKS, Container Instances

See infrastructure templates in `infrastructure/`.

### How do I scale the deployment?

**Horizontal scaling:**
```bash
kubectl scale deployment deployment-service --replicas=5
```

**Auto-scaling:**
```yaml
# Already configured in k8s/deployment.yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### How do I update the model?

```bash
# Register new model
python scripts/register_model.py --model-path models/new_model.pth

# Deploy with zero downtime
python deployment/deploy_cli.py --strategy blue-green
```

### What's the inference latency?

- **Single image**: <200ms
- **Batch (10 images)**: ~500ms
- **Batch (100 images)**: ~3s

Varies based on hardware and configuration.

## Performance

### How can I improve inference speed?

1. **Use GPU**: 3-5x faster than CPU
2. **Batch predictions**: Process multiple images together
3. **Model optimization**: Use ONNX or TensorRT
4. **Caching**: Enable Redis caching
5. **Load balancing**: Distribute across multiple replicas

### How much memory does it use?

- **Training**: 4-8 GB GPU memory
- **Inference**: 2-4 GB RAM per replica
- **Database**: 1-2 GB
- **Storage**: 10-20 GB for models and data

### Can I reduce the model size?

Yes, several options:
1. **Quantization**: Reduce precision (INT8)
2. **Pruning**: Remove unnecessary weights
3. **Knowledge distillation**: Train smaller model
4. **Different architecture**: Use MobileNet or EfficientNet-B0

### How many requests can it handle?

- **Single replica**: 50-100 requests/second
- **With auto-scaling**: 500-1000 requests/second
- **With optimization**: 1000+ requests/second

## Troubleshooting

### Services won't start

```bash
# Check Docker is running
docker info

# Check logs
docker-compose logs

# Restart services
docker-compose restart
```

### Out of memory errors

```bash
# Reduce batch size
# In config/training.yaml:
batch_size: 16  # or 8

# Or increase Docker memory limit
# Docker Desktop > Settings > Resources
```

### Model predictions are wrong

1. **Check image preprocessing**: Ensure correct normalization
2. **Verify model version**: Use the correct model
3. **Check input format**: Ensure correct image format
4. **Review logs**: Check for errors or warnings

### Can't connect to services

```bash
# Check services are running
docker-compose ps

# Check ports
netstat -an | grep 8004

# Check firewall
# Ensure ports 8001-8005 are open
```

### Training is very slow

1. **Use GPU**: Much faster than CPU
2. **Reduce image size**: Smaller images train faster
3. **Reduce batch size**: If memory limited
4. **Use mixed precision**: Enable AMP
5. **Optimize data loading**: Increase num_workers

### Tests are failing

```bash
# Update dependencies
pip install -r requirements-dev.txt --upgrade

# Clear cache
pytest --cache-clear

# Run specific test
pytest tests/test_specific.py -v
```

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Quick ways to contribute:
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests
- Help others in discussions

### I found a bug. What should I do?

1. Check if it's already reported in [Issues](https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops/issues)
2. If not, create a new issue with:
   - Description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
   - Error messages/logs

### I have a feature idea

Great! Please:
1. Check existing feature requests
2. Create a new issue with the `enhancement` label
3. Describe the feature and its benefits
4. Discuss with maintainers before implementing

### How do I submit a pull request?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit PR with clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Do you accept financial contributions?

Currently, we don't have a sponsorship program, but you can:
- Star the repository
- Share with others
- Contribute code or documentation
- Help answer questions

## Additional Questions

### Is this HIPAA compliant?

The system includes HIPAA-ready features, but full compliance requires:
- Proper deployment configuration
- Business Associate Agreements
- Security policies and procedures
- Regular audits
- Staff training

Consult with your compliance team.

### Can I use this commercially?

Yes! The MIT License allows commercial use. However:
- Ensure compliance with medical device regulations
- Conduct thorough validation
- Consider liability and insurance
- Consult legal counsel

### How do I cite this project?

```bibtex
@software{chest_xray_mlops,
  title = {Chest X-Ray Pneumonia Detection MLOps System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops}
}
```

### Where can I get help?

1. **Documentation**: Check [docs/](docs/) folder
2. **Issues**: Search existing issues
3. **Discussions**: Ask in GitHub Discussions
4. **Email**: mlops@example.com

### How often is this updated?

- **Bug fixes**: As needed
- **Security updates**: Immediately
- **Feature releases**: Monthly
- **Major versions**: Quarterly

### Can I hire you for consulting?

Please reach out via email: mlops@example.com

### What's the roadmap?

See [GitHub Projects](https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops/projects) for current roadmap.

Planned features:
- Multi-disease detection
- Federated learning
- Edge deployment
- Mobile app
- Advanced explainability

## Still Have Questions?

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops/discussions)
- **Email**: mlops@example.com

We're here to help! 🚀
