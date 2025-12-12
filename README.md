# 🏥 Chest X-Ray Pneumonia Detection MLOps System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-326CE5.svg)](https://kubernetes.io/)

A comprehensive, production-ready MLOps system for automated chest X-ray analysis and pneumonia detection using deep learning. This system provides end-to-end automation from data ingestion to model deployment, monitoring, and continuous improvement.

> **Note**: This is a complete MLOps implementation showcasing best practices for deploying machine learning models in production healthcare environments.

## 🏥 Overview

This project implements a complete MLOps pipeline for chest X-ray pneumonia detection, featuring:

- **🔄 Automated Data Pipeline**: Data ingestion, validation, preprocessing, and versioning
- **🧠 Advanced Model Training**: EfficientNet-B4 based deep learning model with 87% accuracy
- **🚀 Production Deployment**: FastAPI-based inference server with auto-scaling
- **📊 Comprehensive Monitoring**: Real-time performance monitoring and drift detection
- **🔁 Automated Retraining**: Continuous model improvement based on performance metrics
- **🔒 Enterprise Security**: HIPAA-ready architecture with comprehensive audit trails
- **⚡ High Performance**: <200ms inference time with batch processing support

## 🎯 Key Features

### Core Capabilities
- **Real-time Pneumonia Detection**: Analyze chest X-ray images with 87%+ accuracy
- **Batch Processing**: Process multiple images simultaneously for high throughput
- **Blue-Green Deployment**: Zero-downtime deployments with automatic rollback
- **Auto-scaling**: Kubernetes-based horizontal scaling based on demand
- **Drift Detection**: Automatic detection of data and concept drift
- **Model Registry**: Centralized model versioning and metadata management
- **Audit Trail**: Complete audit logging for compliance and debugging

### Performance Metrics
- **Accuracy**: 87.0% on test dataset
- **Precision**: 85.0% for pneumonia detection
- **Recall**: 89.0% for pneumonia cases
- **F1-Score**: 87.0% overall performance
- **Inference Time**: <200ms per image
- **Throughput**: 50-100 images/second per replica

## Architecture

The system consists of 5 microservices:

1. **Data Pipeline Service** (Port 8001) - Data ingestion and preprocessing
2. **Training Service** (Port 8002) - Model training and evaluation
3. **Model Registry Service** (Port 8003) - Model storage and versioning
4. **Deployment Service** (Port 8004) - Model serving and prediction API
5. **Monitoring Service** (Port 8005) - Performance monitoring and alerting

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Kubernetes (optional, for production deployment)

### Local Development Setup

1. **Clone and setup the project:**
   ```bash
   git clone <repository-url>
   cd chest-xray-pneumonia-mlops
   make setup-dev
   ```

2. **Start local services:**
   ```bash
   make docker-up
   ```

3. **Access services:**
   - Data Pipeline: http://localhost:8001
   - Training Service: http://localhost:8002
   - Model Registry: http://localhost:8003
   - Deployment API: http://localhost:8004
   - Monitoring Service: http://localhost:8005
   - MLflow UI: http://localhost:5000
   - MinIO Console: http://localhost:9001

### Production Deployment

1. **Build Docker images:**
   ```bash
   make build
   ```

2. **Deploy to Kubernetes:**
   ```bash
   make k8s-deploy
   ```

3. **Check deployment status:**
   ```bash
   make k8s-status
   ```

## Development

### Project Structure

```
├── src/
│   ├── data_pipeline/          # Data ingestion and preprocessing service
│   ├── training/              # Model training service
│   ├── model_registry/        # Model storage and versioning service
│   ├── deployment/           # Model serving API service
│   └── monitoring/           # Performance monitoring service
├── docs/                 # Documentation files
├── scripts/              # Utility scripts and batch files
├── tests/               # Test suite
├── k8s/                 # Kubernetes deployment manifests
├── .github/             # GitHub workflows and templates
├── docker-compose.yml   # Local development environment
├── Makefile            # Development commands
└── pyproject.toml      # Python package configuration
```

### Available Commands

```bash
make help           # Show available commands
make install        # Install production dependencies
make install-dev    # Install development dependencies
make test          # Run tests
make lint          # Run linting
make format        # Format code
make build         # Build Docker images
make docker-up     # Start local services
make docker-down   # Stop local services
make k8s-deploy    # Deploy to Kubernetes
make k8s-delete    # Delete Kubernetes resources
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_data_pipeline.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run all quality checks:
```bash
make lint
make format
make test
```

## API Documentation

Once the services are running, you can access the interactive API documentation:

- Data Pipeline API: http://localhost:8001/docs
- Training API: http://localhost:8002/docs
- Model Registry API: http://localhost:8003/docs
- Deployment API: http://localhost:8004/docs
- Monitoring API: http://localhost:8005/docs

## Configuration

Copy `.env.example` to `.env` and adjust the configuration as needed:

```bash
cp .env.example .env
```

Key configuration options:
- Database connection settings
- MinIO/S3 storage configuration
- MLflow tracking server URL
- Service endpoints

## Monitoring and Observability

The system includes comprehensive monitoring:

- **Metrics**: Prometheus metrics for all services
- **Logging**: Structured logging with correlation IDs
- **Tracing**: Distributed tracing for request flows
- **Dashboards**: Grafana dashboards for visualization
- **Alerting**: Automated alerts for performance degradation

## 📚 Documentation

Comprehensive documentation is available:

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Setup Guide](SETUP.md)** - Detailed installation instructions
- **[Architecture](docs/ARCHITECTURE.md)** - System design and components
- **[Reproducibility Guide](docs/REPRODUCIBILITY.md)** - Reproduce reported results
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute
- **[User Guide](docs/USER_GUIDE.md)** - Complete usage documentation
- **[API Documentation](docs/API_DOCUMENTATION.md)** - API reference
- **[Operations Runbook](docs/OPERATIONS_RUNBOOK.md)** - Production operations
- **[CI/CD Setup](docs/CICD_SETUP.md)** - Continuous integration/deployment

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code of conduct
- Development workflow
- Coding standards
- Testing guidelines
- Pull request process

Quick contribution steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📊 Project Status

- ✅ Data Pipeline - Complete
- ✅ Model Training - Complete
- ✅ Model Registry - Complete
- ✅ Deployment API - Complete
- ✅ Monitoring System - Complete
- ✅ CI/CD Pipeline - Complete
- ✅ Documentation - Complete
- 🚧 Multi-model Support - In Progress
- 📋 Federated Learning - Planned

## 🙏 Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle
- Model Architecture: EfficientNet by Google Research
- MLOps Tools: MLflow, DVC, Prometheus, Grafana
- Community: Thanks to all contributors and users

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, issues, or suggestions:

- Open an [Issue](https://github.com/sanjeevi-001/chest-xray-pneumonia-mlops/issues)
- Start a [Discussion](https://github.com/sanjeevi-001/chest-xray-pneumonia-mlops/discussions)
- Email: [sanjeeviraj2018@gmail.com]

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

## 🔗 Related Projects

- [MLOps Best Practices](https://ml-ops.org/)
- [PyTorch Medical Imaging](https://github.com/Project-MONAI/MONAI)
- [MLflow Examples](https://github.com/mlflow/mlflow)

---

**Built with ❤️ for the MLOps and Healthcare AI community**