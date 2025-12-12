# System Architecture

Comprehensive architecture documentation for the Chest X-Ray Pneumonia Detection MLOps System.

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Architecture Diagrams](#architecture-diagrams)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Decisions](#design-decisions)
- [Scalability](#scalability)
- [Security](#security)

## Overview

This MLOps system implements a microservices architecture for end-to-end machine learning operations, from data ingestion to model deployment and monitoring.

### Key Principles

- **Microservices**: Independent, loosely coupled services
- **Containerization**: Docker containers for consistency
- **Orchestration**: Kubernetes for production deployment
- **Observability**: Comprehensive monitoring and logging
- **Automation**: CI/CD pipelines for continuous delivery
- **Scalability**: Horizontal scaling for high availability

## System Components

### 1. Data Pipeline Service (Port 8001)

**Purpose**: Data ingestion, validation, preprocessing, and versioning

**Key Features**:
- Data ingestion from multiple sources
- Image validation and quality checks
- Medical-appropriate augmentation
- Data versioning with DVC
- Storage management (MinIO/S3)

**Technologies**:
- FastAPI for REST API
- OpenCV for image processing
- Albumentations for augmentation
- DVC for data versioning
- MinIO for object storage

**API Endpoints**:
```
POST /ingest          - Ingest new data
GET  /validate        - Validate dataset
POST /preprocess      - Preprocess images
GET  /versions        - List data versions
POST /version/create  - Create new version
```

### 2. Training Service (Port 8002)

**Purpose**: Model training, hyperparameter optimization, and experiment tracking

**Key Features**:
- Distributed training support
- Hyperparameter optimization (Optuna)
- Experiment tracking (MLflow)
- Model checkpointing
- Early stopping
- Learning rate scheduling

**Technologies**:
- PyTorch for deep learning
- EfficientNet-B4 architecture
- MLflow for experiment tracking
- Optuna for hyperparameter tuning
- Ray for distributed training

**API Endpoints**:
```
POST /train           - Start training job
GET  /train/{job_id}  - Get training status
POST /optimize        - Run hyperparameter optimization
GET  /experiments     - List experiments
GET  /metrics         - Get training metrics
```

### 3. Model Registry Service (Port 8003)

**Purpose**: Model versioning, metadata management, and artifact storage

**Key Features**:
- Model versioning
- Metadata tracking
- Model comparison
- Artifact storage
- Model lineage
- Model promotion workflow

**Technologies**:
- MLflow Model Registry
- PostgreSQL for metadata
- MinIO for model artifacts
- FastAPI for REST API

**API Endpoints**:
```
POST /models/register     - Register new model
GET  /models              - List all models
GET  /models/{name}       - Get model details
POST /models/{name}/stage - Promote model stage
GET  /models/compare      - Compare models
```

### 4. Deployment Service (Port 8004)

**Purpose**: Model serving, inference, and prediction API

**Key Features**:
- Real-time inference
- Batch prediction
- Model versioning
- A/B testing support
- Auto-scaling
- Load balancing

**Technologies**:
- FastAPI for REST API
- TorchServe for model serving
- Redis for caching
- NGINX for load balancing

**API Endpoints**:
```
POST /predict         - Single image prediction
POST /predict/batch   - Batch prediction
GET  /models          - List available models
POST /models/load     - Load specific model version
GET  /health          - Health check
GET  /metrics         - Inference metrics
```

### 5. Monitoring Service (Port 8005)

**Purpose**: System monitoring, drift detection, and alerting

**Key Features**:
- Performance monitoring
- Data drift detection
- Model drift detection
- Alerting system
- Audit logging
- Explainability (SHAP, Grad-CAM)

**Technologies**:
- Prometheus for metrics
- Grafana for visualization
- Evidently for drift detection
- SHAP for explainability
- AlertManager for alerts

**API Endpoints**:
```
GET  /metrics         - System metrics
POST /drift/detect    - Detect drift
GET  /alerts          - List active alerts
POST /explain         - Explain prediction
GET  /audit           - Audit trail
```

## Architecture Diagrams

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  (Web UI, Mobile App, External Systems, API Clients)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway / Load Balancer                 │
│                         (NGINX / Ingress)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Data Pipeline│    │  Deployment  │    │  Monitoring  │
│   Service    │    │   Service    │    │   Service    │
│  (Port 8001) │    │ (Port 8004)  │    │ (Port 8005)  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Training   │    │    Model     │    │  Prometheus  │
│   Service    │    │   Registry   │    │   Grafana    │
│ (Port 8002)  │    │ (Port 8003)  │    │              │
└──────┬───────┘    └──────┬───────┘    └──────────────┘
       │                   │
       └────────┬──────────┘
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ MinIO/S3 │  │PostgreSQL│  │  MLflow  │  │  Redis   │       │
│  │ (Objects)│  │(Metadata)│  │(Tracking)│  │ (Cache)  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
┌──────────────┐
│ Raw Data     │
│ (X-Ray Images)│
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Data Pipeline Service│
│ • Validation         │
│ • Cleaning           │
│ • Augmentation       │
│ • Versioning         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Processed Data       │
│ (MinIO Storage)      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Training Service     │
│ • Model Training     │
│ • Hyperparameter Opt │
│ • Experiment Tracking│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Model Registry       │
│ • Version Management │
│ • Metadata Storage   │
│ • Model Artifacts    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Deployment Service   │
│ • Model Loading      │
│ • Inference          │
│ • A/B Testing        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Monitoring Service   │
│ • Performance Track  │
│ • Drift Detection    │
│ • Alerting           │
└──────────────────────┘
```

### Deployment Architecture (Kubernetes)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                          │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    Ingress Controller                   │    │
│  │              (NGINX / Traefik / Istio)                 │    │
│  └────────────────────────┬───────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┼───────────────────────────────┐    │
│  │                  Service Mesh (Optional)                │    │
│  │                    (Istio / Linkerd)                    │    │
│  └────────────────────────┬───────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴───────────────────────────────┐    │
│  │                    Namespace: mlops                     │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │    │
│  │  │ Deployment   │  │ Deployment   │  │ Deployment   │ │    │
│  │  │ data-pipeline│  │   training   │  │ model-registry│ │    │
│  │  │ Replicas: 2  │  │ Replicas: 1  │  │ Replicas: 2  │ │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐                    │    │
│  │  │ Deployment   │  │ Deployment   │                    │    │
│  │  │  deployment  │  │  monitoring  │                    │    │
│  │  │ Replicas: 3  │  │ Replicas: 2  │                    │    │
│  │  └──────────────┘  └──────────────┘                    │    │
│  │                                                          │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │         Horizontal Pod Autoscaler (HPA)          │  │    │
│  │  │  • CPU-based scaling                             │  │    │
│  │  │  • Custom metrics scaling                        │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Persistent Storage                      │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │    │
│  │  │     PVC      │  │     PVC      │  │     PVC      │ │    │
│  │  │  (MinIO)     │  │ (PostgreSQL) │  │  (MLflow)    │ │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline

1. **Data Ingestion**: Raw X-ray images uploaded to Data Pipeline Service
2. **Validation**: Images validated for quality and format
3. **Preprocessing**: Images cleaned, normalized, and augmented
4. **Versioning**: Processed data versioned and stored in MinIO
5. **Training**: Training Service fetches data and trains model
6. **Tracking**: Metrics logged to MLflow during training
7. **Registration**: Trained model registered in Model Registry
8. **Evaluation**: Model evaluated on test set
9. **Promotion**: Model promoted to staging/production

### Inference Pipeline

1. **Request**: Client sends X-ray image to Deployment Service
2. **Preprocessing**: Image preprocessed (resize, normalize)
3. **Inference**: Model performs prediction
4. **Postprocessing**: Results formatted and confidence calculated
5. **Logging**: Prediction logged for monitoring
6. **Response**: Results returned to client
7. **Monitoring**: Metrics collected by Monitoring Service

### Monitoring Pipeline

1. **Metrics Collection**: Services expose Prometheus metrics
2. **Scraping**: Prometheus scrapes metrics periodically
3. **Storage**: Metrics stored in time-series database
4. **Visualization**: Grafana displays dashboards
5. **Drift Detection**: Monitoring Service analyzes data/model drift
6. **Alerting**: Alerts triggered on threshold violations
7. **Notification**: Alerts sent via email/Slack/PagerDuty

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.9+ | Primary development language |
| ML Framework | PyTorch 2.0+ | Deep learning framework |
| Model Architecture | EfficientNet-B4 | CNN architecture |
| API Framework | FastAPI | REST API development |
| Containerization | Docker | Application packaging |
| Orchestration | Kubernetes | Container orchestration |
| Experiment Tracking | MLflow | ML experiment management |
| Data Versioning | DVC | Data version control |
| Object Storage | MinIO/S3 | Artifact storage |
| Database | PostgreSQL | Metadata storage |
| Caching | Redis | Response caching |
| Monitoring | Prometheus | Metrics collection |
| Visualization | Grafana | Metrics visualization |
| CI/CD | GitHub Actions | Automation pipeline |

### Development Tools

| Tool | Purpose |
|------|---------|
| Black | Code formatting |
| isort | Import sorting |
| flake8 | Linting |
| mypy | Type checking |
| pytest | Testing framework |
| pre-commit | Git hooks |

## Design Decisions

### Why Microservices?

**Advantages**:
- Independent scaling of services
- Technology flexibility per service
- Isolated failures
- Easier maintenance and updates
- Team autonomy

**Trade-offs**:
- Increased complexity
- Network latency
- Distributed system challenges

### Why EfficientNet-B4?

**Reasons**:
- Excellent accuracy/efficiency trade-off
- Proven performance on medical imaging
- Reasonable inference time (<200ms)
- Manageable model size (17.6M parameters)
- Transfer learning from ImageNet

**Alternatives Considered**:
- ResNet-50: Lower accuracy
- DenseNet-121: Slower inference
- Vision Transformer: Higher resource requirements

### Why FastAPI?

**Advantages**:
- High performance (async support)
- Automatic API documentation
- Type validation with Pydantic
- Modern Python features
- Easy testing

### Why MLflow?

**Advantages**:
- Comprehensive experiment tracking
- Model registry with versioning
- Framework agnostic
- Open source
- Easy integration

## Scalability

### Horizontal Scaling

**Deployment Service**:
- Stateless design allows easy replication
- Load balanced across multiple replicas
- Auto-scaling based on CPU/memory/requests

**Training Service**:
- Distributed training with Ray
- Multiple training jobs in parallel
- GPU resource management

**Data Pipeline Service**:
- Parallel data processing
- Queue-based job management
- Scalable storage (MinIO/S3)

### Vertical Scaling

**GPU Resources**:
- Training: Scale GPU memory/count
- Inference: Batch size optimization

**Database**:
- PostgreSQL read replicas
- Connection pooling
- Query optimization

### Performance Optimization

**Caching**:
- Redis for frequent predictions
- Model caching in memory
- Response caching

**Batch Processing**:
- Batch inference for throughput
- Dynamic batching
- Async processing

**Model Optimization**:
- Model quantization
- ONNX conversion
- TensorRT optimization

## Security

### Authentication & Authorization

- API key authentication
- JWT tokens for user sessions
- Role-based access control (RBAC)
- Service-to-service authentication

### Data Security

- Encryption at rest (MinIO/S3)
- Encryption in transit (TLS/SSL)
- HIPAA compliance considerations
- Data anonymization

### Network Security

- Network policies in Kubernetes
- Service mesh for mTLS
- Ingress with TLS termination
- Private subnets for databases

### Audit & Compliance

- Comprehensive audit logging
- Request/response logging
- Model prediction logging
- Access logs

### Secrets Management

- Kubernetes Secrets
- External secrets operator
- HashiCorp Vault integration
- Environment-based configuration

## Disaster Recovery

### Backup Strategy

- Database backups (daily)
- Model artifact backups
- Configuration backups
- Automated backup verification

### High Availability

- Multi-replica deployments
- Health checks and readiness probes
- Automatic failover
- Geographic redundancy (production)

### Recovery Procedures

- Documented runbooks
- Automated recovery scripts
- Regular DR drills
- RTO/RPO targets defined

## Future Enhancements

### Planned Improvements

1. **Multi-model Support**: Support for multiple disease detection
2. **Federated Learning**: Privacy-preserving distributed training
3. **Edge Deployment**: Model deployment on edge devices
4. **Real-time Streaming**: Stream processing for continuous inference
5. **Advanced Explainability**: More interpretability features
6. **AutoML**: Automated model selection and tuning

### Scalability Roadmap

1. **Phase 1**: Single region deployment
2. **Phase 2**: Multi-region deployment
3. **Phase 3**: Global CDN integration
4. **Phase 4**: Edge computing integration

## References

- [System Overview](docs/SYSTEM_OVERVIEW.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Operations Runbook](docs/OPERATIONS_RUNBOOK.md)
- [Production Infrastructure](docs/PRODUCTION_INFRASTRUCTURE.md)
