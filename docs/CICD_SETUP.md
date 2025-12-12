# CI/CD Pipeline Documentation

## Overview

This document describes the comprehensive CI/CD pipeline setup for the Chest X-Ray Pneumonia Detection MLOps system. The pipeline implements automated testing, security scanning, model validation, and deployment workflows.

## Pipeline Components

### 1. Main CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` branch
- Manual workflow dispatch

**Jobs:**
1. **Code Quality** - Formatting, linting, and security checks
2. **Testing** - Unit and integration tests across all components
3. **Model Validation** - Automated model performance validation
4. **Build & Scan** - Docker image building and security scanning
5. **Deploy Staging** - Automated deployment to staging environment
6. **Integration Tests** - End-to-end testing on staging

### 2. Security Scanning Pipeline (`.github/workflows/security-scan.yml`)

**Triggers:**
- Daily scheduled runs (2 AM UTC)
- Push to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

**Security Checks:**
- **Dependency Scanning** - Safety and pip-audit for known vulnerabilities
- **Secret Scanning** - TruffleHog and GitLeaks for exposed secrets
- **Code Security** - Bandit and Semgrep for security issues
- **License Compliance** - Automated license compliance checking

### 3. Model Validation Pipeline (`.github/workflows/model-validation.yml`)

**Triggers:**
- Changes to training, models, or data pipeline code
- Manual workflow dispatch with configurable parameters

**Validation Steps:**
1. **Data Validation** - Data quality and integrity checks
2. **Model Performance** - Accuracy, precision, recall, F1-score validation
3. **Bias & Fairness Testing** - Demographic parity and fairness metrics
4. **Robustness Testing** - Noise, adversarial, and drift robustness

### 4. Dependency Updates (`.github/workflows/dependency-update.yml`)

**Triggers:**
- Weekly scheduled runs (Mondays at 9 AM UTC)
- Manual workflow dispatch

**Features:**
- Automated Python dependency updates
- GitHub Actions version updates
- Security audit reporting
- Automated PR creation for updates

## Configuration Files

### Pre-commit Hooks (`.pre-commit-config.yaml`)

Ensures code quality before commits:
- **Code Formatting** - Black, isort
- **Linting** - Flake8, mypy
- **Security** - Bandit, secret detection
- **File Validation** - YAML, JSON, TOML validation

### Renovate Bot (`.github/renovate.json`)

Automated dependency management:
- Scheduled dependency updates
- Security vulnerability alerts
- Grouped updates by category
- Auto-merge for patch updates

## Test Structure

### Test Categories

1. **Data Pipeline Tests** (`tests/test_data_pipeline.py`)
   - Data ingestion and validation
   - Image preprocessing
   - Data versioning

2. **Training Tests** (`tests/test_training_*.py`)
   - Model training components
   - Experiment tracking
   - Hyperparameter optimization

3. **Deployment Tests** (`tests/test_deployment_*.py`)
   - Deployment automation
   - API integration
   - Load balancing

4. **Monitoring Tests** (`tests/test_monitoring_*.py`)
   - Metrics collection
   - Drift detection
   - Performance monitoring

5. **Retraining Tests** (`tests/test_retraining_workflows.py`)
   - Automated retraining triggers
   - Model comparison logic
   - Notification systems

### Test Execution

Use the test runner script:
```bash
# Run all tests
python scripts/run_tests.py all

# Run specific test suite
python scripts/run_tests.py data-pipeline
python scripts/run_tests.py training
python scripts/run_tests.py deployment
python scripts/run_tests.py monitoring
python scripts/run_tests.py retraining

# Run smoke tests
python scripts/run_tests.py smoke

# Run security tests
python scripts/run_tests.py security
```

## Performance Thresholds

### Model Validation Thresholds

- **Minimum Accuracy**: 80%
- **Minimum F1-Score**: 75%
- **Minimum Precision**: 70%
- **Minimum Recall**: 70%
- **Maximum Inference Time**: 2 seconds

### Security Thresholds

- **Critical Vulnerabilities**: 0 allowed
- **High Severity Issues**: Review required
- **License Compliance**: All packages must use approved licenses

## Environment Variables

### Required Secrets (GitHub Repository Settings)

```yaml
# Container Registry
DOCKER_REGISTRY: ghcr.io
GITHUB_TOKEN: <automatic>

# Staging Environment (if using real cluster)
STAGING_CLUSTER_URL: <staging-k8s-api-url>
STAGING_TOKEN: <staging-service-account-token>

# Notifications (optional)
SLACK_WEBHOOK_URL: <slack-webhook-for-notifications>
EMAIL_SMTP_PASSWORD: <smtp-password-for-email-notifications>

# Security Scanning (optional)
GITLEAKS_LICENSE: <gitleaks-license-key>
```

### Environment Configuration

```yaml
env:
  PYTHON_VERSION: '3.10'
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: chest-xray-pneumonia-mlops
  MIN_ACCURACY: 0.80
  MIN_F1_SCORE: 0.75
  MIN_PRECISION: 0.70
  MIN_RECALL: 0.70
```

## Deployment Workflow

### Staging Deployment

1. **Automatic Triggers**:
   - Push to `main` branch
   - All tests pass
   - Security scans complete

2. **Deployment Steps**:
   - Update Kubernetes manifests with new image tags
   - Deploy to staging namespace
   - Run smoke tests
   - Execute integration tests

3. **Validation**:
   - Health check endpoints
   - API functionality tests
   - End-to-end workflow validation

### Production Deployment

Production deployment requires manual approval and is handled by a separate workflow (Task 7.2).

## Monitoring and Alerting

### Pipeline Monitoring

- **Build Status**: GitHub Actions status badges
- **Test Coverage**: Codecov integration
- **Security Alerts**: GitHub Security tab
- **Dependency Updates**: Renovate dashboard

### Notification Channels

- **GitHub Issues**: Automated security audit reports
- **Pull Requests**: Validation results and coverage reports
- **Slack** (optional): Build status and deployment notifications
- **Email** (optional): Critical security alerts

## Troubleshooting

### Common Issues

1. **Test Failures**:
   ```bash
   # Run specific failing test
   python -m pytest tests/test_specific.py::test_function -v
   
   # Check test logs
   python scripts/run_tests.py <component> --verbose
   ```

2. **Security Scan Failures**:
   ```bash
   # Run local security scan
   bandit -r . -f json -o bandit-report.json
   safety check --json --output safety-report.json
   ```

3. **Docker Build Issues**:
   ```bash
   # Test local Docker build
   docker build -f training/Dockerfile -t test-image .
   ```

4. **Kubernetes Deployment Issues**:
   ```bash
   # Validate Kubernetes manifests
   kubectl apply --dry-run=client -f k8s/
   ```

### Validation Script

Run the CI/CD validation script to check setup:
```bash
python scripts/validate_cicd.py
```

This script validates:
- ✅ GitHub Actions workflows
- ✅ Configuration files
- ✅ Test structure
- ✅ Requirements files
- ✅ Docker files
- ✅ Kubernetes manifests
- ✅ Environment variables
- ✅ Security configuration

## Best Practices

### Code Quality

1. **Pre-commit Hooks**: Always run before committing
2. **Test Coverage**: Maintain >80% coverage
3. **Security Scanning**: Address all high/critical issues
4. **Documentation**: Update docs with code changes

### Model Validation

1. **Performance Thresholds**: Set realistic but strict thresholds
2. **Bias Testing**: Regular fairness and bias assessments
3. **Robustness Testing**: Test against various input conditions
4. **Data Quality**: Validate data before training

### Security

1. **Dependency Updates**: Regular automated updates
2. **Secret Management**: Use GitHub Secrets, never commit secrets
3. **Container Scanning**: Scan all container images
4. **License Compliance**: Regular license audits

### Deployment

1. **Staging First**: Always deploy to staging before production
2. **Rollback Plan**: Maintain ability to rollback quickly
3. **Health Checks**: Comprehensive health and readiness probes
4. **Monitoring**: Monitor all deployments closely

## Next Steps

After completing Task 7.1, the next steps are:

1. **Task 7.2**: Implement production deployment pipeline with manual approval gates
2. **Task 7.3**: Write end-to-end integration tests
3. **Task 8.1**: Set up production infrastructure
4. **Task 8.2**: Final integration and validation

## Support

For issues with the CI/CD pipeline:

1. Check the validation script output
2. Review GitHub Actions logs
3. Consult this documentation
4. Check individual workflow files for specific configurations