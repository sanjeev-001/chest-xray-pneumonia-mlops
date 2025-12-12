# Production Infrastructure Documentation

## Overview

This document describes the production infrastructure setup for the Chest X-Ray Pneumonia Detection MLOps system.

## Architecture Components

### AWS Infrastructure (Terraform)
- **EKS Cluster**: Kubernetes cluster with auto-scaling node groups
- **RDS PostgreSQL**: Production database with encryption and backups
- **ElastiCache Redis**: Caching layer for improved performance
- **S3 Buckets**: Model artifacts and data lake storage
- **Application Load Balancer**: Traffic distribution and SSL termination
- **VPC**: Secure network with public/private subnets
- **KMS Keys**: Encryption for all data at rest
- **CloudWatch**: Centralized logging and monitoring
- **ECR**: Container image registry

### Kubernetes Components
- **Monitoring Stack**: Prometheus, Grafana, AlertManager
- **ArgoCD**: GitOps deployment management
- **Backup System**: Automated backup jobs for data and configurations
- **Blue-Green Environments**: Zero-downtime deployment strategy

### Security Features
- **Encryption**: All data encrypted at rest and in transit
- **Network Security**: VPC with security groups and network policies
- **RBAC**: Role-based access control for Kubernetes and ArgoCD
- **Secret Management**: Kubernetes secrets with encryption
- **Container Security**: Image scanning and security policies

## Deployment

### Prerequisites
- AWS CLI configured with appropriate permissions
- Terraform >= 1.0
- kubectl
- Helm
- Docker

### Quick Start
```bash
# Deploy complete infrastructure
python scripts/deploy_production_infrastructure.py

# Deploy only AWS infrastructure
python scripts/deploy_production_infrastructure.py --terraform-only

# Deploy only Kubernetes components
python scripts/deploy_production_infrastructure.py --k8s-only
```

### Manual Deployment Steps
1. **Deploy AWS Infrastructure**:
   ```bash
   cd infrastructure/terraform
   terraform init
   terraform plan
   terraform apply
   ```

2. **Configure kubectl**:
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name <cluster-name>
   ```

3. **Install Monitoring**:
   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm install prometheus prometheus-community/kube-prometheus-stack \
     --namespace monitoring --create-namespace \
     --values infrastructure/kubernetes/monitoring/prometheus-values.yaml
   ```

4. **Install ArgoCD**:
   ```bash
   kubectl create namespace argocd
   kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
   ```

## Access Information

### Service URLs
- **Grafana**: `https://grafana.mlops.example.com`
- **ArgoCD**: `https://argocd.mlops.example.com`
- **Prometheus**: `https://prometheus.mlops.example.com`

### Default Credentials
- **Grafana**: admin / (get from secret)
- **ArgoCD**: admin / (get from secret)

## Monitoring and Alerting

### Key Metrics
- Model inference rate and latency
- Model accuracy and performance
- Infrastructure resource utilization
- Application error rates

### Alert Rules
- Model accuracy below threshold
- High inference latency
- Service unavailability
- Resource exhaustion

## Backup and Disaster Recovery

### Backup Strategy
- **Database**: Daily automated backups with 30-day retention
- **Model Artifacts**: Cross-region replication
- **Kubernetes Resources**: Daily configuration backups
- **Application Data**: Incremental backups

### Recovery Procedures
- **RTO**: 4 hours (Recovery Time Objective)
- **RPO**: 1 hour (Recovery Point Objective)
- **Cross-region failover**: Manual approval required
- **Blue-green rollback**: Automated capability

## Cost Optimization

### Strategies
- **Spot Instances**: For non-critical workloads
- **Auto-scaling**: Based on demand
- **Resource Right-sizing**: Regular optimization
- **Storage Tiering**: Lifecycle policies for S3

### Estimated Monthly Costs
- **EKS Cluster**: $150-300
- **RDS Database**: $100-200
- **ElastiCache**: $50-100
- **S3 Storage**: $20-50
- **Data Transfer**: $30-60
- **Total**: ~$350-710/month

## Security Best Practices

### Network Security
- VPC with private subnets for workloads
- Security groups with least privilege
- Network policies in Kubernetes
- WAF protection for public endpoints

### Data Protection
- Encryption at rest for all storage
- TLS encryption for all communications
- Regular security scans
- Compliance with healthcare regulations

### Access Control
- IAM roles with minimal permissions
- Kubernetes RBAC
- Multi-factor authentication
- Audit logging for all access

## Troubleshooting

### Common Issues
1. **EKS nodes not joining**: Check security groups and IAM roles
2. **Monitoring not working**: Verify service monitors and endpoints
3. **ArgoCD sync issues**: Check repository access and credentials
4. **Backup failures**: Verify IAM permissions and S3 access

### Support Contacts
- **Infrastructure**: devops@company.com
- **Security**: security@company.com
- **On-call**: +1-555-ONCALL

This infrastructure provides a robust, scalable, and secure foundation for the MLOps system.