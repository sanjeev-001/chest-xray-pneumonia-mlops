# Production Deployment Pipeline Documentation

## Overview

This document describes the production deployment pipeline for the Chest X-Ray Pneumonia Detection MLOps system. The pipeline implements blue-green deployment strategy with manual approval gates, ArgoCD for GitOps, and comprehensive rollback mechanisms.

## Architecture

### Blue-Green Deployment Strategy

The production environment uses a blue-green deployment strategy with two identical environments:

- **Blue Environment** (`production-blue` namespace)
- **Green Environment** (`production-green` namespace)

At any time, one environment serves production traffic (active) while the other remains on standby. During deployment, the new version is deployed to the standby environment, validated, and then traffic is switched.

### Components

1. **GitHub Actions Workflows**
   - `production-deployment.yml` - Main production deployment pipeline
   - `rollback.yml` - Emergency and planned rollback procedures

2. **ArgoCD Applications**
   - GitOps-based deployment management
   - Automated sync and self-healing
   - Environment-specific configurations

3. **Environment Configuration**
   - Production-specific resource limits and scaling
   - Security policies and network restrictions
   - Monitoring and logging configuration

4. **Deployment Manager Script**
   - CLI tool for manual deployment operations
   - Blue-green deployment automation
   - Health checks and validation

## Deployment Process

### 1. Automatic Trigger

The production deployment is triggered automatically when:
- The main CI/CD pipeline completes successfully on `main` branch
- All tests pass including security scans
- Model validation meets performance thresholds

### 2. Pre-deployment Validation

```yaml
Jobs:
  - Pre-deployment validation
  - Security and compliance check
  - Manual approval gate
```

**Validation Steps:**
- Verify staging deployment success
- Check deployment conditions
- Generate deployment summary
- Run production-specific security scans
- Validate compliance requirements

### 3. Manual Approval Gate

**Environment:** `production-approval`

**Required Approvals:**
- DevOps team member
- ML Engineering team lead
- Product owner (for major releases)

**Approval Criteria:**
- All automated tests passed
- Security scans completed without critical issues
- Performance validation successful
- Deployment summary reviewed

### 4. Blue-Green Deployment

**Process:**
1. **Environment Detection** - Determine current active environment
2. **Target Preparation** - Prepare standby environment for deployment
3. **Deployment** - Deploy new version to standby environment
4. **Health Checks** - Validate deployment health and functionality
5. **Traffic Switch** - Gradually switch traffic to new environment
6. **Monitoring** - Monitor metrics during and after switch
7. **Cleanup** - Scale down old environment (kept for quick rollback)

### 5. Post-deployment Validation

**Validation Steps:**
- Comprehensive health checks
- Performance validation
- Monitoring system verification
- Backup system validation
- Deployment record creation

## Manual Deployment

### Using GitHub Actions

**Standard Deployment:**
```bash
# Trigger via GitHub UI
# Go to Actions → Production Deployment Pipeline → Run workflow
# Select deployment type: standard
```

**Emergency Deployment:**
```bash
# Trigger via GitHub UI with emergency parameters
# Select deployment type: hotfix
# Provide justification
```

### Using Deployment Manager Script

**Deploy New Version:**
```bash
python scripts/deployment_manager.py deploy --version v2.1.0
```

**Check Environment Status:**
```bash
python scripts/deployment_manager.py status
```

**Run Health Checks:**
```bash
python scripts/deployment_manager.py health-check --environment blue
```

**Switch Traffic:**
```bash
python scripts/deployment_manager.py switch-traffic \
  --from-env green --to-env blue --percentage 50
```

## Rollback Procedures

### Automatic Rollback Triggers

Automatic rollback is triggered when:
- Health checks fail after deployment
- Error rate exceeds 1% for 5 minutes
- Response time exceeds 2 seconds for 5 minutes
- Critical alerts are fired

### Manual Rollback

**Emergency Rollback (GitHub Actions):**
```yaml
Workflow: rollback.yml
Inputs:
  - rollback_target: emergency_rollback
  - rollback_reason: "Critical production issue"
  - emergency: true  # Skips approval gates
```

**Planned Rollback:**
```yaml
Workflow: rollback.yml
Inputs:
  - rollback_target: previous_version
  - rollback_reason: "Performance degradation detected"
  - emergency: false  # Requires approval
```

**Using Deployment Manager:**
```bash
python scripts/deployment_manager.py rollback \
  --version v2.0.5 \
  --reason "Critical bug in v2.1.0"
```

### Rollback Process

1. **Rollback Planning** - Determine target version and environment
2. **Approval** - Manual approval (unless emergency)
3. **Environment Preparation** - Prepare rollback environment
4. **Deployment** - Deploy rollback version
5. **Health Checks** - Validate rollback deployment
6. **Traffic Switch** - Switch traffic to rollback environment
7. **Validation** - Comprehensive post-rollback validation
8. **Cleanup** - Clean up failed environment

## ArgoCD Configuration

### Project Structure

```
argocd/
├── projects/
│   └── production.yaml          # Production project definition
└── applications/
    ├── production-blue.yaml     # Blue environment application
    └── production-green.yaml    # Green environment application
```

### Application Configuration

**Key Features:**
- Automated sync with self-healing
- Sync waves for ordered deployment
- Retry policies for failed syncs
- Revision history tracking
- RBAC integration

**Sync Policy:**
```yaml
syncPolicy:
  automated:
    prune: true
    selfHeal: true
  syncOptions:
  - CreateNamespace=true
  - PrunePropagationPolicy=foreground
```

### Access Control

**Roles:**
- `production-admin` - Full access (DevOps team)
- `production-developer` - Deploy and sync access (Developers)
- `production-viewer` - Read-only access (Stakeholders)

**Sync Windows:**
- **Allowed:** Business hours (9 AM - 5 PM, weekdays)
- **Denied:** Nights, weekends (except emergency)

## Environment Configuration

### Production Values

**Resource Configuration:**
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

replicaCount: 3
minReplicas: 3
maxReplicas: 10
```

**Security Configuration:**
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
```

**Monitoring Configuration:**
```yaml
monitoring:
  enabled: true
  prometheus:
    scrapeInterval: 30s
  alertmanager:
    enabled: true
```

### Environment-Specific Values

**Blue Environment:**
- Namespace: `production-blue`
- Ingress: `api-blue.production.example.com`
- Database: `prod-postgres-blue.internal`

**Green Environment:**
- Namespace: `production-green`
- Ingress: `api-green.production.example.com`
- Database: `prod-postgres-green.internal`

## Monitoring and Alerting

### Deployment Metrics

**Key Metrics:**
- Deployment success rate
- Deployment duration
- Rollback frequency
- Time to recovery (MTTR)

**Alerts:**
- Deployment failure
- Health check failure
- High error rate post-deployment
- Performance degradation

### Dashboards

**Grafana Dashboards:**
- Production Deployment Overview
- Blue-Green Environment Status
- Application Performance Metrics
- Infrastructure Resource Usage

## Security and Compliance

### Security Measures

1. **Container Security**
   - Image vulnerability scanning
   - Signed container images
   - Runtime security policies

2. **Network Security**
   - Network policies
   - Service mesh with mTLS
   - Ingress TLS termination

3. **Access Control**
   - RBAC for Kubernetes
   - ArgoCD role-based access
   - Audit logging

### Compliance Requirements

1. **Audit Trail**
   - All deployment actions logged
   - Approval records maintained
   - Change tracking

2. **Data Protection**
   - Encryption at rest and in transit
   - Backup and recovery procedures
   - Data retention policies

## Troubleshooting

### Common Issues

**1. Deployment Stuck in Pending**
```bash
# Check ArgoCD application status
argocd app get chest-xray-mlops-prod-blue

# Check Kubernetes events
kubectl get events -n production-blue --sort-by='.lastTimestamp'
```

**2. Health Checks Failing**
```bash
# Check pod logs
kubectl logs -n production-blue -l app=chest-xray-mlops

# Check service endpoints
kubectl get endpoints -n production-blue
```

**3. Traffic Switch Issues**
```bash
# Check ingress configuration
kubectl get ingress -n production-blue -o yaml

# Check service mesh configuration
kubectl get virtualservice -n production-blue
```

### Emergency Procedures

**1. Complete System Failure**
```bash
# Emergency rollback to last known good version
python scripts/deployment_manager.py rollback \
  --version last-known-good \
  --reason "Complete system failure"
```

**2. Database Issues**
```bash
# Switch to backup database
kubectl patch configmap app-config \
  -n production-blue \
  --patch '{"data":{"database_url":"backup-db-url"}}'
```

**3. Network Issues**
```bash
# Check network policies
kubectl get networkpolicy -n production-blue

# Verify DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup api-blue.production.example.com
```

## Best Practices

### Deployment Best Practices

1. **Always test in staging first**
2. **Use gradual traffic switching**
3. **Monitor metrics during deployment**
4. **Keep rollback plan ready**
5. **Document all changes**

### Security Best Practices

1. **Scan all container images**
2. **Use least privilege access**
3. **Enable audit logging**
4. **Regular security reviews**
5. **Keep secrets encrypted**

### Operational Best Practices

1. **Automate everything possible**
2. **Monitor continuously**
3. **Practice disaster recovery**
4. **Maintain documentation**
5. **Regular team training**

## Support and Contacts

**Primary Contacts:**
- DevOps Team: devops@company.com
- ML Engineering: ml-engineering@company.com
- On-call: +1-555-ONCALL

**Escalation:**
- Level 1: DevOps Engineer
- Level 2: Senior DevOps Engineer
- Level 3: Engineering Manager
- Level 4: CTO

**Resources:**
- Runbooks: `docs/runbooks/`
- Monitoring: https://grafana.production.example.com
- Logs: https://kibana.production.example.com
- ArgoCD: https://argocd.production.example.com