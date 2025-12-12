#!/usr/bin/env python3
"""
Production Infrastructure Deployment Script
Deploys and configures the complete production infrastructure for MLOps
"""

import argparse
import subprocess
import sys
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class InfrastructureDeployer:
    """Manages production infrastructure deployment"""
    
    def __init__(self, config_path: str = "infrastructure"):
        self.config_path = Path(config_path)
        self.terraform_dir = self.config_path / "terraform"
        self.k8s_dir = self.config_path / "kubernetes"
        
    def validate_prerequisites(self) -> bool:
        """Validate that all required tools are installed"""
        print("🔍 Validating prerequisites...")
        
        required_tools = [
            ("terraform", "Terraform"),
            ("kubectl", "Kubernetes CLI"),
            ("helm", "Helm"),
            ("aws", "AWS CLI")
        ]
        
        missing_tools = []
        for tool, name in required_tools:
            try:
                result = subprocess.run([tool, "--version"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {name} is installed")
                else:
                    missing_tools.append(name)
            except FileNotFoundError:
                missing_tools.append(name)
        
        if missing_tools:
            print(f"❌ Missing required tools: {', '.join(missing_tools)}")
            return False
        
        print("✅ All prerequisites validated")
        return True
    
    def deploy_terraform_infrastructure(self) -> bool:
        """Deploy AWS infrastructure using Terraform"""
        print("🚀 Deploying AWS infrastructure with Terraform...")
        
        try:
            # Change to terraform directory
            os.chdir(self.terraform_dir)
            
            # Initialize Terraform
            print("Initializing Terraform...")
            result = subprocess.run(["terraform", "init"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Terraform init failed: {result.stderr}")
                return False
            
            # Plan deployment
            print("Planning Terraform deployment...")
            result = subprocess.run(["terraform", "plan", "-out=tfplan"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Terraform plan failed: {result.stderr}")
                return False
            
            # Apply deployment
            print("Applying Terraform deployment...")
            result = subprocess.run(["terraform", "apply", "tfplan"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Terraform apply failed: {result.stderr}")
                return False
            
            print("✅ AWS infrastructure deployed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error deploying infrastructure: {e}")
            return False
        finally:
            # Return to original directory
            os.chdir(Path(__file__).parent.parent)
    
    def configure_kubectl(self) -> bool:
        """Configure kubectl to connect to the EKS cluster"""
        print("🔧 Configuring kubectl for EKS cluster...")
        
        try:
            # Get cluster name from Terraform output
            result = subprocess.run([
                "terraform", "output", "-raw", "cluster_id"
            ], cwd=self.terraform_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to get cluster name: {result.stderr}")
                return False
            
            cluster_name = result.stdout.strip()
            
            # Update kubeconfig
            result = subprocess.run([
                "aws", "eks", "update-kubeconfig", 
                "--region", "us-east-1",
                "--name", cluster_name
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to update kubeconfig: {result.stderr}")
                return False
            
            # Test kubectl connection
            result = subprocess.run([
                "kubectl", "get", "nodes"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to connect to cluster: {result.stderr}")
                return False
            
            print("✅ kubectl configured successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error configuring kubectl: {e}")
            return False
    
    def install_monitoring_stack(self) -> bool:
        """Install Prometheus, Grafana, and AlertManager"""
        print("📊 Installing monitoring stack...")
        
        try:
            # Add Prometheus Helm repository
            subprocess.run([
                "helm", "repo", "add", "prometheus-community",
                "https://prometheus-community.github.io/helm-charts"
            ], check=True)
            
            subprocess.run([
                "helm", "repo", "update"
            ], check=True)
            
            # Create monitoring namespace
            subprocess.run([
                "kubectl", "create", "namespace", "monitoring"
            ], capture_output=True)  # Ignore if already exists
            
            # Install kube-prometheus-stack
            values_file = self.k8s_dir / "monitoring" / "prometheus-values.yaml"
            result = subprocess.run([
                "helm", "install", "prometheus",
                "prometheus-community/kube-prometheus-stack",
                "--namespace", "monitoring",
                "--values", str(values_file),
                "--wait", "--timeout", "10m"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to install monitoring stack: {result.stderr}")
                return False
            
            print("✅ Monitoring stack installed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error installing monitoring stack: {e}")
            return False
    
    def setup_backup_system(self) -> bool:
        """Set up backup and disaster recovery system"""
        print("💾 Setting up backup system...")
        
        try:
            # Create backup namespace
            subprocess.run([
                "kubectl", "create", "namespace", "backup-system"
            ], capture_output=True)  # Ignore if already exists
            
            # Apply backup configuration
            backup_config = self.config_path / "backup" / "backup-strategy.yaml"
            result = subprocess.run([
                "kubectl", "apply", "-f", str(backup_config)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to apply backup configuration: {result.stderr}")
                return False
            
            print("✅ Backup system configured successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up backup system: {e}")
            return False
    
    def install_argocd(self) -> bool:
        """Install ArgoCD for GitOps"""
        print("🔄 Installing ArgoCD...")
        
        try:
            # Create argocd namespace
            subprocess.run([
                "kubectl", "create", "namespace", "argocd"
            ], capture_output=True)  # Ignore if already exists
            
            # Install ArgoCD
            result = subprocess.run([
                "kubectl", "apply", "-n", "argocd", "-f",
                "https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to install ArgoCD: {result.stderr}")
                return False
            
            # Wait for ArgoCD to be ready
            print("Waiting for ArgoCD to be ready...")
            subprocess.run([
                "kubectl", "wait", "--for=condition=available",
                "--timeout=300s", "deployment/argocd-server",
                "-n", "argocd"
            ], check=True)
            
            print("✅ ArgoCD installed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error installing ArgoCD: {e}")
            return False
    
    def configure_argocd_applications(self) -> bool:
        """Configure ArgoCD applications for blue-green deployment"""
        print("⚙️ Configuring ArgoCD applications...")
        
        try:
            # Apply ArgoCD project
            project_config = Path("argocd/projects/production.yaml")
            if project_config.exists():
                subprocess.run([
                    "kubectl", "apply", "-f", str(project_config)
                ], check=True)
            
            # Apply ArgoCD applications
            app_configs = [
                "argocd/applications/production-blue.yaml",
                "argocd/applications/production-green.yaml"
            ]
            
            for app_config in app_configs:
                app_path = Path(app_config)
                if app_path.exists():
                    subprocess.run([
                        "kubectl", "apply", "-f", str(app_path)
                    ], check=True)
            
            print("✅ ArgoCD applications configured successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error configuring ArgoCD applications: {e}")
            return False
    
    def validate_deployment(self) -> bool:
        """Validate that all components are deployed correctly"""
        print("🔍 Validating deployment...")
        
        try:
            # Check EKS cluster
            result = subprocess.run([
                "kubectl", "get", "nodes"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print("❌ EKS cluster not accessible")
                return False
            
            node_count = len([line for line in result.stdout.split('\n') 
                            if 'Ready' in line])
            print(f"✅ EKS cluster running with {node_count} nodes")
            
            # Check monitoring stack
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", "monitoring"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Monitoring stack deployed")
            else:
                print("⚠️ Monitoring stack not found")
            
            # Check ArgoCD
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", "argocd"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ ArgoCD deployed")
            else:
                print("⚠️ ArgoCD not found")
            
            # Check backup system
            result = subprocess.run([
                "kubectl", "get", "cronjobs", "-n", "backup-system"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Backup system configured")
            else:
                print("⚠️ Backup system not found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating deployment: {e}")
            return False
    
    def get_access_information(self) -> Dict[str, str]:
        """Get access information for deployed services"""
        print("📋 Gathering access information...")
        
        access_info = {}
        
        try:
            # Get ArgoCD admin password
            result = subprocess.run([
                "kubectl", "-n", "argocd", "get", "secret", "argocd-initial-admin-secret",
                "-o", "jsonpath={.data.password}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                import base64
                password = base64.b64decode(result.stdout).decode('utf-8')
                access_info['argocd_password'] = password
            
            # Get Grafana admin password
            result = subprocess.run([
                "kubectl", "-n", "monitoring", "get", "secret", 
                "prometheus-grafana", "-o", "jsonpath={.data.admin-password}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                import base64
                password = base64.b64decode(result.stdout).decode('utf-8')
                access_info['grafana_password'] = password
            
            # Get load balancer endpoints
            result = subprocess.run([
                "kubectl", "get", "svc", "-n", "monitoring",
                "prometheus-grafana", "-o", "jsonpath={.status.loadBalancer.ingress[0].hostname}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                access_info['grafana_url'] = f"http://{result.stdout}"
            
            return access_info
            
        except Exception as e:
            print(f"⚠️ Error gathering access information: {e}")
            return access_info
    
    def deploy_full_infrastructure(self) -> bool:
        """Deploy the complete production infrastructure"""
        print("🚀 Starting full infrastructure deployment...")
        
        steps = [
            ("Validating prerequisites", self.validate_prerequisites),
            ("Deploying AWS infrastructure", self.deploy_terraform_infrastructure),
            ("Configuring kubectl", self.configure_kubectl),
            ("Installing monitoring stack", self.install_monitoring_stack),
            ("Setting up backup system", self.setup_backup_system),
            ("Installing ArgoCD", self.install_argocd),
            ("Configuring ArgoCD applications", self.configure_argocd_applications),
            ("Validating deployment", self.validate_deployment),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*60}")
            print(f"Step: {step_name}")
            print(f"{'='*60}")
            
            if not step_func():
                print(f"❌ Failed at step: {step_name}")
                return False
            
            print(f"✅ Completed: {step_name}")
        
        # Get access information
        access_info = self.get_access_information()
        
        print(f"\n{'='*60}")
        print("🎉 INFRASTRUCTURE DEPLOYMENT COMPLETED!")
        print(f"{'='*60}")
        
        if access_info:
            print("\n📋 Access Information:")
            for service, info in access_info.items():
                print(f"  {service}: {info}")
        
        print("\n📚 Next Steps:")
        print("1. Configure DNS records for your domain")
        print("2. Set up SSL certificates")
        print("3. Configure monitoring alerts")
        print("4. Test backup and recovery procedures")
        print("5. Deploy your MLOps applications")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Deploy production infrastructure")
    parser.add_argument("--config-path", default="infrastructure",
                       help="Path to infrastructure configuration")
    parser.add_argument("--terraform-only", action="store_true",
                       help="Deploy only Terraform infrastructure")
    parser.add_argument("--k8s-only", action="store_true",
                       help="Deploy only Kubernetes components")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate the deployment")
    
    args = parser.parse_args()
    
    deployer = InfrastructureDeployer(args.config_path)
    
    if args.validate_only:
        success = deployer.validate_deployment()
    elif args.terraform_only:
        success = (deployer.validate_prerequisites() and 
                  deployer.deploy_terraform_infrastructure())
    elif args.k8s_only:
        success = (deployer.configure_kubectl() and
                  deployer.install_monitoring_stack() and
                  deployer.setup_backup_system() and
                  deployer.install_argocd() and
                  deployer.configure_argocd_applications())
    else:
        success = deployer.deploy_full_infrastructure()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()