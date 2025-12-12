#!/usr/bin/env python3
"""
Production Deployment Manager
Manages blue-green deployments, rollbacks, and environment configuration
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import requests


class DeploymentManager:
    """Manages production deployments with blue-green strategy"""
    
    def __init__(self, config_path: str = "config/environments/production"):
        self.config_path = Path(config_path)
        self.argocd_server = "argocd.production.example.com"
        self.kubectl_context = "production"
        
    def get_current_environment_state(self) -> Dict[str, str]:
        """Get current state of blue-green environments"""
        try:
            # Query load balancer or service mesh for current traffic distribution
            # This is a mock implementation
            return {
                "active": "green",
                "standby": "blue",
                "traffic_distribution": {"green": 100, "blue": 0}
            }
        except Exception as e:
            print(f"Error getting environment state: {e}")
            return {"active": "unknown", "standby": "unknown"}
    
    def validate_deployment_readiness(self, environment: str, version: str) -> bool:
        """Validate that an environment is ready for deployment"""
        print(f"Validating deployment readiness for {environment} environment...")
        
        try:
            # Check if namespace exists
            result = subprocess.run([
                "kubectl", "get", "namespace", f"production-{environment}",
                "--context", self.kubectl_context
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Namespace production-{environment} does not exist")
                return False
            
            # Check if required secrets exist
            secrets = ["database-credentials", "redis-credentials", "tls-certificates"]
            for secret in secrets:
                result = subprocess.run([
                    "kubectl", "get", "secret", secret,
                    "-n", f"production-{environment}",
                    "--context", self.kubectl_context
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Required secret {secret} not found in production-{environment}")
                    return False
            
            print(f"✅ Environment {environment} is ready for deployment")
            return True
            
        except Exception as e:
            print(f"Error validating deployment readiness: {e}")
            return False
    
    def deploy_to_environment(self, environment: str, version: str) -> bool:
        """Deploy a specific version to an environment"""
        print(f"Deploying version {version} to {environment} environment...")
        
        try:
            # Update ArgoCD application with new version
            app_name = f"chest-xray-mlops-prod-{environment}"
            
            # Update the application's target revision
            result = subprocess.run([
                "argocd", "app", "set", app_name,
                "--revision", version,
                "--server", self.argocd_server
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Failed to update ArgoCD application: {result.stderr}")
                return False
            
            # Trigger sync
            result = subprocess.run([
                "argocd", "app", "sync", app_name,
                "--server", self.argocd_server
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Failed to sync ArgoCD application: {result.stderr}")
                return False
            
            # Wait for sync to complete
            print("Waiting for deployment to complete...")
            for i in range(30):  # Wait up to 15 minutes
                result = subprocess.run([
                    "argocd", "app", "get", app_name,
                    "--output", "json",
                    "--server", self.argocd_server
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    app_status = json.loads(result.stdout)
                    sync_status = app_status.get("status", {}).get("sync", {}).get("status")
                    health_status = app_status.get("status", {}).get("health", {}).get("status")
                    
                    if sync_status == "Synced" and health_status == "Healthy":
                        print(f"✅ Deployment to {environment} completed successfully")
                        return True
                    elif health_status == "Degraded":
                        print(f"❌ Deployment to {environment} failed - application is degraded")
                        return False
                
                time.sleep(30)
            
            print(f"⏰ Deployment to {environment} timed out")
            return False
            
        except Exception as e:
            print(f"Error deploying to environment: {e}")
            return False
    
    def run_health_checks(self, environment: str) -> bool:
        """Run health checks on an environment"""
        print(f"Running health checks on {environment} environment...")
        
        try:
            # Health check endpoints
            endpoints = [
                f"https://api-{environment}.production.example.com/health",
                f"https://model-{environment}.production.example.com/health"
            ]
            
            for endpoint in endpoints:
                print(f"Checking {endpoint}...")
                try:
                    response = requests.get(endpoint, timeout=10)
                    if response.status_code == 200:
                        print(f"✅ {endpoint} is healthy")
                    else:
                        print(f"❌ {endpoint} returned status {response.status_code}")
                        return False
                except requests.RequestException as e:
                    print(f"❌ {endpoint} is unreachable: {e}")
                    return False
            
            print(f"✅ All health checks passed for {environment}")
            return True
            
        except Exception as e:
            print(f"Error running health checks: {e}")
            return False
    
    def switch_traffic(self, from_env: str, to_env: str, percentage: int = 100) -> bool:
        """Switch traffic between environments"""
        print(f"Switching {percentage}% traffic from {from_env} to {to_env}...")
        
        try:
            # Update load balancer configuration
            # This would typically involve updating:
            # - Istio VirtualService
            # - NGINX Ingress weights
            # - AWS ALB target groups
            # - DNS records
            
            # Mock implementation
            print(f"Updating load balancer configuration...")
            print(f"Setting {to_env} weight to {percentage}%")
            print(f"Setting {from_env} weight to {100 - percentage}%")
            
            # Simulate configuration update
            time.sleep(5)
            
            print(f"✅ Traffic switch completed")
            return True
            
        except Exception as e:
            print(f"Error switching traffic: {e}")
            return False
    
    def rollback_deployment(self, target_version: str, reason: str) -> bool:
        """Rollback to a previous version"""
        print(f"Rolling back to version {target_version}...")
        print(f"Rollback reason: {reason}")
        
        try:
            # Get current environment state
            env_state = self.get_current_environment_state()
            current_active = env_state["active"]
            target_env = "blue" if current_active == "green" else "green"
            
            print(f"Current active environment: {current_active}")
            print(f"Rolling back to: {target_env}")
            
            # Deploy rollback version to target environment
            if not self.deploy_to_environment(target_env, target_version):
                print("❌ Rollback deployment failed")
                return False
            
            # Run health checks
            if not self.run_health_checks(target_env):
                print("❌ Rollback health checks failed")
                return False
            
            # Switch traffic
            if not self.switch_traffic(current_active, target_env):
                print("❌ Traffic switch for rollback failed")
                return False
            
            # Record rollback
            self.record_deployment({
                "type": "rollback",
                "from_environment": current_active,
                "to_environment": target_env,
                "version": target_version,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            print(f"✅ Rollback to version {target_version} completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during rollback: {e}")
            return False
    
    def blue_green_deployment(self, version: str) -> bool:
        """Perform blue-green deployment"""
        print(f"Starting blue-green deployment of version {version}...")
        
        try:
            # Get current environment state
            env_state = self.get_current_environment_state()
            current_active = env_state["active"]
            target_env = "blue" if current_active == "green" else "green"
            
            print(f"Current active environment: {current_active}")
            print(f"Deploying to: {target_env}")
            
            # Validate target environment readiness
            if not self.validate_deployment_readiness(target_env, version):
                print("❌ Target environment is not ready")
                return False
            
            # Deploy to target environment
            if not self.deploy_to_environment(target_env, version):
                print("❌ Deployment failed")
                return False
            
            # Run health checks
            if not self.run_health_checks(target_env):
                print("❌ Health checks failed")
                return False
            
            # Gradual traffic switch (optional)
            print("Starting gradual traffic switch...")
            for percentage in [10, 25, 50, 75, 100]:
                print(f"Switching {percentage}% traffic to {target_env}...")
                if not self.switch_traffic(current_active, target_env, percentage):
                    print(f"❌ Traffic switch to {percentage}% failed")
                    return False
                
                # Monitor for issues
                time.sleep(60)  # Wait 1 minute between switches
                
                # Check health after each switch
                if not self.run_health_checks(target_env):
                    print(f"❌ Health check failed at {percentage}% traffic")
                    # Rollback traffic
                    self.switch_traffic(target_env, current_active, 0)
                    return False
            
            # Record successful deployment
            self.record_deployment({
                "type": "blue_green",
                "from_environment": current_active,
                "to_environment": target_env,
                "version": version,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            print(f"✅ Blue-green deployment of version {version} completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during blue-green deployment: {e}")
            return False
    
    def record_deployment(self, deployment_info: Dict) -> None:
        """Record deployment information"""
        try:
            # Create deployments directory if it doesn't exist
            deployments_dir = Path("deployments/history")
            deployments_dir.mkdir(parents=True, exist_ok=True)
            
            # Create deployment record file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"deployment_{timestamp}.json"
            
            with open(deployments_dir / filename, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            print(f"Deployment record saved: {filename}")
            
        except Exception as e:
            print(f"Error recording deployment: {e}")
    
    def list_deployment_history(self, limit: int = 10) -> List[Dict]:
        """List recent deployment history"""
        try:
            deployments_dir = Path("deployments/history")
            if not deployments_dir.exists():
                return []
            
            # Get all deployment files
            deployment_files = sorted(
                deployments_dir.glob("deployment_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            deployments = []
            for file_path in deployment_files[:limit]:
                with open(file_path, 'r') as f:
                    deployment = json.load(f)
                    deployment['file'] = file_path.name
                    deployments.append(deployment)
            
            return deployments
            
        except Exception as e:
            print(f"Error listing deployment history: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(description="Production Deployment Manager")
    parser.add_argument("action", choices=[
        "deploy", "rollback", "status", "history", "health-check", "switch-traffic"
    ], help="Action to perform")
    
    parser.add_argument("--version", help="Version to deploy or rollback to")
    parser.add_argument("--environment", choices=["blue", "green"], help="Target environment")
    parser.add_argument("--reason", help="Reason for rollback")
    parser.add_argument("--percentage", type=int, default=100, help="Traffic percentage")
    parser.add_argument("--from-env", help="Source environment for traffic switch")
    parser.add_argument("--to-env", help="Target environment for traffic switch")
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    if args.action == "deploy":
        if not args.version:
            print("Error: --version is required for deploy action")
            sys.exit(1)
        
        success = manager.blue_green_deployment(args.version)
        sys.exit(0 if success else 1)
    
    elif args.action == "rollback":
        if not args.version or not args.reason:
            print("Error: --version and --reason are required for rollback action")
            sys.exit(1)
        
        success = manager.rollback_deployment(args.version, args.reason)
        sys.exit(0 if success else 1)
    
    elif args.action == "status":
        env_state = manager.get_current_environment_state()
        print("Current Environment State:")
        print(f"  Active: {env_state['active']}")
        print(f"  Standby: {env_state['standby']}")
        print(f"  Traffic Distribution: {env_state.get('traffic_distribution', {})}")
    
    elif args.action == "history":
        deployments = manager.list_deployment_history()
        print("Recent Deployment History:")
        for deployment in deployments:
            print(f"  {deployment['timestamp']}: {deployment['type']} - {deployment.get('version', 'N/A')}")
    
    elif args.action == "health-check":
        if not args.environment:
            print("Error: --environment is required for health-check action")
            sys.exit(1)
        
        success = manager.run_health_checks(args.environment)
        sys.exit(0 if success else 1)
    
    elif args.action == "switch-traffic":
        if not args.from_env or not args.to_env:
            print("Error: --from-env and --to-env are required for switch-traffic action")
            sys.exit(1)
        
        success = manager.switch_traffic(args.from_env, args.to_env, args.percentage)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()