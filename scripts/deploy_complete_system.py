#!/usr/bin/env python3
"""
Complete MLOps System Deployment Script
Deploys and validates the entire chest X-ray pneumonia detection system
"""

import argparse
import subprocess
import sys
import time
import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLOpsSystemDeployer:
    """Complete MLOps system deployment and validation"""
    
    def __init__(self, config_path: str = ".", model_path: str = "models/best_chest_xray_model.pth"):
        self.config_path = Path(config_path)
        self.model_path = Path(model_path)
        self.deployment_start_time = time.time()
        
        # Service endpoints (will be updated during deployment)
        self.endpoints = {
            'model_server': 'http://localhost:8000',
            'api_gateway': 'http://localhost:8080',
            'monitoring': 'http://localhost:9090',
            'grafana': 'http://localhost:3000'
        }
        
        # Deployment status
        self.deployment_status = {
            'infrastructure': False,
            'model_server': False,
            'api_gateway': False,
            'monitoring': False,
            'validation': False
        }
    
    def validate_prerequisites(self) -> bool:
        """Validate system prerequisites"""
        logger.info("🔍 Validating system prerequisites...")
        
        # Check if trained model exists
        if not self.model_path.exists():
            logger.error(f"❌ Trained model not found at {self.model_path}")
            logger.info("Please ensure the trained model file exists before deployment")
            return False
        
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Found trained model: {self.model_path} ({model_size_mb:.1f}MB)")
        
        # Check Python dependencies
        required_packages = [
            'torch', 'torchvision', 'fastapi', 'uvicorn', 
            'prometheus_client', 'requests', 'pillow', 'numpy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Missing required packages: {missing_packages}")
            logger.info("Install missing packages with: pip install -r requirements.txt")
            return False
        
        logger.info("✅ All Python dependencies available")
        
        # Check Docker (optional)
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Docker available for containerized deployment")
            else:
                logger.info("⚠️ Docker not available (optional)")
        except FileNotFoundError:
            logger.info("⚠️ Docker not available (optional)")
        
        return True
    
    def start_model_server(self) -> bool:
        """Start the model server"""
        logger.info("🚀 Starting model server...")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['MODEL_PATH'] = str(self.model_path)
            env['MODEL_ARCHITECTURE'] = 'efficientnet_b4'
            env['DEVICE'] = 'cpu'  # Use CPU for demo, GPU if available
            
            # Start model server
            cmd = [
                sys.executable, '-m', 'uvicorn',
                'deployment.model_server:app',
                '--host', '0.0.0.0',
                '--port', '8000',
                '--reload'
            ]
            
            logger.info(f"Starting command: {' '.join(cmd)}")
            
            # Start in background
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            max_wait = 60  # seconds
            wait_time = 0
            
            while wait_time < max_wait:
                try:
                    response = requests.get(f"{self.endpoints['model_server']}/health", timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        logger.info("✅ Model server started successfully")
                        logger.info(f"   Status: {health_data.get('status', 'unknown')}")
                        logger.info(f"   Model loaded: {health_data.get('model_loaded', False)}")
                        logger.info(f"   Architecture: {health_data.get('model_architecture', 'unknown')}")
                        self.deployment_status['model_server'] = True
                        return True
                except requests.RequestException:
                    pass
                
                time.sleep(2)
                wait_time += 2
                logger.info(f"   Waiting for model server... ({wait_time}s)")
            
            logger.error("❌ Model server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error starting model server: {e}")
            return False
    
    def start_api_gateway(self) -> bool:
        """Start the API gateway"""
        logger.info("🚀 Starting API gateway...")
        
        try:
            # Start API gateway
            cmd = [
                sys.executable, '-m', 'uvicorn',
                'deployment.api:app',
                '--host', '0.0.0.0',
                '--port', '8080',
                '--reload'
            ]
            
            # Start in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            max_wait = 30
            wait_time = 0
            
            while wait_time < max_wait:
                try:
                    response = requests.get(f"{self.endpoints['api_gateway']}/health", timeout=5)
                    if response.status_code == 200:
                        logger.info("✅ API gateway started successfully")
                        self.deployment_status['api_gateway'] = True
                        return True
                except requests.RequestException:
                    pass
                
                time.sleep(2)
                wait_time += 2
                logger.info(f"   Waiting for API gateway... ({wait_time}s)")
            
            logger.error("❌ API gateway failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error starting API gateway: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """Start monitoring services"""
        logger.info("📊 Starting monitoring services...")
        
        try:
            # Start Prometheus metrics exporter
            cmd = [
                sys.executable, 'monitoring/prometheus_exporter.py'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for startup
            time.sleep(5)
            
            # Check if monitoring is accessible
            try:
                response = requests.get(f"{self.endpoints['monitoring']}/metrics", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Monitoring services started successfully")
                    self.deployment_status['monitoring'] = True
                    return True
            except requests.RequestException:
                pass
            
            logger.info("⚠️ Monitoring services started (metrics endpoint not accessible)")
            self.deployment_status['monitoring'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting monitoring: {e}")
            return False
    
    def validate_system_health(self) -> bool:
        """Validate overall system health"""
        logger.info("🏥 Validating system health...")
        
        health_checks = []
        
        # Check model server
        try:
            response = requests.get(f"{self.endpoints['model_server']}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                health_checks.append({
                    'service': 'Model Server',
                    'status': 'healthy',
                    'details': f"Model: {health_data.get('model_architecture', 'unknown')}"
                })
            else:
                health_checks.append({
                    'service': 'Model Server',
                    'status': 'unhealthy',
                    'details': f"HTTP {response.status_code}"
                })
        except Exception as e:
            health_checks.append({
                'service': 'Model Server',
                'status': 'error',
                'details': str(e)
            })
        
        # Check API gateway
        try:
            response = requests.get(f"{self.endpoints['api_gateway']}/health", timeout=10)
            if response.status_code == 200:
                health_checks.append({
                    'service': 'API Gateway',
                    'status': 'healthy',
                    'details': 'Responding normally'
                })
            else:
                health_checks.append({
                    'service': 'API Gateway',
                    'status': 'unhealthy',
                    'details': f"HTTP {response.status_code}"
                })
        except Exception as e:
            health_checks.append({
                'service': 'API Gateway',
                'status': 'error',
                'details': str(e)
            })
        
        # Display health check results
        all_healthy = True
        for check in health_checks:
            if check['status'] == 'healthy':
                logger.info(f"   ✅ {check['service']}: {check['details']}")
            else:
                logger.error(f"   ❌ {check['service']}: {check['details']}")
                all_healthy = False
        
        if all_healthy:
            logger.info("✅ All system components are healthy")
            self.deployment_status['validation'] = True
        else:
            logger.error("❌ Some system components are unhealthy")
        
        return all_healthy
    
    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        logger.info("🧪 Running integration tests...")
        
        try:
            # Run the final integration tests
            cmd = [
                sys.executable, '-m', 'pytest',
                'tests/test_final_integration.py',
                '-v', '--tb=short'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Integration tests passed")
                return True
            else:
                logger.error("❌ Integration tests failed")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Integration tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running integration tests: {e}")
            return False
    
    def generate_deployment_report(self) -> Dict:
        """Generate deployment report"""
        deployment_time = time.time() - self.deployment_start_time
        
        report = {
            'deployment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'deployment_duration_seconds': round(deployment_time, 2),
            'model_path': str(self.model_path),
            'model_size_mb': round(self.model_path.stat().st_size / (1024 * 1024), 2),
            'deployment_status': self.deployment_status,
            'service_endpoints': self.endpoints,
            'system_health': all(self.deployment_status.values())
        }
        
        return report
    
    def deploy_complete_system(self) -> bool:
        """Deploy the complete MLOps system"""
        logger.info("🚀 Starting complete MLOps system deployment...")
        logger.info(f"Model: {self.model_path}")
        
        deployment_steps = [
            ("Validating prerequisites", self.validate_prerequisites),
            ("Starting model server", self.start_model_server),
            ("Starting API gateway", self.start_api_gateway),
            ("Starting monitoring", self.start_monitoring),
            ("Validating system health", self.validate_system_health),
            ("Running integration tests", self.run_integration_tests),
        ]
        
        for step_name, step_func in deployment_steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Step: {step_name}")
            logger.info(f"{'='*60}")
            
            if not step_func():
                logger.error(f"❌ Failed at step: {step_name}")
                return False
            
            logger.info(f"✅ Completed: {step_name}")
        
        # Generate deployment report
        report = self.generate_deployment_report()
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 MLOPS SYSTEM DEPLOYMENT COMPLETED!")
        logger.info(f"{'='*60}")
        
        logger.info(f"\n📊 Deployment Summary:")
        logger.info(f"   Duration: {report['deployment_duration_seconds']}s")
        logger.info(f"   Model: {report['model_path']} ({report['model_size_mb']}MB)")
        logger.info(f"   System Health: {'✅ Healthy' if report['system_health'] else '❌ Issues detected'}")
        
        logger.info(f"\n🌐 Service Endpoints:")
        for service, endpoint in self.endpoints.items():
            status = "✅" if self.deployment_status.get(service.replace('_', ''), False) else "❌"
            logger.info(f"   {status} {service.replace('_', ' ').title()}: {endpoint}")
        
        logger.info(f"\n📚 Next Steps:")
        logger.info("1. Access the API documentation at http://localhost:8000/docs")
        logger.info("2. Test predictions using the web interface at http://localhost:8080")
        logger.info("3. Monitor system metrics at http://localhost:9090")
        logger.info("4. View logs and performance data")
        
        # Save deployment report
        report_path = Path("deployment_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Deployment report saved to: {report_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Deploy complete MLOps system")
    parser.add_argument("--model-path", default="models/best_chest_xray_model.pth",
                       help="Path to trained model file")
    parser.add_argument("--config-path", default=".",
                       help="Path to configuration files")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip integration tests")
    parser.add_argument("--health-check-only", action="store_true",
                       help="Only run health checks")
    
    args = parser.parse_args()
    
    deployer = MLOpsSystemDeployer(args.config_path, args.model_path)
    
    if args.health_check_only:
        success = deployer.validate_system_health()
    else:
        success = deployer.deploy_complete_system()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()