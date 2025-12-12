#!/usr/bin/env python3
"""
Deployment Integration Test Runner
Comprehensive test runner for all deployment-related functionality
"""

import sys
import os
import subprocess
import time
import tempfile
import shutil
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentTestRunner:
    """Comprehensive deployment test runner"""
    
    def __init__(self):
        self.temp_dir = None
        self.test_results = {}
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created test environment: {self.temp_dir}")
        
        # Create mock model file
        self.mock_model_path = Path(self.temp_dir) / "mock_model.pth"
        with open(self.mock_model_path, 'wb') as f:
            f.write(b'0' * (50 * 1024 * 1024))  # 50MB mock model
        
        # Create test images
        self.create_test_images()
    
    def create_test_images(self):
        """Create test images for API testing"""
        try:
            from PIL import Image
            
            # Normal chest X-ray
            normal_image = Image.new('RGB', (224, 224), color='white')
            self.normal_image_path = Path(self.temp_dir) / "normal_xray.jpg"
            normal_image.save(self.normal_image_path)
            
            # Pneumonia chest X-ray
            pneumonia_image = Image.new('RGB', (224, 224), color='gray')
            self.pneumonia_image_path = Path(self.temp_dir) / "pneumonia_xray.jpg"
            pneumonia_image.save(self.pneumonia_image_path)
            
            logger.info("Created test images")
            
        except ImportError:
            logger.warning("PIL not available, skipping test image creation")
            self.normal_image_path = None
            self.pneumonia_image_path = None
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up test environment")
    
    def run_unit_tests(self):
        """Run unit tests for deployment components"""
        logger.info("Running deployment unit tests...")
        
        test_files = [
            "tests/test_deployment_integration.py",
            "tests/test_deployment_automation.py"
        ]
        
        results = {}
        
        for test_file in test_files:
            if Path(test_file).exists():
                logger.info(f"Running {test_file}...")
                
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
                    ], capture_output=True, text=True, timeout=300)
                    
                    results[test_file] = {
                        'returncode': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'passed': result.returncode == 0
                    }
                    
                    if result.returncode == 0:
                        logger.info(f"✅ {test_file} passed")
                    else:
                        logger.error(f"❌ {test_file} failed")
                        logger.error(f"STDERR: {result.stderr}")
                
                except subprocess.TimeoutExpired:
                    logger.error(f"❌ {test_file} timed out")
                    results[test_file] = {
                        'returncode': -1,
                        'error': 'timeout',
                        'passed': False
                    }
                
                except Exception as e:
                    logger.error(f"❌ {test_file} error: {e}")
                    results[test_file] = {
                        'returncode': -1,
                        'error': str(e),
                        'passed': False
                    }
            else:
                logger.warning(f"Test file not found: {test_file}")
        
        return results
    
    def test_model_server_components(self):
        """Test model server components"""
        logger.info("Testing model server components...")
        
        try:
            # Test imports
            from deployment.model_server import app, load_model
            from deployment.performance_optimizer import PerformanceOptimizer
            from deployment.deployment_manager import DeploymentManager
            
            logger.info("✅ All imports successful")
            
            # Test basic component initialization
            manager = DeploymentManager(base_port=9000)
            logger.info("✅ DeploymentManager initialization successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model server component test failed: {e}")
            return False
    
    def test_deployment_automation_components(self):
        """Test deployment automation components"""
        logger.info("Testing deployment automation components...")
        
        try:
            from deployment.automated_deploy import AutomatedDeployment
            from deployment.load_balancer import LoadBalancer
            from deployment.deploy_cli import main as cli_main
            
            logger.info("✅ All automation imports successful")
            
            # Test basic initialization
            deployment = AutomatedDeployment()
            logger.info("✅ AutomatedDeployment initialization successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment automation test failed: {e}")
            return False
    
    def test_performance_components(self):
        """Test performance optimization components"""
        logger.info("Testing performance components...")
        
        try:
            from deployment.performance_optimizer import (
                PerformanceOptimizer, ModelCache, PerformanceMonitor, 
                ModelOptimizer, MemoryManager
            )
            from deployment.performance_test import PerformanceTester
            from deployment.performance_dashboard import PerformanceDashboard
            
            logger.info("✅ All performance imports successful")
            
            # Test cache functionality
            cache = ModelCache(max_size=10, ttl_seconds=60)
            test_data = b"test_image_data"
            test_result = {"prediction": "NORMAL", "confidence": 0.95}
            
            # Test cache miss
            result = cache.get(test_data)
            assert result is None
            
            # Test cache put and hit
            cache.put(test_data, test_result)
            cached_result = cache.get(test_data)
            assert cached_result is not None
            assert cached_result["prediction"] == "NORMAL"
            
            logger.info("✅ Cache functionality test passed")
            
            # Test performance monitor
            monitor = PerformanceMonitor()
            monitor.record_request(100.0, "NORMAL", error=False)
            stats = monitor.get_stats()
            assert stats["request_count"] == 1
            
            logger.info("✅ Performance monitor test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance components test failed: {e}")
            return False
    
    def test_configuration_loading(self):
        """Test configuration loading"""
        logger.info("Testing configuration loading...")
        
        try:
            # Test deployment config
            config_path = Path(self.temp_dir) / "test_config.json"
            test_config = {
                "base_port": 9000,
                "docker_image": "test-chest-xray-api",
                "validation_tests": {
                    "enabled": True,
                    "max_response_time_ms": 2000
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(test_config, f)
            
            from deployment.automated_deploy import AutomatedDeployment
            deployment = AutomatedDeployment(config_file=str(config_path))
            
            assert deployment.config["base_port"] == 9000
            assert deployment.config["docker_image"] == "test-chest-xray-api"
            
            logger.info("✅ Configuration loading test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration loading test failed: {e}")
            return False
    
    def test_model_validation(self):
        """Test model validation functionality"""
        logger.info("Testing model validation...")
        
        try:
            from deployment.automated_deploy import AutomatedDeployment
            deployment = AutomatedDeployment()
            
            # Test valid model file
            result = deployment.validate_model(str(self.mock_model_path))
            assert result is True
            
            # Test non-existent file
            result = deployment.validate_model("non_existent.pth")
            assert result is False
            
            # Test invalid file (too small)
            small_model_path = Path(self.temp_dir) / "small_model.pth"
            with open(small_model_path, 'wb') as f:
                f.write(b"small")
            
            result = deployment.validate_model(str(small_model_path))
            assert result is False
            
            logger.info("✅ Model validation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model validation test failed: {e}")
            return False
    
    def test_cli_functionality(self):
        """Test CLI functionality"""
        logger.info("Testing CLI functionality...")
        
        try:
            # Test CLI import
            from deployment.deploy_cli import main, cmd_list, cmd_status
            
            # Test that functions are callable
            assert callable(main)
            assert callable(cmd_list)
            assert callable(cmd_status)
            
            logger.info("✅ CLI functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ CLI functionality test failed: {e}")
            return False
    
    def run_integration_tests(self):
        """Run integration tests"""
        logger.info("Running integration tests...")
        
        tests = [
            ("Model Server Components", self.test_model_server_components),
            ("Deployment Automation", self.test_deployment_automation_components),
            ("Performance Components", self.test_performance_components),
            ("Configuration Loading", self.test_configuration_loading),
            ("Model Validation", self.test_model_validation),
            ("CLI Functionality", self.test_cli_functionality)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name}...")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed")
            except Exception as e:
                logger.error(f"❌ {test_name} error: {e}")
                results[test_name] = False
        
        return results
    
    def generate_test_report(self, unit_results, integration_results):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "temp_dir": str(self.temp_dir)
            },
            "unit_tests": unit_results,
            "integration_tests": integration_results,
            "summary": {
                "unit_tests_passed": sum(1 for r in unit_results.values() if r.get('passed', False)),
                "unit_tests_total": len(unit_results),
                "integration_tests_passed": sum(1 for r in integration_results.values() if r),
                "integration_tests_total": len(integration_results),
            }
        }
        
        # Calculate overall success
        unit_success_rate = report["summary"]["unit_tests_passed"] / max(report["summary"]["unit_tests_total"], 1)
        integration_success_rate = report["summary"]["integration_tests_passed"] / max(report["summary"]["integration_tests_total"], 1)
        
        report["summary"]["unit_success_rate"] = unit_success_rate
        report["summary"]["integration_success_rate"] = integration_success_rate
        report["summary"]["overall_success"] = unit_success_rate > 0.8 and integration_success_rate > 0.8
        
        # Save report
        report_path = Path(self.temp_dir) / "deployment_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to: {report_path}")
        
        return report
    
    def print_summary(self, report):
        """Print test summary"""
        print("\n" + "="*60)
        print("DEPLOYMENT INTEGRATION TEST SUMMARY")
        print("="*60)
        
        print(f"Unit Tests: {report['summary']['unit_tests_passed']}/{report['summary']['unit_tests_total']} passed")
        print(f"Integration Tests: {report['summary']['integration_tests_passed']}/{report['summary']['integration_tests_total']} passed")
        print(f"Unit Success Rate: {report['summary']['unit_success_rate']:.1%}")
        print(f"Integration Success Rate: {report['summary']['integration_success_rate']:.1%}")
        
        if report["summary"]["overall_success"]:
            print("\n🎉 OVERALL: TESTS PASSED")
            print("Deployment system is ready for use!")
        else:
            print("\n⚠️ OVERALL: TESTS FAILED")
            print("Some components need attention before deployment.")
        
        print("\nFailed Tests:")
        for test_name, result in report["unit_tests"].items():
            if not result.get('passed', False):
                print(f"  ❌ {test_name}")
        
        for test_name, result in report["integration_tests"].items():
            if not result:
                print(f"  ❌ {test_name}")
        
        print("="*60)
    
    def run_all_tests(self):
        """Run all deployment tests"""
        logger.info("Starting comprehensive deployment testing...")
        
        try:
            # Run unit tests
            unit_results = self.run_unit_tests()
            
            # Run integration tests
            integration_results = self.run_integration_tests()
            
            # Generate report
            report = self.generate_test_report(unit_results, integration_results)
            
            # Print summary
            self.print_summary(report)
            
            return report["summary"]["overall_success"]
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
        
        finally:
            self.cleanup_test_environment()


def main():
    """Main test runner"""
    print("🚀 Starting Deployment Integration Tests")
    print("Testing all deployment components and workflows...")
    print()
    
    runner = DeploymentTestRunner()
    
    try:
        success = runner.run_all_tests()
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        runner.cleanup_test_environment()
        return 1
    
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        runner.cleanup_test_environment()
        return 1


if __name__ == "__main__":
    sys.exit(main())