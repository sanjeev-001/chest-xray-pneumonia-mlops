#!/usr/bin/env python3
"""
Test Runner Script for MLOps CI/CD Pipeline
Runs different test suites based on the component specified
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description or command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def run_data_pipeline_tests():
    """Run data pipeline tests"""
    commands = [
        ("python -m pytest tests/test_data_pipeline.py -v --cov=data_pipeline --cov-report=xml --cov-report=term", 
         "Data Pipeline Tests"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def run_training_tests():
    """Run training pipeline tests"""
    commands = [
        ("python -m pytest tests/test_training_components.py -v --cov=training --cov-report=xml", 
         "Training Components Tests"),
        ("python -m pytest tests/test_training_integration.py -v", 
         "Training Integration Tests"),
        ("python -m pytest tests/test_experiment_tracking.py -v", 
         "Experiment Tracking Tests"),
        ("python -m pytest tests/test_hyperparameter_optimization.py -v", 
         "Hyperparameter Optimization Tests"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def run_deployment_tests():
    """Run deployment tests"""
    commands = [
        ("python -m pytest tests/test_deployment_automation.py -v --cov=deployment --cov-report=xml", 
         "Deployment Automation Tests"),
        ("python -m pytest tests/test_deployment_integration.py -v", 
         "Deployment Integration Tests"),
        ("python -m pytest tests/test_api_integration.py -v", 
         "API Integration Tests"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def run_monitoring_tests():
    """Run monitoring tests"""
    commands = [
        ("python -m pytest tests/test_monitoring_components.py -v --cov=monitoring --cov-report=xml", 
         "Monitoring Components Tests"),
        ("python -m pytest tests/test_monitoring_integration.py -v", 
         "Monitoring Integration Tests"),
        ("python -m pytest tests/test_monitoring_metrics.py -v", 
         "Monitoring Metrics Tests"),
        ("python -m pytest tests/test_audit_explainability.py -v", 
         "Audit and Explainability Tests"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def run_retraining_tests():
    """Run retraining workflow tests"""
    commands = [
        ("python -m pytest tests/test_retraining_workflows.py -v --cov=training --cov-report=xml", 
         "Retraining Workflow Tests"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def run_all_tests():
    """Run all test suites"""
    commands = [
        ("python -m pytest tests/ -v --cov=. --cov-report=xml --cov-report=term --cov-report=html", 
         "All Tests with Coverage"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def run_smoke_tests():
    """Run quick smoke tests"""
    commands = [
        ('python -c "import data_pipeline; print(\'[OK] Data pipeline imports OK\')"', 
         "Data Pipeline Import Test"),
        ('python -c "import training; print(\'[OK] Training module imports OK\')"', 
         "Training Module Import Test"),
        ('python -c "import deployment; print(\'[OK] Deployment module imports OK\')"', 
         "Deployment Module Import Test"),
        ('python -c "import monitoring; print(\'[OK] Monitoring module imports OK\')"', 
         "Monitoring Module Import Test"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def run_security_tests():
    """Run security tests"""
    commands = [
        ("bandit -r . -f json -o bandit-report.json", 
         "Bandit Security Scan"),
        ("safety check --json --output safety-report.json", 
         "Safety Vulnerability Check"),
    ]
    
    success = True
    for command, description in commands:
        # Security tests are informational, don't fail the build
        run_command(command, description)
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Run MLOps test suites")
    parser.add_argument(
        "test_suite", 
        choices=[
            "data-pipeline", "training", "deployment", "monitoring", 
            "retraining", "all", "smoke", "security"
        ],
        help="Test suite to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c", 
        action="store_true", 
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    print(f"🚀 Starting {args.test_suite} test suite...")
    print(f"Working directory: {Path.cwd()}")
    print(f"Python path: {sys.executable}")
    
    # Run the appropriate test suite
    test_functions = {
        "data-pipeline": run_data_pipeline_tests,
        "training": run_training_tests,
        "deployment": run_deployment_tests,
        "monitoring": run_monitoring_tests,
        "retraining": run_retraining_tests,
        "all": run_all_tests,
        "smoke": run_smoke_tests,
        "security": run_security_tests,
    }
    
    success = test_functions[args.test_suite]()
    
    if success:
        print(f"\n🎉 {args.test_suite} tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ {args.test_suite} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()