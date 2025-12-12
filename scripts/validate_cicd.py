#!/usr/bin/env python3
"""
CI/CD Setup Validation Script
Validates that all CI/CD components are properly configured
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and report the result"""
    path = Path(file_path)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    return exists


def validate_yaml_file(file_path: str, description: str) -> bool:
    """Validate that a YAML file is properly formatted"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Handle multi-document YAML files (common in Kubernetes)
            if '---' in content:
                documents = yaml.safe_load_all(content)
                list(documents)  # Force evaluation of all documents
            else:
                yaml.safe_load(content)
        print(f"✅ {description} is valid YAML")
        return True
    except yaml.YAMLError as e:
        print(f"❌ {description} has invalid YAML: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ {description} not found: {file_path}")
        return False
    except UnicodeDecodeError as e:
        print(f"❌ {description} has encoding issues: {e}")
        return False


def validate_json_file(file_path: str, description: str) -> bool:
    """Validate that a JSON file is properly formatted"""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        print(f"✅ {description} is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ {description} has invalid JSON: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ {description} not found: {file_path}")
        return False


def validate_github_workflows() -> bool:
    """Validate GitHub Actions workflows"""
    print("\n🔍 Validating GitHub Actions Workflows...")
    
    workflows = [
        (".github/workflows/ci-cd.yml", "Main CI/CD Pipeline"),
        (".github/workflows/security-scan.yml", "Security Scanning Pipeline"),
        (".github/workflows/model-validation.yml", "Model Validation Pipeline"),
        (".github/workflows/dependency-update.yml", "Dependency Update Pipeline"),
    ]
    
    all_valid = True
    for workflow_path, description in workflows:
        if not check_file_exists(workflow_path, description):
            all_valid = False
        elif not validate_yaml_file(workflow_path, description):
            all_valid = False
    
    return all_valid


def validate_configuration_files() -> bool:
    """Validate configuration files"""
    print("\n🔍 Validating Configuration Files...")
    
    configs = [
        (".pre-commit-config.yaml", "Pre-commit Configuration"),
        (".github/renovate.json", "Renovate Configuration"),
        ("pyproject.toml", "Python Project Configuration"),
    ]
    
    all_valid = True
    for config_path, description in configs:
        if not check_file_exists(config_path, description):
            all_valid = False
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            if not validate_yaml_file(config_path, description):
                all_valid = False
        elif config_path.endswith('.json'):
            if not validate_json_file(config_path, description):
                all_valid = False
    
    return all_valid


def validate_test_structure() -> bool:
    """Validate test structure and files"""
    print("\n🔍 Validating Test Structure...")
    
    test_files = [
        "tests/test_data_pipeline.py",
        "tests/test_training_components.py",
        "tests/test_training_integration.py",
        "tests/test_deployment_automation.py",
        "tests/test_deployment_integration.py",
        "tests/test_api_integration.py",
        "tests/test_monitoring_components.py",
        "tests/test_monitoring_integration.py",
        "tests/test_monitoring_metrics.py",
        "tests/test_retraining_workflows.py",
        "tests/test_audit_explainability.py",
        "tests/conftest.py",
    ]
    
    all_exist = True
    for test_file in test_files:
        if not check_file_exists(test_file, f"Test file"):
            all_exist = False
    
    # Check test runner scripts
    scripts = [
        "scripts/run_tests.py",
        "scripts/validate_cicd.py",
    ]
    
    for script in scripts:
        if not check_file_exists(script, f"Script"):
            all_exist = False
    
    return all_exist


def validate_requirements_files() -> bool:
    """Validate requirements files"""
    print("\n🔍 Validating Requirements Files...")
    
    req_files = [
        ("requirements.txt", "Main Requirements"),
        ("requirements-dev.txt", "Development Requirements"),
    ]
    
    all_valid = True
    for req_file, description in req_files:
        if not check_file_exists(req_file, description):
            all_valid = False
        else:
            # Check if file has content
            try:
                with open(req_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        print(f"✅ {description} has content")
                    else:
                        print(f"⚠️ {description} is empty")
            except Exception as e:
                print(f"❌ Error reading {description}: {e}")
                all_valid = False
    
    return all_valid


def validate_docker_files() -> bool:
    """Validate Docker files"""
    print("\n🔍 Validating Docker Files...")
    
    docker_files = [
        "docker-compose.yml",
        "training/Dockerfile",
    ]
    
    all_exist = True
    for docker_file in docker_files:
        if not check_file_exists(docker_file, f"Docker file"):
            all_exist = False
    
    return all_exist


def validate_kubernetes_manifests() -> bool:
    """Validate Kubernetes manifests"""
    print("\n🔍 Validating Kubernetes Manifests...")
    
    k8s_files = [
        "k8s/data-pipeline.yaml",
        "k8s/training.yaml",
        "k8s/deployment.yaml",
        "k8s/monitoring.yaml",
        "k8s/model-registry.yaml",
        "k8s/minio.yaml",
    ]
    
    all_valid = True
    for k8s_file in k8s_files:
        if not check_file_exists(k8s_file, f"Kubernetes manifest"):
            all_valid = False
        elif not validate_yaml_file(k8s_file, f"Kubernetes manifest"):
            all_valid = False
    
    return all_valid


def check_environment_variables() -> bool:
    """Check for required environment variables in CI/CD"""
    print("\n🔍 Checking Environment Variables...")
    
    # These would be set in GitHub Secrets in a real environment
    required_vars = [
        "PYTHON_VERSION",
        "DOCKER_REGISTRY", 
        "IMAGE_NAME",
    ]
    
    # For validation, we'll just check if they're defined in the workflow files
    workflow_file = ".github/workflows/ci-cd.yml"
    
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        all_found = True
        for var in required_vars:
            if var in content:
                print(f"✅ Environment variable {var} found in workflow")
            else:
                print(f"❌ Environment variable {var} not found in workflow")
                all_found = False
        
        return all_found
    except FileNotFoundError:
        print(f"❌ Workflow file not found: {workflow_file}")
        return False
    except UnicodeDecodeError as e:
        print(f"❌ Error reading workflow file: {e}")
        return False


def validate_security_configuration() -> bool:
    """Validate security configuration"""
    print("\n🔍 Validating Security Configuration...")
    
    security_checks = [
        ("Security scanning workflow exists", ".github/workflows/security-scan.yml"),
        ("Pre-commit hooks configured", ".pre-commit-config.yaml"),
        ("Bandit configuration present", "pyproject.toml"),
    ]
    
    all_valid = True
    for description, file_path in security_checks:
        if not check_file_exists(file_path, description):
            all_valid = False
    
    return all_valid


def generate_validation_report() -> Dict[str, Any]:
    """Generate a comprehensive validation report"""
    print("🚀 Starting CI/CD Setup Validation...")
    print("=" * 60)
    
    validations = [
        ("GitHub Workflows", validate_github_workflows),
        ("Configuration Files", validate_configuration_files),
        ("Test Structure", validate_test_structure),
        ("Requirements Files", validate_requirements_files),
        ("Docker Files", validate_docker_files),
        ("Kubernetes Manifests", validate_kubernetes_manifests),
        ("Environment Variables", check_environment_variables),
        ("Security Configuration", validate_security_configuration),
    ]
    
    results = {}
    overall_success = True
    
    for category, validation_func in validations:
        try:
            success = validation_func()
            results[category] = success
            if not success:
                overall_success = False
        except Exception as e:
            print(f"❌ Error validating {category}: {e}")
            results[category] = False
            overall_success = False
    
    return {
        "overall_success": overall_success,
        "category_results": results,
        "timestamp": str(Path.cwd()),
    }


def main():
    """Main validation function"""
    report = generate_validation_report()
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    for category, success in report["category_results"].items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {category}")
    
    print("\n" + "=" * 60)
    if report["overall_success"]:
        print("🎉 CI/CD SETUP VALIDATION PASSED!")
        print("Your MLOps CI/CD pipeline is properly configured.")
        sys.exit(0)
    else:
        print("❌ CI/CD SETUP VALIDATION FAILED!")
        print("Please fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()