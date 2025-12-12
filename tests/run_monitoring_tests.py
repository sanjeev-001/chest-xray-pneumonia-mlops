"""
Test Runner for Monitoring Components
Runs all monitoring-related tests with detailed reporting
"""

import pytest
import sys
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_monitoring_tests():
    """Run all monitoring component tests"""
    
    print("="*80)
    print("CHEST X-RAY PNEUMONIA DETECTION - MONITORING TESTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test files to run
    test_files = [
        "tests/test_monitoring_components.py",
        "tests/test_audit_explainability.py",
        "tests/test_monitoring_metrics.py"
    ]
    
    # Check if test files exist
    existing_files = []
    for test_file in test_files:
        if (project_root / test_file).exists():
            existing_files.append(test_file)
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    if not existing_files:
        print("❌ No test files found!")
        return False
    
    print(f"Found {len(existing_files)} test files:")
    for file in existing_files:
        print(f"  ✓ {file}")
    print()
    
    # Run tests with detailed output
    all_passed = True
    results = {}
    
    for test_file in existing_files:
        print(f"Running tests in {test_file}...")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--no-header",
                "--disable-warnings"
            ], 
            cwd=project_root,
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout per test file
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ PASSED ({duration:.1f}s)")
                results[test_file] = {"status": "PASSED", "duration": duration}
            else:
                print(f"❌ FAILED ({duration:.1f}s)")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results[test_file] = {"status": "FAILED", "duration": duration}
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT (>300s)")
            results[test_file] = {"status": "TIMEOUT", "duration": 300}
            all_passed = False
            
        except Exception as e:
            print(f"💥 ERROR: {e}")
            results[test_file] = {"status": "ERROR", "duration": 0}
            all_passed = False
        
        print()
    
    # Print summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_duration = sum(r["duration"] for r in results.values())
    
    for test_file, result in results.items():
        status_icon = {
            "PASSED": "✅",
            "FAILED": "❌", 
            "TIMEOUT": "⏰",
            "ERROR": "💥"
        }.get(result["status"], "❓")
        
        print(f"{status_icon} {test_file:<40} {result['status']:<8} ({result['duration']:.1f}s)")
    
    print("-" * 80)
    print(f"Total time: {total_duration:.1f}s")
    
    passed_count = sum(1 for r in results.values() if r["status"] == "PASSED")
    total_count = len(results)
    
    if all_passed:
        print(f"🎉 ALL TESTS PASSED ({passed_count}/{total_count})")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed_count}/{total_count})")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_passed

def run_specific_test_class(test_class: str):
    """Run a specific test class"""
    
    print(f"Running specific test class: {test_class}")
    print("-" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_monitoring_components.py",
            f"-k", test_class,
            "-v", 
            "--tb=short"
        ], 
        cwd=project_root,
        timeout=120
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running test class {test_class}: {e}")
        return False

def run_integration_tests():
    """Run integration tests specifically"""
    
    print("Running Monitoring Integration Tests...")
    print("-" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_monitoring_components.py::TestIntegratedMonitoring",
            "-v", 
            "--tb=short"
        ], 
        cwd=project_root,
        timeout=180
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running integration tests: {e}")
        return False

def check_monitoring_dependencies():
    """Check if monitoring dependencies are available"""
    
    print("Checking monitoring dependencies...")
    print("-" * 40)
    
    dependencies = {
        "numpy": "numpy",
        "psutil": "psutil", 
        "sqlite3": "sqlite3",
        "threading": "threading",
        "datetime": "datetime",
        "json": "json",
        "pathlib": "pathlib",
        "tempfile": "tempfile",
        "unittest.mock": "unittest.mock"
    }
    
    missing = []
    available = []
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            available.append(name)
            print(f"✅ {name}")
        except ImportError:
            missing.append(name)
            print(f"❌ {name}")
    
    print(f"\nDependencies: {len(available)} available, {len(missing)} missing")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        return False
    
    return True

def main():
    """Main test runner"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "deps":
            success = check_monitoring_dependencies()
            sys.exit(0 if success else 1)
            
        elif command == "integration":
            success = run_integration_tests()
            sys.exit(0 if success else 1)
            
        elif command.startswith("class:"):
            test_class = command.split(":", 1)[1]
            success = run_specific_test_class(test_class)
            sys.exit(0 if success else 1)
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  deps        - Check dependencies")
            print("  integration - Run integration tests only")
            print("  class:NAME  - Run specific test class")
            sys.exit(1)
    
    # Run all tests by default
    success = run_monitoring_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()