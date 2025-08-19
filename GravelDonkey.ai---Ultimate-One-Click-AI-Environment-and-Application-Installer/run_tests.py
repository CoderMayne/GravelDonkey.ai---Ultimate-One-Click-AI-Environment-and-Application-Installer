#!/usr/bin/env python3
"""
Comprehensive test runner for the AI Development Environment Installer.
Provides various testing options and generates detailed reports.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import time
import json

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout.strip()}")
        if e.stderr:
            print(f"Stderr: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False

def check_dependencies():
    """Check if required testing dependencies are installed."""
    print("🔍 Checking testing dependencies...")
    
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-html"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements-test.txt")
        return False
    
    print("✅ All testing dependencies are available")
    return True

def run_unit_tests(verbose=False, coverage=True, html_report=True):
    """Run unit tests."""
    print("\n🧪 Running Unit Tests...")
    
    command = ["python", "-m", "pytest", "tests/", "-v"]
    
    if coverage:
        command.extend(["--cov=.", "--cov-report=term-missing"])
    
    if html_report:
        command.extend(["--html=test_reports/unit_tests.html", "--self-contained-html"])
    
    if not verbose:
        command.append("-q")
    
    return run_command(command, "Unit tests")

def run_integration_tests(verbose=False, coverage=True, html_report=True):
    """Run integration tests."""
    print("\n🔗 Running Integration Tests...")
    
    command = ["python", "-m", "pytest", "tests/test_integration_comprehensive.py", "-v"]
    
    if coverage:
        command.extend(["--cov=.", "--cov-report=term-missing"])
    
    if html_report:
        command.extend(["--html=test_reports/integration_tests.html", "--self-contained-html"])
    
    if not verbose:
        command.append("-q")
    
    return run_command(command, "Integration tests")

def run_specific_test(test_file, verbose=False):
    """Run a specific test file."""
    print(f"\n🎯 Running specific test: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    command = ["python", "-m", "pytest", test_file, "-v"]
    
    if not verbose:
        command.append("-q")
    
    return run_command(command, f"Specific test: {test_file}")

def run_performance_tests():
    """Run performance tests."""
    print("\n⚡ Running Performance Tests...")
    
    command = ["python", "-m", "pytest", "tests/", "-m", "benchmark", "-v"]
    
    return run_command(command, "Performance tests")

def run_smoke_tests():
    """Run smoke tests for critical functionality."""
    print("\n💨 Running Smoke Tests...")
    
    command = ["python", "-m", "pytest", "tests/", "-m", "smoke", "-v"]
    
    return run_command(command, "Smoke tests")

def generate_coverage_report():
    """Generate comprehensive coverage report."""
    print("\n📊 Generating Coverage Report...")
    
    # Create test_reports directory if it doesn't exist
    os.makedirs("test_reports", exist_ok=True)
    
    command = [
        "python", "-m", "pytest", "tests/",
        "--cov=.",
        "--cov-report=html:test_reports/coverage_html",
        "--cov-report=xml:test_reports/coverage.xml",
        "--cov-report=term-missing"
    ]
    
    success = run_command(command, "Coverage report generation")
    
    if success:
        print(f"\n📁 Coverage reports generated in test_reports/ directory")
        print(f"   - HTML: test_reports/coverage_html/index.html")
        print(f"   - XML: test_reports/coverage.xml")
    
    return success

def run_all_tests(verbose=False, coverage=True, html_report=True):
    """Run all tests with comprehensive reporting."""
    print("🚀 Running Complete Test Suite...")
    
    # Create test_reports directory
    os.makedirs("test_reports", exist_ok=True)
    
    start_time = time.time()
    
    # Run all tests
    command = ["python", "-m", "pytest", "tests/", "-v"]
    
    if coverage:
        command.extend(["--cov=.", "--cov-report=term-missing"])
    
    if html_report:
        command.extend([
            "--html=test_reports/complete_test_suite.html",
            "--self-contained-html"
        ])
    
    if not verbose:
        command.append("-q")
    
    success = run_command(command, "Complete test suite")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
    
    if success:
        print("✅ All tests completed successfully")
    else:
        print("❌ Some tests failed")
    
    return success

def list_available_tests():
    """List all available test files and test cases."""
    print("\n📋 Available Tests:")
    
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("❌ Tests directory not found")
        return
    
    test_files = list(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("❌ No test files found")
        return
    
    for test_file in test_files:
        print(f"\n📁 {test_file.name}:")
        
        # Try to get test cases from the file
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find test functions and classes
            import re
            test_functions = re.findall(r'def (test_\w+)', content)
            test_classes = re.findall(r'class (Test\w+)', content)
            
            if test_classes:
                print(f"   Classes: {', '.join(test_classes)}")
            
            if test_functions:
                print(f"   Functions: {', '.join(test_functions[:5])}")  # Show first 5
                if len(test_functions) > 5:
                    print(f"   ... and {len(test_functions) - 5} more")
            
        except Exception as e:
            print(f"   Error reading file: {e}")

def create_test_summary():
    """Create a summary of test results."""
    print("\n📈 Creating Test Summary...")
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_suite": "AI Development Environment Installer",
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "coverage": 0,
        "execution_time": 0
    }
    
    # Try to read pytest results if available
    try:
        # This would need to be enhanced to actually parse pytest output
        # For now, just create a basic summary
        summary_file = "test_reports/test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Test summary created: {summary_file}")
    except Exception as e:
        print(f"⚠️  Could not create test summary: {e}")
    
    return summary

def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Development Environment Installer Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit --verbose         # Run unit tests with verbose output
  python run_tests.py --integration            # Run integration tests only
  python run_tests.py --coverage               # Generate coverage report
  python run_tests.py --file tests/test_gpu_simulator.py  # Run specific test file
  python run_tests.py --list                   # List available tests
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-html", action="store_true", help="Disable HTML reports")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    print("🧪 AI Development Environment Installer - Test Runner")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Cannot proceed without required dependencies")
        return
    
    # Create test_reports directory
    os.makedirs("test_reports", exist_ok=True)
    
    # Determine coverage and HTML report settings
    coverage = not args.no_coverage
    html_report = not args.no_html
    
    success = True
    
    try:
        if args.list:
            list_available_tests()
        
        elif args.file:
            success = run_specific_test(args.file, args.verbose)
        
        elif args.unit:
            success = run_unit_tests(args.verbose, coverage, html_report)
        
        elif args.integration:
            success = run_integration_tests(args.verbose, coverage, html_report)
        
        elif args.performance:
            success = run_performance_tests()
        
        elif args.smoke:
            success = run_smoke_tests()
        
        elif args.coverage:
            success = generate_coverage_report()
        
        elif args.all:
            success = run_all_tests(args.verbose, coverage, html_report)
        
        # Create test summary
        create_test_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        success = False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        success = False
    
    # Final status
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test execution completed successfully!")
        print("📁 Check test_reports/ directory for detailed reports")
    else:
        print("❌ Test execution completed with errors")
        print("🔍 Check the output above for details")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

