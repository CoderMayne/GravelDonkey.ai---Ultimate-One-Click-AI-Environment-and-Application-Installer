#!/usr/bin/env python3
"""
Simple test execution script for quick testing of the AI Development Environment Installer.
This script provides a simplified way to run tests without the full test runner.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_simple_tests():
    """Run a simple test suite to verify basic functionality."""
    print("🧪 Running Simple Test Suite...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Tests directory not found. Please run from the project root.")
        return False
    
    # Check if pytest is available
    try:
        import pytest
        print("✅ Pytest is available")
    except ImportError:
        print("❌ Pytest not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest"], check=True)
            print("✅ Pytest installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install pytest")
            return False
    
    # Run basic tests
    test_files = [
        "tests/test_gpu_simulator.py",
        "tests/test_enhanced_hardware_detector.py",
        "tests/test_dependency_utils.py",
        "tests/test_prereq_checker.py"
    ]
    
    success_count = 0
    total_tests = len(test_files)
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n🔍 Testing {test_file}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print(f"✅ {test_file} - PASSED")
                    success_count += 1
                else:
                    print(f"❌ {test_file} - FAILED")
                    if result.stdout:
                        print(f"Output: {result.stdout}")
                    if result.stderr:
                        print(f"Errors: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file} - TIMEOUT")
            except Exception as e:
                print(f"💥 {test_file} - ERROR: {e}")
        else:
            print(f"⚠️  {test_file} - NOT FOUND")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

def run_quick_integration_test():
    """Run a quick integration test to verify system functionality."""
    print("\n🔗 Running Quick Integration Test...")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("🔍 Testing module imports...")
        
        from enhanced_hardware_detector import get_gpu_info
        print("✅ enhanced_hardware_detector imported")
        
        from gpu_simulator import GPUSimulator
        print("✅ gpu_simulator imported")
        
        from dependency_utils import fetch_package_versions
        print("✅ dependency_utils imported")
        
        from prereq_checker import check_wsl, check_docker
        print("✅ prereq_checker imported")
        
        # Test basic functionality
        print("\n🔍 Testing basic functionality...")
        
        # Test GPU simulator
        simulator = GPUSimulator()
        print("✅ GPU Simulator initialized")
        
        # Test hardware detection (should fall back to CPU)
        gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
        print(f"✅ Hardware detection: {gpu_info[0]} ({gpu_info[1]})")
        
        print("\n🎉 Quick integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick integration test failed: {e}")
        return False

def main():
    """Main function."""
    print("🧪 AI Development Environment Installer - Simple Test Runner")
    print("=" * 60)
    
    # Run simple tests
    simple_success = run_simple_tests()
    
    # Run quick integration test
    integration_success = run_quick_integration_test()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 Final Test Summary:")
    print(f"   Simple Tests: {'✅ PASSED' if simple_success else '❌ FAILED'}")
    print(f"   Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    if simple_success and integration_success:
        print("\n🎉 All tests completed successfully!")
        print("🚀 The system is ready for use!")
        return 0
    else:
        print("\n❌ Some tests failed.")
        print("🔍 Check the output above for details.")
        print("📚 Refer to TESTING_README.md for troubleshooting.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

