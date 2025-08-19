#!/usr/bin/env python3
"""
GPU Diagnostic Script
Run this to diagnose GPU detection issues in WSL
"""

import subprocess
import shutil
import os
import sys

def run_command(cmd, description):
    """Run a command and show results"""
    print(f"\n{'='*60}")
    print(f"TESTING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(f"RETURN CODE: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command took too long")
        return False, "", "Timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, "", str(e)

def main():
    print("GPU DIAGNOSTIC SCRIPT")
    print("This will help diagnose why GPU detection isn't working properly")
    print("\nSystem Information:")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Test basic commands
    tests = [
        ("echo $PATH", "Check PATH environment"),
        ("which nvidia-smi", "Check if nvidia-smi is in PATH"),
        ("ls -la /usr/bin/nvidia*", "Check for nvidia binaries in /usr/bin"),
        ("ls -la /usr/local/cuda*/bin/nvidia*", "Check for nvidia binaries in CUDA"),
        ("whereis nvidia-smi", "Find all nvidia-smi locations"),
        
        # Try different nvidia-smi commands
        ("nvidia-smi", "Basic nvidia-smi (should show GPU table)"),
        ("nvidia-smi -L", "List GPUs"),
        ("nvidia-smi --query-gpu=name --format=csv,noheader", "Query GPU name only"),
        ("nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader", "Query detailed info"),
        
        # Alternative detection methods
        ("lspci | grep -i nvidia", "Check PCI devices for NVIDIA"),
        ("cat /proc/driver/nvidia/version", "Check NVIDIA driver version"),
        ("ls -la /dev/nvidia*", "Check NVIDIA device files"),
        
        # WSL-specific checks
        ("uname -r", "Check WSL kernel version"),
        ("cat /proc/version", "Check kernel details"),
        ("ls -la /usr/lib/wsl/drivers/", "Check WSL GPU drivers"),
        
        # Windows nvidia-smi (if accessible)
        ("nvidia-smi.exe", "Try Windows nvidia-smi.exe"),
        ("/mnt/c/Program\\ Files/NVIDIA\\ Corporation/NVSMI/nvidia-smi.exe", "Try direct path to Windows nvidia-smi"),
    ]
    
    results = {}
    
    for cmd, description in tests:
        success, stdout, stderr = run_command(cmd, description)
        results[description] = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    # Check if nvidia-smi is available
    if results["Check if nvidia-smi is in PATH"]['success']:
        print("✅ nvidia-smi is available in PATH")
    else:
        print("❌ nvidia-smi is NOT in PATH")
        print("   This is likely the main issue!")
    
    # Check if basic nvidia-smi works
    if results["Basic nvidia-smi (should show GPU table)"]['success']:
        print("✅ nvidia-smi command works")
    else:
        print("❌ nvidia-smi command fails")
        print("   Error:", results["Basic nvidia-smi (should show GPU table)"]['stderr'])
    
    # Check for NVIDIA hardware
    if results["Check PCI devices for NVIDIA"]['success']:
        print("✅ NVIDIA hardware detected via lspci")
        print("   Hardware:", results["Check PCI devices for NVIDIA"]['stdout'].strip())
    else:
        print("❌ No NVIDIA hardware found via lspci")
    
    # Check WSL drivers
    if results["Check WSL GPU drivers"]['success']:
        print("✅ WSL GPU drivers found")
        print("   Drivers:", results["Check WSL GPU drivers"]['stdout'].strip())
    else:
        print("❌ No WSL GPU drivers found")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if not results["Check if nvidia-smi is in PATH"]['success']:
        print("1. nvidia-smi is not available. Try these solutions:")
        print("   - Install NVIDIA drivers in WSL: sudo apt update && sudo apt install nvidia-driver-535")
        print("   - Or add Windows nvidia-smi to PATH")
        print("   - Or use the Windows version directly")
    
    if results["Check PCI devices for NVIDIA"]['success'] and not results["Basic nvidia-smi (should show GPU table)"]['success']:
        print("2. Hardware is detected but nvidia-smi fails:")
        print("   - This suggests a driver issue")
        print("   - Try: sudo apt update && sudo apt install nvidia-utils-535")
    
    if results["Try Windows nvidia-smi.exe"]['success']:
        print("3. Windows nvidia-smi works! We can use that as fallback")
    
    print("\n4. If all else fails, we can implement Windows nvidia-smi fallback")

if __name__ == "__main__":
    main()