import subprocess
import sys
import os

def _is_wsl_environment():
    is_wsl = False
    if sys.platform == "linux":
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    is_wsl = True
        except FileNotFoundError:
            pass # Not a WSL environment if /proc/version doesn't exist
    print(f"DEBUG: sys.version: {sys.version}")
    print(f"DEBUG: sys.platform: {sys.platform}")
    print(f"DEBUG: _is_wsl_environment() returning: {is_wsl}")
    return is_wsl

def check_wsl():
    if _is_wsl_environment():
        print("DEBUG: check_wsl() - Running in WSL environment branch.")
        return True, "WSL 2: Detected and running (script running inside WSL)."
    else:
        print("DEBUG: check_wsl() - Running in Windows environment branch.")
        try:
            result = subprocess.run(["wsl", "--status"], check=True, capture_output=True, text=True)
            print(f"DEBUG: wsl --status stdout: {result.stdout}")
            print(f"DEBUG: wsl --status stderr: {result.stderr}")
            if "Default Version: 2" in result.stdout:
                return True, "WSL 2: Detected and running."
            else:
                return False, "WSL 2: Not detected or not running (Default Version not 2)."
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"DEBUG: Error running wsl --status: {e}")
            return False, "WSL 2: Not detected or not running."

def check_wsl_distro():
    if _is_wsl_environment():
        print("DEBUG: check_wsl_distro() - Running in WSL environment branch.")
        return True, "WSL Distribution: Detected (script running inside a WSL distro)."
    else:
        print("DEBUG: check_wsl_distro() - Running in Windows environment branch.")
        try:
            result = subprocess.run(["wsl", "-l", "-v"], check=True, capture_output=True, text=True)
            print(f"DEBUG: wsl -l -v stdout: {result.stdout}")
            print(f"DEBUG: wsl -l -v stderr: {result.stderr}")
            if "*" in result.stdout:
                return True, "WSL Distribution: Detected."
            else:
                return False, "WSL Distribution: No default distribution found."
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"DEBUG: Error running wsl -l -v: {e}")
            return False, "WSL Distribution: Not detected."

def check_docker():
    try:
        result = subprocess.run(["docker", "--version"], check=True, capture_output=True, text=True)
        return True, f"Docker: Detected ({result.stdout.strip()})."
    except (subprocess.CalledProcessError, FileNotFoundError):\
        return False, "Docker: Not detected."

def check_docker_wsl_integration():
    try:
        result = subprocess.run(["docker", "context", "show"], check=True, capture_output=True, text=True)
        if "default" in result.stdout:
            return True, "Docker-WSL Integration: Docker context detected."
        else:
            return False, "Docker-WSL Integration: Docker context not detected."
    except (subprocess.CalledProcessError, FileNotFoundError):\
        return False, "Docker-WSL Integration: Not detected."

def check_python():
    try:
        cmd = ["python3", "--version"]
        if not _is_wsl_environment():
            cmd.insert(0, "wsl") # Prepend 'wsl' if not already in WSL
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, f"Python: Detected (running on version {result.stdout.strip()})."
    except (subprocess.CalledProcessError, FileNotFoundError):\
        return False, "Python: Not detected in WSL."

def check_gpu_drivers():
    try:
        cmd = ["nvidia-smi"]
        if not _is_wsl_environment():
            cmd.insert(0, "wsl") # Prepend 'wsl' if not already in WSL
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if "NVIDIA-SMI" in result.stdout:
            return True, "GPU Drivers: Successfully detected a NVIDIA GPU accessible from WSL."
        else:
            return False, "GPU Drivers: NVIDIA GPU not accessible from WSL."
    except (subprocess.CalledProcessError, FileNotFoundError):\
        return False, "GPU Drivers: NVIDIA GPU not accessible from WSL."

def check_teacache():
    try:
        # Attempt to import teacache to check if it's installed
        import teacache
        return True, "Teacache: Detected."
    except ImportError:\
        return False, "Teacache: Not detected."

def check_ubuntu_version():
    if not _is_wsl_environment():
        return True, "Ubuntu Version: Check skipped (not in WSL).", "Unknown"

    detected_version = "Unknown"
    try:
        # Try lsb_release first
        result = subprocess.run(["lsb_release", "-r"], check=True, capture_output=True, text=True)
        version_line = result.stdout.strip()
        detected_version = version_line.replace('Release:        ', '')
        if "24.04" in detected_version:
            return True, f"Ubuntu Version: {detected_version} detected.", detected_version
        elif "22.04" in detected_version:
            return True, f"Ubuntu Version: {detected_version} detected.", detected_version
        else:
            return False, f"Ubuntu Version: {detected_version} detected. Only 22.04 or 24.04 are supported.", detected_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to /etc/os-release if lsb_release is not found
        try:
            with open("/etc/os-release", "r") as f:
                os_release_content = f.read()
            
            os_id = ""
            version_id = ""
            for line in os_release_content.splitlines():
                if line.startswith("ID="):
                    os_id = line.split("=")[1].strip('"')
                if line.startswith("VERSION_ID="):
                    version_id = line.split("=")[1].strip('"')
            
            if os_id == "ubuntu" and (version_id == "24.04" or version_id == "22.04"):
                detected_version = version_id
                return True, f"Ubuntu Version: {detected_version} detected (from /etc/os-release).", detected_version
            else:
                detected_version = version_id if os_id == "ubuntu" else "Unknown"
                return False, f"Ubuntu Version: {detected_version} detected. Only 22.04 or 24.04 are supported.", detected_version
        except FileNotFoundError:
            return False, "Ubuntu Version: Could not determine (lsb_release and /etc/os-release not found).", "Unknown"
    except Exception as e:
        return False, f"Ubuntu Version: An unexpected error occurred: {e}", "Unknown"

if __name__ == "__main__":
    print("Running prerequisite checks...")
    checks = [
        check_wsl,
        check_wsl_distro,
        check_docker,
        check_docker_wsl_integration,
        check_python,
        check_gpu_drivers,
        check_teacache,
        check_ubuntu_version, # Add the new check here
    ]

    all_passed = True
    for check_func in checks:
        passed, message, _ = check_func() # Unpack the third return value
        print(f"- {message}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll prerequisite checks passed.")
    else:
        print("\nSome prerequisite checks failed. Please address the issues above.")
        sys.exit(1)