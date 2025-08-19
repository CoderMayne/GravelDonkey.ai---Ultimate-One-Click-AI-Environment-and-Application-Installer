import subprocess
import webbrowser
import os
import shutil
from prereq_checker import _is_wsl_environment

def install_wsl():
    """Installs WSL 2 and the recommended Ubuntu distribution. This function is intended to be run from Windows, not from within WSL."""
    if _is_wsl_environment():
        return "WSL installation cannot be initiated from within WSL. Please run this from a Windows command prompt."
    try:
        # Install WSL
        subprocess.run(["wsl", "--install", "-d", "Ubuntu"], check=True, capture_output=True, text=True)
        return "WSL and Ubuntu installed successfully. Please restart your computer."
    except subprocess.CalledProcessError as e:
        return f"Error installing WSL: {e.stderr}"

def install_python():
    """Installs Python 3 and pip in the WSL environment."""
    try:
        cmd = ["sudo", "apt", "update"]
        if not _is_wsl_environment():
            cmd.insert(0, "wsl")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        cmd = ["sudo", "apt", "install", "-y", "python3", "python3-pip"]
        if not _is_wsl_environment():
            cmd.insert(0, "wsl")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return "Python 3 and pip installed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error installing Python: {e.stderr}"

def check_nvidia_smi():
    """Checks for the presence of nvidia-smi."""
    try:
        cmd = ["nvidia-smi"]
        if not _is_wsl_environment():
            cmd.insert(0, "wsl")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"nvidia-smi found:\n{result.stdout}"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return f"Error running nvidia-smi: {e}"

def open_docker_website():
    """Opens the Docker Desktop download page."""
    if _is_wsl_environment():
        print("Please open the following URL in your browser: https://www.docker.com/products/docker-desktop")
        return "Please open the Docker website in your browser."
    else:
        webbrowser.open("https://www.docker.com/products/docker-desktop")
        return "Opened Docker website."

def check_docker():
    """Checks Docker version and context."""
    try:
        result = subprocess.run(["docker", "--version"], check=True, capture_output=True, text=True)
        return True, f"Docker: Detected ({result.stdout.strip()})."
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return f"Error checking Docker: {e}"

def open_nvidia_driver_website():
    """Opens the NVIDIA driver download page."""
    if _is_wsl_environment():
        print("Please open the following URL in your browser: https://www.nvidia.com/drivers/")
        return "Please open the NVIDIA driver download page in your browser."
    else:
        webbrowser.open("https://www.nvidia.com/drivers/")
        return "Opened NVIDIA driver download page. Please download and install the latest drivers for your GPU."

def open_cuda_toolkit_website():
    """Opens the CUDA Toolkit download page."""
    if _is_wsl_environment():
        print("Please open the following URL in your browser: https://developer.nvidia.com/cuda-downloads")
        return "Please open the CUDA Toolkit download page in your browser."
    else:
        webbrowser.open("https://developer.nvidia.com/cuda-downloads")
        return "Opened CUDA Toolkit download page. Please download and install the appropriate CUDA Toolkit."

def open_wsl_integration_guide():
    """Opens a guide for enabling WSL 2 integration in Docker Desktop."""
    # This is a placeholder. A direct link to a specific setting is not feasible.
    # Instead, we link to Docker Desktop's WSL 2 integration documentation.
    if _is_wsl_environment():
        print("Please open the following URL in your browser: https://docs.docker.com/desktop/wsl/")
        return "Please open the Docker Desktop WSL 2 integration guide in your browser."
    else:
        webbrowser.open("https://docs.docker.com/desktop/wsl/")
        return "Opened Docker Desktop WSL 2 integration guide. Please follow the instructions to enable WSL 2 integration for your Linux distro."

def install_teacache():
    """Installs Teacache from its GitHub repository."""
    try:
        # Check if git is installed
        subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Error: Git is not installed. Please install Git to proceed with Teacache installation."

    temp_dir = os.path.join(os.getcwd(), "temp_teacache_install")
    repo_url = "https://github.com/LiewFeng/TeaCache"
    
    # Ensure the temp directory is clean
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    original_cwd = os.getcwd()
    try:
        # Clone the repository
        subprocess.run(["git", "clone", repo_url, temp_dir], check=True, capture_output=True, text=True)
        
        # Change to the repository directory
        os.chdir(temp_dir)
        
        # Install in editable mode
        cmd = ["pip", "install", "-e", "."]
        if not _is_wsl_environment():
            # If not in WSL, assume pip is available in the Windows environment
            pass
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        return "Teacache installed successfully from GitHub."
    except subprocess.CalledProcessError as e:
        return f"Error installing Teacache from GitHub: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred during Teacache installation: {e}"
    finally:
        # Change back to original directory
        os.chdir(original_cwd)
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)