import subprocess
import re
import shutil
import json
import os

def _is_command_available(command):
    """Check if a command is available in the system's PATH."""
    return shutil.which(command) is not None

def _run_command(command):
    """Runs a command and returns its decoded output or None."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=False,
            encoding='utf-8',
            errors='ignore',
            timeout=15
        )
        if result.returncode != 0:
            print(f"DEBUG: Command '{command}' failed with exit code {result.returncode}")
            print(f"DEBUG: Stderr: {result.stderr}")
            return None
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"DEBUG: Exception running command '{command}': {e}")
        return None

def _parse_gpu_memory(nvidia_smi_output):
    """Extract GPU memory information from nvidia-smi output."""
    try:
        # Look for memory usage pattern: "1234MiB / 24576MiB"
        memory_matches = re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB', nvidia_smi_output)
        if memory_matches:
            used_mb, total_mb = memory_matches[0]
            return int(total_mb), int(used_mb)
        
        # Alternative pattern for memory
        memory_alt = re.search(r'Memory-Usage.*?(\d+)MiB\s*/\s*(\d+)MiB', nvidia_smi_output, re.DOTALL)
        if memory_alt:
            used_mb, total_mb = memory_alt.groups()
            return int(total_mb), int(used_mb)
    except (ValueError, IndexError):
        pass
    return None, None

def _get_gpu_architecture(gpu_name):
    """Determine GPU architecture from name for optimization recommendations."""
    gpu_name_lower = gpu_name.lower()
    
    # NVIDIA architectures
    if any(x in gpu_name_lower for x in ['rtx 40', '4090', '4080', '4070', '4060']):
        return "Ada Lovelace", "sm_89"
    elif any(x in gpu_name_lower for x in ['rtx 30', '3090', '3080', '3070', '3060']):
        return "Ampere", "sm_86"
    elif any(x in gpu_name_lower for x in ['rtx 20', '2080', '2070', '2060', 'titan rtx']):
        return "Turing", "sm_75"
    elif any(x in gpu_name_lower for x in ['gtx 16', '1660', '1650']):
        return "Turing", "sm_75"
    elif any(x in gpu_name_lower for x in ['gtx 10', '1080', '1070', '1060', '1050']):
        return "Pascal", "sm_61"
    elif any(x in gpu_name_lower for x in ['tesla v100']):
        return "Volta", "sm_70"
    elif any(x in gpu_name_lower for x in ['tesla p100']):
        return "Pascal", "sm_60"
    elif any(x in gpu_name_lower for x in ['tesla k80', 'tesla k40']):
        return "Kepler", "sm_37"
    elif any(x in gpu_name_lower for x in ['h100', 'a100']):
        return "Hopper/Ampere", "sm_90"
    
    # AMD architectures
    elif any(x in gpu_name_lower for x in ['rx 9070', '9070 xt']):
        return "RDNA 4", "gfx1200"
    elif any(x in gpu_name_lower for x in ['rx 7900', 'rx 7800', 'rx 7700', 'rx 7600']):
        return "RDNA 3", "gfx1100"
    elif any(x in gpu_name_lower for x in ['rx 6950', 'rx 6900', 'rx 6800', 'rx 6700', 'rx 6600', 'rx 6500']):
        return "RDNA 2", "gfx1030"
    elif any(x in gpu_name_lower for x in ['rx 5700', 'rx 5600', 'rx 5500']):
        return "RDNA", "gfx1010"
    elif any(x in gpu_name_lower for x in ['vega 64', 'vega 56', 'radeon vii']):
        return "GCN 5.0", "gfx900"
    
    # Intel architectures
    elif any(x in gpu_name_lower for x in ['arc a770', 'arc a750', 'arc a580']):
        return "Xe-HPG", "DG2"
    elif any(x in gpu_name_lower for x in ['arc a380', 'arc a310']):
        return "Xe-HPG", "DG2"
    elif 'iris xe' in gpu_name_lower:
        return "Xe-LP", "Gen12"
    
    return "Unknown", "unknown"

def _get_cuda_version_from_nvcc():
    """Tries to get the CUDA version from nvcc."""
    print("DEBUG: Trying to get CUDA version from nvcc...")
    if _is_command_available("nvcc"):
        output = _run_command("nvcc --version")
        if output:
            match = re.search(r"release (\d+\.\d+)", output)
            if match:
                version = match.group(1)
                print(f"DEBUG: Found CUDA version {version} from nvcc.")
                return version
    print("DEBUG: nvcc not found or version not parsable.")
    return None

def _get_nvidia_gpu_info():
    """Gets NVIDIA GPU model, CUDA version, memory, and architecture using multiple WSL-optimized methods."""
    print("DEBUG: Checking for NVIDIA GPU in WSL environment...")
    
    # Method 1: Try comprehensive nvidia-smi query
    if _is_command_available("nvidia-smi"):
        print("DEBUG: nvidia-smi found in PATH, trying comprehensive query...")
        output = _run_command("nvidia-smi --query-gpu=gpu_name,memory.total,memory.used,driver_version,cuda_version --format=csv,noheader,nounits")
        if output and output.strip():
            print(f"DEBUG: nvidia-smi comprehensive output: '{output}'")
            lines = output.strip().split('\n')
            if lines and lines[0].strip():
                try:
                    parts = [part.strip() for part in lines[0].split(',')] 
                    if len(parts) >= 3 and parts[0] and parts[0] != "N/A":
                        gpu_model = parts[0]
                        memory_total = int(parts[1]) if parts[1].isdigit() else None
                        memory_used = int(parts[2]) if parts[2].isdigit() else None
                        driver_version = parts[3] if len(parts) > 3 and parts[3] != "N/A" else "Unknown"
                        cuda_version = parts[4] if len(parts) > 4 and parts[4] != "N/A" else "Unknown"
                        
                        architecture, compute_capability = _get_gpu_architecture(gpu_model)
                        
                        # If CUDA version is still unknown, try another method before returning
                        if cuda_version == "Unknown":
                            print("DEBUG: CUDA version not found in comprehensive query, trying nvcc...")
                            nvcc_version = _get_cuda_version_from_nvcc()
                            if nvcc_version:
                                cuda_version = nvcc_version

                        print(f"DEBUG: Successfully parsed - GPU: {gpu_model}, Memory: {memory_total}MB, CUDA: {cuda_version}, Arch: {architecture}")
                        return gpu_model, cuda_version, driver_version, memory_total, memory_used, architecture, compute_capability
                except Exception as e:
                    print(f"DEBUG: Error parsing comprehensive nvidia-smi output: {e}")
        
        # Method 1b: Try basic nvidia-smi and parse full output
        print("DEBUG: Trying basic nvidia-smi and parsing...")
        output = _run_command("nvidia-smi")
        if output:
            print(f"DEBUG: Basic nvidia-smi output (first 500 chars): {output[:500]}")
            
            # Parse GPU name
            gpu_name = None
            gpu_patterns = [
                r'\|\s+\d+\s+([^|]+?)\s+(?:On|Off)\s+\|',
                r'GeForce [^|]+',
                r'RTX [^|]+',
                r'GTX [^|]+',
                r'Tesla [^|]+',
                r'Quadro [^|]+'
            ]
            
            for pattern in gpu_patterns:
                match = re.search(pattern, output)
                if match:
                    gpu_name = match.group(1).strip() if '|' in pattern else match.group(0).strip()
                    break
            
            if gpu_name:
                # Parse memory
                memory_total, memory_used = _parse_gpu_memory(output)
                
                # Parse CUDA version
                cuda_match = re.search(r'CUDA Version:\s*(\d+\.\d+)', output)
                cuda_version = cuda_match.group(1) if cuda_match else "Unknown"

                # Fallback to nvcc if needed
                if cuda_version == "Unknown":
                    print("DEBUG: CUDA version not found in nvidia-smi output, trying nvcc...")
                    nvcc_version = _get_cuda_version_from_nvcc()
                    if nvcc_version:
                        cuda_version = nvcc_version
                
                # Parse driver version
                driver_match = re.search(r'Driver Version:\s*([0-9.]+)', output)
                driver_version = driver_match.group(1) if driver_match else "Unknown"
                
                architecture, compute_capability = _get_gpu_architecture(gpu_name)
                
                print(f"DEBUG: Parsed from basic nvidia-smi - GPU: {gpu_name}, Memory: {memory_total}MB, CUDA: {cuda_version}")
                return gpu_name, cuda_version, driver_version, memory_total, memory_used, architecture, compute_capability
    
    # Method 2: Try Windows nvidia-smi.exe
    print("DEBUG: nvidia-smi not working, trying Windows nvidia-smi.exe...")
    windows_paths = [
        "nvidia-smi.exe",
        "/mnt/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe",
        "/mnt/c/Windows/System32/nvidia-smi.exe",
        "/mnt/c/Program Files (x86)/NVIDIA Corporation/NVSMI/nvidia-smi.exe"
    ]
    
    for path in windows_paths:
        if path == "nvidia-smi.exe" and not _is_command_available("nvidia-smi.exe"):
            continue
        
        print(f"DEBUG: Trying Windows nvidia-smi at: {path}")
        if " " in path and not path.startswith('"'):
            cmd = f'"{path}" --query-gpu=gpu_name,memory.total,driver_version,cuda_version --format=csv,noheader'
        else:
            cmd = f'{path} --query-gpu=gpu_name,memory.total,driver_version,cuda_version --format=csv,noheader'
        
        output = _run_command(cmd)
        if output and output.strip():
            print(f"DEBUG: Windows nvidia-smi output: '{output}'")
            lines = output.strip().split('\n')
            if lines and lines[0].strip():
                try:
                    parts = [part.strip() for part in lines[0].split(',')] 
                    if len(parts) >= 2 and parts[0] and parts[0] != "N/A":
                        gpu_model = parts[0]
                        memory_total = int(parts[1]) if len(parts) > 1 and parts[1].replace(' MiB', '').isdigit() else None
                        driver_version = parts[2] if len(parts) > 2 and parts[2] != "N/A" else "Unknown"
                        cuda_version = parts[3] if len(parts) > 3 and parts[3] != "N/A" else "Unknown"
                        
                        architecture, compute_capability = _get_gpu_architecture(gpu_model)

                        if cuda_version == "Unknown":
                            print("DEBUG: CUDA version not found in Windows nvidia-smi, trying nvcc...")
                            nvcc_version = _get_cuda_version_from_nvcc()
                            if nvcc_version:
                                cuda_version = nvcc_version
                        
                        print(f"DEBUG: Successfully parsed Windows nvidia-smi - GPU: {gpu_model}, CUDA: {cuda_version}")
                        return gpu_model, cuda_version, driver_version, memory_total, None, architecture, compute_capability
                except Exception as e:
                    print(f"DEBUG: Error parsing Windows nvidia-smi output: {e}")
    
    # Method 3: Try lspci fallback
    print("DEBUG: nvidia-smi methods failed, trying lspci...")
    if _is_command_available("lspci"):
        lspci_output = _run_command("lspci | grep -i nvidia")
        if lspci_output:
            print(f"DEBUG: lspci NVIDIA output: {lspci_output}")
            
            gpu_patterns = [
                r'GeForce [^[(\n]+',
                r'RTX [^[(\n]+',
                r'GTX [^[(\n]+',
                r'Tesla [^[(\n]+',
                r'Quadro [^[(\n]+'
            ]
            
            for line in lspci_output.strip().split('\n'):
                if 'VGA compatible controller' in line or '3D controller' in line:
                    for pattern in gpu_patterns:
                        match = re.search(pattern, line)
                        if match:
                            gpu_model = match.group(0).strip()
                            architecture, compute_capability = _get_gpu_architecture(gpu_model)
                            # CUDA version is likely unknown here, but we can try nvcc as a last resort
                            cuda_version = _get_cuda_version_from_nvcc() or "Unknown"
                            return gpu_model, cuda_version, "Unknown", None, None, architecture, compute_capability
            
            # Generic NVIDIA detection
            return "NVIDIA GPU (Model Unknown)", "Unknown", "Unknown", None, None, "Unknown", "unknown"
    
    print("DEBUG: No NVIDIA GPU detected")
    return None, None, None, None, None, None, None

def _get_amd_gpu_info():
    """Gets AMD GPU model and architecture."""
    print("DEBUG: Checking for AMD GPU...")
    
    # Method 1: Try rocm-smi
    if _is_command_available("rocm-smi"):
        print("DEBUG: rocm-smi found...")
        output = _run_command("rocm-smi --showproductname --csv")
        if output and output.strip():
            lines = output.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip():
                    gpu_name = line.strip()
                    architecture, compute_capability = _get_gpu_architecture(gpu_name)
                    print(f"DEBUG: AMD GPU from rocm-smi: {gpu_name}")
                    return gpu_name, architecture, compute_capability

    # Method 2: Try lspci
    if _is_command_available("lspci"):
        print("DEBUG: Checking lspci for AMD...")
        lspci_output = _run_command("lspci | grep -i 'vga\\|3d\\|2d' | grep -i 'amd\\|ati'")
        if lspci_output:
            print(f"DEBUG: lspci AMD output: {lspci_output}")
            
            # Extract AMD GPU model
            patterns = [
                r'Radeon [^[(\n]+',
                r'RX [^[(\n]+',
                r'Vega [^[(\n]+'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, lspci_output, re.IGNORECASE)
                if match:
                    gpu_name = f"AMD {match.group(0).strip()}"
                    architecture, compute_capability = _get_gpu_architecture(gpu_name)
                    return gpu_name, architecture, compute_capability
            
            # Generic AMD detection
            match = re.search(r'AMD/ATI\s+(.+?)(\[|$)', lspci_output, re.IGNORECASE)
            if match:
                gpu_name = f"AMD {match.group(1).strip()}"
                architecture, compute_capability = _get_gpu_architecture(gpu_name)
                return gpu_name, architecture, compute_capability
            
            return "AMD GPU (Model Unknown)", "Unknown", "unknown"
    
    print("DEBUG: No AMD GPU detected")
    return None, None, None

def _get_intel_gpu_info():
    """Gets Intel GPU model and architecture."""
    print("DEBUG: Checking for Intel GPU...")
    
    # Method 1: Try lspci
    if _is_command_available("lspci"):
        print("DEBUG: Checking lspci for Intel...")
        lspci_output = _run_command("lspci | grep -i 'vga\\|3d\\|2d' | grep -i intel")
        if lspci_output:
            print(f"DEBUG: lspci Intel output: {lspci_output}")
            
            # Extract Intel GPU model
            patterns = [
                r'Arc [^[(\n]+',
                r'UHD Graphics [^[(\n]+',
                r'Iris [^[(\n]+',
                r'HD Graphics [^[(\n]+',
                r'Intel Corporation\s+(.+?)(\[|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, lspci_output, re.IGNORECASE)
                if match:
                    if 'Intel Corporation' in pattern:
                        gpu_name = f"Intel {match.group(1).strip()}"
                    else:
                        gpu_name = f"Intel {match.group(0).strip()}"
                    architecture, compute_capability = _get_gpu_architecture(gpu_name)
                    return gpu_name, architecture, compute_capability
            
            return "Intel GPU (Model Unknown)", "Unknown", "unknown"
    
    print("DEBUG: No Intel GPU detected")
    return None, None, None

def get_gpu_info():
    """
    Detects the GPU vendor, model, memory, and architecture, optimized for WSL environments.
    Returns: (gpu_type, gpu_model, cuda_version, driver_version, memory_total, memory_used, architecture, compute_capability)
    """
    print("DEBUG: Starting comprehensive GPU detection (WSL-optimized)...")
    
    # Check for NVIDIA first (highest priority for AI workloads)
    gpu_model, cuda_version, driver_version, memory_total, memory_used, architecture, compute_capability = _get_nvidia_gpu_info()
    if gpu_model:
        print(f"DEBUG: Final result - NVIDIA: {gpu_model}, CUDA: {cuda_version}, Memory: {memory_total}MB, Arch: {architecture}")
        return "nvidia", gpu_model, cuda_version, driver_version, memory_total, memory_used, architecture, compute_capability

    # Check for AMD
    amd_model, architecture, compute_capability = _get_amd_gpu_info()
    if amd_model:
        print(f"DEBUG: Final result - AMD: {amd_model}, Arch: {architecture}")
        return "amd", amd_model, None, None, None, None, architecture, compute_capability

    # Check for Intel
    intel_model, architecture, compute_capability = _get_intel_gpu_info()
    if intel_model:
        print(f"DEBUG: Final result - Intel: {intel_model}, Arch: {architecture}")
        return "intel", intel_model, None, None, None, None, architecture, compute_capability

    # Fallback to CPU
    print("DEBUG: Final result - No GPU detected, defaulting to CPU")
    return "cpu", "N/A", None, None, None, None, "CPU", "cpu"

def get_recommended_dependency(gpu_type, gpu_model, architecture, memory_total):
    """
    Returns recommended dependency configuration based on detected hardware.
    """
    recommendations = {
        "nvidia": {
            "Ada Lovelace": "CUDA 12.6 + cuDNN 9.0 (Recommended)",  # RTX 40 series
            "Ampere": "CUDA 12.4 + cuDNN 8.9",  # RTX 30 series  
            "Turing": "CUDA 12.1 + cuDNN 8.8",  # RTX 20/GTX 16 series
            "Pascal": "CUDA 11.8 + cuDNN 8.7",  # GTX 10 series
            "Volta": "CUDA 12.1 + cuDNN 8.8",   # Tesla V100
            "Kepler": "CUDA 11.8 + cuDNN 8.7",  # Older Tesla
            "Hopper/Ampere": "CUDA 12.6 + cuDNN 9.0 (Recommended)"  # H100/A100
        },
        "amd": {
            "RDNA 3": "ROCm 6.2 (Recommended)",  # RX 7000 series
            "RDNA 2": "ROCm 6.1",  # RX 6000 series
            "RDNA": "ROCm 6.0",    # RX 5000 series
            "GCN 5.0": "ROCm 5.7 (Legacy)"  # Vega series
        },
        "intel": {
            "Xe-HPG": "Intel XPU 2024.2 (Recommended)",  # Arc series
            "Xe-LP": "Intel XPU 2024.1",  # Iris Xe
            "Gen12": "Intel CPU Optimized"
        },
        "cpu": {
            "CPU": "CPU Standard (Recommended)"
        }
    }
    
    return recommendations.get(gpu_type, {}).get(architecture, 
           list(recommendations.get(gpu_type, {"default": "Standard"}).values())[0])

if __name__ == '__main__':
    # For testing purposes
    print("=== Enhanced WSL GPU Detection Test ===")
    gpu_type, gpu_model, cuda_version, driver_version, memory_total, memory_used, architecture, compute_capability = get_gpu_info()
    
    print(f"\n=== COMPREHENSIVE RESULTS ===")
    print(f"Detected GPU Type: {gpu_type.upper()}")
    if gpu_model and gpu_model != "N/A":
        print(f"Detected GPU Model: {gpu_model}")
    if cuda_version and cuda_version != "Unknown":
        print(f"Detected CUDA Version: {cuda_version}")
    if driver_version and driver_version != "Unknown":
        print(f"Driver Version: {driver_version}")
    if memory_total:
        print(f"GPU Memory: {memory_total} MB total")
        if memory_used:
            print(f"GPU Memory Used: {memory_used} MB")
    if architecture != "Unknown":
        print(f"GPU Architecture: {architecture}")
        print(f"Compute Capability: {compute_capability}")
    
    recommended = get_recommended_dependency(gpu_type, gpu_model, architecture, memory_total)
    print(f"Recommended Configuration: {recommended}")
    
    print(f"\nThis system would use the '{gpu_type}' dependency configuration.")