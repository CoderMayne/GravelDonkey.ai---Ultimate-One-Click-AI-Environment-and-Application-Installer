import subprocess
import re
import shutil
import json
import os
from typing import Dict, Tuple, Optional, List
from gpu_simulator import GPUSimulator

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
    if any(x in gpu_name_lower for x in ['rtx 50', '5090', '5080', '5070', '5060']):
        return "Blackwell", "sm_90"
    elif any(x in gpu_name_lower for x in ['rtx 40', '4090', '4080', '4070', '4060']):
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
    elif any(x in gpu_name_lower for x in ['rx 90', '9060', '9070']):
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
    elif any(x in gpu_name_lower for x in ['arc b', 'b770', 'b750', 'b580', 'b380']):
        return "Xe-HPG", "DG3"
    elif any(x in gpu_name_lower for x in ['arc a770', 'arc a750', 'arc a580']):
        return "Xe-HPG", "DG2"
    elif any(x in gpu_name_lower for x in ['arc a380', 'arc a310']):
        return "Xe-HPG", "DG2"
    elif 'iris xe' in gpu_name_lower:
        return "Xe-LP", "Gen12"
    
    return "Unknown", "unknown"

def _get_model_key(gpu_name):
    """Generates a sanitized model key from the GPU name."""
    if not gpu_name:
        return None
    # Convert to lowercase, replace spaces and non-alphanumeric with underscores
    key = re.sub(r'[^a-z0-9]+', '_', gpu_name.lower()).strip('_')
    # Specific replacements for common NVIDIA names to match gpu_database.json
    key = key.replace('nvidia_geforce_rtx_', 'rtx_')
    key = key.replace('nvidia_geforce_gtx_', 'gtx_')
    key = key.replace('amd_radeon_rx_', 'rx_')
    key = key.replace('intel_arc_a', 'arc_a')
    # More specific replacements if needed to match exact keys in gpu_database.json
    return key

def _get_nvidia_gpu_info():
    """Gets NVIDIA GPU model, CUDA version, memory, and architecture using multiple WSL-optimized methods."""
    print("DEBUG: Checking for NVIDIA GPU in WSL environment...")
    
    # Method 1: Try comprehensive nvidia-smi query
    if _is_command_available("nvidia-smi"):
        print("DEBUG: nvidia-smi found...")
        output = _run_command("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader,nounits")
        if output and output.strip():
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')] 
                    if len(parts) >= 4:
                        gpu_name = parts[0]
                        driver_version = parts[1]
                        memory_total = int(parts[2]) if parts[2].isdigit() else None
                        memory_used = int(parts[3]) if parts[3].isdigit() else None
                        architecture, compute_capability = _get_gpu_architecture(gpu_name)
                        
                        # Get CUDA version
                        cuda_output = _run_command("nvidia-smi --query-gpu=cuda_version --format=csv,noheader")
                        cuda_version = cuda_output.strip() if cuda_output else "Unknown"
                        
                        print(f"DEBUG: NVIDIA GPU from nvidia-smi: {gpu_name}")
                        return "nvidia", gpu_name, cuda_version, driver_version, memory_total, memory_used, architecture, compute_capability
    
    # Method 2: Try lspci for basic detection
    if _is_command_available("lspci"):
        print("DEBUG: Checking lspci for NVIDIA...")
        lspci_output = _run_command("lspci | grep -i 'vga\\|3d\\|2d' | grep -i nvidia")
        if lspci_output:
            print(f"DEBUG: lspci NVIDIA output: {lspci_output}")
            match = re.search(r'NVIDIA.*?\\[(.+?)\\]', lspci_output, re.IGNORECASE)
            if match:
                gpu_name = f"NVIDIA {match.group(1).strip()}"
                architecture, compute_capability = _get_gpu_architecture(gpu_name)
                return gpu_name, "Unknown", "Unknown", None, None, architecture, compute_capability
    
    print("DEBUG: No NVIDIA GPU detected")
    return None, None, None, None, None, None, None

def _get_nvidia_gpu_info_with_sim(simulator: Optional[GPUSimulator] = None, simulated_model: str = "rtx_4090"):
    """Get NVIDIA GPU information with optional simulation support."""
    if simulator:
        print("DEBUG: Using NVIDIA GPU simulator...")
        gpu_name, architecture, compute_cap, memory_total, memory_used = simulator.simulate_nvidia_gpu_detection(simulated_model)
        cuda_version = simulator.get_ai_recommendations("nvidia", simulated_model).get("cuda", "12.4")
        driver_version = "580.88"
        return ("nvidia", gpu_name, cuda_version, driver_version, memory_total, memory_used, architecture, compute_cap)
    
    # Real NVIDIA detection logic
    return _get_nvidia_gpu_info_real()

def _get_nvidia_gpu_info_real():
    """Real NVIDIA GPU detection logic."""
    # Original NVIDIA detection code from hardware_detector.py
    if not _is_command_available("nvidia-smi"):
        return ("cpu", "N/A", "Unknown", "Unknown", 0, 0, "Unknown", "unknown")
    
    try:
        # Get GPU name
        result = _run_command("nvidia-smi --query-gpu=name --format=csv,noheader")
        if result and result.strip():
            gpu_name = result.strip()
        else:
            return ("cpu", "N/A", "Unknown", "Unknown", 0, 0, "Unknown", "unknown")
        
        # Get CUDA version
        result = _run_command("nvidia-smi --query-gpu=cuda_version --format=csv,noheader")
        cuda_version = result.strip() if result else "Unknown"
        
        # Get driver version
        result = _run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        driver_version = result.strip() if result else "Unknown"
        
        # Get memory info
        result = _run_command("nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader")
        memory_total, memory_used = _parse_gpu_memory(result) if result else (0, 0)
        
        # Get architecture info
        architecture, compute_capability = _get_gpu_architecture(gpu_name)
        
        print(f"DEBUG: Final result - NVIDIA: {gpu_name}, CUDA: {cuda_version}, Memory: {memory_total}MB, Arch: {architecture}")
        return "nvidia", gpu_name, cuda_version, driver_version, memory_total, memory_used, architecture, compute_capability
        
    except Exception as e:
        print(f"DEBUG: Error detecting NVIDIA GPU: {e}")
        return ("cpu", "N/A", "Unknown", "Unknown", 0, 0, "Unknown", "unknown")

def _get_amd_gpu_info(simulator: Optional[GPUSimulator] = None, simulated_model: str = "rx_7900_xtx"):
    """Gets AMD GPU model and architecture, with optional simulation support."""
    print("DEBUG: Checking for AMD GPU...")
    
    # If simulator is provided, use it for testing
    if simulator:
        print("DEBUG: Using AMD GPU simulator...")
        gpu_name, architecture, compute_capability, memory_total, memory_used = simulator.simulate_amd_gpu_detection(simulated_model)
        return gpu_name, architecture, compute_capability, memory_total, memory_used
    
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
                    return gpu_name, architecture, compute_capability, None, None

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
                    return gpu_name, architecture, compute_capability, None, None
            
            # Generic AMD detection
            match = re.search(r'AMD/ATI\\s+(.+?)(\\[|$)', lspci_output, re.IGNORECASE)
            if match:
                gpu_name = f"AMD {match.group(1).strip()}"
                architecture, compute_capability = _get_gpu_architecture(gpu_name)
                return gpu_name, architecture, compute_capability, None, None
            
            return "AMD GPU (Model Unknown)", "Unknown", "unknown", None, None
    
    print("DEBUG: No AMD GPU detected")
    return None, None, None, None, None

def _get_intel_gpu_info(simulator: Optional[GPUSimulator] = None, simulated_model: str = "arc_a770"):
    """Gets Intel GPU model and architecture, with optional simulation support."""
    print("DEBUG: Checking for Intel GPU...")
    
    # If simulator is provided, use it for testing
    if simulator:
        print("DEBUG: Using Intel GPU simulator...")
        gpu_name, architecture, compute_capability, memory_total, memory_used = simulator.simulate_intel_gpu_detection(simulated_model)
        return gpu_name, architecture, compute_capability, memory_total, memory_used
    
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
                r'Intel Corporation\\s+(.+?)(\\[|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, lspci_output, re.IGNORECASE)
                if match:
                    if 'Intel Corporation' in pattern:
                        gpu_name = f"Intel {match.group(1).strip()}"
                    else:
                        gpu_name = f"Intel {match.group(0).strip()}"
                    architecture, compute_capability = _get_gpu_architecture(gpu_name)
                    return gpu_name, architecture, compute_capability, None, None
            
            return "Intel GPU (Model Unknown)", "Unknown", "unknown", None, None
    
    print("DEBUG: No Intel GPU detected")
    return None, None, None, None, None

def get_gpu_info(simulation_mode: str = "auto", simulated_gpu_type: str = "nvidia", simulated_gpu_model: str = "rtx_4090") -> Dict:
    """
    Detects the GPU vendor and model, and returns a dictionary with all hardware info and AI recommendations.

    Args:
        simulation_mode: "auto", "force_nvidia", "force_amd", "force_intel", "force_cpu"
        simulated_gpu_type: The GPU type to simulate.
        simulated_gpu_model: The specific GPU model to simulate.

    Returns:
        A dictionary containing all GPU information and recommendations.
    """
    print(f"DEBUG: get_gpu_info called with mode: {simulation_mode}")
    simulator = GPUSimulator()
    info = {
        "gpu_type": "cpu",
        "gpu_model": "N/A",
        "gpu_model_key": None, # Added gpu_model_key
        "cuda_version": None,
        "driver_version": None,
        "memory_total": None,
        "memory_used": None,
        "architecture": "CPU",
        "compute_capability": "cpu",
        "recommendations": {} # This will be removed as recommendations are now looked up in dockerfile_generator.py
    }

    detected_info = None
    if simulation_mode == "force_nvidia":
        detected_info = _get_nvidia_gpu_info_with_sim(simulator, simulated_gpu_model)
    elif simulation_mode == "force_amd":
        detected_info = _get_amd_gpu_info(simulator, simulated_gpu_model)
    elif simulation_mode == "force_intel":
        detected_info = _get_intel_gpu_info(simulator, simulated_gpu_model)
    elif simulation_mode == "auto":
        # Try NVIDIA first, then AMD, then Intel
        detected_info = _get_nvidia_gpu_info() or _get_amd_gpu_info() or _get_intel_gpu_info()

    if detected_info:
        info["gpu_type"] = detected_info[0]
        info["gpu_model"] = detected_info[1]
        info["gpu_model_key"] = _get_model_key(info["gpu_model"]) # Set gpu_model_key
        if detected_info[0] == 'nvidia':
            info["cuda_version"] = detected_info[2]
            info["driver_version"] = detected_info[3]
            info["memory_total"] = detected_info[4]
            info["memory_used"] = detected_info[5]
            info["architecture"] = detected_info[6]
            info["compute_capability"] = detected_info[7]

    # Removed the recommendations lookup here as it's now done in dockerfile_generator.py
    # model_key = simulator.get_model_key(info["gpu_model"], info["gpu_type"])
    # if model_key:
    #     info["recommendations"] = simulator.get_ai_recommendations(info["gpu_type"], model_key)

    print(f"DEBUG: Returning GPU Info: {info}")
    return info

def get_recommended_dependency(gpu_type, gpu_model, architecture, memory_total):
    """
    Returns recommended dependency configuration based on detected hardware.
    """
    # This function is now deprecated as recommendations are handled by gpu_database.json and dockerfile_generator.py
    # Keeping it for now, but it should be removed or refactored later.
    recommendations = {
        "nvidia": {
            "Blackwell": "CUDA 13.0 + cuDNN 9.2 (Recommended)",  # RTX 50 series
            "Ada Lovelace": "CUDA 12.6 + cuDNN 9.0 (Recommended)",  # RTX 40 series
            "Ampere": "CUDA 12.4 + cuDNN 8.9",  # RTX 30 series  
            "Turing": "CUDA 12.1 + cuDNN 8.8",  # RTX 20/GTX 16 series
            "Pascal": "CUDA 11.8 + cuDNN 8.7",  # GTX 10 series
            "Volta": "CUDA 12.1 + cuDNN 8.8",   # Tesla V100
            "Kepler": "CUDA 11.8 + cuDNN 8.7",  # Older Tesla
            "Hopper/Ampere": "CUDA 12.6 + cuDNN 9.0 (Recommended)"  # H100/A100
        },
        "amd": {
            "RDNA 4": "ROCm 7.0.0 (Recommended)",  # RX 9000 series
            "RDNA 3": "ROCm 6.2 (Recommended)",  # RX 7000 series
            "RDNA 2": "ROCm 6.1",  # RX 6000 series
            "RDNA": "ROCm 6.0",    # RX 5000 series
            "GCN 5.0": "ROCm 5.7 (Legacy)"  # Vega series
        },
        "intel": {
            "Xe-HPG": "Intel XPU 2025.1.0 (Recommended)",  # Arc B series
            "DG3": "Intel XPU 2025.1.0 (Recommended)",  # Arc B series
            "DG2": "Intel XPU 2024.2",  # Arc A series
            "Xe-LP": "Intel XPU 2024.1",  # Iris Xe
            "Gen12": "Intel CPU Optimized"
        },
        "cpu": {
            "CPU": "CPU Standard (Recommended)"
        }
    }
    
    return recommendations.get(gpu_type, {}).get(architecture, 
           list(recommendations.get(gpu_type, {"default": "Standard"}).values())[0])

def test_simulation_modes():
    """Test all simulation modes to demonstrate functionality."""
    print("=== Enhanced Hardware Detector - Simulation Test ===")
    
    # Test real hardware detection
    print("\n1. Testing Real Hardware Detection (Auto Mode):")
    gpu_info = get_gpu_info("auto")
    print(f"   Detected: {gpu_info['gpu_type'].upper()} - {gpu_info['gpu_model']} (Key: {gpu_info['gpu_model_key']})")
    
    # Test AMD simulation
    print("\n2. Testing AMD GPU Simulation:")
    gpu_info = get_gpu_info("force_amd", "amd", "rx_7900_xtx")
    print(f"   Simulated: {gpu_info['gpu_type'].upper()} - {gpu_info['gpu_model']} (Key: {gpu_info['gpu_model_key']})")
    print(f"   Architecture: {gpu_info['architecture']}")
    print(f"   Memory: {gpu_info['memory_total']}MB")
    
    # Test Intel simulation
    print("\n3. Testing Intel GPU Simulation:")
    gpu_info = get_gpu_info("force_intel", "intel", "arc_a770")
    print(f"   Simulated: {gpu_info['gpu_type'].upper()} - {gpu_info['gpu_model']} (Key: {gpu_info['gpu_model_key']})")
    print(f"   Architecture: {gpu_info['architecture']}")
    print(f"   Memory: {gpu_info['memory_total']}MB")
    
    # Test different AMD models
    print("\n4. Testing Different AMD Models:")
    amd_models = ["rx_7900_xtx", "rx_6800_xt", "rx_5700_xt"]
    for model in amd_models:
        gpu_info = get_gpu_info("force_amd", "amd", model)
        print(f"   {model}: {gpu_info['gpu_model']} - {gpu_info['architecture']} - {gpu_info['memory_total']}MB (Key: {gpu_info['gpu_model_key']})")
    
    # Test different Intel models
    print("\n5. Testing Different Intel Models:")
    intel_models = ["arc_a770", "arc_a750", "iris_xe"]
    for model in intel_models:
        gpu_info = get_gpu_info("force_intel", "intel", model)
        print(f"   {model}: {gpu_info['gpu_model']} - {gpu_info['architecture']} - {gpu_info['memory_total']}MB (Key: {gpu_info['gpu_model_key']})")

if __name__ == '__main__':
    # Run simulation tests
    test_simulation_modes()
