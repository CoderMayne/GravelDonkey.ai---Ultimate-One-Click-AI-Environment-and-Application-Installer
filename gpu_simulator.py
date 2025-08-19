import subprocess
import re
import shutil
import json
import os
from typing import Dict, Tuple, Optional, List

class GPUSimulator:
    """
    GPU Simulator for testing AMD and Intel GPU detection without actual hardware.
    This allows developers to test the installer with different GPU configurations.
    """
    
    def __init__(self, simulation_mode: str = "auto"):
        """
        Initialize the GPU simulator.
        
        Args:
            simulation_mode: "auto" (detect real hardware first), "force_amd", "force_intel", "force_cpu"
        """
        self.simulation_mode = simulation_mode
        self.simulated_gpus = self._get_simulated_gpu_configs()
        
    def _get_simulated_gpu_configs(self) -> Dict[str, Dict]:
        """Loads simulated GPU configurations from the JSON database."""
        try:
            with open(os.path.join(os.path.dirname(__file__), 'data', 'gpu_database.json'), 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading GPU database: {e}")
            return {}
    
    def simulate_amd_gpu_detection(self, gpu_model: str = "rx_7900_xtx") -> Tuple[str, str, str, int, int]:
        """
        Simulate AMD GPU detection.
        
        Args:
            gpu_model: The simulated GPU model to use
            
        Returns:
            Tuple of (gpu_name, architecture, compute_capability, memory_total, memory_used)
        """
        if gpu_model not in self.simulated_gpus["amd"]:
            gpu_model = "rx_7900_xtx"  # Default to high-end model
            
        gpu_config = self.simulated_gpus["amd"][gpu_model]
        
        print(f"SIMULATION: Detecting AMD GPU - {gpu_config['name']}")
        print(f"SIMULATION: Architecture: {gpu_config['architecture']}")
        print(f"SIMULATION: Memory: {gpu_config['memory_total']}MB total, {gpu_config['memory_used']}MB used")
        
        return (
            gpu_config['name'],
            gpu_config['architecture'],
            gpu_config['compute_capability'],
            gpu_config['memory_total'],
            gpu_config['memory_used']
        )
    
    def simulate_intel_gpu_detection(self, gpu_model: str = "arc_a770") -> Tuple[str, str, str, int, int]:
        """
        Simulate Intel GPU detection.
        
        Args:
            gpu_model: The simulated GPU model to use
            
        Returns:
            Tuple of (gpu_name, architecture, compute_capability, memory_total, memory_used)
        """
        if gpu_model not in self.simulated_gpus["intel"]:
            gpu_model = "arc_a770"  # Default to high-end model
            
        gpu_config = self.simulated_gpus["intel"][gpu_model]
        
        print(f"SIMULATION: Detecting Intel GPU - {gpu_config['name']}")
        print(f"SIMULATION: Architecture: {gpu_config['architecture']}")
        print(f"SIMULATION: Memory: {gpu_config['memory_total']}MB total, {gpu_config['memory_used']}MB used")
        
        return (
            gpu_config['name'],
            gpu_config['architecture'],
            gpu_config['compute_capability'],
            gpu_config['memory_total'],
            gpu_config['memory_used']
        )
    
    def get_simulated_rocm_smi_output(self, gpu_model: str = "rx_7900_xtx") -> str:
        """Generate simulated rocm-smi output for AMD GPUs."""
        gpu_config = self.simulated_gpus["amd"][gpu_model]
        
        return f"""GPU  Name
0    {gpu_config['name']}
"""
    
    def get_simulated_lspci_output(self, gpu_type: str, gpu_model: str) -> str:
        """Generate simulated lspci output for GPU detection."""
        if gpu_type == "amd":
            gpu_config = self.simulated_gpus["amd"][gpu_model]
            return f"01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] {gpu_config['name']} [1002:1234]"
        elif gpu_type == "intel":
            gpu_config = self.simulated_gpus["intel"][gpu_model]
            return f"00:02.0 VGA compatible controller: Intel Corporation {gpu_config['name']} [8086:1234]"
        
        return ""
    
    def list_available_simulated_gpus(self) -> Dict[str, List[str]]:
        """List all available simulated GPU models."""
        return {
            "nvidia": list(self.simulated_gpus["nvidia"].keys()),
            "amd": list(self.simulated_gpus["amd"].keys()),
            "intel": list(self.simulated_gpus["intel"].keys())
        }
    
    def get_ai_recommendations(self, gpu_type: str, gpu_model: str) -> Dict:
        """Get AI workload recommendations for a specific GPU."""
        if gpu_type in self.simulated_gpus and gpu_model in self.simulated_gpus[gpu_type]:
            gpu_config = self.simulated_gpus[gpu_type][gpu_model]
            return gpu_config.get("ai_recommendations", {})
        return {}
    
    def simulate_nvidia_gpu_detection(self, gpu_model: str = "rtx_4090") -> Tuple[str, str, str, int, int]:
        """
        Simulate NVIDIA GPU detection.
        
        Args:
            gpu_model: The simulated GPU model to use
            
        Returns:
            Tuple of (gpu_name, architecture, compute_capability, memory_total, memory_used)
        """
        if gpu_model not in self.simulated_gpus["nvidia"]:
            gpu_model = "rtx_4090"  # Default to high-end model
            
        gpu_config = self.simulated_gpus["nvidia"][gpu_model]
        
        print(f"SIMULATION: Detecting NVIDIA GPU - {gpu_config['name']}")
        print(f"SIMULATION: Architecture: {gpu_config['architecture']}")
        print(f"SIMULATION: Memory: {gpu_config['memory_total']}MB total, {gpu_config['memory_used']}MB used")
        
        return (
            gpu_config['name'],
            gpu_config['architecture'],
            gpu_config['compute_capability'],
            gpu_config['memory_total'],
            gpu_config['memory_used']
        )
    
    def get_simulated_gpu_info(self, gpu_type: str, gpu_model: str) -> Dict:
        """Get complete simulated GPU information."""
        if gpu_type not in self.simulated_gpus or gpu_model not in self.simulated_gpus[gpu_type]:
            return {}
        
        return self.simulated_gpus[gpu_type][gpu_model].copy()

    def get_model_key(self, gpu_model: str, gpu_type: str) -> Optional[str]:
        """
        Find the model key for a given GPU model name and type.
        
        Args:
            gpu_model: The full name of the GPU model.
            gpu_type: The type of the GPU (e.g., "nvidia", "amd", "intel").
            
        Returns:
            The model key if found, otherwise None.
        """
        if gpu_type in self.simulated_gpus:
            for model_key, gpu_config in self.simulated_gpus[gpu_type].items():
                if gpu_config["name"].lower() in gpu_model.lower():
                    return model_key
        return None

def create_simulation_environment(gpu_type: str = "amd", gpu_model: str = "rx_7900_xtx"):
    """
    Create a simulation environment for testing GPU detection.
    
    Args:
        gpu_type: "amd" or "intel"
        gpu_model: The specific GPU model to simulate
    """
    simulator = GPUSimulator()
    
    print(f"=== GPU Simulation Environment ===")
    print(f"Simulating: {gpu_type.upper()} GPU - {gpu_model}")
    print()
    
    if gpu_type == "amd":
        gpu_name, architecture, compute_cap, memory_total, memory_used = simulator.simulate_amd_gpu_detection(gpu_model)
        rocm_output = simulator.get_simulated_rocm_smi_output(gpu_model)
        lspci_output = simulator.get_simulated_lspci_output(gpu_type, gpu_model)
        
        print("Simulated rocm-smi output:")
        print(rocm_output)
        print("Simulated lspci output:")
        print(lspci_output)
        
    elif gpu_type == "intel":
        gpu_name, architecture, compute_cap, memory_total, memory_used = simulator.simulate_intel_gpu_detection(gpu_model)
        lspci_output = simulator.get_simulated_lspci_output(gpu_type, gpu_model)
        
        print("Simulated lspci output:")
        print(lspci_output)
    
    print(f"\nSimulated GPU Info:")
    print(f"  Name: {gpu_name}")
    print(f"  Architecture: {architecture}")
    print(f"  Compute Capability: {compute_cap}")
    print(f"  Memory: {memory_total}MB total, {memory_used}MB used")
    
    return simulator

if __name__ == "__main__":
    # Test the simulator
    print("=== GPU Simulator Test ===")
    
    # Test AMD simulation
    print("\n1. Testing AMD GPU Simulation:")
    amd_sim = create_simulation_environment("amd", "rx_7900_xtx")
    
    # Test Intel simulation
    print("\n2. Testing Intel GPU Simulation:")
    intel_sim = create_simulation_environment("intel", "arc_a770")
    
    # List available models
    print("\n3. Available Simulated GPU Models:")
    for gpu_type, models in amd_sim.list_available_simulated_gpus().items():
        print(f"  {gpu_type.upper()}: {', '.join(models)}")
