# GPU Simulation for AMD and Intel GPUs

This directory contains GPU simulation tools that allow you to test AMD and Intel GPU detection without requiring actual hardware. This is particularly useful for development, testing, and demonstration purposes.

## 🎯 Overview

The GPU simulation system consists of several components:

- **`gpu_simulator.py`** - Core simulation engine with predefined GPU configurations
- **`enhanced_hardware_detector.py`** - Enhanced hardware detection with simulation support
- **`test_gpu_simulation.py`** - Comprehensive test suite for all simulation features
- **`demo_simulation.py`** - Interactive Gradio demo for testing GPU simulation

## 🚀 Quick Start

### 1. Test Basic Simulation

```bash
# Test the GPU simulator directly
python3 gpu_simulator.py

# Run comprehensive tests
python3 test_gpu_simulation.py
```

### 2. Launch Interactive Demo

```bash
# Start the Gradio demo interface
python3 demo_simulation.py
```

The demo will be available at `http://localhost:7860`

### 3. Use in Your Code

```python
from enhanced_hardware_detector import get_gpu_info

# Simulate AMD RX 7900 XTX
gpu_info = get_gpu_info("force_amd", "amd", "rx_7900_xtx")
print(f"GPU: {gpu_info[1]}, Architecture: {gpu_info[6]}")

# Simulate Intel Arc A770
gpu_info = get_gpu_info("force_intel", "intel", "arc_a770")
print(f"GPU: {gpu_info[1]}, Architecture: {gpu_info[6]}")

# Detect real hardware
gpu_info = get_gpu_info("auto")
print(f"Real GPU: {gpu_info[1]}")
```

## 📋 Supported GPU Models

### AMD Radeon Series

| Model | Architecture | Memory | Compute Capability |
|-------|--------------|--------|-------------------|
| RX 7900 XTX | RDNA 3 | 24GB | gfx1100 |
| RX 7800 XT | RDNA 3 | 16GB | gfx1100 |
| RX 6800 XT | RDNA 2 | 16GB | gfx1030 |
| RX 6700 XT | RDNA 2 | 12GB | gfx1030 |
| RX 5700 XT | RDNA | 8GB | gfx1010 |

### Intel Arc Series

| Model | Architecture | Memory | Compute Capability |
|-------|--------------|--------|-------------------|
| Arc A770 | Xe-HPG | 16GB | DG2 |
| Arc A750 | Xe-HPG | 8GB | DG2 |
| Arc A580 | Xe-HPG | 8GB | DG2 |
| Arc A380 | Xe-HPG | 6GB | DG2 |
| Iris Xe | Xe-LP | 4GB | Gen12 |

## 🔧 API Reference

### GPUSimulator Class

```python
from gpu_simulator import GPUSimulator

# Initialize simulator
simulator = GPUSimulator()

# Simulate AMD GPU detection
gpu_name, architecture, compute_cap, memory_total, memory_used = simulator.simulate_amd_gpu_detection("rx_7900_xtx")

# Simulate Intel GPU detection
gpu_name, architecture, compute_cap, memory_total, memory_used = simulator.simulate_intel_gpu_detection("arc_a770")

# List available models
available_gpus = simulator.list_available_simulated_gpus()
```

### Enhanced Hardware Detection

```python
from enhanced_hardware_detector import get_gpu_info, get_recommended_dependency

# Simulation modes:
# - "auto": Detect real hardware (default)
# - "force_amd": Force AMD simulation
# - "force_intel": Force Intel simulation
# - "force_cpu": Force CPU fallback

# Get GPU information with simulation
gpu_info = get_gpu_info(
    simulation_mode="force_amd",  # or "force_intel", "auto"
    simulated_gpu_type="amd",     # "amd" or "intel"
    simulated_gpu_model="rx_7900_xtx"  # specific model
)

# Get recommended dependencies
recommended = get_recommended_dependency(
    gpu_info[0],  # gpu_type
    gpu_info[1],  # gpu_model
    gpu_info[6],  # architecture
    gpu_info[4]   # memory_total
)
```

## 🎮 Interactive Demo Features

The Gradio demo (`demo_simulation.py`) provides:

- **GPU Type Selection**: Choose between Real Hardware, AMD, or Intel
- **Model Selection**: Dynamic dropdown with available models for each GPU type
- **Real-time Detection**: Click to simulate GPU detection
- **Detailed Results**: Shows architecture, memory, compute capability, and recommendations
- **Simulation Notice**: Clear indication when using simulated data

## 🔍 Integration with Installer

To integrate GPU simulation into your existing installer:

1. **Replace hardware detection calls**:
   ```python
   # Instead of:
   from hardware_detector import get_gpu_info
   
   # Use:
   from enhanced_hardware_detector import get_gpu_info
   ```

2. **Add simulation mode parameter**:
   ```python
   # For testing with AMD simulation
   gpu_info = get_gpu_info("force_amd", "amd", "rx_7900_xtx")
   
   # For testing with Intel simulation
   gpu_info = get_gpu_info("force_intel", "intel", "arc_a770")
   
   # For real hardware detection
   gpu_info = get_gpu_info("auto")
   ```

3. **Add simulation controls to UI** (optional):
   ```python
   # Add to your Gradio interface
   simulation_mode = gr.Dropdown(
       choices=["Real Hardware", "AMD Simulation", "Intel Simulation"],
       value="Real Hardware",
       label="Detection Mode"
   )
   ```

## 🧪 Testing

### Run All Tests

```bash
# Comprehensive test suite
python3 test_gpu_simulation.py
```

### Individual Component Tests

```bash
# Test GPU simulator
python3 gpu_simulator.py

# Test enhanced hardware detector
python3 enhanced_hardware_detector.py
```

### Command Line Testing

```bash
# Test specific GPU simulation
python3 -c "
from enhanced_hardware_detector import get_gpu_info
print('AMD RX 7900 XTX:', get_gpu_info('force_amd', 'amd', 'rx_7900_xtx'))
print('Intel Arc A770:', get_gpu_info('force_intel', 'intel', 'arc_a770'))
print('Real Hardware:', get_gpu_info('auto'))
"
```

## 📊 Example Output

### AMD Simulation
```
SIMULATION: Detecting AMD GPU - AMD Radeon RX 7900 XTX
SIMULATION: Architecture: RDNA 3
SIMULATION: Memory: 24576MB total, 1024MB used

GPU Type: AMD
GPU Model: AMD Radeon RX 7900 XTX
Architecture: RDNA 3
Compute Capability: gfx1100
Memory: 24576MB
Recommended Configuration: ROCm 6.2 (Recommended)
```

### Intel Simulation
```
SIMULATION: Detecting Intel GPU - Intel Arc A770
SIMULATION: Architecture: Xe-HPG
SIMULATION: Memory: 16384MB total, 1024MB used

GPU Type: INTEL
GPU Model: Intel Arc A770
Architecture: Xe-HPG
Compute Capability: DG2
Memory: 16384MB
Recommended Configuration: Intel XPU 2024.2 (Recommended)
```

## 🔧 Customization

### Adding New GPU Models

To add new GPU models to the simulation:

1. Edit `gpu_simulator.py`
2. Add new configurations to `_get_simulated_gpu_configs()`
3. Update architecture detection in `_get_gpu_architecture()`

Example:

```python
"rx_7600": {
    "name": "AMD Radeon RX 7600",
    "architecture": "RDNA 3",
    "compute_capability": "gfx1100",
    "memory_total": 8192,  # 8GB
    "memory_used": 1024,
    "driver_version": "23.40.2.01",
    "rocm_version": "6.2.0"
}
```

### Custom Simulation Modes

You can create custom simulation modes by extending the `GPUSimulator` class:

```python
class CustomGPUSimulator(GPUSimulator):
    def simulate_custom_gpu(self, custom_config):
        # Custom simulation logic
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all files are in the same directory
2. **Gradio Not Found**: Install with `pip install gradio`
3. **Simulation Not Working**: Check that simulation mode is set correctly

### Debug Mode

Enable debug output by setting environment variable:

```bash
export DEBUG_GPU_SIMULATION=1
python3 test_gpu_simulation.py
```

## 📝 License

This GPU simulation system is part of the AI Installer project and follows the same licensing terms.

## 🤝 Contributing

To contribute to the GPU simulation system:

1. Add new GPU models to the simulator
2. Improve detection accuracy
3. Add new simulation features
4. Update documentation
5. Add more test cases

## 📞 Support

For issues or questions about GPU simulation:

1. Check the troubleshooting section
2. Run the test suite to verify functionality
3. Review the example outputs
4. Check the interactive demo for visual verification
