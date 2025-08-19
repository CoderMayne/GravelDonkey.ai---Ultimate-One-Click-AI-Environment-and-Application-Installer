"""
Unit tests for the Enhanced Hardware Detector module.
Tests GPU detection with both real hardware and simulation capabilities.
"""

import pytest
import subprocess
import sys
import os
from unittest.mock import Mock, patch, mock_open, MagicMock
from enhanced_hardware_detector import (
    get_gpu_info,
    _is_command_available,
    _run_command,
    _parse_gpu_memory,
    _get_gpu_architecture
)


class TestHardwareDetectionUtilities:
    """Test cases for utility functions."""
    
    def test_is_command_available(self):
        """Test command availability checking."""
        with patch('shutil.which') as mock_which:
            # Test available command
            mock_which.return_value = "/usr/bin/nvidia-smi"
            assert _is_command_available("nvidia-smi") is True
            
            # Test unavailable command
            mock_which.return_value = None
            assert _is_command_available("nonexistent-command") is False
    
    def test_run_command_success(self):
        """Test successful command execution."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Command output"
        mock_result.stderr = ""
        
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            result = _run_command("test-command")
            assert result == "Command output"
            mock_run.assert_called_once()
    
    def test_run_command_failure(self):
        """Test failed command execution."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            result = _run_command("test-command")
            assert result is None
            mock_run.assert_called_once()
    
    def test_run_command_timeout(self):
        """Test command execution timeout."""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("test-command", 15)):
            result = _run_command("test-command")
            assert result is None
    
    def test_run_command_file_not_found(self):
        """Test command execution when command not found."""
        with patch('subprocess.run', side_effect=FileNotFoundError):
            result = _run_command("nonexistent-command")
            assert result is None
    
    def test_parse_gpu_memory_standard_format(self):
        """Test GPU memory parsing with standard format."""
        nvidia_output = "Memory-Usage: 1024MiB / 24576MiB"
        total, used = _parse_gpu_memory(nvidia_output)
        assert total == 24576
        assert used == 1024
    
    def test_parse_gpu_memory_alternative_format(self):
        """Test GPU memory parsing with alternative format."""
        nvidia_output = "1234MiB / 16384MiB"
        total, used = _parse_gpu_memory(nvidia_output)
        assert total == 16384
        assert used == 1234
    
    def test_parse_gpu_memory_invalid_format(self):
        """Test GPU memory parsing with invalid format."""
        nvidia_output = "Invalid memory format"
        total, used = _parse_gpu_memory(nvidia_output)
        assert total is None
        assert used is None
    
    def test_parse_gpu_memory_empty_string(self):
        """Test GPU memory parsing with empty string."""
        total, used = _parse_gpu_memory("")
        assert total is None
        assert used is None
    
    @pytest.mark.parametrize("gpu_name,expected_arch,expected_sm", [
        ("RTX 5090", "Blackwell", "sm_90"),
        ("RTX 4090", "Ada Lovelace", "sm_89"),
        ("RTX 3090", "Ampere", "sm_86"),
        ("RTX 2080", "Turing", "sm_75"),
        ("GTX 1660", "Turing", "sm_75"),
        ("GTX 1080", "Pascal", "sm_61"),
        ("Tesla V100", "Volta", "sm_70"),
        ("Tesla P100", "Pascal", "sm_60"),
        ("Tesla K80", "Kepler", "sm_37"),
        ("H100", "Hopper/Ampere", "sm_90"),
        ("A100", "Hopper/Ampere", "sm_90"),
    ])
    def test_get_gpu_architecture_nvidia(self, gpu_name, expected_arch, expected_sm):
        """Test NVIDIA GPU architecture detection."""
        arch, sm = _get_gpu_architecture(gpu_name)
        assert arch == expected_arch
        assert sm == expected_sm
    
    @pytest.mark.parametrize("gpu_name,expected_arch,expected_sm", [
        ("RX 9060", "RDNA 4", "gfx1200"),
        ("RX 7900 XTX", "RDNA 3", "gfx1100"),
        ("RX 6800 XT", "RDNA 2", "gfx1030"),
        ("RX 5700 XT", "RDNA", "gfx1010"),
        ("Vega 64", "GCN 5.0", "gfx900"),
        ("Radeon VII", "GCN 5.0", "gfx900"),
    ])
    def test_get_gpu_architecture_amd(self, gpu_name, expected_arch, expected_sm):
        """Test AMD GPU architecture detection."""
        arch, sm = _get_gpu_architecture(gpu_name)
        assert arch == expected_arch
        assert sm == expected_sm
    
    @pytest.mark.parametrize("gpu_name,expected_arch,expected_sm", [
        ("Arc B770", "Xe-HPG", "DG3"),
        ("Arc A770", "Xe-HPG", "DG2"),
        ("Arc A380", "Xe-HPG", "DG2"),
        ("Iris Xe Max", "Xe-LP", "Gen12"),
        ("Iris Xe", "Xe-LP", "Gen12"),
    ])
    def test_get_gpu_architecture_intel(self, gpu_name, expected_arch, expected_sm):
        """Test Intel GPU architecture detection."""
        arch, sm = _get_gpu_architecture(gpu_name)
        assert arch == expected_arch
        assert sm == expected_sm
    
    def test_get_gpu_architecture_unknown(self):
        """Test GPU architecture detection with unknown GPU."""
        arch, sm = _get_gpu_architecture("Unknown GPU Model")
        assert arch == "Unknown"
        assert sm == "Unknown"
    
    def test_get_gpu_architecture_case_insensitive(self):
        """Test GPU architecture detection is case insensitive."""
        # Test different case variations
        arch1, sm1 = _get_gpu_architecture("rtx 4090")
        arch2, sm2 = _get_gpu_architecture("RTX 4090")
        arch3, sm3 = _get_gpu_architecture("Rtx 4090")
        
        assert arch1 == arch2 == arch3 == "Ada Lovelace"
        assert sm1 == sm2 == sm3 == "sm_89"


class TestGPUDetection:
    """Test cases for GPU detection functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up common mocks."""
        self.mock_subprocess = patch('subprocess.run')
        self.mock_shutil = patch('shutil.which')
        self.mock_gpu_simulator = patch('enhanced_hardware_detector.GPUSimulator')
        
        self.subprocess_mock = self.mock_subprocess.start()
        self.shutil_mock = self.mock_shutil.start()
        self.gpu_simulator_mock = self.mock_gpu_simulator.start()
        
        yield
        
        self.mock_subprocess.stop()
        self.mock_shutil.stop()
        self.mock_gpu_simulator.stop()
    
    def test_nvidia_gpu_detection(self):
        """Test NVIDIA GPU detection."""
        # Mock nvidia-smi command
        self.shutil_mock.return_value = "/usr/bin/nvidia-smi"
        
        # Mock successful nvidia-smi execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 535.86.10    Driver Version: 535.86.10    CUDA Version: 12.2     |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |                               |                      |               MIG M. |
        |===============================+======================+======================|
        |   0  NVIDIA GeForce RTX 4090  On   | 00000000:01:00.0  On |                  N/A |
        | 30%   45C    P8    45W /  450W|    1024MiB / 24576MiB |      0%      Default |
        |                               |                      |                  N/A |
        +-------------------------------+----------------------+----------------------+
        """
        mock_result.stderr = ""
        self.subprocess_mock.return_value = mock_result
        
        # Mock CUDA version detection
        with patch('enhanced_hardware_detector._get_cuda_version', return_value="12.2"):
            gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
        
        assert gpu_info[0] == "NVIDIA GeForce RTX 4090"
        assert gpu_info[1] == "nvidia"
        assert gpu_info[2] == "Ada Lovelace"
        assert gpu_info[3] == "sm_89"
        assert gpu_info[4] == 24576
        assert gpu_info[5] == 1024
        assert gpu_info[6] == "12.2"
    
    def test_amd_gpu_detection(self):
        """Test AMD GPU detection."""
        # Mock rocm-smi command
        self.shutil_mock.side_effect = lambda cmd: "/usr/bin/rocm-smi" if cmd == "rocm-smi" else None
        
        # Mock successful rocm-smi execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "GPU  Name\n0    AMD Radeon RX 7900 XTX\n"
        mock_result.stderr = ""
        self.subprocess_mock.return_value = mock_result
        
        # Mock lspci for AMD GPU
        with patch('enhanced_hardware_detector._run_command') as mock_run_cmd:
            mock_run_cmd.side_effect = [
                "01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] AMD Radeon RX 7900 XTX [1002:1234]",
                "24576"  # Memory size
            ]
            
            gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
        
        assert gpu_info[0] == "AMD Radeon RX 7900 XTX"
        assert gpu_info[1] == "amd"
        assert gpu_info[2] == "RDNA 3"
        assert gpu_info[3] == "gfx1100"
        assert gpu_info[4] == 24576
    
    def test_intel_gpu_detection(self):
        """Test Intel GPU detection."""
        # Mock lspci command
        self.shutil_mock.return_value = "/usr/bin/lspci"
        
        # Mock successful lspci execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "01:00.0 VGA compatible controller: Intel Corporation Intel Arc A770 [8086:1234]"
        mock_result.stderr = ""
        self.subprocess_mock.return_value = mock_result
        
        # Mock memory detection
        with patch('enhanced_hardware_detector._run_command') as mock_run_cmd:
            mock_run_cmd.side_effect = ["16384"]  # Memory size
            
            gpu_info = get_gpu_info("auto", "intel", "arc_a770")
        
        assert gpu_info[0] == "Intel Arc A770"
        assert gpu_info[1] == "intel"
        assert gpu_info[2] == "Xe-HPG"
        assert gpu_info[3] == "DG2"
        assert gpu_info[4] == 16384
    
    def test_cpu_fallback_detection(self):
        """Test CPU fallback when no GPU is detected."""
        # Mock no GPU commands available
        self.shutil_mock.return_value = None
        
        gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
        
        assert gpu_info[0] == "CPU Only"
        assert gpu_info[1] == "cpu"
        assert gpu_info[2] == "x86_64"
        assert gpu_info[3] == "Unknown"
        assert gpu_info[4] == 0
        assert gpu_info[5] == 0
    
    def test_simulation_mode_amd(self):
        """Test AMD GPU simulation mode."""
        # Mock GPU simulator
        mock_simulator = Mock()
        mock_simulator.simulate_amd_gpu_detection.return_value = (
            "AMD Radeon RX 7900 XTX", "RDNA 3", "gfx1100", 24576, 1024
        )
        self.gpu_simulator_mock.return_value = mock_simulator
        
        gpu_info = get_gpu_info("force_amd", "amd", "rx_7900_xtx")
        
        assert gpu_info[0] == "AMD Radeon RX 7900 XTX"
        assert gpu_info[1] == "amd"
        assert gpu_info[2] == "RDNA 3"
        assert gpu_info[3] == "gfx1100"
        assert gpu_info[4] == 24576
        assert gpu_info[5] == 1024
        
        # Verify simulator was called
        mock_simulator.simulate_amd_gpu_detection.assert_called_once_with("rx_7900_xtx")
    
    def test_simulation_mode_intel(self):
        """Test Intel GPU simulation mode."""
        # Mock GPU simulator
        mock_simulator = Mock()
        mock_simulator.simulate_intel_gpu_detection.return_value = (
            "Intel Arc A770", "Xe-HPG", "DG2", 16384, 256
        )
        self.gpu_simulator_mock.return_value = mock_simulator
        
        gpu_info = get_gpu_info("force_intel", "intel", "arc_a770")
        
        assert gpu_info[0] == "Intel Arc A770"
        assert gpu_info[1] == "intel"
        assert gpu_info[2] == "Xe-HPG"
        assert gpu_info[3] == "DG2"
        assert gpu_info[4] == 16384
        assert gpu_info[5] == 256
        
        # Verify simulator was called
        mock_simulator.simulate_intel_gpu_detection.assert_called_once_with("arc_a770")
    
    def test_simulation_mode_cpu(self):
        """Test CPU simulation mode."""
        gpu_info = get_gpu_info("force_cpu", "amd", "rx_7900_xtx")
        
        assert gpu_info[0] == "CPU Only"
        assert gpu_info[1] == "cpu"
        assert gpu_info[2] == "x86_64"
        assert gpu_info[3] == "Unknown"
        assert gpu_info[4] == 0
        assert gpu_info[5] == 0
    
    def test_command_execution_errors(self):
        """Test handling of command execution errors."""
        # Mock command available but execution fails
        self.shutil_mock.return_value = "/usr/bin/nvidia-smi"
        self.subprocess_mock.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
        
        gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
        
        # Should fall back to CPU mode
        assert gpu_info[0] == "CPU Only"
        assert gpu_info[1] == "cpu"
    
    def test_memory_parsing_edge_cases(self):
        """Test memory parsing with edge cases."""
        # Test with various memory formats
        test_cases = [
            ("1024MiB / 24576MiB", 24576, 1024),
            ("0MiB / 16384MiB", 16384, 0),
            ("16384MiB / 16384MiB", 16384, 16384),
            ("512 MiB / 8192 MiB", 8192, 512),  # With spaces
        ]
        
        for input_str, expected_total, expected_used in test_cases:
            total, used = _parse_gpu_memory(input_str)
            assert total == expected_total
            assert used == expected_used
    
    def test_gpu_architecture_edge_cases(self):
        """Test GPU architecture detection with edge cases."""
        # Test with various naming conventions
        test_cases = [
            ("GeForce RTX 4090", "Ada Lovelace", "sm_89"),
            ("RTX 4090", "Ada Lovelace", "sm_89"),
            ("rtx 4090", "Ada Lovelace", "sm_89"),  # Lowercase
            ("Rtx 4090", "Ada Lovelace", "sm_89"),  # Mixed case
            ("  RTX 4090  ", "Ada Lovelace", "sm_89"),  # With spaces
        ]
        
        for gpu_name, expected_arch, expected_sm in test_cases:
            arch, sm = _get_gpu_architecture(gpu_name)
            assert arch == expected_arch
            assert sm == expected_sm


class TestHardwareDetectionIntegration:
    """Integration tests for hardware detection."""
    
    def test_end_to_end_nvidia_detection(self):
        """Test complete NVIDIA GPU detection flow."""
        with patch('shutil.which', return_value="/usr/bin/nvidia-smi"), \
             patch('subprocess.run') as mock_run, \
             patch('enhanced_hardware_detector._get_cuda_version', return_value="12.2"):
            
            # Mock nvidia-smi output
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Memory-Usage: 1024MiB / 24576MiB"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
            
            assert len(gpu_info) == 7  # All fields should be present
            assert gpu_info[1] == "nvidia"
            assert gpu_info[6] == "12.2"  # CUDA version
    
    def test_end_to_end_amd_detection(self):
        """Test complete AMD GPU detection flow."""
        with patch('shutil.which', side_effect=lambda cmd: "/usr/bin/rocm-smi" if cmd == "rocm-smi" else None), \
             patch('subprocess.run') as mock_run, \
             patch('enhanced_hardware_detector._run_command') as mock_run_cmd:
            
            # Mock rocm-smi output
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "GPU  Name\n0    AMD Radeon RX 7900 XTX\n"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            # Mock lspci output
            mock_run_cmd.side_effect = [
                "01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] AMD Radeon RX 7900 XTX [1002:1234]",
                "24576"
            ]
            
            gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
            
            assert len(gpu_info) == 7
            assert gpu_info[1] == "amd"
            assert gpu_info[2] == "RDNA 3"
    
    def test_simulation_integration(self):
        """Test integration between detection and simulation."""
        with patch('enhanced_hardware_detector.GPUSimulator') as mock_simulator_class:
            mock_simulator = Mock()
            mock_simulator.simulate_amd_gpu_detection.return_value = (
                "AMD Radeon RX 7900 XTX", "RDNA 3", "gfx1100", 24576, 1024
            )
            mock_simulator_class.return_value = mock_simulator
            
            gpu_info = get_gpu_info("force_amd", "amd", "rx_7900_xtx")
            
            # Verify simulation was used
            mock_simulator_class.assert_called_once_with("force_amd")
            mock_simulator.simulate_amd_gpu_detection.assert_called_once_with("rx_7900_xtx")
            
            # Verify results
            assert gpu_info[0] == "AMD Radeon RX 7900 XTX"
            assert gpu_info[1] == "amd"
    
    def test_fallback_chain(self):
        """Test the complete fallback chain when no GPU is detected."""
        with patch('shutil.which', return_value=None), \
             patch('enhanced_hardware_detector._run_command', return_value=None):
            
            gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
            
            # Should fall back to CPU mode
            assert gpu_info[0] == "CPU Only"
            assert gpu_info[1] == "cpu"
            assert gpu_info[2] == "x86_64"
            assert gpu_info[3] == "Unknown"
            assert gpu_info[4] == 0
            assert gpu_info[5] == 0
            assert gpu_info[6] == "Unknown"
    
    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        with patch('shutil.which', return_value="/usr/bin/nvidia-smi"), \
             patch('subprocess.run', side_effect=Exception("Unexpected error")):
            
            gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
            
            # Should handle unexpected errors gracefully
            assert gpu_info[0] == "CPU Only"
            assert gpu_info[1] == "cpu"
    
    def test_performance_under_load(self):
        """Test performance when making multiple detection calls."""
        with patch('shutil.which', return_value=None), \
             patch('enhanced_hardware_detector._run_command', return_value=None):
            
            import time
            start_time = time.time()
            
            # Make multiple detection calls
            for _ in range(10):
                gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
                assert gpu_info[1] == "cpu"  # Should always fall back to CPU
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete 10 calls in reasonable time
            assert execution_time < 1.0
            assert execution_time > 0

