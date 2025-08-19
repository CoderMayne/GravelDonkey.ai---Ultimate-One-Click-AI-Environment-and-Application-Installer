"""
Pytest configuration and fixtures for the AI Development Environment Installer tests.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add the parent directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="ai_installer_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def mock_gpu_database():
    """Provide mock GPU database for testing."""
    return {
        "amd": {
            "rx_7900_xtx": {
                "name": "AMD Radeon RX 7900 XTX",
                "architecture": "RDNA 3",
                "compute_capability": "gfx1100",
                "memory_total": 24576,
                "memory_used": 1024
            },
            "rx_6800_xt": {
                "name": "AMD Radeon RX 6800 XT",
                "architecture": "RDNA 2",
                "compute_capability": "gfx1030",
                "memory_total": 16384,
                "memory_used": 512
            }
        },
        "intel": {
            "arc_a770": {
                "name": "Intel Arc A770",
                "architecture": "Xe-HPG",
                "compute_capability": "DG2",
                "memory_total": 16384,
                "memory_used": 256
            },
            "arc_a750": {
                "name": "Intel Arc A750",
                "architecture": "Xe-HPG",
                "compute_capability": "DG2",
                "memory_total": 8192,
                "memory_used": 128
            }
        }
    }

@pytest.fixture(scope="function")
def mock_subprocess():
    """Mock subprocess calls for testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        yield mock_run

@pytest.fixture(scope="function")
def mock_requests():
    """Mock requests for testing HTTP calls."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {
                "1.0.0": [],
                "2.0.0": [],
                "3.0.0": []
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get

@pytest.fixture(scope="function")
def mock_file_system():
    """Mock file system operations."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('builtins.open', create=True) as mock_open:
        
        mock_exists.return_value = True
        mock_mkdir.return_value = None
        mock_open.return_value.__enter__.return_value = Mock()
        mock_open.return_value.__exit__.return_value = None
        
        yield {
            'exists': mock_exists,
            'mkdir': mock_mkdir,
            'open': mock_open
        }

@pytest.fixture(scope="function")
def mock_gradio():
    """Mock Gradio components for testing."""
    with patch('gradio.Interface') as mock_interface, \
         patch('gradio.Radio') as mock_radio, \
         patch('gradio.Dropdown') as mock_dropdown, \
         patch('gradio.Button') as mock_button, \
         patch('gradio.Markdown') as mock_markdown, \
         patch('gradio.Audio') as mock_audio:
        
        # Mock Gradio components
        mock_interface.return_value = Mock()
        mock_radio.return_value = Mock()
        mock_dropdown.return_value = Mock()
        mock_button.return_value = Mock()
        mock_markdown.return_value = Mock()
        mock_audio.return_value = Mock()
        
        yield {
            'Interface': mock_interface,
            'Radio': mock_radio,
            'Dropdown': mock_dropdown,
            'Button': mock_button,
            'Markdown': mock_markdown,
            'Audio': mock_audio
        }

@pytest.fixture(scope="function")
def mock_platform():
    """Mock platform detection for testing."""
    with patch('platform.system') as mock_system, \
         patch('sys.platform') as mock_platform_attr:
        
        mock_system.return_value = "Windows"
        mock_platform_attr = "win32"
        
        yield {
            'system': mock_system,
            'platform': mock_platform_attr
        }

@pytest.fixture(scope="function")
def sample_applications():
    """Provide sample applications for testing."""
    return [
        {
            "name": "Test App 1",
            "description": "A test application",
            "category": "AI/ML",
            "folder_name": "test-app-1"
        },
        {
            "name": "Test App 2",
            "description": "Another test application",
            "category": "Data Science",
            "folder_name": "test-app-2"
        }
    ]

@pytest.fixture(scope="function")
def sample_dependencies():
    """Provide sample dependencies for testing."""
    return {
        "nvidia": {
            "name": "NVIDIA CUDA",
            "docker_base_image": "nvidia/cuda:12.1-devel-ubuntu22.04",
            "torch_options": ["torch==2.1.0", "torch==2.0.0"],
            "xformers_options": ["xformers==0.0.22", "xformers==0.0.21"]
        },
        "amd": {
            "name": "AMD ROCm",
            "docker_base_image": "rocm/dev-ubuntu22.04",
            "torch_options": ["torch==2.1.0+rocm5.6", "torch==2.0.0+rocm5.6"],
            "xformers_options": ["xformers==0.0.22", "xformers==0.0.21"]
        }
    }

@pytest.fixture(scope="function")
def temp_workspace():
    """Provide a temporary workspace directory."""
    temp_dir = tempfile.mkdtemp(prefix="ai_installer_workspace_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def mock_environment():
    """Mock environment variables and system state."""
    with patch.dict(os.environ, {
        'PATH': '/usr/bin:/usr/local/bin',
        'CUDA_HOME': '/usr/local/cuda',
        'ROCM_HOME': '/opt/rocm'
    }):
        yield

@pytest.fixture(scope="function")
def mock_gpu_detection():
    """Mock GPU detection results."""
    return {
        'nvidia': {
            'gpu_name': 'NVIDIA GeForce RTX 4090',
            'architecture': 'Ada Lovelace',
            'compute_capability': 'sm_89',
            'memory_total': 24576,
            'memory_used': 1024,
            'cuda_version': '12.1'
        },
        'amd': {
            'gpu_name': 'AMD Radeon RX 7900 XTX',
            'architecture': 'RDNA 3',
            'compute_capability': 'gfx1100',
            'memory_total': 24576,
            'memory_used': 1024,
            'rocm_version': '5.6'
        },
        'intel': {
            'gpu_name': 'Intel Arc A770',
            'architecture': 'Xe-HPG',
            'compute_capability': 'DG2',
            'memory_total': 16384,
            'memory_used': 256
        }
    }

