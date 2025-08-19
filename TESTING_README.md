# Testing Documentation for AI Development Environment Installer

## Overview

This document provides comprehensive information about the testing framework implemented for the AI Development Environment Installer. The testing suite includes unit tests, integration tests, and comprehensive test utilities to ensure code quality and reliability.

## 🏗️ Testing Architecture

### Test Structure
```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── test_gpu_simulator.py       # GPU Simulator unit tests
├── test_enhanced_hardware_detector.py  # Hardware detection tests
├── test_dependency_utils.py    # Dependency management tests
├── test_prereq_checker.py      # Prerequisite checker tests
└── test_integration_comprehensive.py   # Integration tests
```

### Test Categories

#### 1. Unit Tests
- **GPU Simulator Tests**: Test individual GPU simulation functionality
- **Hardware Detection Tests**: Test GPU detection and architecture parsing
- **Dependency Utils Tests**: Test package version fetching and parsing
- **Prerequisite Checker Tests**: Test WSL, Docker, and Python detection

#### 2. Integration Tests
- **System Integration**: Test complete workflows and component interactions
- **Workflow Integration**: Test end-to-end installation processes
- **Data Consistency**: Test data flow between components

#### 3. Performance Tests
- **Load Testing**: Test system performance under multiple operations
- **Memory Management**: Test memory usage and leak detection
- **Response Time**: Test execution time for various operations

## 🚀 Quick Start

### 1. Install Testing Dependencies

```bash
pip install -r requirements-test.txt
```

### 2. Run All Tests

```bash
python run_tests.py --all
```

### 3. Run Specific Test Categories

```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only
python run_tests.py --integration

# Performance tests
python run_tests.py --performance

# Smoke tests
python run_tests.py --smoke
```

### 4. Generate Coverage Reports

```bash
python run_tests.py --coverage
```

## 🧪 Test Runner Options

### Command Line Arguments

| Option | Description | Example |
|--------|-------------|---------|
| `--all` | Run complete test suite | `python run_tests.py --all` |
| `--unit` | Run unit tests only | `python run_tests.py --unit` |
| `--integration` | Run integration tests only | `python run_tests.py --integration` |
| `--performance` | Run performance tests | `python run_tests.py --performance` |
| `--smoke` | Run smoke tests | `python run_tests.py --smoke` |
| `--coverage` | Generate coverage report | `python run_tests.py --coverage` |
| `--file` | Run specific test file | `python run_tests.py --file tests/test_gpu_simulator.py` |
| `--list` | List available tests | `python run_tests.py --list` |
| `--verbose` | Enable verbose output | `python run_tests.py --unit --verbose` |
| `--no-html` | Disable HTML reports | `python run_tests.py --all --no-html` |
| `--no-coverage` | Disable coverage reporting | `python run_tests.py --all --no-coverage` |

### Examples

```bash
# Run all tests with verbose output and coverage
python run_tests.py --all --verbose

# Run only unit tests with HTML reports
python run_tests.py --unit --verbose

# Run specific test file
python run_tests.py --file tests/test_gpu_simulator.py

# Generate comprehensive coverage report
python run_tests.py --coverage

# List all available tests
python run_tests.py --list
```

## 📊 Test Coverage

### Coverage Metrics

The testing framework provides comprehensive coverage reporting:

- **Line Coverage**: Percentage of code lines executed
- **Branch Coverage**: Percentage of code branches executed
- **Function Coverage**: Percentage of functions called
- **HTML Reports**: Interactive coverage reports in `test_reports/coverage_html/`
- **XML Reports**: Machine-readable coverage data in `test_reports/coverage.xml`

### Coverage Targets

- **Minimum Coverage**: 80% (configured in `pytest.ini`)
- **Target Coverage**: 90%+
- **Critical Paths**: 100% coverage required

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=.
    --cov-report=html
    --cov-report=term-missing
    --cov-report=xml
    --cov-fail-under=80
    --html=test_reports/report.html
    --self-contained-html
    --benchmark-only
    --benchmark-skip
    --randomly-seed=42
    --randomly-dont-reset-seed
```

### Test Markers

| Marker | Description | Usage |
|--------|-------------|-------|
| `@pytest.mark.unit` | Unit tests | `@pytest.mark.unit` |
| `@pytest.mark.integration` | Integration tests | `@pytest.mark.integration` |
| `@pytest.mark.slow` | Slow running tests | `@pytest.mark.slow` |
| `@pytest.mark.gpu` | GPU-related tests | `@pytest.mark.gpu` |
| `@pytest.mark.simulation` | GPU simulation tests | `@pytest.mark.simulation` |
| `@pytest.mark.hardware` | Hardware detection tests | `@pytest.mark.hardware` |
| `@pytest.mark.ui` | UI/Gradio tests | `@pytest.mark.ui` |
| `@pytest.mark.docker` | Docker-related tests | `@pytest.mark.docker` |
| `@pytest.mark.prereq` | Prerequisite checker tests | `@pytest.mark.prereq` |
| `@pytest.mark.dependency` | Dependency management tests | `@pytest.mark.dependency` |
| `@pytest.mark.benchmark` | Performance benchmark tests | `@pytest.mark.benchmark` |
| `@pytest.mark.smoke` | Smoke tests for critical functionality | `@pytest.mark.smoke` |

## 🎯 Test Categories in Detail

### 1. GPU Simulator Tests (`test_gpu_simulator.py`)

Tests the GPU simulation capabilities for AMD and Intel GPUs.

**Key Test Areas:**
- GPU simulator initialization
- AMD GPU simulation
- Intel GPU simulation
- Error handling and edge cases
- Performance and memory management

**Example Test:**
```python
def test_simulate_amd_gpu_detection(self):
    """Test AMD GPU simulation."""
    gpu_name, architecture, compute_capability, memory_total, memory_used = \
        self.simulator.simulate_amd_gpu_detection("rx_7900_xtx")
    
    assert gpu_name == "AMD Radeon RX 7900 XTX"
    assert architecture == "RDNA 3"
    assert compute_capability == "gfx1100"
    assert memory_total == 24576
    assert memory_used == 1024
```

### 2. Hardware Detection Tests (`test_enhanced_hardware_detector.py`)

Tests GPU detection with both real hardware and simulation capabilities.

**Key Test Areas:**
- NVIDIA GPU detection
- AMD GPU detection
- Intel GPU detection
- CPU fallback detection
- Simulation mode integration
- Error handling and recovery

**Example Test:**
```python
def test_nvidia_gpu_detection(self):
    """Test NVIDIA GPU detection."""
    with patch('shutil.which', return_value="/usr/bin/nvidia-smi"):
        gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
        
        assert gpu_info[0] == "NVIDIA GeForce RTX 4090"
        assert gpu_info[1] == "nvidia"
        assert gpu_info[2] == "Ada Lovelace"
        assert gpu_info[3] == "sm_89"
```

### 3. Dependency Utils Tests (`test_dependency_utils.py`)

Tests package version fetching and version parsing functionality.

**Key Test Areas:**
- PyPI API integration
- Version parsing and validation
- Error handling for network issues
- Performance under load
- Memory management

**Example Test:**
```python
def test_fetch_package_versions_success(self, mock_requests):
    """Test successful package version fetching."""
    versions = fetch_package_versions("test-package")
    
    assert versions == ["3.0.0", "2.1.0", "2.0.0", "1.0.0"]
    mock_requests.assert_called_once_with(
        "https://pypi.org/pypi/test-package/json",
        timeout=5
    )
```

### 4. Prerequisite Checker Tests (`test_prereq_checker.py`)

Tests WSL, Docker, Python, and GPU driver detection functionality.

**Key Test Areas:**
- WSL environment detection
- Docker installation verification
- Python availability checking
- GPU driver detection
- Cross-platform compatibility

**Example Test:**
```python
def test_check_wsl_windows_wsl2_detected(self):
    """Test WSL detection on Windows with WSL 2."""
    with patch('sys.platform', 'win32'):
        success, message = check_wsl()
        
        assert success is True
        assert "WSL 2: Detected and running" in message
```

### 5. Integration Tests (`test_integration_comprehensive.py`)

Tests the complete system integration and workflow.

**Key Test Areas:**
- End-to-end workflows
- Component interactions
- Data flow between modules
- Error recovery and fallbacks
- Performance under load

**Example Test:**
```python
def test_complete_installation_workflow(self):
    """Test the complete installation workflow from start to finish."""
    # Step 1: Prerequisite checking
    wsl_check = check_wsl()
    docker_check = check_docker()
    python_check = check_python()
    
    # All checks should pass
    assert wsl_check[0] is True
    assert docker_check[0] is True
    assert python_check[0] is True
    
    # Step 2: Hardware detection
    gpu_info = get_gpu_info("auto", "amd", "rx_7900_xtx")
    assert gpu_info[1] == "nvidia"
```

## 🧩 Test Fixtures and Utilities

### Common Fixtures (`conftest.py`)

The testing framework provides comprehensive fixtures for common testing scenarios:

#### Mock GPU Database
```python
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
            }
        }
    }
```

#### Mock Subprocess
```python
@pytest.fixture(scope="function")
def mock_subprocess():
    """Mock subprocess calls for testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        yield mock_run
```

#### Mock Requests
```python
@pytest.fixture(scope="function")
def mock_requests():
    """Mock requests for testing HTTP calls."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {"releases": {"1.0.0": []}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get
```

## 📈 Performance Testing

### Benchmark Tests

The testing framework includes performance benchmarks to ensure the system meets performance requirements:

```python
def test_simulator_performance(self):
    """Test simulator performance with multiple calls."""
    import time
    start_time = time.time()
    
    for _ in range(100):
        simulator.simulate_amd_gpu_detection("rx_7900_xtx")
        simulator.simulate_intel_gpu_detection("arc_a770")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Should complete 200 calls in reasonable time
    assert execution_time < 1.0
    assert execution_time > 0
```

### Memory Management Tests

```python
def test_simulator_memory_usage(self):
    """Test that simulator doesn't leak memory."""
    import gc
    
    gc.collect()
    initial_objects = len(gc.get_objects())
    
    # Create and destroy multiple simulators
    for _ in range(10):
        simulator = GPUSimulator()
        del simulator
    
    gc.collect()
    final_objects = len(gc.get_objects())
    
    # Should not have created too many persistent objects
    assert final_objects - initial_objects < 100
```

## 🚨 Error Handling and Edge Cases

### Comprehensive Error Testing

The testing framework covers various error scenarios:

#### Network Failures
```python
def test_fetch_package_versions_connection_error(self, mock_requests):
    """Test package version fetching with connection error."""
    mock_requests.side_effect = requests.exceptions.ConnectionError("Connection failed")
    
    versions = fetch_package_versions("test-package")
    assert versions == []
```

#### Command Execution Failures
```python
def test_run_command_failure(self):
    """Test failed command execution."""
    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error message"
    
    with patch('subprocess.run', return_value=mock_result):
        result = _run_command("test-command")
        assert result is None
```

#### Invalid Data Handling
```python
def test_initialization_with_invalid_json(self):
    """Test simulator initialization with invalid JSON data."""
    with patch('builtins.open', mock_open(read_data="invalid json")):
        simulator = GPUSimulator()
        assert simulator.simulated_gpus == {}
```

## 🔍 Debugging and Troubleshooting

### Common Test Issues

#### 1. Import Errors
```bash
# Ensure all files are in the same directory
ls -la *.py

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 2. Mock Configuration Issues
```python
# Ensure mocks are properly configured
with patch('module.function') as mock_func:
    mock_func.return_value = expected_value
    # Test code here
    mock_func.assert_called_once()
```

#### 3. Test Isolation Problems
```python
# Use fixtures with proper scope
@pytest.fixture(scope="function")  # Fresh for each test
def clean_environment():
    # Setup
    yield
    # Cleanup
```

### Debug Mode

Enable debug output for troubleshooting:

```bash
# Run tests with maximum verbosity
python run_tests.py --all --verbose

# Run specific test with debug output
python -m pytest tests/test_gpu_simulator.py -v -s
```

## 📊 Test Reporting

### HTML Reports

The testing framework generates comprehensive HTML reports:

- **Test Results**: Detailed test execution results
- **Coverage Reports**: Interactive coverage visualization
- **Performance Metrics**: Execution time and resource usage
- **Error Details**: Stack traces and failure information

### Coverage Reports

```bash
# Generate coverage report
python run_tests.py --coverage

# View HTML coverage report
open test_reports/coverage_html/index.html
```

### Test Summary

```bash
# View test summary
cat test_reports/test_summary.json
```

## 🚀 Continuous Integration

### GitHub Actions Integration

The testing framework is designed to integrate with CI/CD pipelines:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
      - name: Run tests
        run: |
          python run_tests.py --all --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Pre-commit Hooks

Configure pre-commit hooks to run tests automatically:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python run_tests.py --unit
        language: system
        pass_filenames: false
        always_run: true
```

## 📚 Best Practices

### 1. Test Organization
- Group related tests in classes
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

### 2. Mocking Strategy
- Mock external dependencies
- Use appropriate mock scopes
- Verify mock interactions

### 3. Test Data Management
- Use fixtures for common test data
- Clean up test resources
- Avoid test interdependencies

### 4. Error Testing
- Test both success and failure paths
- Verify error messages and codes
- Test edge cases and boundary conditions

### 5. Performance Considerations
- Keep tests fast and focused
- Use appropriate test markers
- Monitor test execution time

## 🔮 Future Enhancements

### Planned Testing Features

1. **Property-Based Testing**: Using Hypothesis for property-based testing
2. **Mutation Testing**: Stryker for mutation testing
3. **Visual Regression Testing**: For UI components
4. **Load Testing**: For performance validation
5. **Security Testing**: For vulnerability detection

### Testing Metrics Dashboard

- Real-time test execution status
- Coverage trends over time
- Performance regression detection
- Test failure analysis

## 📞 Support and Contributing

### Getting Help

- **Documentation**: This file and inline code comments
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions

### Contributing Tests

1. **Fork the repository**
2. **Create a feature branch**
3. **Write comprehensive tests**
4. **Ensure all tests pass**
5. **Submit a pull request**

### Test Review Process

- All new tests must pass
- Coverage should not decrease
- Tests must follow established patterns
- Performance impact should be minimal

---

**🎯 Ready to ensure code quality and reliability through comprehensive testing!**

For questions or issues, please refer to the testing documentation or create a GitHub issue.

