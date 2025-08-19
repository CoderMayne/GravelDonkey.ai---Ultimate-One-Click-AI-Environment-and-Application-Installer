"""
Unit tests for the Dependency Utils module.
Tests package version fetching and version parsing functionality.
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from packaging.version import parse as parse_version, InvalidVersion
from dependency_utils import fetch_package_versions


class TestPackageVersionFetching:
    """Test cases for package version fetching functionality."""
    
    def test_fetch_package_versions_success(self, mock_requests):
        """Test successful package version fetching."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {
                "1.0.0": [],
                "2.0.0": [],
                "3.0.0": [],
                "2.1.0": [],
                "2.0.1": []
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        versions = fetch_package_versions("test-package")
        
        # Should return versions in descending order
        assert versions == ["3.0.0", "2.1.0", "2.0.1", "2.0.0", "1.0.0"]
        
        # Verify request was made to correct URL
        mock_requests.assert_called_once_with(
            "https://pypi.org/pypi/test-package/json",
            timeout=5
        )
    
    def test_fetch_package_versions_with_invalid_versions(self, mock_requests):
        """Test package version fetching with invalid version strings."""
        # Mock response with some invalid versions
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {
                "1.0.0": [],
                "2.0.0": [],
                "invalid-version": [],
                "2004d": [],  # Invalid version
                "3.0.0": [],
                "alpha-1.0": [],
                "1.0.0rc1": []
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        versions = fetch_package_versions("test-package")
        
        # Should filter out invalid versions and return valid ones in order
        assert versions == ["3.0.0", "2.0.0", "1.0.0"]
        
        # Invalid versions should be filtered out
        assert "invalid-version" not in versions
        assert "2004d" not in versions
        assert "alpha-1.0" not in versions
        assert "1.0.0rc1" not in versions
    
    def test_fetch_package_versions_empty_releases(self, mock_requests):
        """Test package version fetching with empty releases."""
        mock_response = Mock()
        mock_response.json.return_value = {"releases": {}}
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        versions = fetch_package_versions("test-package")
        
        assert versions == []
    
    def test_fetch_package_versions_http_error(self, mock_requests):
        """Test package version fetching with HTTP error."""
        # Mock HTTP error
        mock_requests.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        versions = fetch_package_versions("test-package")
        
        assert versions == []
    
    def test_fetch_package_versions_connection_error(self, mock_requests):
        """Test package version fetching with connection error."""
        # Mock connection error
        mock_requests.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        versions = fetch_package_versions("test-package")
        
        assert versions == []
    
    def test_fetch_package_versions_timeout_error(self, mock_requests):
        """Test package version fetching with timeout error."""
        # Mock timeout error
        mock_requests.side_effect = requests.exceptions.Timeout("Request timed out")
        
        versions = fetch_package_versions("test-package")
        
        assert versions == []
    
    def test_fetch_package_versions_request_exception(self, mock_requests):
        """Test package version fetching with general request exception."""
        # Mock general request exception
        mock_requests.side_effect = requests.exceptions.RequestException("General error")
        
        versions = fetch_package_versions("test-package")
        
        assert versions == []
    
    def test_fetch_package_versions_json_decode_error(self, mock_requests):
        """Test package version fetching with JSON decode error."""
        # Mock response that raises JSON decode error
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        versions = fetch_package_versions("test-package")
        
        assert versions == []
    
    def test_fetch_package_versions_malformed_response(self, mock_requests):
        """Test package version fetching with malformed response."""
        # Mock response missing required fields
        mock_response = Mock()
        mock_response.json.return_value = {"info": {}}  # Missing "releases" key
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        versions = fetch_package_versions("test-package")
        
        assert versions == []
    
    def test_fetch_package_versions_version_sorting(self, mock_requests):
        """Test that versions are properly sorted in descending order."""
        # Mock response with versions in random order
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {
                "1.0.0": [],
                "10.0.0": [],
                "2.0.0": [],
                "1.1.0": [],
                "2.1.0": [],
                "1.0.1": []
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        versions = fetch_package_versions("test-package")
        
        # Should be sorted in descending order
        expected_order = ["10.0.0", "2.1.0", "2.0.0", "1.1.0", "1.0.1", "1.0.0"]
        assert versions == expected_order
    
    def test_fetch_package_versions_semantic_versioning(self, mock_requests):
        """Test handling of semantic versioning formats."""
        # Mock response with various semantic versioning formats
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {
                "1.0.0": [],
                "1.0.0-alpha": [],
                "1.0.0-beta": [],
                "1.0.0-rc.1": [],
                "1.0.0+local": [],
                "1.0.0-alpha.1": [],
                "1.0.0-beta.2": []
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        versions = fetch_package_versions("test-package")
        
        # Should handle semantic versioning properly
        assert "1.0.0" in versions
        assert "1.0.0-alpha" in versions
        assert "1.0.0-beta" in versions
        assert "1.0.0-rc.1" in versions
    
    def test_fetch_package_versions_edge_cases(self, mock_requests):
        """Test package version fetching with edge case version strings."""
        # Mock response with edge case versions
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {
                "0.0.1": [],
                "0.1.0": [],
                "1.0.0": [],
                "999.999.999": [],
                "0.0.0": [],
                "1": [],
                "1.0": []
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        versions = fetch_package_versions("test-package")
        
        # Should handle various version formats
        assert "999.999.999" in versions
        assert "0.0.0" in versions
        assert "1" in versions
        assert "1.0" in versions
    
    def test_fetch_package_versions_performance(self, mock_requests):
        """Test performance with large number of versions."""
        # Mock response with many versions
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {f"{i}.0.0": [] for i in range(1000)}
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        import time
        start_time = time.time()
        
        versions = fetch_package_versions("test-package")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time
        assert execution_time < 1.0
        assert len(versions) == 1000
        assert versions[0] == "999.0.0"  # Highest version first
        assert versions[-1] == "0.0.0"   # Lowest version last
    
    def test_fetch_package_versions_memory_usage(self, mock_requests):
        """Test memory usage with large number of versions."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Mock response with many versions
        mock_response = Mock()
        mock_response.json.return_value = {
            "releases": {f"{i}.0.0": [] for i in range(1000)}
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        # Fetch versions
        versions = fetch_package_versions("test-package")
        del versions  # Clean up
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not leak memory significantly
        # Allow some tolerance for test overhead
        assert final_objects - initial_objects < 200


class TestVersionParsing:
    """Test cases for version parsing functionality."""
    
    def test_valid_version_parsing(self):
        """Test parsing of valid version strings."""
        valid_versions = [
            "1.0.0",
            "2.1.3",
            "10.20.30",
            "0.0.1",
            "999.999.999",
            "1.0.0-alpha",
            "1.0.0-beta.1",
            "1.0.0+local"
        ]
        
        for version_str in valid_versions:
            try:
                parsed = parse_version(version_str)
                assert str(parsed) == version_str
            except InvalidVersion:
                pytest.fail(f"Failed to parse valid version: {version_str}")
    
    def test_invalid_version_parsing(self):
        """Test parsing of invalid version strings."""
        invalid_versions = [
            "invalid",
            "1.0.0.0",  # Too many components
            "1.0",       # Missing patch version
            "1",         # Missing minor and patch
            "1.0.0-",    # Incomplete pre-release
            "1.0.0+",    # Incomplete build
            "1.0.0.alpha",  # Invalid pre-release format
            "1.0.0..1",  # Double dots
            "1.0.0-",    # Trailing dash
            "1.0.0+",    # Trailing plus
        ]
        
        for version_str in invalid_versions:
            with pytest.raises(InvalidVersion):
                parse_version(version_str)
    
    def test_version_comparison(self):
        """Test version comparison functionality."""
        # Test basic version comparison
        v1 = parse_version("1.0.0")
        v2 = parse_version("2.0.0")
        v3 = parse_version("1.1.0")
        
        assert v1 < v2
        assert v1 < v3
        assert v3 < v2
        assert v1 <= v1
        assert v2 >= v1
        
        # Test pre-release versions
        v1_alpha = parse_version("1.0.0-alpha")
        v1_beta = parse_version("1.0.0-beta")
        v1_rc = parse_version("1.0.0-rc.1")
        v1_final = parse_version("1.0.0")
        
        assert v1_alpha < v1_beta
        assert v1_beta < v1_rc
        assert v1_rc < v1_final
        
        # Test build metadata
        v1_local = parse_version("1.0.0+local")
        v1_other = parse_version("1.0.0+other")
        
        # Build metadata should not affect comparison
        assert v1_final == v1_local
        assert v1_local == v1_other
    
    def test_version_components(self):
        """Test access to version components."""
        version = parse_version("1.2.3-alpha.1+build.123")
        
        assert version.major == 1
        assert version.minor == 2
        assert version.micro == 3
        assert version.pre == ("alpha", 1)
        assert version.dev is None
        assert version.local == "build.123"
    
    def test_version_string_representation(self):
        """Test version string representation."""
        version = parse_version("1.2.3-alpha.1+build.123")
        
        # String representation should match input
        assert str(version) == "1.2.3-alpha.1+build.123"
        
        # Public version should exclude build metadata
        assert version.public == "1.2.3-alpha.1"
        
        # Base version should exclude pre-release and build metadata
        assert version.base_version == "1.2.3"


class TestDependencyUtilsIntegration:
    """Integration tests for dependency utilities."""
    
    def test_real_package_fetching(self):
        """Test fetching versions for a real package (with mocked requests)."""
        with patch('requests.get') as mock_get:
            # Mock a realistic PyPI response
            mock_response = Mock()
            mock_response.json.return_value = {
                "releases": {
                    "1.0.0": [{"filename": "test-1.0.0.tar.gz"}],
                    "1.1.0": [{"filename": "test-1.1.0.tar.gz"}],
                    "2.0.0": [{"filename": "test-2.0.0.tar.gz"}],
                    "2.1.0": [{"filename": "test-2.1.0.tar.gz"}]
                }
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            versions = fetch_package_versions("gradio")
            
            # Should return versions in descending order
            assert versions == ["2.1.0", "2.0.0", "1.1.0", "1.0.0"]
            
            # Verify request was made to PyPI
            mock_get.assert_called_once_with(
                "https://pypi.org/pypi/gradio/json",
                timeout=5
            )
    
    def test_error_handling_integration(self):
        """Test error handling in real-world scenarios."""
        with patch('requests.get') as mock_get:
            # Test network timeout
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
            
            versions = fetch_package_versions("gradio")
            assert versions == []
            
            # Test HTTP error
            mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
            
            versions = fetch_package_versions("gradio")
            assert versions == []
            
            # Test connection error
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            versions = fetch_package_versions("gradio")
            assert versions == []
    
    def test_version_filtering_integration(self):
        """Test version filtering with realistic data."""
        with patch('requests.get') as mock_get:
            # Mock response with realistic version data
            mock_response = Mock()
            mock_response.json.return_value = {
                "releases": {
                    "1.0.0": [{"filename": "test-1.0.0.tar.gz"}],
                    "1.0.0-alpha": [{"filename": "test-1.0.0-alpha.tar.gz"}],
                    "1.0.0-beta": [{"filename": "test-1.0.0-beta.tar.gz"}],
                    "1.0.0-rc.1": [{"filename": "test-1.0.0-rc.1.tar.gz"}],
                    "2.0.0": [{"filename": "test-2.0.0.tar.gz"}],
                    "2.0.0-dev": [{"filename": "test-2.0.0-dev.tar.gz"}],
                    "2.1.0": [{"filename": "test-2.1.0.tar.gz"}],
                    "invalid-version": [{"filename": "test-invalid.tar.gz"}],
                    "2004d": [{"filename": "test-2004d.tar.gz"}]
                }
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            versions = fetch_package_versions("test-package")
            
            # Should filter out invalid versions
            assert "invalid-version" not in versions
            assert "2004d" not in versions
            
            # Should include valid versions
            assert "2.1.0" in versions
            assert "2.0.0" in versions
            assert "1.0.0" in versions
            
            # Should be sorted in descending order
            assert versions == ["2.1.0", "2.0.0", "2.0.0-dev", "1.0.0", "1.0.0-rc.1", "1.0.0-beta", "1.0.0-alpha"]
    
    def test_performance_under_load(self):
        """Test performance when fetching multiple packages."""
        with patch('requests.get') as mock_get:
            # Mock successful responses
            mock_response = Mock()
            mock_response.json.return_value = {"releases": {"1.0.0": []}}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            import time
            start_time = time.time()
            
            # Fetch versions for multiple packages
            packages = ["gradio", "torch", "transformers", "numpy", "pandas"]
            for package in packages:
                versions = fetch_package_versions(package)
                assert len(versions) > 0
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete multiple fetches in reasonable time
            assert execution_time < 2.0
            assert execution_time > 0
            
            # Verify correct number of requests
            assert mock_get.call_count == len(packages)

