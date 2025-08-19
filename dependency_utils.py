import requests
from packaging.version import parse as parse_version, InvalidVersion

def fetch_package_versions(package_name):
    """Fetches available versions for a given package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        valid_versions = []
        for v_str in data["releases"].keys():
            try:
                valid_versions.append(parse_version(v_str))
            except InvalidVersion:
                # Silently ignore invalid versions like '2004d'
                pass
        
        # Sort the valid versions and convert them back to strings
        sorted_versions = sorted(valid_versions, reverse=True)
        return [str(v) for v in sorted_versions]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching versions for {package_name}: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    versions = fetch_package_versions("gradio")
    print(f"Gradio versions: {versions[:5]}") # Print top 5 versions

    versions = fetch_package_versions("nonexistent-package-12345")
    print(f"Non-existent package versions: {versions}")
