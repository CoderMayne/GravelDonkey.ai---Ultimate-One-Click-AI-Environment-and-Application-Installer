import gradio as gr
import json
from pathlib import Path
import zipfile
import os
import subprocess
import platform
import sys
import re
from enhanced_hardware_detector import get_gpu_info
from dockerfile_generator import generate_build_artifacts, sanitize_name
from prereq_checker import (
    check_wsl,
    check_wsl_distro,
    check_docker,
    check_docker_wsl_integration,
    check_python,
    check_gpu_drivers,
    check_ubuntu_version,
)
from dependency_utils import fetch_package_versions

# --- Constants and Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
APPLICATIONS_PATH = DATA_DIR / 'applications.json'
DEPENDENCIES_PATH = DATA_DIR / 'dependencies.json'
README_PATH = BASE_DIR / 'README.md'
OUTPUT_DIR = BASE_DIR / 'build'

# Ensure the output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Audio and UI Configuration ---
def play_donkey_sound():
    """Play a donkey sound when the application exits."""
    try:
        # Try to use gradio.Audio for web playback
        import gradio as gr
        donkey_sound_path = BASE_DIR / "Donkey.mp3"
        
        if donkey_sound_path.exists():
            # Create a temporary audio component for playback
            print(f"🐴 Playing donkey sound from {donkey_sound_path}")
            # For web interface, we'll use HTML5 audio
            return str(donkey_sound_path)
        else:
            print(f"Warning: Donkey.mp3 not found at {donkey_sound_path}")
            
        # Fallback to system sounds
        if platform.system() == "Windows":
            import winsound
            # Play donkey-like sound sequence
            for i in range(2):
                winsound.Beep(600, 300)  # Hee
                winsound.Beep(400, 400)  # Haw
        else:
            print("🐴 *HEE-HAW! HEE-HAW!* 🐴")
            
    except ImportError as e:
        print(f"Audio library not available: {e}")
        if platform.system() == "Windows":
            try:
                import winsound
                for i in range(2):
                    winsound.Beep(600, 300)
                    winsound.Beep(400, 400)
            except:
                print("🐴 *HEE-HAW! HEE-HAW!* 🐴")
        else:
            print("🐴 *HEE-HAW! HEE-HAW!* 🐴")
    except Exception as e:
        print(f"Could not play donkey sound: {e}")
        print("🐴 *HEE-HAW! HEE-HAW!* 🐴")
    
    return None

def exit_application():
    """Play donkey sound and provide exit instructions."""
    donkey_sound_path = BASE_DIR / "Donkey.mp3"
    if donkey_sound_path.exists():
        return gr.Audio(value=str(donkey_sound_path), autoplay=True, visible=True), gr.Markdown("Please close this browser tab to exit the application.", visible=True)
    else:
        return gr.Audio(value=None, visible=False), gr.Markdown("Donkey.mp3 not found. Please close this browser tab to exit the application.", visible=True)

# --- Prerequisites Auto-Installation ---

def run_checks():
    """Runs all prerequisite checks and returns a formatted string, a success flag, and detected_ubuntu_version."""
    
    # Call check_ubuntu_version separately to get the detected_version
    ubuntu_check_success, ubuntu_check_message, detected_ubuntu_version = check_ubuntu_version()

    checks = [
        check_wsl(),
        check_wsl_distro(),
        check_docker(),
        check_docker_wsl_integration(),
        check_python(),
        check_gpu_drivers(),
        (ubuntu_check_success, ubuntu_check_message) # Add the ubuntu check result to the list
    ]
    results = ""
    all_passed = True
    for success, message in checks:
        results += f"{'✅' if success else '❌'} {message}\n"
        if not success:
            all_passed = False
    return results, all_passed, detected_ubuntu_version # Return detected_ubuntu_version


def load_json_data(path, name):
    """Loads and validates a JSON file."""
    if not path.exists():
        print(f"FATAL: {name} file not found at '{path}'.")
        return None
    try:
        with open(path, 'r', encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"FATAL: Could not parse {name} file at '{path}'. It may be corrupt. Error: {e}")
        return None

# Try to load comprehensive applications first, then extended, then original
COMPREHENSIVE_APPLICATIONS_PATH = DATA_DIR / 'applications_comprehensive.json'
EXTENDED_APPLICATIONS_PATH = DATA_DIR / 'applications_extended.json'
if COMPREHENSIVE_APPLICATIONS_PATH.exists():
    APPLICATIONS = load_json_data(COMPREHENSIVE_APPLICATIONS_PATH, "Comprehensive Applications")
elif EXTENDED_APPLICATIONS_PATH.exists():
    APPLICATIONS = load_json_data(EXTENDED_APPLICATIONS_PATH, "Extended Applications")
else:
    APPLICATIONS = load_json_data(APPLICATIONS_PATH, "Applications")

DEPENDENCIES = load_json_data(DEPENDENCIES_PATH, "Dependencies")

def get_all_dependency_versions():
    """Reads requirements.txt and fetches all available versions for each package from PyPI."""
    dependency_versions = {}
    special_packages = ["Teacache", "sage-attention-2", "sage-attention-3"]
    try:
        with open(BASE_DIR / 'requirements.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
                    if package_name in special_packages:
                        continue
                    versions = fetch_package_versions(package_name)
                    if versions:
                        dependency_versions[package_name] = versions
                    else:
                        print(f"Warning: Could not fetch versions for {package_name}. Skipping.")
    except FileNotFoundError:
        print("Error: requirements.txt not found.")
    return dependency_versions

# Call this once at startup
ALL_DEPENDENCY_VERSIONS = get_all_dependency_versions()

# Create a mapping from CUDA version to recommended defaults
cuda_to_defaults = {
    "13.0": (
        "9.0 (Recommended for CUDA 13.0)",
        "2.3.0 (Recommended for CUDA 13.0)",
        "1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)",
        "1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)",
        "0.0.24 (Recommended for CUDA 13.0/12.9/12.8/12.7)"
    ),
    "12.9": (
        "8.9 (Recommended for CUDA 12.9/12.8/12.7)",
        "2.2.2 (Recommended for CUDA 12.9/12.8/12.7)",
        "1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)",
        "1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)",
        "0.0.24 (Recommended for CUDA 13.0/12.9/12.8/12.7)"
    ),
    "12.8": (
        "8.9 (Recommended for CUDA 12.9/12.8/12.7)",
        "2.2.2 (Recommended for CUDA 12.9/12.8/12.7)",
        "1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)",
        "1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)",
        "0.0.24 (Recommended for CUDA 13.0/12.9/12.8/12.7)"
    ),
    "12.7": (
        "8.9 (Recommended for CUDA 12.9/12.8/12.7)",
        "2.2.2 (Recommended for CUDA 12.9/12.8/12.7)",
        "1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)",
        "1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)",
        "0.0.24 (Recommended for CUDA 13.0/12.9/12.8/12.7)"
    )
}

def find_selected_dependency(dependency_name_with_gpu):
    """Finds the dependency details and its GPU type from a formatted name."""
    if not dependency_name_with_gpu:
        return None, None
        
    match = re.match(r'^\(([^)]+)\)\s*(.*)', dependency_name_with_gpu)
    if not match:
        # Fallback for old format or if something goes wrong
        for gpu_type, deps in DEPENDENCIES.items():
            for dep in deps:
                if dep['name'] == dependency_name_with_gpu:
                    return gpu_type, dep
        return None, None

    gpu_type_str = match.group(1).lower()
    original_name = match.group(2)

    for gpu_type, deps in DEPENDENCIES.items():
        if gpu_type.lower() == gpu_type_str:
            for dep in deps:
                if dep['name'] == original_name:
                    return gpu_type, dep
    return None, None

# --- UI Functions ---

def detect_hardware(simulation_mode="auto", nvidia_model="rtx_5080", amd_model="rx_9060_xt", intel_model="arc_b770"):
    """
    Runs hardware detection, formats a summary, and returns UI updates.
    Supports simulation modes for testing without actual hardware.
    """
    global cuda_to_defaults
    print(f"DEBUG: detect_hardware called with simulation_mode={simulation_mode}, nvidia_model={nvidia_model}, amd_model={amd_model}, intel_model={intel_model})")
    # Determine which GPU type to simulate based on mode
    if simulation_mode == "force_nvidia":
        simulated_gpu_type = "nvidia"
        simulated_gpu_model = nvidia_model
    elif simulation_mode == "force_amd":
        simulated_gpu_type = "amd"
        simulated_gpu_model = amd_model
    elif simulation_mode == "force_intel":
        simulated_gpu_type = "intel"
        simulated_gpu_model = intel_model
    else:
        simulated_gpu_type = "nvidia"
        simulated_gpu_model = "rtx_5080"
    
    gpu_info = get_gpu_info(simulation_mode, simulated_gpu_type, simulated_gpu_model)
    print(f"DEBUG: get_gpu_info returned: {gpu_info}")
    gpu_type = gpu_info["gpu_type"]
    gpu_model = gpu_info["gpu_model"]
    cuda_version = gpu_info["cuda_version"]
    recommendations = gpu_info["recommendations"]

    # Add simulation indicator
    simulation_indicator = ""
    if simulation_mode != "auto":
        simulation_indicator = f"\n\n> ⚙⚙ **GPU Override / Simulation Mode:** `{simulation_mode}` - Using simulated {simulated_gpu_type.upper()} GPU"
    
    markdown_output = f"""### ✅ Hardware Detection Complete{simulation_indicator}
- **Detected GPU Type:** `{gpu_type.upper()}`"""
    if gpu_model and "N/A" not in gpu_model:
        markdown_output += f"\n- **Detected GPU Model:** `{gpu_model}`"
    if cuda_version and "Unknown" not in cuda_version:
        markdown_output += f"\n- **Detected CUDA Version:** `{cuda_version}`"

    all_dependency_names = []
    for gpu, deps in DEPENDENCIES.items():
        for dep in deps:
            all_dependency_names.append(f"({gpu.upper()}) {dep['name']}")
    
    dependency_names = sorted(all_dependency_names)

    default_dependency = None
    detected_deps = DEPENDENCIES.get(gpu_type, [])
    if detected_deps:
        recommended_dep_name = next((dep['name'] for dep in detected_deps if "recommended" in dep['name'].lower()), detected_deps[0]['name'])
        default_dependency = f"({gpu_type.upper()}) {recommended_dep_name}"

    # --- Augment Dropdown Lists ---
    
    # Base lists with recommended labels
    cuda_versions = ["13.0 (Recommended for RTX 5080, Blackwell, sm_120)", "12.9 (Recommended)", "12.8 (Recommended)", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8", "11.7", "11.6", "11.5", "Nightly"]
    cudnn_versions = ["9.0 (Recommended for CUDA 13.0)", "8.9 (Recommended for CUDA 12.9/12.8/12.7)", "8.8", "8.7", "8.6", "8.5", "8.4", "8.3", "8.2", "8.1", "8.0", "Nightly"]
    torch_versions = ["2.3.0 (Recommended for CUDA 13.0)", "2.2.2 (Recommended for CUDA 12.9/12.8/12.7)", "2.2.0", "2.1.2", "2.1.0", "2.0.1", "2.0.0", "1.13.1", "1.12.1", "Nightly"]
    xformers_versions = ["0.0.24 (Recommended for CUDA 13.0/12.9/12.8/12.7)", "0.0.23", "0.0.22", "0.0.21", "Nightly"]
    sage_attn2_versions = ["1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)", "0.9.0", "0.8.0", "Nightly"]
    sage_attn3_versions = ["1.0.0 (Recommended for CUDA 13.0/12.9/12.8/12.7)", "0.9.0", "Nightly"]

    # --- Augment with versions from dependencies.json ---
    
    def augment_list(base_list, key, parse_regex=None):
        existing_bases = {v.split()[0] for v in base_list}
        all_versions = set()
        for gpu_deps in DEPENDENCIES.values():
            for dep in gpu_deps:
                if parse_regex:
                    if dep.get(key):
                        for option in dep[key]:
                            match = re.search(parse_regex, option)
                            if match:
                                all_versions.add(match.group(1))
                elif dep.get(key):
                    all_versions.add(dep[key])
        
        for v in sorted(list(all_versions), reverse=True):
            if v not in existing_bases:
                base_list.append(v)
        return base_list

    cuda_versions = augment_list(cuda_versions, "cuda_version")
    cudnn_versions = augment_list(cudnn_versions, "cudnn_version")
    torch_versions = augment_list(torch_versions, "torch_options", r"torch==([0-9.]+)")
    xformers_versions = augment_list(xformers_versions, "xformers_options", r"xformers==([0-9.a-z]+)")


    # Special handling for RTX 5080 (Blackwell, sm_90)
    is_rtx_5080 = False
    if gpu_model and ("5080" in gpu_model or "Blackwell" in gpu_model or "sm_90" in gpu_model):
        is_rtx_5080 = True

    if is_rtx_5080:
        cuda_note = "\n> **Note:** NVIDIA RTX 5080 (Blackwell, sm_90) requires CUDA 12.8 or later. Older versions are not compatible."
    else:
        cuda_note = ""
    
    # Set defaults based on detected values (if available)
    def get_cuda_base(ver):
        return ver.split()[0] if ver else "13.0"
    
    show_recommendations = gr.update(visible=True)
    show_dependency_group = gr.update(visible=True)
    detected_cuda_base = get_cuda_base(cuda_version)
    
    if is_rtx_5080:
        for v in cuda_versions:
            if v.startswith("13.0"):
                default_cuda = v
                break
            elif v.startswith("12.9"):
                default_cuda = v
                break
            elif v.startswith("12.8"):
                default_cuda = v
                break
        else:
            default_cuda = cuda_versions[0]
    else:
        default_cuda = next((v for v in cuda_versions if v.startswith(detected_cuda_base)), cuda_versions[0])
        
    defaults = cuda_to_defaults.get(get_cuda_base(default_cuda), (cudnn_versions[0], torch_versions[0], sage_attn2_versions[0], sage_attn3_versions[0], xformers_versions[0]))
    default_cudnn, default_torch, default_sage2, default_sage3, default_xformers = defaults

    # Get choices for other packages from ALL_DEPENDENCY_VERSIONS
    triton_versions = ALL_DEPENDENCY_VERSIONS.get("triton", ["3.0.0"])
    pandas_versions = ALL_DEPENDENCY_VERSIONS.get("pandas", ["2.2.0"])
    numpy_versions = ALL_DEPENDENCY_VERSIONS.get("numpy", ["1.26.0"])
    transformers_versions = ALL_DEPENDENCY_VERSIONS.get("transformers", ["4.40.0"])
    nltk_versions = ALL_DEPENDENCY_VERSIONS.get("nltk", ["3.8.1"])
    spacy_versions = ALL_DEPENDENCY_VERSIONS.get("spacy", ["3.8.0"])
    gensim_versions = ALL_DEPENDENCY_VERSIONS.get("gensim", ["4.3.2"])
    xgboost_versions = ALL_DEPENDENCY_VERSIONS.get("xgboost", ["2.0.3"])

    # Get default values from recommendations
    default_triton = recommendations.get('triton', '3.0.0')
    default_pandas = recommendations.get('pandas', '2.2.0')
    default_numpy = recommendations.get('numpy', '1.26.0')
    default_transformers = recommendations.get('transformers', '4.40.0')
    default_nltk = recommendations.get('nltk', '3.8.1')
    default_spacy = recommendations.get('spacy', '3.8.0')
    default_gensim = recommendations.get('gensim', '4.3.2')
    default_xgboost = recommendations.get('xgboost', '2.0.3')

    debug_text = f"DEBUG: GPU Type: {gpu_type}, Model: {gpu_model}, CUDA: {cuda_version}"
    if cuda_note:
        markdown_output += cuda_note
    
    output_tuple = (
        markdown_output,
        gpu_type,
        gr.Dropdown(choices=dependency_names, value=default_dependency, label="Select Dependency Version", interactive=True, visible=True),
        gr.Dropdown(choices=cuda_versions, value=default_cuda, label="Select CUDA Version", interactive=True, visible=True),
        gr.Dropdown(choices=cudnn_versions, value=default_cudnn, label="Select cuDNN Version", interactive=True, visible=True),
        gr.Dropdown(choices=torch_versions, value=default_torch, label="Select PyTorch Version", interactive=True, visible=True),
        gr.Dropdown(choices=sage_attn2_versions, value=default_sage2, label="Select Sage Attention 2 Version", interactive=True, visible=True),
        gr.Dropdown(choices=sage_attn3_versions, value=default_sage3, label="Select Sage Attention 3 Version", interactive=True, visible=True),
        gr.Dropdown(choices=xformers_versions, value=default_xformers, label="Select Xformers Version", interactive=True, visible=True),
        show_dependency_group,
        show_recommendations,
        # New Dropdowns
        gr.Dropdown(choices=triton_versions, value=default_triton, label="Triton Version", interactive=True),
        gr.Dropdown(choices=pandas_versions, value=default_pandas, label="Pandas Version", interactive=True),
        gr.Dropdown(choices=numpy_versions, value=default_numpy, label="NumPy Version", interactive=True),
        gr.Dropdown(choices=transformers_versions, value=default_transformers, label="Transformers Version", interactive=True),
        gr.Dropdown(choices=nltk_versions, value=default_nltk, label="NLTK Version", interactive=True),
        gr.Dropdown(choices=spacy_versions, value=default_spacy, label="SpaCy Version", interactive=True),
        gr.Dropdown(choices=gensim_versions, value=default_gensim, label="Gensim Version", interactive=True),
        gr.Dropdown(choices=xgboost_versions, value=default_xgboost, label="XGBoost Version", interactive=True),
    )
    print(f"DEBUG: detect_hardware returning: {output_tuple}")
    return output_tuple


def update_sub_dependency_dropdowns(gpu_type, dependency_name):
    selected_dependency = next((dep for dep in DEPENDENCIES.get(gpu_type, []) if dep['name'] == dependency_name), None)
    torch_options = selected_dependency.get('torch_options', []) if selected_dependency else []
    xformers_options = selected_dependency.get('xformers_options', []) if selected_dependency else []
    return (
        gr.Dropdown(choices=torch_options, value=torch_options[0] if torch_options else None, interactive=True, visible=True),
        gr.Dropdown(choices=xformers_options, value=xformers_options[0] if xformers_options else None, interactive=True, visible=True)
    )

def prepare_summary(gpu_type_from_state, dependency_version, selected_cuda_version, selected_cudnn_version, selected_torch_version, selected_sage2_version, selected_sage3_version, selected_xformers_version, selected_triton_version, selected_pandas_version, selected_numpy_version, selected_transformers_version, selected_nltk_version, selected_spacy_version, selected_gensim_version, selected_xgboost_version, *app_ui_inputs):
    """Generates a summary of user selections."""
    selected_app_names = _get_selected_app_names_from_ui(*app_ui_inputs)
    gpu_type, selected_dependency = find_selected_dependency(dependency_version)

    if not gpu_type:
        return "Hardware not detected. Please click 'Detect Hardware' first.", gr.Group(visible=False)
    if not dependency_version:
        return "Please select a dependency version.", gr.Group(visible=False)

    if not selected_dependency:
        return f"Error: Could not find details for the selected dependency version: {dependency_version}", gr.Group(visible=False)

    summary_text = f"""### 

Review Your Configuration
- **GPU Type:** `{gpu_type.upper()}`
- **Dependency Version:** `{dependency_version}`
  - **CUDA Version:** `{selected_dependency.get('cuda_version', 'N/A')}`
  - **cuDNN Version:** `{selected_dependency.get('cudnn_version', 'N/A')}`
  - **PyTorch Version:** `{selected_torch_version}`
  - **Sage Attention 2 Version:** `{selected_sage2_version}`
  - **Sage Attention 3 Version:** `{selected_sage3_version}`
  - **Xformers Version:** `{selected_xformers_version}`
  - **Triton Version:** `{selected_triton_version}`
  - **Pandas Version:** `{selected_pandas_version}`
  - **NumPy Version:** `{selected_numpy_version}`
  - **Transformers Version:** `{selected_transformers_version}`
  - **NLTK Version:** `{selected_nltk_version}`
  - **SpaCy Version:** `{selected_spacy_version}`
  - **Gensim Version:** `{selected_gensim_version}`
  - **XGBoost Version:** `{selected_xgboost_version}`
  - **Base Docker Image:** `{selected_dependency.get('docker_base_image', 'N/A')}`
  - **Notes:** `{selected_dependency.get('notes', 'N/A')}`
- **Applications to Install:**
"""
    summary_text += "\n".join([f"  - **{name}**" for name in selected_app_names]) if selected_app_names else "  - None"

    if len(selected_app_names) >= 5:
        summary_text += "\n\n### **Disclaimer: Selecting many applications will significantly increase the Docker image build time.**"

    return summary_text, gr.Group(visible=True)

def _get_selected_app_names_from_ui(*app_ui_inputs):
    """Helper to extract selected app names from UI checkbox inputs."""
    selected_app_names = []
    app_table_data = app_ui_inputs[0] if app_ui_inputs else None
    if app_table_data is not None:
        for row in app_table_data.itertuples():
            if row.Install:
                selected_app_names.append(row.Name)
    return selected_app_names

def generate_files(gpu_type_from_state, dependency_version, selected_cuda_version, selected_cudnn_version, selected_torch_version, selected_sage2_version, selected_sage3_version, selected_xformers_version, selected_triton_version, selected_pandas_version, selected_numpy_version, selected_transformers_version, selected_nltk_version, selected_spacy_version, selected_gensim_version, selected_xgboost_version, container_name, image_tag, custom_run_commands, custom_env_vars, *app_ui_inputs):
    """Generates Dockerfile, start.sh, and a zip archive for download."""
    gpu_type, selected_dependency = find_selected_dependency(dependency_version)
    if not selected_dependency:
        error_message = f"Error: Could not find details for dependency '{dependency_version}'."
        return f"# {error_message}", gr.DownloadButton(visible=False), f"**Status:** {error_message}"

    apps_to_install = []
    app_table_data = app_ui_inputs[0] if app_ui_inputs else None
    if app_table_data is not None:
        selected_names = {row[1] for row in app_table_data.itertuples(index=False) if row[0]}
        apps_to_install = [app for app in APPLICATIONS if app['name'] in selected_names]

    # Pass all selected versions to generate_build_artifacts
    artifacts = generate_build_artifacts(
        gpu_type,
        apps_to_install,
        selected_dependency,
        selected_cuda_version,
        selected_cudnn_version,
        selected_torch_version,
        selected_sage2_version,
        selected_sage3_version,
        selected_xformers_version,
        selected_triton_version,
        selected_pandas_version,
        selected_numpy_version,
        selected_transformers_version,
        selected_nltk_version,
        selected_spacy_version,
        selected_gensim_version,
        selected_xgboost_version,
        custom_run_commands=custom_run_commands,
        custom_env_vars=custom_env_vars
    )
    dockerfile_content = artifacts["Dockerfile"]
    start_script_content = artifacts["start.sh"]

    image_tag = sanitize_name(image_tag) or "my-ai-environment"
    container_name = sanitize_name(container_name) or "my-ai-container"

    # Save files to the output directory
    dockerfile_path = OUTPUT_DIR / "Dockerfile"
    start_script_path = OUTPUT_DIR / "start.sh"
    dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
    start_script_path.write_text(start_script_content, encoding="utf-8")

    # Create a zip file for download
    zip_path = OUTPUT_DIR / "ai_environment_build_files.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(dockerfile_path, arcname="Dockerfile")
        zf.write(start_script_path, arcname="start.sh")

    unique_ports = sorted({app['port'] for app in apps_to_install if app.get("port")})
    port_mapping_suggestion = " ".join([f"-p {port}:{port}" for port in unique_ports])

    next_steps_md = f"""### ✅ Success! Your Build Files are Ready.
**1. Download the Files:**
Use the download button that just appeared to get your `Dockerfile` and `start.sh` in a zip archive.

**2. Build the Docker Image:**
```bash
docker build -t {image_tag} .
```

**3. Run Your Container:**
```bash
docker run -it --gpus all {port_mapping_suggestion} --name {container_name} {image_tag}
```"""

    return dockerfile_content, gr.DownloadButton("Download Build Files (Zip)", value=str(zip_path), visible=True), next_steps_md

with gr.Blocks(theme=None, title="GravelDonkey.ai - Ultimate One-Click AI Environment and Application Installer") as demo:
    # Custom CSS for Elephant Pro font and styling with enhanced readability
    gr.HTML("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Elephant+Pro:wght@400;700&display=swap');
    
    /* Global font size improvements */
    body {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    
    /* Title styling */
    .donkey-title {
        font-family: 'Elephant Pro', serif;
        font-size: 3.2em;
        font-weight: 700;
        text-align: center;
        margin: 25px 0;
        color: #2c3e50;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        line-height: 1.2;
    }
    
    .donkey-description {
        font-size: 1.5em;
        text-align: center;
        margin: 15px 0 35px 0;
        color: #7f8c8d;
        font-style: italic;
        line-height: 1.4;
    }
    
    .donkey-icon {
        font-size: 4em;
        text-align: center;
        margin: 15px 0;
    }
    
    /* Enhanced readability for all text elements */
    .markdown-text {
        font-size: 18px !important;
        line-height: 1.7 !important;
    }
    
    /* Larger headings */
    h1, h2, h3, h4, h5, h6 {
        font-size: 1.4em !important;
        line-height: 1.3 !important;
        margin: 20px 0 15px 0 !important;
    }
    
    h1 { font-size: 2.2em !important; }
    h2 { font-size: 1.9em !important; }
    h3 { font-size: 1.6em !important; }
    h4 { font-size: 1.4em !important; }
    
    /* Larger buttons */
    button {
        font-size: 16px !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
    }
    
    /* Larger dropdowns and inputs */
    select, input, textarea {
        font-size: 16px !important;
        padding: 10px 12px !important;
        border-radius: 6px !important;
    }
    
    /* Larger labels */
    label {
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    /* Enhanced table readability */
    table {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    
    th, td {
        padding: 12px 16px !important;
        font-size: 16px !important;
    }
    
    /* Larger accordion text */
    .accordion-content {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    
    /* Enhanced tab styling */
    .tab-nav {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Larger code blocks */
    pre, code {
        font-size: 15px !important;
        line-height: 1.5 !important;
    }
    
    /* Enhanced spacing */
    .container {
        padding: 20px !important;
    }
    
    /* Better contrast for readability */
    .text-content {
        color: #2c3e50 !important;
        background-color: #ffffff !important;
    }
    
    /* Responsive design for better readability */
    @media (max-width: 768px) {
        .donkey-title {
            font-size: 2.5em;
        }
        .donkey-description {
            font-size: 1.3em;
        }
        body {
            font-size: 15px !important;
        }
    }
    </style>
    """)
    
    # Title with donkey image
    gr.HTML('<div class="donkey-icon">🐴</div>')
    gr.HTML('<div class="donkey-title">GravelDonkey.ai - Ultimate One-Click AI Environment and Application Installer</div>')
    gr.HTML('<div class="donkey-description">Your comprehensive AI development environment setup tool with GPU optimization and 1000+ applications</div>')
    if not APPLICATIONS or not DEPENDENCIES:
        gr.Markdown("## ❌ FATAL ERROR: Could not load data files.")
        gr.Markdown("The application cannot start because `data/applications.json` or `data/dependencies.json` could not be loaded. Please check the terminal for errors.")
    else:
        # Shared UI components
        gpu_info_state = gr.State()
        gpu_type_state = gr.State()
        ubuntu_version_state = gr.State()
        dependency_version_selector = gr.Dropdown(label="Select Dependency Version", interactive=False, visible=False)
        cuda_version_selector = gr.Dropdown(label="Select CUDA Version", interactive=False, visible=False)
        cudnn_version_selector = gr.Dropdown(label="Select cuDNN Version", interactive=False, visible=False)
        torch_version_selector = gr.Dropdown(label="Select PyTorch Version", interactive=False, visible=False)
        sage2_version_selector = gr.Dropdown(label="Select Sage Attention 2 Version", interactive=False, visible=False)
        sage3_version_selector = gr.Dropdown(label="Select Sage Attention 3 Version", interactive=False, visible=False)
        xformers_version_selector = gr.Dropdown(label="Select Xformers Version", interactive=False, visible=False)
        triton_version_selector = gr.Dropdown(label="Select Triton Version", interactive=False, visible=False)
        pandas_version_selector = gr.Dropdown(label="Select Pandas Version", interactive=False, visible=False)
        numpy_version_selector = gr.Dropdown(label="Select NumPy Version", interactive=False, visible=False)
        transformers_version_selector = gr.Dropdown(label="Select Transformers Version", interactive=False, visible=False)
        nltk_version_selector = gr.Dropdown(label="Select NLTK Version", interactive=False, visible=False)
        spacy_version_selector = gr.Dropdown(label="Select SpaCy Version", interactive=False, visible=False)
        gensim_version_selector = gr.Dropdown(label="Select Gensim Version", interactive=False, visible=False)
        xgboost_version_selector = gr.Dropdown(label="Select XGBoost Version", interactive=False, visible=False)
        all_app_inputs = []
        donkey_audio_output = gr.Audio(label="Donkey Sound", autoplay=False, visible=False)
        exit_message_output = gr.Markdown("", visible=False)
        
        with gr.Tabs():
            with gr.TabItem("How to Use This Application"):
                gr.HTML('<div class="markdown-text">')
                gr.Markdown("## Step 2: How to Use This Application")
                gr.HTML('</div>')
                try:
                    readme_content = README_PATH.read_text(encoding="utf-8")
                    gr.Code(value=readme_content, language="markdown", interactive=False, label="README.md")
                except FileNotFoundError:
                    gr.Markdown("## Error: README.md not found.\nPlease refer to the project documentation for instructions.")
                exit_btn_tab2 = gr.Button("🐴 Exit Application", variant="stop", size="lg")
                exit_btn_tab2.click(fn=exit_application, outputs=[donkey_audio_output, exit_message_output])

            with gr.TabItem("Step 1: Prerequisites Check"):
                gr.HTML('<div class="markdown-text">')
                gr.Markdown("## Fix Missing Prerequisites")
                gr.Markdown("Use the buttons below to automatically install missing prerequisites.")
                gr.HTML('</div>')
                
                # Check prerequisites first to show current status
                fix_check_btn = gr.Button("🔍 Check Prerequisites Status", variant="secondary")
                fix_status_md = gr.Markdown("Click 'Check Prerequisites Status' to see what needs to be fixed.")
                
                # Installation buttons - listed vertically
                install_wsl_btn = gr.Button("Install WSL", variant="primary")
                install_docker_btn = gr.Button("Install Docker Desktop", variant="primary")
                install_python_btn = gr.Button("Install Python 3.12", variant="primary")
                install_gpu_drivers_btn = gr.Button("Install GPU Drivers", variant="primary")
                install_ffmpeg_wsl_btn = gr.Button("Install FFmpeg (WSL)", variant="primary", visible=False)
                
                fix_results_md = gr.Markdown("Installation results will appear here.")
                
                def check_prereqs_for_fix():
                    """Check prerequisites and show what needs to be fixed."""
                    results, all_passed, detected_ubuntu_version = run_checks()
                    if all_passed:
                        return "✅ All prerequisites are already installed!", detected_ubuntu_version
                    else:
                        return f"❌ Some prerequisites need attention:\n\n{results}", detected_ubuntu_version
                
                def install_wsl():
                    """Install WSL."""
                    try:
                        if platform.system() == "Windows":
                            result = subprocess.run(["powershell", "-Command", "wsl --install"], 
                                                  capture_output=True, text=True, check=True)
                            return "✅ WSL installation initiated. Please restart your computer when prompted and then start WSL by opening a terminal with 'wsl' command before running this application again."
                        else:
                            return "❌ WSL installation is only available on Windows."
                    except subprocess.CalledProcessError as e:
                        return f"❌ Failed to install WSL: {e.stderr}\n\n💡 Tip: If WSL is already installed, please start it by running 'wsl' in a new terminal window before proceeding."
                    except Exception as e:
                        return f"❌ Error installing WSL: {str(e)}"
                
                def install_docker():
                    """Provide Docker installation instructions."""
                    return """📋 To install Docker Desktop:
1. Visit: https://www.docker.com/products/docker-desktop
2. Download Docker Desktop for Windows
3. Run the installer and follow the setup wizard
4. Enable WSL 2 backend during installation
5. Restart your computer when prompted"""
                
                def install_python():
                    """Provide Python installation instructions."""
                    return """📋 To install Python:
1. Visit: https://www.python.org/downloads/
2. Download Python version 3.12 for Windows
3. Run the installer
4. ✅ IMPORTANT: Check "Add Python to PATH" during installation
5. Complete the installation"""
                
                def install_gpu_drivers():
                    """Provide GPU driver installation instructions."""
                    return """📋 To install GPU Drivers:
**For NVIDIA GPUs:**
1. Visit: https://www.nvidia.com/drivers/
2. Select your GPU model and download the latest driver
3. Run the installer

**For AMD GPUs:**
1. Visit: https://www.amd.com/support
2. Select your GPU model and download the latest driver
3. Run the installer

**For Intel GPUs:**
1. Visit: https://www.intel.com/content/www/us/en/support/products/80939/graphics.html
2. Download Intel Graphics Driver
3. Run the installer"""
                
                def install_ffmpeg_wsl():
                    """Install FFmpeg in WSL."""
                    try:
                        # Check if we're already in WSL
                        if os.path.exists('/proc/version'):
                            with open('/proc/version', 'r') as f:
                                if 'microsoft' in f.read().lower():
                                    # We're in WSL, run directly
                                    result = subprocess.run(["sudo", "apt", "update"], capture_output=True, text=True, check=True)
                                    result = subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], capture_output=True, text=True, check=True)
                                    return "✅ FFmpeg installed successfully in WSL!"
                        
                        # Try from Windows to WSL
                        command = "sudo apt update && sudo apt install -y ffmpeg"
                        result = subprocess.run(["wsl", "bash", "-c", command], 
                                              capture_output=True, text=True, check=True)
                        return "✅ FFmpeg installed successfully in WSL!"
                    except subprocess.CalledProcessError as e:
                        return f"❌ Failed to install FFmpeg in WSL: {e.stderr}"
                    except FileNotFoundError:
                        return "❌ WSL not found. Please install WSL first or run this from within WSL."
                    except Exception as e:
                        return f"❌ Error installing FFmpeg in WSL: {str(e)}"
                
                # Connect buttons to functions
                fix_check_btn.click(fn=check_prereqs_for_fix, outputs=[fix_status_md, ubuntu_version_state])
                install_wsl_btn.click(fn=install_wsl, outputs=[fix_results_md])
                install_docker_btn.click(fn=install_docker, outputs=[fix_results_md])
                install_python_btn.click(fn=install_python, outputs=[fix_results_md])
                install_gpu_drivers_btn.click(fn=install_gpu_drivers, outputs=[fix_results_md])
                # install_ffmpeg_wsl_btn.click(fn=install_ffmpeg_wsl, outputs=[fix_results_md])
                exit_btn_tab3 = gr.Button("🐴 Exit Application", variant="stop", size="lg")
                exit_btn_tab3.click(fn=exit_application, outputs=[donkey_audio_output, exit_message_output])

            with gr.TabItem("Step 2: Detect Your Hardware"):
                gr.HTML('<div class="markdown-text">')
                gr.Markdown("## Step 2: Detect Your Hardware")
                gr.Markdown("Click the button below to automatically detect your GPU and get optimized dependency recommendations.")
                gr.HTML('</div>')
                
                with gr.Row():
                    with gr.Column(scale=2):
                        detect_btn = gr.Button("🔍 Detect Hardware", variant="primary", size="lg")
                
                hardware_results_md = gr.Markdown("Your hardware information will appear here after detection.")
                
                with gr.Accordion("GPU Override / Simulation (Advanced)", open=False):
                    simulation_mode = gr.Dropdown(
                        choices=["auto", "force_nvidia", "force_amd", "force_intel"],
                        value="auto",
                        label="Simulation Mode (for testing)",
                        interactive=True
                    )
                    nvidia_model_selector = gr.Dropdown(label="NVIDIA Models", choices=["rtx_5090", "rtx_5080", "rtx_4090", "rtx_4080", "rtx_3090", "rtx_3080"], value="rtx_5080", interactive=True, visible=True)
                    amd_model_selector = gr.Dropdown(label="AMD Models", choices=["rx_9070_xt", "rx_9070", "rx_9060_xt", "rx_9060", "rx_7900_xtx", "rx_6800_xt"], value="rx_9070_xt", interactive=True, visible=True)
                    intel_model_selector = gr.Dropdown(label="Intel Models", choices=["arc_b770", "arc_a770", "arc_a750"], value="arc_b770", interactive=True, visible=True)

                with gr.Group(visible=False) as dependency_group:
                    gr.Markdown("### Select Your Dependency Configuration")
                    dependency_version_selector = gr.Dropdown(label="Select Dependency Version", interactive=True)
                    
                    with gr.Row():
                        cuda_version_selector = gr.Dropdown(label="Select CUDA Version", interactive=True, visible=False)
                        cudnn_version_selector = gr.Dropdown(label="Select cuDNN Version", interactive=True, visible=False)
                    
                    with gr.Row():
                        torch_version_selector = gr.Dropdown(label="Select PyTorch Version", interactive=True, visible=False)
                        sage2_version_selector = gr.Dropdown(label="Select Sage Attention 2 Version", interactive=True, visible=False)
                    
                    with gr.Row():
                        sage3_version_selector = gr.Dropdown(label="Select Sage Attention 3 Version", interactive=True, visible=False)
                        xformers_version_selector = gr.Dropdown(label="Select Xformers Version", interactive=True, visible=False)

                with gr.Group(visible=False) as recommendations_group:
                    gr.Markdown("### 🤖 AI Recommendations")
                    gr.Markdown("**Hardware-optimized dependency recommendations based on your GPU:**")
                    with gr.Row():
                        triton_version_selector = gr.Dropdown(label="Triton Version", interactive=True)
                        pandas_version_selector = gr.Dropdown(label="Pandas Version", interactive=True)
                        numpy_version_selector = gr.Dropdown(label="NumPy Version", interactive=True)
                    with gr.Row():
                        transformers_version_selector = gr.Dropdown(label="Transformers Version", interactive=True)
                        nltk_version_selector = gr.Dropdown(label="NLTK Version", interactive=True)
                        spacy_version_selector = gr.Dropdown(label="SpaCy Version", interactive=True)
                    with gr.Row():
                        gensim_version_selector = gr.Dropdown(label="Gensim Version", interactive=True)
                        xgboost_version_selector = gr.Dropdown(label="XGBoost Version", interactive=True)

                detect_btn.click(
                    fn=detect_hardware,
                    inputs=[simulation_mode, nvidia_model_selector, amd_model_selector, intel_model_selector],
                    outputs=[
                        hardware_results_md, 
                        gpu_type_state, 
                        dependency_version_selector, 
                        cuda_version_selector, 
                        cudnn_version_selector,
                        torch_version_selector, 
                        sage2_version_selector, 
                        sage3_version_selector, 
                        xformers_version_selector,
                        dependency_group, 
                        recommendations_group, 
                        triton_version_selector,
                        pandas_version_selector,
                        numpy_version_selector,
                        transformers_version_selector,
                        nltk_version_selector,
                        spacy_version_selector,
                        gensim_version_selector,
                        xgboost_version_selector,
                    ]
                )
                exit_btn_tab4 = gr.Button("🐴 Exit Application", variant="stop", size="lg")
                exit_btn_tab4.click(fn=exit_application, outputs=[donkey_audio_output, exit_message_output])

            with gr.TabItem("Step 3: Choose Applications to Install"):
                gr.HTML('<div class="markdown-text">')
                gr.Markdown("## Step 3: Choose Applications to Install")
                gr.Markdown("Select the AI applications you want to include in your Docker environment.")
                gr.HTML('</div>')
                
                if APPLICATIONS:
                    categories = sorted(list(set(app.get("category", "Other") for app in APPLICATIONS)))
                    
                    with gr.Row():
                        category_filter = gr.Dropdown(choices=["All"] + categories, value="All", label="Filter by Category", interactive=True)
                    
                    # Sort applications by rating (highest first) and add rating column
                    sorted_apps = sorted(APPLICATIONS, key=lambda x: x.get('rating', 0), reverse=True)
                    app_data = [[False, app['name'], app.get('description', 'No description'), app.get('port', 'N/A'), f"{app.get('rating', 'N/A')}/10"] for app in sorted_apps]
                    app_table = gr.Dataframe(
                        headers=["Install", "Name", "Description", "Port", "Rating"],
                        datatype=["bool", "str", "str", "str", "str"],
                        value=app_data,
                        interactive=True,
                        label="Available Applications (Sorted by Rating)"
                    )
                    all_app_inputs.append(app_table)

                    def filter_apps(category):
                        if category == "All":
                            filtered_apps = APPLICATIONS
                        else:
                            filtered_apps = [app for app in APPLICATIONS if app.get("category") == category]
                        
                        # Sort by rating within category
                        filtered_apps = sorted(filtered_apps, key=lambda x: x.get('rating', 0), reverse=True)
                        new_data = [[False, app['name'], app.get('description', 'No description'), app.get('port', 'N/A'), f"{app.get('rating', 'N/A')}/10"] for app in filtered_apps]
                        return gr.Dataframe(value=new_data)

                    category_filter.change(fn=filter_apps, inputs=category_filter, outputs=app_table)

                exit_btn_tab5 = gr.Button("🐴 Exit Application", variant="stop", size="lg")
                exit_btn_tab5.click(fn=exit_application, outputs=[donkey_audio_output, exit_message_output])

            with gr.TabItem("⚙️ Step 4: Configuration & Build"):
                gr.HTML('<div class="markdown-text">')
                gr.HTML('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 10px 0; color: #333; text-align: center;">')
                gr.HTML('<h2 style="color: #333; margin: 0; font-size: 2em;">⚙️ Step 4: Configuration & Build</h2>')
                gr.HTML('<p style="color: #333; margin: 10px 0; font-size: 1.2em;">Review your selections and generate the Docker build files</p>')
                gr.HTML('</div>')
                gr.HTML('</div>')
                
                with gr.Row():
                    container_name_input = gr.Textbox(label="Container Name", value="my-ai-container", interactive=True)
                    image_tag_input = gr.Textbox(label="Docker Image Tag", value="my-ai-environment", interactive=True)
                
                with gr.Accordion("Advanced Options", open=False):
                    custom_run_commands = gr.Textbox(
                        label="Custom Run Commands (one per line)",
                        placeholder="pip install custom-package\napt-get install -y custom-tool",
                        lines=3,
                        interactive=True
                    )
                    custom_env_vars = gr.Textbox(
                        label="Custom Environment Variables (KEY=VALUE, one per line)",
                        placeholder="CUSTOM_VAR=value\nANOTHER_VAR=another_value",
                        lines=3,
                        interactive=True
                    )
                
                with gr.Row():
                    summary_btn = gr.Button("📋 Review Configuration", variant="secondary", size="lg", scale=2)
                    gr.HTML('<div style="padding: 10px;"></div>')  # Spacer
                
                summary_md = gr.Markdown("Click 'Review Configuration' to see your selections.", elem_classes=["markdown-text"])
                
                with gr.Group(visible=False) as build_group:
                    gr.HTML('<div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 10px 0;">')
                    gr.HTML('<h3 style="color: #2e7d32; margin: 0;">✅ Configuration Approved - Ready to Build!</h3>')
                    gr.HTML('</div>')
                    
                    with gr.Row():
                        build_btn = gr.Button("🚀 Generate Build Files", variant="primary", size="lg", scale=2)
                        gr.HTML('<div style="padding: 10px;"></div>')  # Spacer
                    
                    dockerfile_output = gr.Code(language="dockerfile", label="Generated Dockerfile", interactive=False)
                    download_btn = gr.DownloadButton("📦 Download Build Files (Zip)", visible=False, variant="primary", size="lg")
                    next_steps_md = gr.Markdown(elem_classes=["markdown-text"])

                summary_btn.click(
                    fn=prepare_summary,
                    inputs=[
                        gpu_type_state, dependency_version_selector, cuda_version_selector, cudnn_version_selector,
                        torch_version_selector, sage2_version_selector, sage3_version_selector, xformers_version_selector,
                        triton_version_selector, pandas_version_selector, numpy_version_selector, transformers_version_selector,
                        nltk_version_selector, spacy_version_selector, gensim_version_selector, xgboost_version_selector
                    ] + all_app_inputs,
                    outputs=[summary_md, build_group]
                )

                build_btn.click(
                    fn=generate_files,
                    inputs=[
                        gpu_type_state, dependency_version_selector, cuda_version_selector, cudnn_version_selector,
                        torch_version_selector, sage2_version_selector, sage3_version_selector, xformers_version_selector,
                        triton_version_selector, pandas_version_selector, numpy_version_selector, transformers_version_selector,
                        nltk_version_selector, spacy_version_selector, gensim_version_selector, xgboost_version_selector,
                        container_name_input, image_tag_input, custom_run_commands, custom_env_vars
                    ] + all_app_inputs,
                    outputs=[dockerfile_output, download_btn, next_steps_md]
                )
                exit_btn_tab6 = gr.Button("🐴 Exit Application", variant="stop", size="lg")
                exit_btn_tab6.click(fn=exit_application, outputs=[donkey_audio_output, exit_message_output])

            with gr.TabItem("🎓 Tutorials"):
                gr.HTML('<div class="markdown-text">')
                gr.Markdown("## 🎓 Tutorials and Learning Resources")
                gr.Markdown("Learn how to use AI applications and get the most out of your environment.")
                gr.HTML('</div>')
                
                with gr.Accordion("📚 Getting Started with Stable Diffusion", open=False):
                    gr.Markdown("""
                    ### Quick Start Guide for Image Generation
                    
                    1. **Install AUTOMATIC1111 WebUI** from the Applications tab
                    2. **Download a model**: Visit [Civitai](https://civitai.com) or [Hugging Face](https://huggingface.co/models)
                    3. **Place models** in the `models/Stable-diffusion/` folder
                    4. **Start generating**: Use prompts like "a beautiful landscape, highly detailed, 8k"
                    5. **Experiment with settings**: Try different samplers, steps, and CFG scales
                    
                    **Pro Tips:**
                    - Use negative prompts to avoid unwanted elements
                    - Start with 20-30 steps for good quality
                    - CFG scale 7-12 works well for most images
                    - Enable xformers for faster generation
                    """
                )
                
                with gr.Accordion("🤖 Running Large Language Models", open=False):
                    gr.Markdown("""
                    ### LLM Setup and Usage Guide
                    
                    1. **Choose your interface**: Oobabooga WebUI (recommended for beginners)
                    2. **Download models**: Use Hugging Face or GPT44All model zoo
                    3. **Model formats**: GGUF files are recommended for efficiency
                    4. **Memory requirements**: 
                       - 7B models: 8GB+ RAM
                       - 13B models: 16GB+ RAM
                       - 30B+ models: 32GB+ RAM
                    5. **Optimization**: Use quantized models for better performance
                    
                    **Popular Models to Try:**
                    - Llama 2 7B/13B (General purpose)
                    - Code Llama (Programming tasks)
                    - Mistral 7B (Efficient and capable)
                    - Vicuna (Conversational AI)
                    """
                )
                
                with gr.Accordion("🎧 Audio Processing Workflows", open=False):
                    gr.Markdown("""
                    ### Voice and Audio AI Guide
                    
                    **Speech Recognition with Whisper:**
                    1. Install OpenAI Whisper from Applications
                    2. Supported formats: MP3, WAV, M4A, FLAC
                    3. Command: `whisper audio_file.mp3 --model medium`
                    4. Models: tiny, base, small, medium, large
                    
                    **Text-to-Speech with Coqui TTS:**
                    1. Install Coqui TTS
                    2. List available models: `tts --list_models`
                    3. Generate speech: `tts --text "Hello world" --out_path output.wav`
                    
                    **Voice Cloning:**
                    1. Use Real-Time Voice Cloning or RVC
                    2. Prepare 5-10 minutes of clean audio
                    3. Follow the training process in the WebUI
                    4. Generate new speech with the cloned voice
                    """
                )
                
                exit_btn_tutorials = gr.Button("🐴 Exit Application", variant="stop", size="lg")
                exit_btn_tutorials.click(fn=exit_application, outputs=[donkey_audio_output, exit_message_output])

            with gr.TabItem("🐴 Exit Application"):
                gr.HTML('<div class="markdown-text">')
                gr.HTML('<div style="text-align: center; padding: 40px;">')
                gr.HTML('<div class="donkey-icon">🐴</div>')
                gr.Markdown("## 🐴 Exit GravelDonkey.ai")
                gr.Markdown("Thank you for using GravelDonkey.ai! Click the button below to exit with a donkey sound.")
                gr.Markdown("*HEE-HAW! Come back soon!*")
                gr.HTML('</div>')
                gr.HTML('</div>')
                
                # Add audio element for donkey sound
                donkey_audio_path = BASE_DIR / "Donkey.mp3"
                if donkey_audio_path.exists():
                    gr.Audio(str(donkey_audio_path), label="🐴 Donkey Sound Preview", autoplay=False)
                
                exit_btn = gr.Button("🐴 Exit with Donkey Sound", variant="stop", size="lg")
                exit_btn.click(
                    fn=exit_application,
                    inputs=[],
                    outputs=[donkey_audio_output, exit_message_output]
                )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7865,
        share=False,
        debug=True,
        show_error=True,
        inbrowser=False
    )