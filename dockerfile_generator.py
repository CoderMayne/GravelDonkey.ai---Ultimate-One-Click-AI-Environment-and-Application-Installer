import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import json

def sanitize_name(name):
    """Converts a string to a safe, lowercase, hyphenated identifier."""
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

def _find_matching_package(selection_string, package_options):
    """
    Finds the best matching package string from a list of options.
    e.g., selection_string="2.3.0 (Recommended)" should match a package containing "torch==2.3.0".
    """
    if not selection_string or not package_options:
        return "None"
        
    # Handle "Nightly" case
    if "nightly" in selection_string.lower():
        for option in package_options:
            if "nightly" in option:
                return option
        return package_options[-1] # Fallback to last option for nightly

    # Extract version number
    version_match = re.match(r'^[0-9.]+', selection_string)
    if not version_match:
        # If no version number, it might be a direct package name like "flash-attn"
        for option in package_options:
            if selection_string in option:
                return option
        return selection_string # Fallback to the selection itself

    selected_version = version_match.group(0)

    for option in package_options:
        if f"=={selected_version}" in option:
            return option
            
    # Fallback if no exact match is found, return the first option
    return package_options[0]

def generate_build_artifacts(
    gpu_type,
    selected_apps,
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
    custom_run_commands=None,
    custom_env_vars=None
):
    """
    Generates build artifacts (Dockerfile, start.sh) as strings using specific dependency versions.
    """
    BASE_DIR = Path(__file__).resolve().parent
    TEMPLATE_DIR = BASE_DIR / 'templates'

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)
    dockerfile_template = env.get_template('Dockerfile.j2')

    for app in selected_apps:
        app['folder_name'] = sanitize_name(app['name'])
        app['env_var_name'] = app['folder_name'].upper().replace('-', '_') + "_PATH"

    docker_base_image = selected_dependency.get('docker_base_image', 'ubuntu:24.04')

    # Find the full package string based on user's selection
    torch_package = _find_matching_package(selected_torch_version, selected_dependency.get('torch_options'))
    
    attention_package = "None"
    if selected_xformers_version and selected_xformers_version.lower() != 'none':
        attention_package = _find_matching_package(selected_xformers_version, selected_dependency.get('xformers_options') or selected_dependency.get('attention_options'))

    # Prepare extra packages for installation
    extra_pip_packages = []
    if selected_triton_version: extra_pip_packages.append(f'triton=={selected_triton_version}')
    if selected_pandas_version: extra_pip_packages.append(f'pandas=={selected_pandas_version}')
    if selected_numpy_version: extra_pip_packages.append(f'numpy=={selected_numpy_version}')
    if selected_transformers_version: extra_pip_packages.append(f'transformers=={selected_transformers_version}')
    if selected_nltk_version: extra_pip_packages.append(f'nltk=={selected_nltk_version}')
    if selected_spacy_version: extra_pip_packages.append(f'spacy=={selected_spacy_version}')
    if selected_gensim_version: extra_pip_packages.append(f'gensim=={selected_gensim_version}')
    if selected_xgboost_version: extra_pip_packages.append(f'xgboost=={selected_xgboost_version}')

    # Prepend extra package installations to custom run commands
    if extra_pip_packages:
        pip_install_command = f"pip install --no-cache-dir {' '.join(extra_pip_packages)}"
        if custom_run_commands:
            custom_run_commands = f"{pip_install_command}\n{custom_run_commands}"
        else:
            custom_run_commands = pip_install_command

    # Render the Dockerfile template
    dockerfile_content = dockerfile_template.render(
        docker_base_image=docker_base_image,
        selected_apps=selected_apps,
        torch_package=torch_package,
        attention_package=attention_package,
        custom_run_commands=custom_run_commands,
        custom_env_vars=custom_env_vars,
        # Pass other versions for potential use in template
        cuda_version=selected_cuda_version,
        cudnn_version=selected_cudnn_version,
        sage2_version=selected_sage2_version,
        sage3_version=selected_sage3_version
    )

    # --- Generate start.sh content ---
    start_script_parts = [
        "#!/bin/bash",
        "set -e",
        'echo "========================================"',
        'echo " AI Environment - Application Launcher"',
        'echo "========================================"',
        ""
    ]

    for i, app in enumerate(selected_apps):
        start_script_parts.append(f'echo "{i+1}) {app["name"]}"')
    start_script_parts.append(f'echo "{len(selected_apps) + 1}) Quit"')
    start_script_parts.append("")
    start_script_parts.append('read -p "Please select an application to start: " choice')
    start_script_parts.append("")

    for i, app in enumerate(selected_apps):
        app_dir = f"/app/{app['folder_name']}"
        start_command = app.get("start_command", "echo 'No start command defined.'")
        
        script = f'''if [ "$choice" -eq "{i+1}" ]; then
    echo "Starting {app["name"]}..."
    APP_DIR="{app_dir}"
    echo "Changing to directory: $APP_DIR"
    cd "$APP_DIR"
    
    START_COMMAND="{start_command}"
    echo "Executing command: $START_COMMAND"
    
    if [[ "$START_COMMAND" == *.sh* ]]; then
        chmod +x $(echo $START_COMMAND | awk '{{print $1}}')
    fi

    exec $START_COMMAND
fi'''
        start_script_parts.append(script)

    start_script_parts.append(f'if [ "$choice" -eq "{len(selected_apps) + 1}" ]; then')
    start_script_parts.append('    echo "Quitting."')
    start_script_parts.append('    exit 0')
    start_script_parts.append('fi')
    start_script_parts.append("")
    start_script_parts.append('echo "Invalid selection."')
    start_script_parts.append('exit 1')

    start_script_content = "\n".join(start_script_parts)

    return {
        "Dockerfile": dockerfile_content,
        "start.sh": start_script_content
    }
