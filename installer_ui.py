import gradio as gr
from fix_prereqs import install_wsl, install_python, open_docker_website, open_nvidia_driver_website, open_cuda_toolkit_website, open_wsl_integration_guide, install_teacache
from prereq_checker import check_wsl, check_wsl_distro, check_docker as check_docker_prereq, check_docker_wsl_integration, check_python, check_gpu_drivers, check_teacache
from dependency_utils import fetch_package_versions


prerequisites_config = [
    {
        "id": "wsl",
        "label": "WSL 2",
        "check_func": check_wsl,
        "install_func": install_wsl,
        "install_button_label": "Install WSL and Ubuntu",
    },
    {
        "id": "wsl_distro",
        "label": "WSL Distribution",
        "check_func": check_wsl_distro,
        "install_func": install_wsl, # Re-use install_wsl for distro
        "install_button_label": "Install WSL and Ubuntu (Distro)",
    },
    {
        "id": "docker",
        "label": "Docker",
        "check_func": check_docker_prereq,
        "install_func": open_docker_website,
        "install_button_label": "Install Docker Desktop",
    },
    {
        "id": "docker_wsl_integration",
        "label": "Docker WSL Integration",
        "check_func": check_docker_wsl_integration,
        "install_func": open_wsl_integration_guide,
        "install_button_label": "Enable Docker WSL Integration",
    },
    {
        "id": "python",
        "label": "Python",
        "check_func": check_python,
        "install_func": install_python,
        "install_button_label": "Install Python in WSL",
    },
    {
        "id": "gpu_drivers",
        "label": "GPU Drivers",
        "check_func": check_gpu_drivers,
        "install_func": open_nvidia_driver_website,
        "install_button_label": "Install NVIDIA Drivers",
    },
    {
        "id": "teacache",
        "label": "Teacache",
        "check_func": check_teacache,
        "install_func": install_teacache,
        "install_button_label": "Install Teacache",
    },
]

# Dictionary to hold Gradio components for dynamic access
prereq_components = {}

def check_and_display_prereqs():
    updates = []
    for prereq in prerequisites_config:
        status, msg = prereq["check_func"]()
        btn_visible = not status
        updates.extend([
            gr.Textbox(value=msg),
            gr.Button(visible=btn_visible),
            gr.Checkbox(value=True), # Default to checked
            gr.State(status)
        ])
    return updates

def update_install_button_visibility(checkbox_value, prereq_met_status):
    return checkbox_value and (not prereq_met_status)

with gr.Blocks() as demo:
    gr.Markdown("# AI Environment Installer")

    with gr.Tabs() as tabs:
        with gr.TabItem("How to Use This Application", id="how_to_use_tab"):
            gr.Markdown("### Instructions on how to use this application.")
            # Content for How to Use tab

        with gr.TabItem("Step 1: Prerequisites Check", id="fix_prereqs_tab"):
            gr.Markdown("### Check and fix system prerequisites.")
            with gr.Column():
                for prereq in prerequisites_config:
                    with gr.Row():
                        gr.Markdown(f"**{prereq['label']}**")
                        prereq_components[f"{prereq['id']}_status_text"] = gr.Textbox(label="Status", interactive=False, scale=2)
                        prereq_components[f"{prereq['id']}_install_btn"] = gr.Button(prereq["install_button_label"], visible=False)
                        prereq_components[f"{prereq['id']}_checkbox"] = gr.Checkbox(label="Prerequisite Met", value=False, visible=False)
                        prereq_components[f"{prereq['id']}_met_status"] = gr.State(False) # Hidden state to store the actual status

                        # Link button click to install function
                        prereq_components[f"{prereq['id']}_install_btn"].click(
                            fn=prereq["install_func"],
                            inputs=[],
                            outputs=[],
                        )

            check_prereqs_button = gr.Button("Re-check Prerequisites")
            check_prereqs_button.click(
                fn=check_and_display_prereqs,
                inputs=[],
                outputs=[
                    prereq_components[f"{p['id']}_status_text"] for p in prerequisites_config
                ] +
                [
                    prereq_components[f"{p['id']}_install_btn"] for p in prerequisites_config
                ] +
                [
                    prereq_components[f"{p['id']}_checkbox"] for p in prerequisites_config
                ] +
                [
                    prereq_components[f"{p['id']}_met_status"] for p in prerequisites_config
                ]
            )

        with gr.TabItem("Step 2: Detect Your Hardware", id="hardware_detection_tab"):
            gr.Markdown("### Detect your system\'s hardware for optimal configuration.")
            # Content for Hardware Detection tab

        with gr.TabItem("Step 3: Choose Applications to Install", id="app_selection_tab"):
            gr.Markdown("### Select the AI applications you wish to install.")
            with gr.Column():
                gr.Markdown("#### Hardware Optimized Dependencies")
                with gr.Row():
                    package_dropdown = gr.Dropdown(
                        label="Select Package",
                        choices=["gradio", "tensorflow", "torch", "Teacache", "sage-attention-2", "sage-attention-3"],
                        value="gradio"
                    )
                    version_dropdown = gr.Dropdown(label="Select Version", choices=["latest"])
                    add_dependency_button = gr.Button("Add Dependency")

                added_dependencies_textbox = gr.Textbox(label="Added Dependencies", interactive=False)
                info_textbox = gr.Textbox(label="Info", visible=False, interactive=False)

                def update_versions(package_name):
                    special_packages = {
                        "Teacache": "Not a PyPI package. Install from GitHub.",
                        "sage-attention-2": "Not a PyPI package. Install from GitHub.",
                        "sage-attention-3": "Not a PyPI package. Install from GitHub."
                    }

                    if package_name in special_packages:
                        return gr.Dropdown(choices=[], value="N/A", interactive=False), gr.Textbox(value=special_packages[package_name], visible=True)
                    else:
                        versions = fetch_package_versions(package_name)
                        return gr.Dropdown(choices=versions if versions else ["latest"], interactive=True), gr.Textbox(visible=False)

                package_dropdown.change(
                    fn=update_versions,
                    inputs=package_dropdown,
                    outputs=[version_dropdown, info_textbox]
                )

                def add_dependency(package, version, current_deps):
                    if version == "N/A":
                        new_dep = f"{package} (from GitHub)"
                    else:
                        new_dep = f"{package}=={version}"

                    if current_deps:
                        return f"{current_deps}\n{new_dep}"
                    return new_dep

                add_dependency_button.click(
                    fn=add_dependency,
                    inputs=[package_dropdown, version_dropdown, added_dependencies_textbox],
                    outputs=added_dependencies_textbox
                )

        with gr.TabItem("Step 4: Configuration & Build", id="config_build_tab"):
            gr.Markdown("### Configure your Dockerfile and build the image.")
            # Content for Configuration & Build tab

        with gr.TabItem("Tutorials", id="tutorials_tab"):
            gr.Markdown("### Helpful tutorials and guides.")
            # Content for Tutorials tab

        with gr.TabItem("API Connections", id="api_tab"):
            gr.Markdown("### Connect to various AI API Providers.")
            with gr.Column():
                gr.Textbox(label="API Key", placeholder="Enter your API key here...")
                gr.Button("Connect to Provider")

        with gr.TabItem("Exit", id="exit_tab"):
            gr.Markdown("### Exit the application.")
            exit_button = gr.Button("Exit with Donkey Sound")
            donkey_audio = gr.Audio(value="Donkey.mp3", autoplay=False, visible=False)
            exit_message = gr.Markdown("", visible=False)

            def play_donkey_and_exit():
                return gr.Audio(autoplay=True, visible=True), gr.Markdown("Please close this browser tab to exit the application.", visible=True), gr.Exit()

            exit_button.click(
                fn=play_donkey_and_exit,
                inputs=[],
                outputs=[donkey_audio, exit_message, gr.Exit()]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=False)