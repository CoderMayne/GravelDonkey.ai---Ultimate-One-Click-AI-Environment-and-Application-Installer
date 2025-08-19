
import json

file_path = "c:/Scripts/WSL2 + Docker Container Approach/Gradio App - Ai Aio/2nd take/data/dependencies.json"

with open(file_path, 'r') as f:
    data = json.load(f)

new_torch_version = "2.8.0"
new_torchvision_version = "0.23.0"
new_torchaudio_version = "2.8.0"

for hardware_type in data:
    for entry in data[hardware_type]:
        # Update torch_package
        if "torch_package" in entry:
            current_package = entry["torch_package"]
            # Extract index-url if present
            index_url = ""
            if "--index-url" in current_package:
                parts = current_package.split("--index-url")
                index_url = "--index-url" + parts[1]
            
            if "nightly" in current_package:
                entry["torch_package"] = f"--pre torch torchvision torchaudio {index_url}"
            else:
                entry["torch_package"] = f"torch=={new_torch_version} torchvision=={new_torchvision_version} torchaudio=={new_torchaudio_version} {index_url}".strip()

        # Update torch_options
        if "torch_options" in entry:
            new_option_base = f"torch=={new_torch_version} torchvision=={new_torchvision_version} torchaudio=={new_torchaudio_version}"
            
            # Add new option to the beginning if not already present
            # Check if a 2.8.x version is already in the options
            found_2_8 = False
            for option in entry["torch_options"]:
                if option.startswith("torch==2.8"):
                    found_2_8 = True
                    break
            
            if not found_2_8:
                # Construct the new option with the correct index-url
                if "nightly" in entry["torch_options"][0]: # Assuming nightly is always the first option if present
                    new_option = f"--pre torch torchvision torchaudio {index_url}"
                else:
                    # Find the index-url from an existing option if available
                    existing_index_url = ""
                    for opt in entry["torch_options"]:
                        if "--index-url" in opt:
                            existing_index_url = "--index-url" + opt.split("--index-url")[1]
                            break
                    new_option = f"{new_option_base} {existing_index_url}".strip()
                
                entry["torch_options"].insert(0, new_option)

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Dependencies updated successfully.")
