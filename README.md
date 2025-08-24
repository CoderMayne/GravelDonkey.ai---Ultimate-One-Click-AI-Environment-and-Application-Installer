# GravelDonkey.ai - Ultimate One-Click AI Environment and Application Installer

## Features

- **Cross-Platform Support**: Works on both Windows (with WSL2) and native Linux.
- **Automated Prerequisite Checks**: Verifies your system is ready for installation on your specific OS.
- **Hardware-Aware**: Auto-detects your GPU and recommends optimized configurations.
- **GPU Simulation**: Allows testing configurations for different GPUs (NVIDIA, AMD, Intel).
- **Customizable Builds**: Select from several Popular A.I applications to include in your build.
- **Dockerfile Generation**: Creates a ready-to-use Dockerfile for building your custom AI environment.

## Prerequisites

This application requires the following software to be installed based on your operating system.

### Windows

- **Windows 11**
- **Windows Subsystem for Linux (WSL) 2**: The application runs within a WSL environment. The WSL distribution should be Ubuntu 22.04 or 24.04.
- **Docker Desktop**: Must be configured to use the WSL 2 backend.
- **Python 3.12.10 on Windows**: For running the Gradio-based installer application.
- **Up-to-date GPU Drivers**: Ensure you have the latest drivers for your NVIDIA, AMD, or Intel GPU.

### Linux
- **Ubuntu 22.04 / 24.04** (or a compatible Debian-based distribution).
- **Docker Engine**: The standard Docker installation for Linux.
- **Python 3.12.10**: For running the Gradio-based installer application.
- **Up-to-date GPU Drivers**: Ensure you have the latest drivers for your NVIDIA, AMD, or Intel GPU.

The application includes a "Prerequisites Check" tab that can help you verify your setup on either platform.

## Installation

1. **Clone the repository:**
  Create a folder somewhere on your PC. Then use Terminal in that Folder
   git clone https://github.com/CoderMayne/GravelDonkey.ai---Ultimate-One-Click-AI-Environment-and-Application-Installer.git
   In Terminal.... cd (folder you created)
   ```

2. **Create a Python virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # venv\Scripts\activate  # On Windows
   ```

3. **Install Python dependencies:**
   ```bash
   pip install gradio pandas
   ```
   *Note: The application will manage the dependencies for the AI environment it creates, this is just for the installer UI.*

## How to Use

1. **Launch the Installer:**
   - On **Windows**, double-click the `launch.bat` script.
   - On **Linux**, run the `launch.sh` script from your terminal:

     ```bash
     bash launch.sh
     ```

   This will start the Gradio web interface, which will open in your default web browser.

2. **Step 1: Prerequisites Check:**
   Navigate to the "Prerequisites Check" tab to ensure your system is configured correctly.

3. **Step 2: Detect Your Hardware:**
   Go to the "Detect Your Hardware" tab and click the "Detect Hardware" button.

4. **Step 3: Choose Applications:**
   Select the AI applications and tools you want to include in your environment.

5. **Step 4: Configure and Build:**
   - Go to the "Configuration & Build" tab and review your selections.
   - Click "Generate Build Files" to create a `Dockerfile` and other scripts in the `build` directory.

6. **Build the Docker Image:**
   - Open a terminal in the project directory.
   - Run the `docker build` command provided by the application, for example:

     ```bash
     docker build -t my-ai-environment .
     ```

7. **Run the Docker Container:**
   - Once the build is complete, run your container using the `docker run` command provided by the application.

## Documentation

- [GPU Simulation Guide](GPU_SIMULATION_README.md)
- [Integration Docs](INTEGRATION_README.md)

## Troubleshooting

### Error: 'wsl.exe' is not recognized as an internal or external command

This error means the installer script cannot find the `wsl.exe` executable. This usually happens when `C:\Windows\System32` is not in your system's PATH environment variable.

**Quick Fix:**

The `launch.bat` script has been updated to use the full path to `wsl.exe`. If you encounter this issue running `wsl` commands manually, you can use the full path: `C:\Windows\System32\wsl.exe`.

**Permanent Fix:**

Add `C:\Windows\System32` to your system's `PATH` environment variable.

### Error: 'docker' is not recognized as the name of a cmdlet, function, script file, or operable program

This is a similar issue, where the `docker` command is not in your system's PATH.

**Quick Fix:**

Run the docker command using the full path to the `docker.exe` executable. In PowerShell, you can use:

```powershell
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" build -t my-ai-environment .

```

**Permanent Fix:**

Add the Docker binary directory to your system's `PATH` environment variable. This is typically `C:\Program Files\Docker\Docker\resources\bin`.

### Docker Error: 'Service quota exceeded: driver not connecting'

This error comes from the Docker daemon and usually indicates a problem with the Docker Desktop installation or its configuration.

Here are some steps to resolve this:

1. **Restart Docker Desktop:** Quit Docker Desktop from the system tray icon, wait a moment, and then restart it.
2. **Check Docker Desktop Status:** Ensure Docker Desktop is running and the icon in the system tray is stable.
3. **Check WSL 2 Integration:**
   - Open Docker Desktop settings.
   - Go to **Resources > WSL Integration**.
   - Make sure "Enable integration with my default WSL distro" is checked.
4. **Prune Docker System (Use with caution):** This command cleans up unused Docker data. **Warning:** This will remove all unused containers, networks, and images.

   ```powershell
   & "C:\Program Files\Docker\Docker\resources\bin\docker.exe" system prune -a
   ```

5. **Check Docker Resources:**
   - In Docker Desktop settings, go to **Resources > Advanced**.
   - Consider increasing the allocated Memory, CPU, or Disk image size.

<!-- Test Comment -->
