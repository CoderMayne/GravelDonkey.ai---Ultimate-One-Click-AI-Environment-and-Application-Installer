import subprocess

script_path = r"c:\Scripts\WSL2 + Docker Container Approach\Gradio App - Ai Aio\2nd take\prereq_checker.py"

try:
    # Use shell=True and pass the entire command as a single string
    # Ensure the path is correctly quoted for the shell
    command = f"python \"{script_path}\"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
    print("Stdout:", result.stdout)
    print("Stderr:", result.stderr)
    print("Return Code:", result.returncode)
except Exception as e:
    print(f"An error occurred: {e}")

