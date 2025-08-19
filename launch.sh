#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check for venv directory
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Python virtual environment not found. Please create it first."
    echo "Example: python3 -m venv venv"
    exit 1
fi

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run the installer application
python3 "$SCRIPT_DIR/installer_app.py"
