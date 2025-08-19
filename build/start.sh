#!/bin/bash
set -e
echo "========================================"
echo " AI Environment - Application Launcher"
echo "========================================"

echo "1) ComfyUI"
echo "2) Hugging Face Transformers"
echo "3) Quit"

read -p "Please select an application to start: " choice

if [ "$choice" -eq "1" ]; then
    echo "Starting ComfyUI..."
    APP_DIR="/app/comfyui"
    echo "Changing to directory: $APP_DIR"
    cd "$APP_DIR"
    
    START_COMMAND="python main.py --listen"
    echo "Executing command: $START_COMMAND"
    
    if [[ "$START_COMMAND" == *.sh* ]]; then
        chmod +x $(echo $START_COMMAND | awk '{print $1}')
    fi

    exec $START_COMMAND
fi
if [ "$choice" -eq "2" ]; then
    echo "Starting Hugging Face Transformers..."
    APP_DIR="/app/hugging-face-transformers"
    echo "Changing to directory: $APP_DIR"
    cd "$APP_DIR"
    
    START_COMMAND="echo 'No start command defined.'"
    echo "Executing command: $START_COMMAND"
    
    if [[ "$START_COMMAND" == *.sh* ]]; then
        chmod +x $(echo $START_COMMAND | awk '{print $1}')
    fi

    exec $START_COMMAND
fi
if [ "$choice" -eq "3" ]; then
    echo "Quitting."
    exit 0
fi

echo "Invalid selection."
exit 1