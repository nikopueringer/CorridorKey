#!/bin/bash
# Corridor Key Launcher - Local Linux/macOS

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Ensure uv is installed and in PATH
if ! command -v uv &> /dev/null; then
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv not found. Installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        if [ -f "$HOME/.cargo/bin/uv" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        elif [ -f "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
fi
LOCAL_SCRIPT="$SCRIPT_DIR/corridorkey_cli.py"

# SAFETY CHECK: Ensure a folder was provided as an argument
if [ -z "$1" ]; then
    echo "[ERROR] No target folder provided."
    echo ""
    echo "USAGE:"
    echo "You can either run this script from the terminal and provide a path:"
    echo "  ./CorridorKey_DRAG_CLIPS_HERE_local.sh /path/to/your/clip/folder"
    echo ""
    echo "Or, in many Linux/macOS desktop environments, you can simply"
    echo "DRAG AND DROP a folder onto this script icon to process it."
    echo ""
    read -p "Press enter to exit..."
    exit 1
fi

# Folder dragged or provided via CLI? Use it as the target path.
TARGET_PATH="$1"

# Strip trailing slash if present
TARGET_PATH="${TARGET_PATH%/}"

echo "Starting Corridor Key locally..."
echo "Target: $TARGET_PATH"

# Run the python script via uv (handles the virtual environment automatically)
uv run python "$LOCAL_SCRIPT" --action wizard --win_path "$TARGET_PATH"

read -p "Press enter to close..."
