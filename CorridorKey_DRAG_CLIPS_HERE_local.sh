#!/usr/bin/env bash
# Corridor Key Launcher - Local Linux/macOS

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
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

# Ensure uv is available before attempting to run
if ! command -v uv &> /dev/null; then
    echo "[ERROR] 'uv' is not installed or not on PATH."
    echo ""
    echo "Install uv by running:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Then reopen your terminal and try again."
    read -p "Press enter to exit..."
    exit 1
fi

echo "Starting Corridor Key locally..."
echo "Target: $TARGET_PATH"

# Run via uv entry point (handles the virtual environment automatically)
uv run corridorkey wizard "$TARGET_PATH"

read -p "Press enter to close..."
