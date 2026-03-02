#!/bin/bash
# Corridor Key Launcher - macOS / Linux

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOCAL_SCRIPT="$SCRIPT_DIR/corridorkey_cli.py"

# Activate virtual environment
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "[WARNING] No virtual environment found at $SCRIPT_DIR/venv"
    echo "Run ./Install_CorridorKey_macOS.sh first."
    read -p "Press enter to exit..."
    exit 1
fi

# If no folder was provided as an argument, ask the user
TARGET_PATH="$1"
if [ -z "$TARGET_PATH" ]; then
    echo "==================================================="
    echo "    CorridorKey"
    echo "==================================================="
    echo ""
    echo "Paste or type the path to your clips folder,"
    echo "then press Enter:"
    echo ""
    read -r TARGET_PATH
fi

# Validate
TARGET_PATH="${TARGET_PATH%/}"
if [ -z "$TARGET_PATH" ]; then
    echo "[ERROR] No path provided."
    read -p "Press enter to exit..."
    exit 1
fi

if [ ! -d "$TARGET_PATH" ]; then
    echo "[ERROR] Not a valid folder: $TARGET_PATH"
    read -p "Press enter to exit..."
    exit 1
fi

echo ""
echo "Starting Corridor Key..."
echo "Target: $TARGET_PATH"
echo ""

python "$LOCAL_SCRIPT" --action wizard --win_path "$TARGET_PATH"

read -p "Press enter to close..."
