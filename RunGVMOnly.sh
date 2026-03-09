#!/bin/bash

# Ensure script stops on error
set -e

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Enable OpenEXR Support
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Starting Coarse Alpha Generation..."
echo "Scanning ClipsForInference..."

# Run Manager (uv handles the virtual environment automatically)
uv run python "${SCRIPT_DIR}/corridorkey_cli.py" --action generate_alphas

echo "Done."
