#!/usr/bin/env bash

# Ensure script stops on error
set -e

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Enable OpenEXR Support
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Starting Coarse Alpha Generation..."
echo "Scanning ClipsForInference..."

# Run Manager (uv handles the virtual environment automatically)
uv run python "${SCRIPT_DIR}/corridorkey_cli.py" --action generate_alphas

echo "Done."
