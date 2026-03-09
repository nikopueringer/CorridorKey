#!/usr/bin/env bash

# Ensure script stops on error
set -e

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Enable OpenEXR Support
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Starting CorridorKey Inference..."
echo "Scanning ClipsForInference for Ready Clips (Input + Alpha)..."

# Run Manager (uv handles the virtual environment automatically)
uv run python "${SCRIPT_DIR}/corridorkey_cli.py" --action run_inference

echo "Inference Complete."
