#!/usr/bin/env bash

cd "$(dirname "$0")"

# Set the Terminal window title
echo -n -e "\033]0;VideoMaMa Setup Wizard\007"
echo "==================================================="
echo "   VideoMaMa (AlphaHint Generator) - Auto-Installer"
echo "==================================================="
echo ""

# Check that uv sync has been run (the .venv directory should exist)
# Note: I changed the name in the error message to match your Mac installer!
if [ ! -d ".venv" ]; then
    echo "[ERROR] Project environment not found."
    echo "Please run Install_CorridorKey_Linux_Mac.sh first!"
    read -p "Press [Enter] to exit..."
    exit 1
fi

# 1. Download Weights
echo "[1/1] Downloading VideoMaMa Model Weights..."
mkdir -p "VideoMaMaInferenceModule/checkpoints"

echo "Downloading VideoMaMa weights from HuggingFace..."
uv run hf download SammyLim/VideoMaMa --local-dir "VideoMaMaInferenceModule/checkpoints"

echo ""
echo "==================================================="
echo "  VideoMaMa Setup Complete!"
echo "==================================================="
read -p "Press [Enter] to close..."