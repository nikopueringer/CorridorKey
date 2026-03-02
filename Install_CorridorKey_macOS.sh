#!/bin/bash

# Always run from the directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "==================================================="
echo "    CorridorKey - macOS Auto-Installer"
echo "==================================================="
echo ""
echo "Installing from: $SCRIPT_DIR"
echo ""

# 1. Find a suitable Python (3.10+)
# Try common locations: python3 on PATH, then versioned Homebrew binaries
PYTHON_CMD=""
for candidate in python3 python3.13 python3.12 python3.11 python3.10 \
                 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
                 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10; do
    if command -v "$candidate" &> /dev/null; then
        VERSION=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD="$candidate"
            echo "Found Python $VERSION ($candidate)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.10+ is required but not found!"
    echo "Install it via: brew install python@3.12"
    echo "Or download from https://www.python.org/downloads/"
    exit 1
fi

# 2. Create Virtual Environment
echo ""
echo "[1/3] Setting up Python Virtual Environment..."
if [ ! -d "venv" ]; then
    "$PYTHON_CMD" -m venv venv
else
    # Check if existing venv meets the version requirement
    VENV_VERSION=$(venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
    VENV_MAJOR=$(echo "$VENV_VERSION" | cut -d. -f1)
    VENV_MINOR=$(echo "$VENV_VERSION" | cut -d. -f2)
    if [ "$VENV_MAJOR" -lt 3 ] || ([ "$VENV_MAJOR" -eq 3 ] && [ "$VENV_MINOR" -lt 10 ]); then
        echo "Existing venv uses Python $VENV_VERSION — recreating with $("$PYTHON_CMD" --version)..."
        rm -rf venv
        "$PYTHON_CMD" -m venv venv
    else
        echo "Virtual environment already exists (Python $VENV_VERSION)."
    fi
fi

source venv/bin/activate

# 3. Install Requirements
echo ""
echo "[2/3] Installing Dependencies (this might take a while)..."
pip install --upgrade pip
pip install .

# On Apple Silicon, torch installs with MPS support automatically.
# Verify MPS is available:
python3 -c "
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('[OK] Apple Silicon MPS backend is available!')
elif torch.cuda.is_available():
    print('[OK] CUDA backend is available!')
else:
    print('[WARNING] No GPU acceleration detected. Inference will run on CPU (slow).')
"

# 4. Download Weights
echo ""
echo "[3/3] Downloading CorridorKey Model Weights..."
mkdir -p CorridorKeyModule/checkpoints

if [ ! -f "CorridorKeyModule/checkpoints/CorridorKey.pth" ]; then
    echo "Downloading CorridorKey.pth..."
    curl -L -o "CorridorKeyModule/checkpoints/CorridorKey.pth" \
        "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
else
    echo "CorridorKey.pth already exists!"
fi

echo ""
echo "==================================================="
echo "  Setup Complete! You are ready to key!"
echo "  Run: ./CorridorKey_DRAG_CLIPS_HERE_local.sh <clip_folder>"
echo "==================================================="
