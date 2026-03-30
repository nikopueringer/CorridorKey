#!/usr/bin/env bash

cd "$(dirname "$0")"

echo "==================================================="
echo "    CorridorKey - MacOS/Linux Auto-Installer"
echo "==================================================="
echo ""

# Detect the Operating System
OS="$(uname -s)"
if [ "$OS" != "Darwin" ] && [ "$OS" != "Linux" ]; then
    echo "[ERROR] Unsupported operating system: $OS"
    read -p "Press [Enter] to exit..."
    exit 1
fi

# 1. Check for uv — install it automatically if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] uv is not installed. Installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install uv. Please visit https://docs.astral.sh/uv/ for manual instructions."
        read -p "Press [Enter] to exit..."
        exit 1
    fi

    # uv installer adds to PATH, but the current terminal session
    # doesn't see it yet. Add the default install location so we can continue.
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] uv was installed but cannot be found on PATH."
        echo "Please close this window, open a new terminal, and run this script again."
        read -p "Press [Enter] to exit..."
        exit 1
    fi
    echo "[INFO] uv installed successfully."
    echo ""
fi

# 2. Install all dependencies
echo "[1/2] Installing Dependencies (This might take a while on first run)..."
echo "      uv will automatically download Python if needed."

if [ "$OS" = "Darwin" ]; then
    echo "[INFO] macOS detected. Installing with MLX support..."
    uv sync --extra mlx
elif [ "$OS" = "Linux" ]; then
    echo "[INFO] Linux detected. Installing with CUDA support..."
    uv sync --extra cuda
fi

if [ $? -ne 0 ]; then
    echo "[ERROR] uv sync failed. Please check the output above for details."
    read -p "Press [Enter] to exit..."
    exit 1
fi

# 3. Download Weights
echo ""
echo "[2/2] Downloading CorridorKey Model Weights..."

# Use -p to create the folder only if it doesn't exist
mkdir -p "CorridorKeyModule/checkpoints"

if [ ! -f "CorridorKeyModule/checkpoints/CorridorKey.pth" ]; then
    echo "Downloading CorridorKey.pth..."
    curl -L -o "CorridorKeyModule/checkpoints/CorridorKey.pth" "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
else
    echo "CorridorKey.pth already exists!"
fi

echo ""
echo "==================================================="
echo "  Setup Complete! You are ready to key!"
echo "==================================================="
read -p "Press [Enter] to close..."