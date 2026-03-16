#!/usr/bin/env bash
# CorridorKey Installer - macOS / Linux
#
# Usage:
#   curl -sSf https://corridorkey.dev/install.sh | bash
# Or locally:
#   bash install.sh

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

step() { echo -e "\n${CYAN}>>> $*${RESET}"; }
ok() { echo -e "    ${GREEN}[OK]${RESET} $*"; }
warn() { echo -e "    ${YELLOW}[WARN]${RESET} $*"; }
fail() { echo -e "\n    ${RED}[ERROR]${RESET} $*"; }

SYSTEM="$(uname -s)"
ARCH="$(uname -m)"

echo ""
echo -e "${CYAN}===================================================${RESET}"
echo -e "${CYAN}    CorridorKey - AI Green Screen Keyer${RESET}"
echo -e "${CYAN}    Installer${RESET}"
echo -e "${CYAN}===================================================${RESET}"

echo ""

# On Apple Silicon, MLX is the only sensible GPU option - skip the NVIDIA choice
if [[ "$SYSTEM" == "Darwin" && "$ARCH" == "arm64" ]]; then
  echo "Which build would you like to install?"
  echo ""
  echo "  [1] Apple Silicon - M1/M2/M3/M4 (MLX)  <-- recommended for this Mac"
  echo "  [2] CPU only"
  echo ""
  CHOICES=("1" "2")
  CHOICE=""
  while [[ ! " ${CHOICES[*]} " =~ $CHOICE ]]; do
    read -rp "Enter choice [1/2]: " CHOICE
  done
  case "$CHOICE" in
  1)
    PACKAGE="corridorkey-cli[mlx]"
    BACKEND="Apple Silicon (MLX)"
    ;;
  2)
    PACKAGE="corridorkey-cli"
    BACKEND="CPU"
    ;;
  esac
else
  echo "Which GPU do you have?"
  echo ""
  echo "  [1] NVIDIA GPU (CUDA)"
  echo "  [2] No GPU / CPU only"
  echo ""
  CHOICES=("1" "2")
  CHOICE=""
  while [[ ! " ${CHOICES[*]} " =~ $CHOICE ]]; do
    read -rp "Enter choice [1/2]: " CHOICE
  done
  case "$CHOICE" in
  1)
    PACKAGE="corridorkey-cli[cuda]"
    BACKEND="NVIDIA (CUDA)"
    ;;
  2)
    PACKAGE="corridorkey-cli"
    BACKEND="CPU"
    ;;
  esac
fi

echo ""
ok "Selected: $BACKEND"

step "Checking for uv package manager..."

if ! command -v uv &>/dev/null; then
  echo "    uv not found. Installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv &>/dev/null; then
    fail "uv was installed but is not on PATH."
    echo "    Close this terminal, open a new one, and run this script again."
    exit 1
  fi
fi
ok "uv is ready."

step "Installing $PACKAGE..."

if ! uv tool install "$PACKAGE" --python 3.13; then
  fail "Installation failed."
  echo "    Try the CPU build: uv tool install corridorkey-cli"
  exit 1
fi
ok "corridorkey-cli installed."

step "Running first-time setup..."
echo "    You will be asked whether to download the inference model (~400 MB)."
echo ""

corridorkey init

step "Creating Desktop launcher..."

DESKTOP_DIR=""
if [[ "$SYSTEM" == "Darwin" ]]; then
  DESKTOP_DIR="$HOME/Desktop"
elif [[ -d "$HOME/Desktop" ]]; then
  DESKTOP_DIR="$HOME/Desktop"
elif [[ -n "${XDG_DESKTOP_DIR:-}" && -d "$XDG_DESKTOP_DIR" ]]; then
  DESKTOP_DIR="$XDG_DESKTOP_DIR"
fi

if [[ -z "$DESKTOP_DIR" ]]; then
  warn "Could not locate Desktop directory. Skipping launcher creation."
  warn "Run manually with: corridorkey wizard /path/to/clips"
else
  if [[ "$SYSTEM" == "Darwin" ]]; then
    LAUNCHER="$DESKTOP_DIR/CorridorKey.command"
    cat >"$LAUNCHER" <<'LAUNCHER_EOF'
#!/usr/bin/env bash
# CorridorKey launcher - drag a clips folder onto this in Finder
if [[ -z "${1:-}" ]]; then
    echo "[ERROR] No folder provided."
    echo ""
    echo "Drag and drop a clips folder onto this file in Finder."
    echo "Or run: ./CorridorKey.command /path/to/clips"
    read -rp "Press Enter to exit..."
    exit 1
fi
TARGET="${1%/}"
if [[ ! -d "$TARGET" ]]; then
    echo "[ERROR] Not a directory: $TARGET"
    read -rp "Press Enter to exit..."
    exit 1
fi
echo "Starting CorridorKey..."
echo "Target: $TARGET"
echo ""
corridorkey wizard "$TARGET"
read -rp "Press Enter to close..."
LAUNCHER_EOF
    chmod +x "$LAUNCHER"
    ok "Launcher created: $LAUNCHER"
    echo "    Drag a clips folder onto 'CorridorKey.command' on your Desktop."

  else
    # Linux .desktop file
    LAUNCHER="$DESKTOP_DIR/CorridorKey.desktop"
    cat >"$LAUNCHER" <<DESKTOP_EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=CorridorKey
Comment=AI Green Screen Keyer
Exec=bash -c 'corridorkey wizard "%f"; read -rp "Press Enter to close..."'
Icon=utilities-terminal
Terminal=true
Categories=Graphics;Video;
MimeType=inode/directory;
DESKTOP_EOF
    chmod +x "$LAUNCHER"
    ok "Launcher created: $LAUNCHER"
    echo "    Drag a clips folder onto 'CorridorKey' on your Desktop."
    warn "Some Linux desktops may ask you to trust the launcher on first run."
  fi
fi

echo ""
echo -e "${GREEN}===================================================${RESET}"
echo -e "${GREEN}    Setup complete! You are ready to key.${RESET}"
echo -e "${GREEN}===================================================${RESET}"
echo ""
