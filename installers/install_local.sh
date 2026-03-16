#!/usr/bin/env bash
# CorridorKey Local Installer - macOS / Linux
#
# For contributors and testers who have cloned the repo.
# Installs directly from the local workspace - no PyPI needed.
#
# Usage (run from the repo root):
#   bash installers/install_local.sh

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

# Resolve repo root - script lives in <repo>/installers/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CLI_PATH="$REPO_ROOT/packages/corridorkey-cli"

echo ""
echo -e "${CYAN}===================================================${RESET}"
echo -e "${CYAN}    CorridorKey - AI Green Screen Keyer${RESET}"
echo -e "${CYAN}    Local Installer (from repo)${RESET}"
echo -e "${CYAN}===================================================${RESET}"
echo -e "    Repo: ${REPO_ROOT}"

# Sanity check
if [[ ! -d "$CLI_PATH" ]]; then
  fail "Could not find packages/corridorkey-cli at: $CLI_PATH"
  echo "    Make sure you run this script from inside the cloned repo."
  exit 1
fi

echo ""

EXTRA=""
BACKEND=""

if [[ "$SYSTEM" == "Darwin" && "$ARCH" == "arm64" ]]; then
  echo "Which build would you like to install?"
  echo ""
  echo "  [1] Apple Silicon - M1/M2/M3/M4 (MLX)  <-- recommended for this Mac"
  echo "  [2] CPU only"
  echo ""
  CHOICE=""
  while [[ "$CHOICE" != "1" && "$CHOICE" != "2" ]]; do
    read -rp "Enter choice [1/2]: " CHOICE
  done
  case "$CHOICE" in
  1)
    EXTRA="mlx"
    BACKEND="Apple Silicon (MLX)"
    ;;
  2)
    EXTRA=""
    BACKEND="CPU"
    ;;
  esac
else
  echo "Which GPU do you have?"
  echo ""
  echo "  [1] NVIDIA GPU (CUDA)"
  echo "  [2] No GPU / CPU only"
  echo ""
  CHOICE=""
  while [[ "$CHOICE" != "1" && "$CHOICE" != "2" ]]; do
    read -rp "Enter choice [1/2]: " CHOICE
  done
  case "$CHOICE" in
  1)
    EXTRA="cuda"
    BACKEND="NVIDIA (CUDA)"
    ;;
  2)
    EXTRA=""
    BACKEND="CPU"
    ;;
  esac
fi

PACKAGE="${CLI_PATH}${EXTRA:+[$EXTRA]}"

echo ""
ok "Selected: $BACKEND"
ok "Source:   $CLI_PATH"

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

step "Installing from local workspace..."

if ! uv tool install "$PACKAGE" --python 3.13; then
  fail "Installation failed."
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
if [[ -z "${1:-}" ]]; then
    echo "[ERROR] No folder provided."
    echo "Drag and drop a clips folder onto this file in Finder."
    read -rp "Press Enter to exit..."; exit 1
fi
TARGET="${1%/}"
if [[ ! -d "$TARGET" ]]; then
    echo "[ERROR] Not a directory: $TARGET"
    read -rp "Press Enter to exit..."; exit 1
fi
echo "Starting CorridorKey..."
echo "Target: $TARGET"
echo ""
corridorkey wizard "$TARGET"
read -rp "Press Enter to close..."
LAUNCHER_EOF
    chmod +x "$LAUNCHER"
    ok "Launcher created: $LAUNCHER"
  else
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
    warn "Some Linux desktops may ask you to trust the launcher on first run."
  fi
  echo "    Drag a clips folder onto 'CorridorKey' on your Desktop to start."
fi

echo ""
echo -e "${GREEN}===================================================${RESET}"
echo -e "${GREEN}    Setup complete! You are ready to key.${RESET}"
echo -e "${GREEN}===================================================${RESET}"
echo ""
