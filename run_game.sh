#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# GAME CONFIGURATION
# =============================================================================

# Display settings (WSL)
DISPLAY_SERVER=":0"              # X11 display server (e.g., ":0" for WSLg)
QT_PLATFORM="xcb"                # Qt platform: "xcb" for X11, "wayland" for Wayland

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Change to project directory so relative paths work correctly
cd "$PROJECT_DIR"

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Console logging
LOG_DIR="${PROJECT_DIR}/logs/console"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/console_${LOG_TIMESTAMP}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

# WSL display configuration
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    echo "WSL detected; configuring display..."
    if [[ -z "${DISPLAY:-}" ]]; then
        export DISPLAY="${DISPLAY_SERVER}"
    fi
    # Set Qt plugin path for PyQt6 in conda environment
    CONDA_QT_PLUGINS="${CONDA_PREFIX}/lib/python3.11/site-packages/PyQt6/Qt6/plugins"
    if [[ -d "$CONDA_QT_PLUGINS" ]]; then
        export QT_QPA_PLATFORM_PLUGIN_PATH="${CONDA_QT_PLUGINS}/platforms"
        echo "Using conda Qt plugins: $CONDA_QT_PLUGINS"
    fi
    # Use specified Qt platform
    export QT_QPA_PLATFORM="${QT_PLATFORM}"
    echo "Set QT_QPA_PLATFORM=${QT_PLATFORM} and DISPLAY=$DISPLAY"
fi

echo "=== Filipino Dama ==="
echo "Python: $(python --version)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo ""

exec python -m dama
