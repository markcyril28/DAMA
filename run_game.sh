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

# Model discovery
MODEL_PATH="${PROJECT_DIR}/models/latest.pt"
if [[ -f "$MODEL_PATH" ]]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    MODEL_DATE=$(date -r "$MODEL_PATH" +"%Y-%m-%d %H:%M")
    echo "ML model:      ${MODEL_PATH} (${MODEL_SIZE}, ${MODEL_DATE})"
    MODEL_PATH="$MODEL_PATH" python - <<'PYEOF' 2>/dev/null || echo "  Checkpoint:   (could not read metadata)"
import os, torch
cp = torch.load(os.environ["MODEL_PATH"], map_location="cpu", weights_only=False)
step = cp.get("step", "?")
loss = cp.get("loss")
arch = cp.get("arch_params", {})
parts = [f"step {step}"]
if loss is not None:
    parts.append(f"loss {loss:.4f}")
if arch:
    parts.append(f'{arch.get("channels", "?")}-ch {arch.get("num_blocks", "?")}-blk')
print("  Checkpoint:  ", ", ".join(parts))
PYEOF
else
    echo "ML model:      not found (game will use algorithmic AI only)"
fi
echo ""

exec python -m dama
