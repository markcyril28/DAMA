#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LOCAL TRAINING LAUNCHER
# =============================================================================
# LOCAL SPECS:
#   GPU: NVIDIA RTX 5050 (8GB VRAM)
#   RAM: 32 GB
#   CPU Cores: 12
#
# All training parameters are in config/training_config_local_retrain.yaml.
# This script only handles session-level overrides (resume, duration, etc.).
# =============================================================================

# -----------------------------------------------------------------------------
# SESSION SETTINGS — Edit these per-run
# -----------------------------------------------------------------------------
RESUME_LATEST=false              # Set true to resume from latest checkpoint in models/checkpoints/
                                 # NOTE: Set to false for fresh retrain — old checkpoints may be
                                 # degenerate. See config/training_config_local_retrain.yaml.
RESUME=""                        # Or set a specific checkpoint path
TRAIN_DURATION=""                # Train for this duration (empty = use config's time_limit)
                                 # Examples: "2d", "4h", "30m", "1d12h"

# =============================================================================
# END OF PARAMETERS - Do not edit below this line
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Minimize CPU threads for numpy/MKL/OpenMP — self-play workers need the cores.
# The training thread uses GPU for all compute; these libraries would otherwise
# spawn threads that compete with the 11 self-play worker processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Console logging
LOG_DIR="${PROJECT_DIR}/logs/console"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/console_${LOG_TIMESTAMP}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Filipino Dama - ML Training (Local) ==="
echo ""

# Verify CUDA is available
python -W ignore::FutureWarning -c "import torch; assert torch.cuda.is_available(), 'CUDA not available. Training requires GPU.'" || {
    echo ""
    echo "ERROR: CUDA is not available."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure NVIDIA GPU driver is installed on Windows (not inside WSL)"
    echo "  2. Run 'nvidia-smi' to verify GPU access"
    echo "  3. Update WSL: 'wsl --update'"
    echo "  4. Reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
}

echo "CUDA verified. Starting training..."
echo "Config: config/training_config_local_retrain.yaml"
echo ""

# Build command arguments — only session-level overrides
ARGS="--config ${PROJECT_DIR}/config/training_config_local_retrain.yaml"

# Resume settings
if [ -n "$RESUME" ]; then
    ARGS+=" --resume ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    ARGS+=" --resume-latest"
fi

# Time-based stopping (override config's time_limit if set)
if [ -n "$TRAIN_DURATION" ]; then
    ARGS+=" --train-duration ${TRAIN_DURATION}"
fi

exec python -W ignore::FutureWarning -m dama.ai.ml.trainer ${ARGS}
