#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================
# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   RAM: 1TB DDR4
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#   CPU Threads: 64
#===============================================================================

# =============================================================================
# TRAINING PARAMETERS - Optimized for MI210 64GB + 128 CPU threads + 1TB RAM
# =============================================================================
# All training parameters are now defined in the YAML config file.
# Edit the config file to change training settings.
# =============================================================================

# Config file to use (comment/uncomment to switch)
# Note: These are relative to PROJECT_DIR, resolved after SCRIPT_DIR is set
_CONFIG_FILE="config/training_config_server_retrain.yaml"
# _CONFIG_FILE="config/training_config_server.yaml"
# _CONFIG_FILE="config/training_config.yaml"

# Optional: Profile to apply from the config file
# CONFIG_PROFILE="server"
# CONFIG_PROFILE="local"
# CONFIG_PROFILE="cpu"
CONFIG_PROFILE=""

# -----------------------------------------------------------------------------
# Resume Settings (can override config file)
# -----------------------------------------------------------------------------
SET_PROCESS_TITLE=true           # Set to false to disable custom process title in htop
PROCESS_TITLE="micro"            # Process name shown in htop or btop (requires 'setproctitle' package)
RESUME=""                        # Path to checkpoint to resume from
RESUME_LATEST=false              # Set true to resume from latest checkpoint in models/checkpoints/
                                 # NOTE: Set to false for fresh retrain — old checkpoints are
                                 # degenerate. See config/training_config_server_retrain.yaml.

# -----------------------------------------------------------------------------
# Time-based Stopping (can override config file)
# -----------------------------------------------------------------------------
TRAIN_DURATION=""                # Override config duration (e.g., "2h", "1d", "30m")
                                 # Leave empty to use duration from config file

# =============================================================================
# END OF PARAMETERS - Do not edit below this line
# =============================================================================

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Resolve config file path (relative paths are relative to PROJECT_DIR)
CONFIG_FILE="${PROJECT_DIR}/${_CONFIG_FILE}"
[ -f "$CONFIG_FILE" ] || { echo "ERROR: Config file not found: $CONFIG_FILE"; exit 1; }

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# =============================================================================
# GPU Vendor Settings
# =============================================================================
# Suppress logging for AMD ROCm (harmless no-ops on NVIDIA/CPU)
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
export AMD_LOG_LEVEL=0
export ROCBLAS_LAYER=0

# =============================================================================
# torch.compile optimization - cache compiled models for faster subsequent runs
# =============================================================================
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.torch_cache"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Skip CUDAGraph for dynamic shapes - prevents overhead from recording many graphs
export TORCHINDUCTOR_CUDAGRAPH_SKIP_DYNAMIC=1

# Enable Parallel Compilation - scale to available cores
CPU_COUNT=$(python3 -c "import os; print(os.cpu_count() or 4)" 2>/dev/null || echo 4)
export MAX_JOBS=$(( CPU_COUNT > 8 ? CPU_COUNT / 2 : CPU_COUNT ))
export TORCHINDUCTOR_MAX_AUTOTUNE_PROCESSES=$(( MAX_JOBS > 4 ? MAX_JOBS * 3 / 4 : MAX_JOBS ))

# Caching improvements
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1

# =============================================================================
# CPU/Memory Optimizations
# =============================================================================
# Minimize CPU threads for numpy/MKL/OpenMP - self-play workers need the cores.
# The training thread uses GPU for all compute; these libraries would otherwise
# spawn threads that compete with the configured self-play worker processes.
if [ "$SET_PROCESS_TITLE" = true ]; then
    export PROCESS_TITLE
    # Self-heal: ensure 'setproctitle' is installed in the active env, otherwise
    # the trainer's rename is a silent no-op and htop shows 'python3'.
    if ! python3 -c "import setproctitle" >/dev/null 2>&1; then
        echo "Installing 'setproctitle' (required for PROCESS_TITLE='${PROCESS_TITLE}')..."
        python3 -m pip install --quiet setproctitle || \
            echo "[warn] Failed to install 'setproctitle'; process will show as 'python3' in htop."
    fi
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Increase file descriptor limit (best-effort, may fail on some systems)
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# Save a timestamped copy of the config file for reproducibility
CONFIG_ARCHIVE_DIR="${PROJECT_DIR}/models/configs_used"
mkdir -p "$CONFIG_ARCHIVE_DIR"
CONFIG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
cp "$CONFIG_FILE" "${CONFIG_ARCHIVE_DIR}/$(basename "${CONFIG_FILE}" .yaml)_${CONFIG_TIMESTAMP}.yaml"

# Console logging
LOG_DIR="${PROJECT_DIR}/logs/server/console"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/console_${LOG_TIMESTAMP}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Dama - ML Training ==="
echo ""

# Verify GPU is available
python3 -c "import torch; assert torch.cuda.is_available(), 'GPU not available. Training requires GPU.'" || {
    echo ""
    echo "ERROR: GPU is not available."
    echo ""
    echo "Troubleshooting:"
    echo "  NVIDIA: run 'nvidia-smi', install CUDA toolkit, pip install torch --index-url https://download.pytorch.org/whl/cu121"
    echo "  AMD:    run 'rocm-smi', install ROCm 6.x, pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
    exit 1
}

echo "GPU verified. Starting training..."
echo ""

# =============================================================================
# Auto-detect valid checkpoint (skip corrupted ones with NaN/Inf)
# =============================================================================
if [ "$RESUME_LATEST" = true ] && [ -z "$RESUME" ]; then
    echo "Checking checkpoints for corruption..."
    
    VALID_CHECKPOINT=$(CHECKPOINT_DIR="${PROJECT_DIR}/models/checkpoints" python3 -W ignore << 'PYEOF' 2>/dev/null | head -1
import sys, os
from pathlib import Path
import torch

checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
if not checkpoint_dir.exists():
    sys.exit(0)

checkpoints = sorted(checkpoint_dir.glob("model_step_*.pt"), 
                     key=lambda p: int(p.stem.split('_')[-1]), 
                     reverse=True)

for ckpt in checkpoints:
    try:
        c = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'model_state_dict' not in c:
            continue
        is_valid = all(torch.isfinite(p).all() for p in c['model_state_dict'].values())
        if is_valid:
            print(str(ckpt))
            sys.exit(0)
    except Exception:
        continue
PYEOF
)
    
    if [ -n "$VALID_CHECKPOINT" ]; then
        echo "  Found valid checkpoint: $(basename "$VALID_CHECKPOINT")"
        RESUME="$VALID_CHECKPOINT"
        RESUME_LATEST=false
    else
        echo "  No valid checkpoints found. Starting fresh."
        RESUME_LATEST=false
    fi
    echo ""
fi

# Build command arguments
ARGS=""

# Config file (required)
ARGS+=" --config ${CONFIG_FILE}"

# Profile (optional)
if [ -n "$CONFIG_PROFILE" ]; then
    ARGS+=" --profile ${CONFIG_PROFILE}"
fi

# Resume settings (override config if specified)
if [ -n "$RESUME" ]; then
    ARGS+=" --resume ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    ARGS+=" --resume-latest"
fi

# Time-based stopping (override config if specified)
if [ -n "$TRAIN_DURATION" ]; then
    ARGS+=" --train-duration ${TRAIN_DURATION}"
fi

echo "Training Configuration:"
echo ""
echo "  Config File:     ${CONFIG_FILE}"
if [ -n "$CONFIG_PROFILE" ]; then
    echo "  Profile:         ${CONFIG_PROFILE}"
fi
echo ""
echo "  [Session Overrides]"
if [ -n "$TRAIN_DURATION" ]; then
    echo "    Train Duration:    ${TRAIN_DURATION}"
else
    echo "    Train Duration:    (from config file)"
fi
if [ -n "$RESUME" ]; then
    echo "    Resume from:       ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    echo "    Resume from:       latest checkpoint"
else
    echo "    Resume from:       (from config file)"
fi
echo ""
echo "  See ${CONFIG_FILE} for all training parameters."
echo ""

exec -a "python3" python3 -W ignore::FutureWarning -m dama.ai.ml.trainer ${ARGS}
