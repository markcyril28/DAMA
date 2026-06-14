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
SET_PROCESS_TITLE=true           # Set to false to disable custom process title in htop
PROCESS_TITLE="micro-trainer"            # Process name shown in htop (requires 'setproctitle' package)
RESUME_LATEST=false            # FRESH START (mandatory): embed_norm arch fix makes ALL old checkpoints obsolete. See Journal Pass 102.
                                 # NOTE: this flag is the EFFECTIVE control — --resume-latest overrides YAML resume.enabled
                                 # (trainer.py:3982). Pass 100 left this 'true' by mistake (changed only the comment), so the
                                 # "fresh" run still resumed; the real fix was the model, not the checkpoint. Re-enable ONLY after
                                 # a healthy (>0% WR) checkpoint on the NEW arch AND archiving old-arch checkpoints (user-action).
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
# spawn threads that compete with the 6 self-play worker processes.
if [ "$SET_PROCESS_TITLE" = true ]; then export PROCESS_TITLE; fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Console logging
LOG_DIR="${PROJECT_DIR}/logs/local/console"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/console_${LOG_TIMESTAMP}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Filipino Dama - ML Training (Local) ==="
echo ""

# Cython staleness guard (fail-safe). The compiled .so files are committed but
# can lag their .pyx sources (a stale committed _fast_search.so silently ran
# the old alpha-beta search in Jun 2026). Rebuild in-place if any .pyx is newer
# than its .so. No-op when fresh; never blocks training on a build failure.
_cython_stale=false
while IFS= read -r _pyx; do
    _so="$(ls -t "${_pyx%.pyx}".*.so 2>/dev/null | head -1 || true)"
    if [ -z "$_so" ] || [ "$_pyx" -nt "$_so" ]; then _cython_stale=true; fi
done < <(find "${PROJECT_DIR}/src" -name '*.pyx' -not -path '*/build/*' 2>/dev/null)
if [ "$_cython_stale" = true ]; then
    echo "Cython sources changed — rebuilding extensions..."
    ( cd "${PROJECT_DIR}/src" && python setup_cython.py build_ext --inplace ) \
        || echo "[warn] Cython rebuild failed — using existing .so files."
    echo ""
fi

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
