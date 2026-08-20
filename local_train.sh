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
# All training parameters are in the selected config below.
# This script only handles session-level overrides (resume, duration, etc.).
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIG SELECTION — Uncomment exactly one
# -----------------------------------------------------------------------------
#TRAINING_CONFIG="config/training_config_local_retrain.yaml"
# TRAINING_CONFIG="config/training_config_local.yaml"
# TRAINING_CONFIG="config/training_config.yaml"
# TRAINING_CONFIG="config/training_config_server.yaml"
# TRAINING_CONFIG="config/training_config_server_retrain.yaml"
TRAINING_CONFIG="config/training_config_policy_distillation.yaml"

# -----------------------------------------------------------------------------
# SESSION SETTINGS — Edit these per-run
# -----------------------------------------------------------------------------
CONDA_ENV="dama"                 # Conda env auto-activated if not already active (see setup_conda.sh)
SET_PROCESS_TITLE=true           # Set to false to disable custom process title in htop
PROCESS_TITLE="micro-trainer"            # Process name shown in htop (requires 'setproctitle' package)
RESUME_LATEST="auto"          # auto = policy baseline, latest for other configs; true/false remain supported.
RESUME=""                     # Optional checkpoint. Enhanced stage requires the recorded promoted checkpoint.
ENHANCED_STAGE=false          # true only after policy-only promotion; trainer verifies registry, hash, and suite.
ENHANCED_INFERENCE_DEPTH=2    # Enhanced stage only: supported values are 2 or 3.
TRAIN_DURATION=""                # Train for this duration (empty = use config's time_limit)
                                 # Examples: "2d", "4h", "30m", "1d12h"

# =============================================================================
# END OF PARAMETERS - Do not edit below this line
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Resolve the config before creating logs or invoking the trainer. The policy
# recovery is resume-only and tied to one preserved checkpoint. Other configs
# retain their normal latest, specific-checkpoint, and fresh-start modes.
if [[ "$TRAINING_CONFIG" = /* ]]; then
    SELECTED_CONFIG_PATH="$TRAINING_CONFIG"
else
    SELECTED_CONFIG_PATH="${PROJECT_DIR}/${TRAINING_CONFIG}"
fi
if [ ! -f "$SELECTED_CONFIG_PATH" ]; then
    echo "ERROR: Training config not found: $SELECTED_CONFIG_PATH" >&2
    exit 1
fi
SELECTED_CONFIG_PATH="$(readlink -f "$SELECTED_CONFIG_PATH")"

POLICY_RECOVERY_CONFIG="$(readlink -f "${PROJECT_DIR}/config/training_config_policy_distillation.yaml")"
POLICY_RECOVERY_BASELINE="${PROJECT_DIR}/models/checkpoints_policy_distillation/model_step_134000.pt"
POLICY_RECOVERY_SHA256="7238CD80F2EF6DC9D8487D2579DE4BDF35AF4B85DCB2B3BD271659E795B14D27"
POLICY_RECOVERY_LEGACY_STATS="${PROJECT_DIR}/models/training_stats_policy_distillation.json"
POLICY_RECOVERY_STATS="${PROJECT_DIR}/models/training_stats_policy_distillation_recovery_wd1e4.json"
IS_POLICY_RECOVERY=false

case "$RESUME_LATEST" in
    auto|true|false) ;;
    *)
        echo "ERROR: RESUME_LATEST must be auto, true, or false." >&2
        exit 1
        ;;
esac
case "$ENHANCED_STAGE" in
    true|false) ;;
    *)
        echo "ERROR: ENHANCED_STAGE must be true or false." >&2
        exit 1
        ;;
esac
if [ "$ENHANCED_STAGE" = true ] && [ "$SELECTED_CONFIG_PATH" != "$POLICY_RECOVERY_CONFIG" ]; then
    echo "ERROR: ENHANCED_STAGE is available only with training_config_policy_distillation.yaml." >&2
    exit 1
fi
if [ "$ENHANCED_STAGE" = true ] &&
   [ "$ENHANCED_INFERENCE_DEPTH" != 2 ] &&
   [ "$ENHANCED_INFERENCE_DEPTH" != 3 ]; then
    echo "ERROR: ENHANCED_INFERENCE_DEPTH must be 2 or 3." >&2
    exit 1
fi

if [ "$SELECTED_CONFIG_PATH" = "$POLICY_RECOVERY_CONFIG" ]; then
    IS_POLICY_RECOVERY=true
    if [ "$RESUME_LATEST" = true ]; then
        echo "ERROR: --resume-latest is disabled for the policy-distillation recovery experiment." >&2
        exit 1
    fi
    if [ ! -f "$POLICY_RECOVERY_BASELINE" ]; then
        echo "ERROR: Policy recovery baseline not found: $POLICY_RECOVERY_BASELINE" >&2
        exit 1
    fi

    if [ "$ENHANCED_STAGE" = true ]; then
        if [ -z "$RESUME" ]; then
            echo "ERROR: ENHANCED_STAGE requires the recorded promoted policy-only checkpoint in RESUME." >&2
            exit 1
        fi
        if [[ "$RESUME" = /* ]]; then
            _resume_candidate="$RESUME"
        else
            _resume_candidate="${PROJECT_DIR}/${RESUME}"
        fi
        if [ ! -f "$_resume_candidate" ]; then
            echo "ERROR: Promoted policy checkpoint not found: $_resume_candidate" >&2
            exit 1
        fi
        RESUME="$(readlink -f "$_resume_candidate")"
    else
        if [ -z "$RESUME" ]; then
            if [ "$RESUME_LATEST" != auto ]; then
                echo "ERROR: Fresh start is disabled for policy recovery. Resume from model_step_134000.pt." >&2
                exit 1
            fi
            RESUME="$POLICY_RECOVERY_BASELINE"
        else
            if [[ "$RESUME" = /* ]]; then
                _resume_candidate="$RESUME"
            else
                _resume_candidate="${PROJECT_DIR}/${RESUME}"
            fi
            if [ ! -f "$_resume_candidate" ]; then
                echo "ERROR: Recovery checkpoint not found: $_resume_candidate" >&2
                exit 1
            fi
            _resume_candidate="$(readlink -f "$_resume_candidate")"
            if [ "$_resume_candidate" != "$(readlink -f "$POLICY_RECOVERY_BASELINE")" ]; then
                echo "ERROR: Policy recovery must resume from: $POLICY_RECOVERY_BASELINE" >&2
                exit 1
            fi
            RESUME="$POLICY_RECOVERY_BASELINE"
        fi
    fi

    _actual_recovery_sha256="$(sha256sum "$POLICY_RECOVERY_BASELINE" | awk '{print toupper($1)}')"
    if [ "$_actual_recovery_sha256" != "$POLICY_RECOVERY_SHA256" ]; then
        echo "ERROR: Policy recovery baseline SHA-256 mismatch." >&2
        echo "Expected: $POLICY_RECOVERY_SHA256" >&2
        echo "Found:    $_actual_recovery_sha256" >&2
        echo "Training was not started." >&2
        exit 1
    fi
fi

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
if [ "$IS_POLICY_RECOVERY" = true ]; then
    echo "Recovery baseline verified: ${POLICY_RECOVERY_BASELINE}"
    echo "Recovery SHA-256: ${POLICY_RECOVERY_SHA256}"
    if [ "$ENHANCED_STAGE" = false ] &&
       [ ! -e "$POLICY_RECOVERY_STATS" ] &&
       [ -f "$POLICY_RECOVERY_LEGACY_STATS" ]; then
        cp -- "$POLICY_RECOVERY_LEGACY_STATS" "$POLICY_RECOVERY_STATS"
        echo "Recovery stats seeded without modifying the legacy stats file: ${POLICY_RECOVERY_STATS}"
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# Conda environment guard
# -----------------------------------------------------------------------------
# Everything below runs bare `python`, so the script must work when launched from
# `base`, from another env, or from a non-interactive shell with no env at all
# (cron / scripts/runner.sh, where .bashrc never runs and conda is not on PATH).
# Activate $CONDA_ENV when it isn't already active. Failures only warn: a working
# non-conda interpreter stays usable, and the CUDA check below is the real gate.
if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
    echo "Conda env: ${CONDA_ENV} (already active)"
else
    _prev_env="${CONDA_DEFAULT_ENV:-none}"

    # Locate the conda installation: exported CONDA_EXE, then PATH, then the
    # usual install prefixes (the only branch that works under cron).
    _conda_bin="${CONDA_EXE:-}"
    [ -x "$_conda_bin" ] || _conda_bin="$(command -v conda 2>/dev/null || true)"
    CONDA_BASE=""
    [ -x "$_conda_bin" ] && CONDA_BASE="$(dirname "$(dirname "$_conda_bin")")"
    if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        CONDA_BASE=""
        for _b in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                  "$HOME/mambaforge" "$HOME/conda" "/opt/conda"; do
            if [ -f "$_b/etc/profile.d/conda.sh" ]; then CONDA_BASE="$_b"; break; fi
        done
    fi

    if [ -n "$CONDA_BASE" ]; then
        set +eu                      # conda.sh / activate are not -eu clean
        . "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        _activate_rc=$?
        set -euo pipefail
        if [ "$_activate_rc" -eq 0 ]; then
            echo "Conda env: ${CONDA_ENV} (activated; was '${_prev_env}')"
        else
            echo "[warn] Could not activate conda env '${CONDA_ENV}' (was '${_prev_env}')."
            echo "[warn] Create it with: bash setup_conda.sh; continuing with $(command -v python 2>/dev/null || echo python)"
        fi
    else
        echo "[warn] conda installation not found; continuing with $(command -v python 2>/dev/null || echo python)"
    fi
fi
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
    echo "  0. Ensure the '${CONDA_ENV}' conda env exists and has torch: bash setup_conda.sh"
    echo "  1. Ensure NVIDIA GPU driver is installed on Windows (not inside WSL)"
    echo "  2. Run 'nvidia-smi' to verify GPU access"
    echo "  3. Update WSL: 'wsl --update'"
    echo "  4. Reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
}

echo "CUDA verified. Starting training..."
echo "Config: ${SELECTED_CONFIG_PATH}"
echo ""

# Build command arguments — only session-level overrides
ARGS=(--config "$SELECTED_CONFIG_PATH")

# Resume settings
if [ -n "$RESUME" ]; then
    ARGS+=(--resume "$RESUME")
elif [ "$RESUME_LATEST" = true ] || [ "$RESUME_LATEST" = auto ]; then
    ARGS+=(--resume-latest)
fi
if [ "$ENHANCED_STAGE" = true ]; then
    ARGS+=(--enhanced-stage --inference-depth "$ENHANCED_INFERENCE_DEPTH")
fi

# Time-based stopping (override config's time_limit if set)
if [ -n "$TRAIN_DURATION" ]; then
    ARGS+=(--train-duration "$TRAIN_DURATION")
fi

exec python -W ignore::FutureWarning -m dama.ai.ml.trainer "${ARGS[@]}"
