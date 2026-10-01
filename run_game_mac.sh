#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# GAME CONFIGURATION (macOS / Apple Silicon)
# =============================================================================
# macOS counterpart of run_game.sh. Qt draws through Cocoa here, so none of
# run_game.sh's WSL/X11 display setup applies. Unlike run_game.sh, this script
# activates the conda env itself, so it also works from a fresh Terminal.
# =============================================================================

CONDA_ENV="dama"                 # Conda env auto-activated if not already active (see setup_conda_mac.sh)

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# sysctl lives in /usr/sbin, which cron's default PATH (/usr/bin:/bin) leaves
# out; without it the checks below would misread this Mac.
PATH="$PATH:/usr/sbin"
if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: run_game_mac.sh is for macOS. Use run_game.sh on Linux/WSL." >&2
    exit 1
fi
# A terminal running under Rosetta makes every universal binary it starts run
# as x86_64 too. Re-run natively so the game's Python is never translated.
# _DAMA_ARM64_REEXEC stops a loop if the re-exec is still translated somehow.
if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
    if [ -z "${_DAMA_ARM64_REEXEC:-}" ]; then
        echo "Terminal is running under Rosetta (x86_64); re-running natively as arm64..."
        export _DAMA_ARM64_REEXEC=1
        exec arch -arm64 /bin/bash "$0" "$@"
    fi
    echo "[warn] Still running under Rosetta after re-exec; continuing as x86_64."
fi

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

# Conda environment guard: same discovery as local_train_mac.sh. A Terminal
# window only has conda on PATH after `conda init zsh`, and double-clicked or
# scripted launches have no env active at all.
if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
    echo "Conda env: ${CONDA_ENV} (already active)"
else
    _prev_env="${CONDA_DEFAULT_ENV:-none}"
    _conda_bin="${CONDA_EXE:-}"
    [ -x "$_conda_bin" ] || _conda_bin="$(command -v conda 2>/dev/null || true)"
    CONDA_BASE=""
    [ -x "$_conda_bin" ] && CONDA_BASE="$(dirname "$(dirname "$_conda_bin")")"
    if [ -z "$CONDA_BASE" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        CONDA_BASE=""
        for _b in "/opt/homebrew/Caskroom/miniforge/base" "/opt/homebrew/Caskroom/miniconda/base" \
                  "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" \
                  "/opt/miniconda3" "/opt/anaconda3"; do
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
            echo "[warn] Create it with: bash setup_conda_mac.sh; continuing with $(command -v python 2>/dev/null || echo python)"
        fi
    else
        echo "[warn] conda installation not found; continuing with $(command -v python 2>/dev/null || echo python)"
    fi
fi
# Activating from base swaps the env's bin/ into base's PATH slot rather than
# prepending it, so a pyenv shim ahead of base would still answer to `python`.
if [ "$(basename "${CONDA_PREFIX:-/}")" = "$CONDA_ENV" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PATH="${CONDA_PREFIX}/bin:$PATH"
fi

echo "=== Filipino Dama ==="
echo "Python: $(python --version)"
# The board and the Calculating Opponent run without torch; only the Learning
# Opponent needs it, so a missing torch is reported rather than fatal.
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())" \
    || echo "PyTorch: not importable (the Learning Opponent is unavailable)"

# Model discovery. The game loads models/latest.pt unless another model was
# picked in its settings. local_train_mac.sh's config writes models/latest_mac.pt
# and models/checkpoints_mac/, which the settings' model list also offers.
_found_model=false
for MODEL_PATH in "${PROJECT_DIR}/models/latest.pt" "${PROJECT_DIR}/models/latest_mac.pt"; do
    [[ -f "$MODEL_PATH" ]] || continue
    _found_model=true
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
done
if [[ "$_found_model" = false ]]; then
    echo "ML model:      not found (game will use algorithmic AI only)"
fi
echo ""

exec python -m dama
