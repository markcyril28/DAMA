#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SETUP CONFIGURATION (macOS / Apple Silicon)
# =============================================================================
# MAC SPECS:
#   Chip: Apple M4 Max (10 performance + 4 efficiency cores)
#   Memory: 36 GB unified (shared by CPU and GPU)
#   GPU backend: Metal Performance Shaders (MPS) -- there is no CUDA on macOS
#
# macOS counterpart of setup_conda.sh. The CUDA wheel index, nvidia-ml-py and
# the Linux Qt/X11 packages that script installs do not exist for macOS; the
# stock PyPI torch wheel for arm64 ships with MPS support built in.
# =============================================================================

# Environment settings
ENV_NAME="dama"                  # Name of the conda environment (local_train_mac.sh activates it)
PYTHON_VERSION="3.11"            # Python version (must match environment.yml)
RECREATE_ENV=true                # true = remove and rebuild an existing '$ENV_NAME' env

# Optional components
INSTALL_MATPLOTLIB=true          # Install matplotlib for plotting (eval_checkpoints_mac.sh)

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# sysctl lives in /usr/sbin, which cron's default PATH (/usr/bin:/bin) leaves
# out; without it the checks below would misread this Mac.
PATH="$PATH:/usr/sbin"
if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: setup_conda_mac.sh is for macOS. Use setup_conda.sh on Linux/WSL." >&2
    exit 1
fi
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" != 1 ]; then
    echo "ERROR: Apple Silicon (M-series) Mac required. PyTorch no longer ships" >&2
    echo "       Intel macOS wheels, and MPS needs an Apple GPU." >&2
    exit 1
fi
# A terminal running under Rosetta makes every child an x86_64 process, so pip
# and the Cython build would target the wrong architecture. Re-run natively.
# _DAMA_ARM64_REEXEC stops a loop if the re-exec is still translated somehow;
# the conda-platform and env-arch checks below still refuse an x86_64 env.
if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
    if [ -z "${_DAMA_ARM64_REEXEC:-}" ]; then
        echo "Terminal is running under Rosetta (x86_64); re-running natively as arm64..."
        export _DAMA_ARM64_REEXEC=1
        exec arch -arm64 /bin/bash "$0" "$@"
    fi
    echo "[warn] Still running under Rosetta after re-exec; continuing as x86_64."
fi

echo "=== Dama - Conda Environment Setup (macOS / Apple Silicon) ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Locate conda. A non-interactive shell does not run the conda init hook from
# ~/.zshrc, so fall back to the usual Homebrew / installer prefixes.
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
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: conda not found. Install Miniforge (arm64) first:" >&2
    echo "       brew install --cask miniforge" >&2
    exit 1
fi
set +u                               # conda.sh / activate are not -u clean
. "${CONDA_BASE}/etc/profile.d/conda.sh"
set -u

# An Intel (osx-64) conda installs x86_64 packages that only run under Rosetta
# and cannot use MPS. Parsed with conda's own interpreter: a bare `python3` is
# Apple's /usr/bin/python3 stub here, which on a Mac without the Command Line
# Tools opens an installer dialog and fails, silently skipping this check.
_conda_platform="$(conda info --json | "${CONDA_BASE}/bin/python" -c 'import json, sys; print(json.load(sys.stdin).get("platform", ""))' 2>/dev/null || true)"
if [ -n "$_conda_platform" ] && [ "$_conda_platform" != "osx-arm64" ]; then
    echo "ERROR: conda at ${CONDA_BASE} is a '${_conda_platform}' build; Apple Silicon needs osx-arm64." >&2
    echo "       Install Miniforge (arm64): brew install --cask miniforge" >&2
    exit 1
fi

PKG_MGR="conda"
command -v mamba &> /dev/null && PKG_MGR="mamba"
echo "Using ${PKG_MGR} from ${CONDA_BASE} (platform: ${_conda_platform:-unknown})"
echo "Project directory: $PROJECT_DIR"
echo "Environment name: $ENV_NAME"
echo ""

if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "$RECREATE_ENV" = true ]; then
        # conda refuses to remove the active environment ("Cannot remove
        # current environment"), so a re-run from a Terminal where it is still
        # active (the first of the next steps printed below) stopped here.
        # This only deactivates it inside this script.
        _active_prefix="${CONDA_PREFIX:-}"
        if [ "${_active_prefix##*/}" = "$ENV_NAME" ]; then
            set +u
            conda deactivate || true
            set -u
        fi
        echo "Removing existing '$ENV_NAME' environment..."
        $PKG_MGR env remove -n "$ENV_NAME" -y
    else
        echo "Environment '$ENV_NAME' exists; updating it in place (RECREATE_ENV=false)..."
    fi
fi

if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    $PKG_MGR env update -f "$PROJECT_DIR/environment.yml" -n "$ENV_NAME"
else
    echo "Creating conda environment '$ENV_NAME'..."
    $PKG_MGR env create -f "$PROJECT_DIR/environment.yml" -n "$ENV_NAME" -y
fi

echo ""
echo "Activating environment..."
set +u
conda activate "$ENV_NAME"
set -u
echo "Active environment: ${CONDA_PREFIX:-unknown}"

_py_arch="$(python -c 'import platform; print(platform.machine())')"
if [ "$_py_arch" != "arm64" ]; then
    echo "ERROR: env python is ${_py_arch}, expected arm64. Recreate the env with an arm64 conda." >&2
    exit 1
fi

echo ""
echo "Installing PyTorch (stable, arm64 wheel with MPS support)..."
# The default PyPI index is the right one on macOS: download.pytorch.org's
# cu*/rocm* indexes carry no macOS builds.
pip install --upgrade torch

if [[ "$INSTALL_MATPLOTLIB" = true ]]; then
    echo ""
    echo "Installing plotting dependencies..."
    pip install matplotlib
fi

echo ""
echo "Installing Cython for accelerated preprocessing and search..."
# Pinned to the version that generated the tracked .c files ("Generated by
# Cython 3.2.9"): a rebuild with any other version rewrites all three of them.
pip install "cython==3.2.9"

# Build Cython extensions in place (encoding: ~7x speedup, search: ~130x).
# build_ext --inplace from src/ drops each .so next to its .pyx, which is what
# local_train_mac.sh's staleness guard looks for (Apple clang via Xcode CLT).
echo ""
echo "Building Cython extensions..."
if ! xcode-select -p &> /dev/null; then
    echo "[warn] Xcode Command Line Tools not found; install with: xcode-select --install"
fi
if (cd "$PROJECT_DIR/src" && python setup_cython.py build_ext --inplace); then
    echo "Cython extensions built and installed."
else
    echo "Warning: Cython extension build failed (training will use the slower Python fallback)."
fi

echo ""
echo "=== Verification ==="
echo "Python: $(python --version) ($(python -c 'import platform; print(platform.machine())'))"
# PROJECT_DIR, not a cwd-relative "src": run from any other directory, the
# import below would fail and report every built extension as MISSING.
PROJECT_DIR="$PROJECT_DIR" python - <<'PYCHECK'
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.environ["PROJECT_DIR"], "src"))
import torch

print("PyTorch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(256, 256, device="mps")
    torch.mps.synchronize()
    print("MPS matmul OK:", tuple((x @ x).shape))
for name in ("dama.ai.algorithmic._fast_search", "dama.ai.ml._fast_encode", "dama.ai.ml._fast_score"):
    try:
        importlib.import_module(name)
        print(f"Cython {name}: OK")
    except Exception as exc:  # noqa: BLE001 - report every missing extension
        print(f"Cython {name}: MISSING ({type(exc).__name__}: {exc})")
from PyQt6.QtCore import QT_VERSION_STR
print("Qt GUI available:", QT_VERSION_STR)
PYCHECK

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: conda activate $ENV_NAME"
echo "  2. Run the game:             bash run_game_mac.sh"
echo "  3. Train the ML model:       bash local_train_mac.sh"
