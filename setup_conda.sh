#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================

# Environment settings
ENV_NAME="dama"                  # Name of the conda environment
PYTHON_VERSION="3.11"            # Python version (must match environment.yml)

# PyTorch settings
FORCE_PYTORCH_NIGHTLY=false      # Force nightly PyTorch even on older GPUs
CUDA_VERSION_STABLE="cu121"      # CUDA version for stable PyTorch (cu118, cu121)
CUDA_VERSION_NIGHTLY="cu128"     # CUDA version for nightly PyTorch

# Optional components
INSTALL_QT_DEPS=true             # Install Qt/X11 system dependencies
INSTALL_MATPLOTLIB=true          # Install matplotlib for plotting

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

echo "=== Dama - Conda Environment Setup ==="
echo ""

# Determine which package manager to use (mamba > conda)
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
    echo "Using mamba"
elif command -v conda &> /dev/null; then
    PKG_MGR="conda"
    echo "Using conda"
else
    echo "ERROR: No package manager found. Please install mamba or conda first."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "Project directory: $PROJECT_DIR"
echo "Environment name: $ENV_NAME"
echo ""

# Remove existing environment if it exists
echo "Checking for existing '$ENV_NAME' environment..."
if $PKG_MGR env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    echo "Removing existing '$ENV_NAME' environment..."
    $PKG_MGR env remove -n "$ENV_NAME" -y
fi

# Create conda environment
echo "Creating conda environment '$ENV_NAME'..."
$PKG_MGR env create -f "$PROJECT_DIR/environment.yml" -n "$ENV_NAME" -y

echo ""
echo "Activating environment..."
if [[ "$PKG_MGR" == "mamba" ]]; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate "$ENV_NAME"
else
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
fi

# Verify we're in the right environment
echo "Active environment: ${CONDA_PREFIX:-unknown}"

echo ""
echo "Installing PyTorch with CUDA support..."
echo "Note: RTX 50-series GPUs require PyTorch nightly builds for full support."

# Try to detect GPU compute capability
COMPUTE_CAP=$(python -c "import subprocess; result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True, text=True); print(result.stdout.strip().split('.')[0] if result.returncode == 0 else '0')" 2>/dev/null || echo "0")

if [[ "$COMPUTE_CAP" -ge "10" ]] || [[ "$FORCE_PYTORCH_NIGHTLY" = true ]]; then
    # RTX 50-series (Blackwell) or newer - use nightly builds
    if [[ "$COMPUTE_CAP" -ge "10" ]]; then
        echo "Detected RTX 50-series or newer GPU (compute capability $COMPUTE_CAP.x)"
    else
        echo "Forcing PyTorch nightly build as requested"
    fi
    echo "Installing PyTorch nightly with CUDA ${CUDA_VERSION_NIGHTLY} support..."
    pip install --pre torch torchvision --index-url "https://download.pytorch.org/whl/nightly/${CUDA_VERSION_NIGHTLY}"
else
    # Older GPUs - use stable release
    echo "Installing PyTorch stable with CUDA ${CUDA_VERSION_STABLE} support..."
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION_STABLE}"
fi

if [[ "$INSTALL_MATPLOTLIB" = true ]]; then
    echo ""
    echo "Installing plotting dependencies..."
    pip install matplotlib
fi

echo ""
echo "Installing Cython for accelerated preprocessing..."
pip install cython

echo ""
echo "Installing nvidia-ml-py for fast GPU monitoring (thermal/utilization)..."
pip install nvidia-ml-py

# Build Cython extensions (encoding: ~7x speedup, search: ~130x speedup)
echo "Building Cython extensions..."
if [[ -f "$PROJECT_DIR/src/setup_cython.py" ]]; then
    (cd "$PROJECT_DIR/src" && python setup_cython.py build --build-base=build build_ext --inplace 2>/dev/null || true)
    # Copy .so files to correct locations (setuptools targets wrong dir with src layout)
    BUILD_LIB="$PROJECT_DIR/src/build/lib.linux-$(uname -m)-cpython-${PYTHON_VERSION//./}"
    PYVER="${PYTHON_VERSION//./}"
    ARCH="$(uname -m)"
    SUFFIX="cpython-${PYVER}-${ARCH}-linux-gnu.so"
    built_any=false
    # Copy every built extension (_fast_encode, _fast_score, _fast_search, ...)
    while IFS= read -r -d '' so_src; do
        # Derive destination from relative path within build dir
        rel="${so_src#$BUILD_LIB/}"
        dest="$PROJECT_DIR/src/$rel"
        mkdir -p "$(dirname "$dest")"
        cp "$so_src" "$dest"
        built_any=true
    done < <(find "$BUILD_LIB" -name "*.${SUFFIX}" -print0 2>/dev/null)
    if [[ "$built_any" = true ]]; then
        echo "Cython extensions built and installed."
    else
        echo "Warning: Cython extension build failed (training will use Python fallback)."
    fi
fi

# Install Qt GUI dependencies (required for PyQt6 on Linux/WSL)
if [[ "$INSTALL_QT_DEPS" = true ]]; then
    echo ""
    echo "Installing Qt GUI dependencies..."
    if [[ "$(uname)" == "Linux" ]]; then
        echo "Detected Linux - installing system Qt dependencies..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq \
                libxcb-xinerama0 \
                libxcb-cursor0 \
                libxcb-xkb1 \
                libxkbcommon-x11-0 \
                libxcb-render-util0 \
                libxcb-icccm4 \
                libxcb-image0 \
                libxcb-keysyms1 \
                libxcb-randr0 \
                libxcb-shape0 \
                libxcb-sync1 \
                libxcb-xfixes0 \
                libegl1 \
                libgl1
            echo "Qt system dependencies installed."
        else
            echo "Warning: apt-get not found. Please install Qt xcb dependencies manually."
        fi
    fi
fi

echo ""
echo "=== Verification ==="
echo "Python: $(python --version)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# WSL detection and hints
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    echo ""
    echo "=== WSL2 Detected ==="
    echo "Distribution: $WSL_DISTRO_NAME"
    echo ""
    echo "GUI Support:"
    echo "  - Windows 11 / recent Windows 10: WSLg should work automatically."
    echo "  - Older Windows 10: Install VcXsrv, run it, then 'export DISPLAY=:0'"
    echo ""
    echo "Qt Troubleshooting:"
    echo "  - If Qt plugin errors occur, the run_game.sh script auto-configures Qt."
    echo "  - The xcb platform is used for X11 display compatibility."
    echo "  - Verify WSLg is working: echo \$WAYLAND_DISPLAY"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: $PKG_MGR activate $ENV_NAME"
echo "  2. Run the game: bash run_game.sh"
echo "  3. Train the ML model: bash train.sh"
