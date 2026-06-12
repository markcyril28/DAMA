#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================
# SERVER SPECS (Dell Server with AMD ROCm):
#   GPU: AMD Instinct MI210 (Aldebaran/MI200)
#   VRAM: 64GB HBM2e
#   Architecture: gfx90a (CDNA2)
#   Driver: amdgpu
#   Compute Platform: ROCm 6.x
#   CPU Threads: 32
#===============================================================================

: << 'SERVER_SPECS'
=================================== Product Info ======================================
GPU[0]          : Card Model:           0x740f
GPU[0]          : Card Vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]          : Card SKU:             D67301V
GPU[0]          : GFX Version:          gfx90a
================================== Memory Info ========================================
GPU[0]          : VRAM Total Memory (B): 68702699520 (~64GB)
GPU[0]          : Temperature (Sensor junction) (C): 35.0
GPU[0]          : Average Graphics Package Power (W): 42.0
==========================================================================================
SERVER_SPECS

# Environment settings
ENV_NAME="train"                  # Name of the conda environment
PYTHON_VERSION="3.11"            # Python version (must match environment.yml)

# PyTorch settings for AMD ROCm
ROCM_VERSION="rocm6.2"           # ROCm version for PyTorch (matches ROCm 6.x platform)

# Optional components
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

# Check if environment already exists
echo "Checking for existing '$ENV_NAME' environment..."
if $PKG_MGR env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Updating..."
    $PKG_MGR env update -f "$PROJECT_DIR/environment.yml" -n "$ENV_NAME"
else
    # Create conda environment
    echo "Creating conda environment '$ENV_NAME'..."
    $PKG_MGR env create -f "$PROJECT_DIR/environment.yml" -n "$ENV_NAME" -y
fi

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
echo "Installing PyTorch with AMD ROCm support..."
echo "Target GPU: AMD Instinct MI210 (gfx90a/CDNA2)"

# Verify ROCm is available
if command -v rocm-smi &> /dev/null; then
    echo "ROCm detected:"
    rocm-smi --showproductname 2>/dev/null || true
else
    echo "Warning: rocm-smi not found. Ensure ROCm 6.x is installed."
fi

# Install PyTorch for ROCm
echo "Installing PyTorch stable with ROCm ${ROCM_VERSION} support..."
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${ROCM_VERSION}"

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
    BUILD_LIB="$PROJECT_DIR/src/build/lib.linux-$(uname -m)-cpython-${PYTHON_VERSION//./}"
    PYVER="${PYTHON_VERSION//./}"
    ARCH="$(uname -m)"
    SUFFIX="cpython-${PYVER}-${ARCH}-linux-gnu.so"
    built_any=false
    # Copy every built extension (_fast_encode, _fast_score, _fast_search, ...)
    while IFS= read -r -d '' so_src; do
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

echo ""
echo "=== Verification ==="
echo "Python: $(python --version)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('ROCm/HIP available:', torch.cuda.is_available())"
python -c "import torch; print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
python -c "import torch; print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import torch; print('ROCm version:', torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')"

# ROCm environment hints
echo ""
echo "=== AMD ROCm Environment ==="
echo "GPU: AMD Instinct MI210 (64GB HBM2e)"
echo "Architecture: gfx90a (CDNA2)"
echo "ROCm Platform: 6.x"
echo ""
echo "Useful ROCm commands:"
echo "  - rocm-smi              : Monitor GPU status"
echo "  - rocm-smi --showuse    : Show GPU utilization"
echo "  - rocm-smi --showmemuse : Show memory usage"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: $PKG_MGR activate $ENV_NAME"
echo "  2. Verify GPU: rocm-smi"
echo "  3. Train the ML model: bash train.sh"
