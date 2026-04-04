#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================
# LOCAL SPECS:
#   GPU: NVIDIA RTX 5050 (8GB VRAM)
#   RAM: 32 GB
#   CPU Cores: 12
#===============================================================================

# =============================================================================
# TRAINING PARAMETERS - Edit these values as needed
# =============================================================================

# -----------------------------------------------------------------------------
# Device Settings
# -----------------------------------------------------------------------------
DEVICE="cuda"                    # Device to train on: "cuda" or "cpu"
NO_AMP=false                     # ENABLED: RTX 5050 has excellent FP16 support, ~2x speedup
COMPILE_MODEL=false              # Disabled: RTX 5050 Triton kernel compilation fails (invalid kernel image)

# -----------------------------------------------------------------------------
# Self-play Settings
# -----------------------------------------------------------------------------
CPU_WORKERS=10                  # 10 of 12 cores for self-play; dataloader uses RAM cache (minimal CPU)
SELFPLAY_GAMES=100               # Fewer games = faster iterations, more frequent model updates
FOCUS_SIDE="both"                # Focus side: "white", "black", or "both"
OPPONENT_FOCUS="both"            # Opponent focus: "ml", "algorithm", or "both"
SELFPLAY_DIFFICULTIES="easy,medium,hard,self"  # Comma-separated difficulties to cycle through
                                               # Options: "easy", "medium", "hard", "self" (AI model)
                                               # Use single value for fixed difficulty, or multiple to alternate
NOISE_PROB=0.15                  # Slightly lower noise for faster convergence
MAX_MOVES_PER_GAME=150           # Shorter games = faster data generation

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
BATCH_SIZE=1024                  # Larger batch with AMP enabled, maximizes GPU utilization
LEARNING_RATE=4e-4               # Higher LR with larger batch for faster convergence
WEIGHT_DECAY=1e-5                # Weight decay for regularization (0 to disable)
GRAD_CLIP_NORM=1.0               # Gradient clipping norm (empty to disable)
TRAIN_STEPS=1000000000           # Total training steps (empty = train indefinitely)
CHECKPOINT_EVERY=50000           # More frequent checkpoints for safety

# -----------------------------------------------------------------------------
# DataLoader Settings
# -----------------------------------------------------------------------------
DATALOADER_WORKERS=4             # Reduced to avoid CPU contention with self-play workers
PIN_MEMORY=true                  # Pin memory for faster GPU transfer

# -----------------------------------------------------------------------------
# Model Testing Settings (ML vs Algorithm)
# -----------------------------------------------------------------------------
TEST_VS_ALGO=true                # Enable periodic testing against algorithm
TEST_EVERY=50000                # Less frequent testing = more time training
TEST_GAMES=30                    # Fewer test games for faster evaluation
TEST_DIFFICULTY="medium"           # Algorithm difficulty for testing: "easy", "medium", "hard"

# -----------------------------------------------------------------------------
# Resume Settings
# -----------------------------------------------------------------------------
RESUME=""                        # Path to checkpoint to resume from (leave empty to start fresh)
RESUME_LATEST=true               # Set to true to resume from latest checkpoint in models/checkpoints/

# -----------------------------------------------------------------------------
# Time-based Stopping
# -----------------------------------------------------------------------------
TRAIN_DURATION=""                # Train for this duration (empty = no limit)
                                 # Examples: "2d" (2 days), "4h" (4 hours), "30m" (30 min), "1d12h" (1 day 12 hours)

# =============================================================================
# END OF PARAMETERS - Do not edit below this line
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Console logging
LOG_DIR="${PROJECT_DIR}/logs/console"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/console_${LOG_TIMESTAMP}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Filipino Dama - ML Training ==="
echo ""

# Verify CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available. Training requires GPU.'" || {
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
echo ""

# Build command arguments
ARGS=""

# Device settings
ARGS+=" --device ${DEVICE}"
if [ "$NO_AMP" = true ]; then
    ARGS+=" --no-amp"
fi
if [ "$COMPILE_MODEL" = true ]; then
    ARGS+=" --compile-model"
fi

# Self-play settings
ARGS+=" --cpu-workers ${CPU_WORKERS}"
ARGS+=" --selfplay-games ${SELFPLAY_GAMES}"
ARGS+=" --focus-side ${FOCUS_SIDE}"
ARGS+=" --opponent-focus ${OPPONENT_FOCUS}"
ARGS+=" --selfplay-difficulties ${SELFPLAY_DIFFICULTIES}"
ARGS+=" --noise-prob ${NOISE_PROB}"
ARGS+=" --max-moves ${MAX_MOVES_PER_GAME}"

# Training settings
ARGS+=" --batch-size ${BATCH_SIZE}"
ARGS+=" --learning-rate ${LEARNING_RATE}"
ARGS+=" --weight-decay ${WEIGHT_DECAY}"
if [ -n "$GRAD_CLIP_NORM" ]; then
    ARGS+=" --grad-clip-norm ${GRAD_CLIP_NORM}"
fi
if [ -n "$TRAIN_STEPS" ] && [ "$TRAIN_STEPS" -gt 0 ] 2>/dev/null; then
    ARGS+=" --train-steps ${TRAIN_STEPS}"
fi
ARGS+=" --checkpoint-every ${CHECKPOINT_EVERY}"

# DataLoader settings
ARGS+=" --dataloader-workers ${DATALOADER_WORKERS}"
if [ "$PIN_MEMORY" = true ]; then
    ARGS+=" --pin-memory"
fi

# Model testing settings
if [ "$TEST_VS_ALGO" = true ]; then
    ARGS+=" --test-vs-algo"
    ARGS+=" --test-every ${TEST_EVERY}"
    ARGS+=" --test-games ${TEST_GAMES}"
    ARGS+=" --test-difficulty ${TEST_DIFFICULTY}"
fi

# Resume settings
if [ -n "$RESUME" ]; then
    ARGS+=" --resume ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    ARGS+=" --resume-latest"
fi

# Time-based stopping
if [ -n "$TRAIN_DURATION" ]; then
    ARGS+=" --train-duration ${TRAIN_DURATION}"
fi

echo "Training parameters:"
echo ""
echo "  [Device Settings]"
echo "    Device:            ${DEVICE}"
echo "    Mixed Precision:   $([ "$NO_AMP" = true ] && echo "disabled" || echo "enabled")"
echo "    Model Compile:     $([ "$COMPILE_MODEL" = true ] && echo "enabled" || echo "disabled")"
echo ""
echo "  [Self-play Settings]"
echo "    CPU Workers:       ${CPU_WORKERS}"
echo "    Self-play Games:   ${SELFPLAY_GAMES}"
echo "    Focus Side:        ${FOCUS_SIDE}"
echo "    Opponent Focus:    ${OPPONENT_FOCUS}"
echo "    Difficulties:      ${SELFPLAY_DIFFICULTIES} (cycling)"
echo "    Noise Probability: ${NOISE_PROB}"
echo "    Max Moves/Game:    ${MAX_MOVES_PER_GAME}"
echo ""
echo "  [Training Settings]"
echo "    Batch Size:        ${BATCH_SIZE}"
echo "    Learning Rate:     ${LEARNING_RATE}"
echo "    Weight Decay:      ${WEIGHT_DECAY}"
if [ -n "$GRAD_CLIP_NORM" ]; then
    echo "    Gradient Clip:     ${GRAD_CLIP_NORM}"
else
    echo "    Gradient Clip:     (disabled)"
fi
if [ -n "$TRAIN_STEPS" ] && [ "$TRAIN_STEPS" -gt 0 ] 2>/dev/null; then
    echo "    Train Steps:       ${TRAIN_STEPS}"
else
    echo "    Train Steps:       (indefinite)"
fi
echo "    Checkpoint:        every ${CHECKPOINT_EVERY} steps"
echo ""
echo "  [DataLoader Settings]"
echo "    Workers:           ${DATALOADER_WORKERS}"
echo "    Pin Memory:        $([ "$PIN_MEMORY" = true ] && echo "enabled" || echo "disabled")"
echo ""
echo "  [Model Testing]"
if [ "$TEST_VS_ALGO" = true ]; then
    echo "    Test vs Algorithm: enabled"
    echo "    Test Frequency:    every ${TEST_EVERY} steps"
    echo "    Test Games:        ${TEST_GAMES}"
    echo "    Test Difficulty:   ${TEST_DIFFICULTY}"
else
    echo "    Test vs Algorithm: disabled"
fi
echo ""
echo "  [Session]"
if [ -n "$TRAIN_DURATION" ]; then
    echo "    Train Duration:    ${TRAIN_DURATION}"
else
    echo "    Train Duration:    (no limit)"
fi
if [ -n "$RESUME" ]; then
    echo "    Resume from:       ${RESUME}"
elif [ "$RESUME_LATEST" = true ]; then
    echo "    Resume from:       latest checkpoint"
else
    echo "    Resume from:       (starting fresh)"
fi
echo ""

exec python -m dama.ai.ml.trainer --config "${PROJECT_DIR}/config/training_config_local.yaml" ${ARGS}
