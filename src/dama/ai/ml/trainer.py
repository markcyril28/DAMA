"""Training loop for the move scorer model."""

import os
import sys
import json
import time
import random
import argparse
import tempfile
import warnings
import platform
import threading
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Set multiprocessing start method before any other multiprocessing imports
# 'fork' is faster on Linux but can cause issues with CUDA
# 'spawn' is safer but slower
if platform.system() != 'Windows':
    try:
        mp.set_start_method('fork', force=False)
    except RuntimeError:
        pass  # Already set

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

# Default matmul precision — overridden per config in Trainer.__init__().
# 'medium' allows aggressive use of TF32 tensor cores on Ampere+ GPUs.
torch.set_float32_matmul_precision('medium')

# Enable cuDNN autotuning — finds the fastest convolution algorithm for
# the fixed input shapes (8x8 board, constant batch size with drop_last=True).
# One-time benchmark cost on first forward pass, then faster for all subsequent.
torch.backends.cudnn.benchmark = True

# Allow FP16 accumulation in matmuls for faster tensor core throughput.
# Minimal precision impact for this model's score magnitudes.
try:
    torch.backends.cuda.matmul.allow_fp16_reduction = True
except AttributeError:
    pass  # Older PyTorch versions

# Configure torch.compile CUDAGraph settings to avoid overhead from dynamic shapes
# This prevents recording too many graphs for varying input sizes
try:
    import torch._inductor.config as inductor_config
    inductor_config.triton.cudagraph_skip_dynamic_graphs = True
    inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
    # Silence inductor/dynamo info and warning messages
    import logging
    logging.getLogger("torch._inductor.compile_fx").setLevel(logging.ERROR)
    logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
except (ImportError, AttributeError):
    pass  # Older PyTorch versions may not have this config

# Suppress torch.compile inductor warnings and pynvml deprecation
warnings.filterwarnings('ignore', message='.*TensorFloat32.*')
warnings.filterwarnings('ignore', message='.*max_autotune_gemm.*')
warnings.filterwarnings('ignore', message='.*CUDAGraph.*dynamic shapes.*')
warnings.filterwarnings('ignore', message='.*skipping cudagraphs.*')
warnings.filterwarnings('ignore', message='.*pynvml package is deprecated.*')

from .model import MoveScorerNet, create_model, save_model, load_model
from .replay import ReplayBuffer
from .selfplay import (
    _play_game_worker_algo_vs_algo,
    _play_games_batch_worker_algo, _play_games_batch_worker_full,
)
from .dataset import (
    create_dataloader, create_dataloader_from_dataset, prepare_training_data,
    CachedTensorDataset, CUDAPrefetcher, FastBatchIterator,
)
from .stats_collector import StatsCollector


def _make_compiled_fwd_loss(model, compile_mode):
    """Build and compile a fused forward_padded + loss function.

    Keeping forward and loss in one compiled graph lets the inductor fuse
    kernels across the boundary and capture everything in a single CUDAGraph,
    eliminating per-kernel launch overhead from separate forward and loss calls.

    NaN/Inf loss is replaced with 0.0 inside the graph via nan_to_num, so the
    caller never needs ``torch.isfinite(loss)`` (which forces a CUDA sync).
    Zero loss produces zero gradients; the optimizer step becomes a near-no-op.
    """
    def _fwd_loss(boards, move_features, move_counts, targets, reward_weights):
        # Forward — model.forward_padded is inlined by the compiler
        scores = model.forward_padded(boards, move_features, move_counts)

        # Loss — inlined from _compute_loss_padded for single-graph capture.
        # move_counts/targets are int32 (sufficient for 0-32 range);
        # cast to int64 once for gather (which requires LongTensor).
        no_moves = move_counts == 0
        stable = scores.masked_fill(no_moves.unsqueeze(1), 0.0)
        log_probs = nn.functional.log_softmax(stable.float(), dim=1)
        log_probs = torch.clamp(log_probs, min=-100.0)
        max_m = scores.shape[1]
        safe_tgt = targets.long().clamp(0, max_m - 1).unsqueeze(1)
        chosen_lp = log_probs.gather(1, safe_tgt).squeeze(1)
        valid = (move_counts > 0) & (targets >= 0) & (targets < move_counts)
        w = reward_weights.float() * valid.float()
        tw = w.sum().clamp(min=1.0)
        loss = -(chosen_lp * w).sum() / tw

        # Sanitize NaN/Inf loss inside the graph to eliminate the per-step
        # ``not torch.isfinite(loss)`` CUDA sync in the training loop.
        # Zero loss → zero gradients → optimizer applies only weight decay
        # (negligible for the rare NaN batch).
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss, scores.detach()

    return torch.compile(_fwd_loss, mode=compile_mode, fullgraph=True)


def parse_duration(duration_str: Optional[str]) -> Optional[datetime]:
    """Parse duration string and return the stop time.
    
    Supported formats:
    - Nd or Ndays (e.g., 2d, 2days) - N days
    - Nh or Nhours (e.g., 4h, 4hours) - N hours  
    - Nm or Nmin (e.g., 30m, 30min) - N minutes
    - Combined (e.g., 1d12h, 2d6h30m) - multiple units
    """
    if not duration_str:
        return None
    
    duration_str = duration_str.strip().lower()
    
    import re
    from datetime import timedelta
    
    total_seconds = 0
    
    # Match patterns like 2d, 4h, 30m
    patterns = [
        (r'(\d+)\s*d(?:ays?)?', 86400),   # days
        (r'(\d+)\s*h(?:ours?)?', 3600),    # hours
        (r'(\d+)\s*m(?:in(?:utes?)?)?', 60),  # minutes
        (r'(\d+)\s*s(?:ec(?:onds?)?)?', 1),   # seconds
    ]
    
    matched_any = False
    for pattern, multiplier in patterns:
        for match in re.finditer(pattern, duration_str):
            total_seconds += int(match.group(1)) * multiplier
            matched_any = True
    
    if not matched_any:
        # Try parsing as plain number (assume hours)
        try:
            hours = float(duration_str)
            total_seconds = int(hours * 3600)
            matched_any = True
        except ValueError:
            pass
    
    if not matched_any or total_seconds <= 0:
        raise ValueError(f"Could not parse duration '{duration_str}'. Use format like: 2d, 4h, 30m, 1d12h, 2days, etc.")
    
    stop_time = datetime.now() + timedelta(seconds=total_seconds)
    return stop_time


def _parse_rest_duration(duration_str: str) -> int:
    """Parse a duration string like '5m', '1m30s', '10m' into total seconds."""
    import re
    duration_str = duration_str.strip().lower()
    total = 0
    for match in re.finditer(r'(\d+)\s*([hms])', duration_str):
        value, unit = int(match.group(1)), match.group(2)
        if unit == 'h':
            total += value * 3600
        elif unit == 'm':
            total += value * 60
        elif unit == 's':
            total += value
    if total == 0:
        # Fallback: try as plain integer (assume minutes)
        try:
            total = int(float(duration_str)) * 60
        except ValueError:
            total = 300  # default 5 minutes
    return total


def _eval_expr(value):
    """Evaluate simple arithmetic expressions from YAML config values.

    YAML doesn't support math, so ``8192*2`` is parsed as the string
    ``"8192*2"`` instead of ``16384``.  This helper safely evaluates
    expressions that contain only integers, floats, and the operators
    ``+ - * / // **`` (plus parentheses and whitespace).

    Non-string values and strings that aren't arithmetic expressions are
    returned unchanged.
    """
    if not isinstance(value, str):
        return value
    import re
    # Only allow digits, decimal points, operators, parens, whitespace
    if not re.fullmatch(r'[\d\s\+\-\*\/\.\(\)]+', value):
        return value
    try:
        result = eval(value, {"__builtins__": {}})  # noqa: S307
        # Preserve int when possible (e.g. 8192*2 -> 16384, not 16384.0)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return result
    except Exception:
        return value


_STATS_HISTORY_CAP = 10000  # Max entries per history list (trim oldest on overflow)


@dataclass
class TrainingStats:
    """Training statistics tracking."""
    start_time: str = ""
    end_time: str = ""
    total_steps: int = 0
    epochs_completed: int = 0
    best_loss: float = float('inf')
    best_val_loss: float = float('inf')

    # History lists (stored as lists for JSON serialization)
    loss_history: list = None
    val_loss_history: list = None
    lr_history: list = None
    gpu_mem_history: list = None
    step_times: list = None
    test_history: list = None  # Model vs algorithm test results
    
    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []
        if self.val_loss_history is None:
            self.val_loss_history = []
        if self.lr_history is None:
            self.lr_history = []
        if self.gpu_mem_history is None:
            self.gpu_mem_history = []
        if self.step_times is None:
            self.step_times = []
        if self.test_history is None:
            self.test_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns shallow copies of all mutable lists so the dict is safe to
        pass to a background thread (e.g., for async checkpoint writes) while
        the main thread continues to append to the originals.
        """
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_steps': self.total_steps,
            'epochs_completed': self.epochs_completed,
            'best_loss': self.best_loss,
            'best_val_loss': self.best_val_loss,
            'loss_history': list(self.loss_history),
            'val_loss_history': list(self.val_loss_history),
            'lr_history': list(self.lr_history),
            'gpu_mem_history': list(self.gpu_mem_history),
            'step_times': list(self.step_times),
            'test_history': list(self.test_history),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingStats':
        """Create from dictionary."""
        return cls(
            start_time=data.get('start_time', ''),
            end_time=data.get('end_time', ''),
            total_steps=data.get('total_steps', 0),
            epochs_completed=data.get('epochs_completed', 0),
            best_loss=data.get('best_loss', float('inf')),
            best_val_loss=data.get('best_val_loss', float('inf')),
            loss_history=data.get('loss_history', []),
            val_loss_history=data.get('val_loss_history', []),
            lr_history=data.get('lr_history', []),
            gpu_mem_history=data.get('gpu_mem_history', []),
            step_times=data.get('step_times', []),
            test_history=data.get('test_history', []),
        )


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Device settings
    device: str = 'cuda'
    amp: bool = True
    amp_dtype: str = 'float16'  # 'float16' or 'bfloat16' (bfloat16 recommended for AMD MI210)
    compile_model: bool = True
    compile_mode: str = 'reduce-overhead'
    matmul_precision: str = 'medium'  # 'highest', 'high', 'medium' — medium enables TF32

    # Model architecture settings
    model_channels: int = 64       # CNN channels in ResidualBlocks
    model_blocks: int = 4          # Number of residual blocks
    model_embedding: int = 128     # Board embedding size
    model_hidden: int = 64         # MoveScorer MLP hidden size

    # Self-play settings
    cpu_workers: int = field(default_factory=lambda: max(2, (os.cpu_count() or 2)))
    selfplay_games: int = 500
    selfplay_difficulties: list = field(default_factory=lambda: ['medium'])  # List of difficulties to cycle
    selfplay_focus_side: str = 'both'  # white, black, both
    selfplay_opponent_focus: str = 'both'  # ml, algorithm, both
    selfplay_noise_prob: float = 0.1  # Probability of random move for exploration
    selfplay_max_moves: int = 200  # Maximum moves per game
    pipeline_mode: str = 'simultaneous'  # 'simultaneous' or 'alternate'
    max_stale_epochs: int = 0  # Max epochs on unchanged data before yielding (0 = unlimited)

    # Algo-vs-algo data generation (pure algorithmic games as training data)
    algo_vs_algo_enabled: bool = False
    algo_vs_algo_games: int = 100  # Games per self-play epoch
    algo_vs_algo_difficulties: list = field(default_factory=lambda: ['easy', 'medium', 'hard'])

    # Training settings
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5  # Weight decay for regularization
    train_steps: int = 999999999  # Default to essentially indefinite
    checkpoint_every: int = 1000

    # Reward scoring mode:
    #   'scoring'  - always use the scoring system reward weights
    #   'none'     - never use scoring (uniform weights, classic behavior)
    #   'cycle'    - alternate epochs: odd epochs use scoring, even epochs don't
    reward_mode: str = 'cycle'

    # Gradient accumulation: effective batch = batch_size * gradient_accumulation_steps.
    # Allows larger effective batches on memory-constrained hardware without
    # increasing VRAM usage.  Optimizer steps every N mini-batches.
    gradient_accumulation_steps: int = 1

    # Learning rate scheduler
    lr_scheduler_enabled: bool = False
    lr_scheduler_type: str = 'cosine_warm_restarts'  # CosineAnnealingWarmRestarts
    lr_scheduler_T0: int = 500       # Steps for first cosine cycle
    lr_scheduler_T_mult: int = 2     # Cycle length multiplier after each restart
    lr_scheduler_eta_min: float = 1e-5  # Minimum LR
    lr_warmup_steps: int = 0         # Linear warmup from 0 to base LR over N steps

    # Value head / TD learning
    value_head_enabled: bool = False
    value_head_hidden: int = 128
    value_weight: float = 0.5  # Weight of value loss relative to policy loss

    # DataLoader settings
    dataloader_workers: int = field(default_factory=lambda: max(2, (os.cpu_count() or 2)))
    pin_memory: bool = True
    ram_cache_enabled: bool = True
    ram_cache_threshold_gb: float = 8.0
    replay_max_entries: int = 100000  # Max entries to sample from replay buffer per epoch
    clear_replay_after_load: bool = False  # Delete replay files after loading into memory
    max_moves_per_sample: int = 32  # Padding width for move features (max legal moves ~20)

    # Stability
    grad_clip_norm: Optional[float] = 1.0

    # Model testing settings
    test_vs_algo: bool = False
    test_every: int = 5000  # Run tests every N steps
    test_games: int = 50  # Number of test games per evaluation
    test_difficulty: str = 'medium'

    # Statistics collection settings
    stats_enabled: bool = True
    stats_record_every: int = 10        # Record loss/grad every N steps
    stats_system_every: int = 500       # Record GPU/CPU metrics every N steps
    stats_model_health_every: int = 2000  # Record param norms every N steps
    stats_score_dist_every: int = 50    # Record score distribution every N steps
    stats_buffer_size: int = 50000      # Ring buffer size for high-frequency metrics
    stats_flush_every: int = 5000       # Flush incremental stats to disk
    stats_output_dir: str = 'logs/stats'

    # Paths
    checkpoint_dir: str = 'models/checkpoints'
    latest_path: str = 'models/latest.pt'
    replay_dir: str = 'data/replay'
    log_dir: str = 'logs'
    stats_file: str = 'models/training_stats.json'

    # Thermal protection
    thermal_enabled: bool = False
    thermal_temp_limit_c: int = 90        # Temperature threshold in Celsius
    thermal_rest_seconds: int = 300       # Rest duration in seconds (parsed from duration string)
    thermal_check_every: int = 30         # Check temperature every N seconds

    # Resume
    resume: Optional[str] = None

    # Time-based stopping
    stop_time: Optional[datetime] = None  # Calculated from train_duration


class Trainer:
    """
    Trainer for the move scorer model.

    Supports:
    - Self-play data generation
    - Imitation learning from algorithmic AI
    - Checkpoint saving and resume
    - Mixed precision training
    - IPC control for GUI integration
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Set up device
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("ERROR: CUDA requested but not available.")
            print("\nTroubleshooting:")
            print("  1. Ensure NVIDIA GPU driver is installed on Windows")
            print("  2. Run 'nvidia-smi' to verify GPU access in WSL")
            print("  3. Reinstall PyTorch with CUDA support")
            sys.exit(1)

        self.device = torch.device(config.device)
        # Apply config-driven matmul precision (overrides module-level default)
        torch.set_float32_matmul_precision(config.matmul_precision)
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Set thread count for CPU operations — minimize to free cores for self-play.
        # With GPU training (compiled forward+loss, GPU-resident data), the training
        # thread does almost zero CPU work. 1 thread is sufficient for the rare
        # CPU-side tensor ops (loss casting, index creation). Every extra thread here
        # steals a core from self-play workers that need them.
        _total_cores = os.cpu_count() or 1
        if self.device.type == 'cuda':
            cpu_threads = 1  # GPU handles all compute; 1 CPU thread for housekeeping
        else:
            cpu_threads = max(4, min(64, _total_cores // 2))
        torch.set_num_threads(cpu_threads)
        # Inter-op threads for parallel independent ops (e.g. data loading + compute)
        interop_threads = 1 if self.device.type == 'cuda' else max(2, min(4, _total_cores // 16))
        try:
            torch.set_num_interop_threads(interop_threads)
        except Exception:
            pass
        print(f"CPU threads: {cpu_threads} intra-op, {interop_threads} inter-op (of {_total_cores} cores)")

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = create_model(
            embedding_size=config.model_embedding,
            num_blocks=config.model_blocks,
            hidden_size=config.model_hidden,
            channels=config.model_channels,
            value_head_enabled=config.value_head_enabled,
            value_head_hidden=config.value_head_hidden,
        )
        self.model.to(self.device)
        # NHWC memory format for conv weights — cuDNN selects faster kernels
        if self.device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)

        # Determine AMP dtype (must come before optimizer — fused AdamW is
        # incompatible with GradScaler's grad_scale/found_inf kwargs).
        self.amp_dtype = torch.float16
        if config.amp:
            if config.amp_dtype == 'bfloat16':
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                    print("Using BFloat16 mixed precision (no GradScaler needed)")
                else:
                    print("BFloat16 not supported, falling back to Float16")
                    self.amp_dtype = torch.float16
            else:
                print("Using Float16 mixed precision with GradScaler")

        # GradScaler only needed for float16, not bfloat16
        _needs_scaler = config.amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler() if _needs_scaler else None

        # Use fused AdamW for faster training on CUDA (PyTorch 2.0+).
        # Fused AdamW runs the entire optimizer step in a single CUDA kernel
        # (vs ~N kernels for N parameter groups), reducing launch overhead.
        # Disabled when GradScaler is active: the fused kernel can fall back to
        # _single_tensor_adam which rejects GradScaler's grad_scale/found_inf.
        use_fused = False
        if config.device == 'cuda' and not _needs_scaler:
            try:
                _test_opt = optim.AdamW([torch.zeros(1, device='cuda')], fused=True)
                del _test_opt
                use_fused = True
            except Exception:
                pass
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=use_fused,
        )
        if use_fused:
            print("Using fused AdamW optimizer for faster training")

        # Learning rate scheduler with optional warmup
        self.scheduler = None
        if config.lr_scheduler_enabled:
            main_scheduler = None
            if config.lr_scheduler_type == 'cosine_warm_restarts':
                main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=config.lr_scheduler_T0,
                    T_mult=config.lr_scheduler_T_mult,
                    eta_min=config.lr_scheduler_eta_min,
                )
                sched_desc = (f"CosineAnnealingWarmRestarts "
                              f"(T_0={config.lr_scheduler_T0}, T_mult={config.lr_scheduler_T_mult}, "
                              f"eta_min={config.lr_scheduler_eta_min})")
            else:
                print(f"Warning: Unknown LR scheduler type '{config.lr_scheduler_type}'")

            if main_scheduler is not None:
                if config.lr_warmup_steps > 0:
                    warmup_scheduler = optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1e-3,  # start at 0.1% of base LR
                        end_factor=1.0,
                        total_iters=config.lr_warmup_steps,
                    )
                    self.scheduler = optim.lr_scheduler.SequentialLR(
                        self.optimizer,
                        schedulers=[warmup_scheduler, main_scheduler],
                        milestones=[config.lr_warmup_steps],
                    )
                    print(f"LR Scheduler: {config.lr_warmup_steps}-step warmup → {sched_desc}")
                else:
                    self.scheduler = main_scheduler
                    print(f"LR Scheduler: {sched_desc}")

        self.replay_buffer = ReplayBuffer(config.replay_dir)

        # Training state
        self.step = 0
        self.best_loss = float('inf')
        self.epoch = 0

        # Control flags for IPC
        self._paused = False
        self._stopped = False

        # Background self-play state
        self._bg_selfplay_thread: Optional[threading.Thread] = None
        self._bg_selfplay_entries: Optional[list] = None
        self._bg_selfplay_dataset: Optional[CachedTensorDataset] = None
        self._bg_selfplay_incremental: Optional[CachedTensorDataset] = None
        self._bg_selfplay_lock = threading.Lock()
        self._last_selfplay_dicts: Optional[list] = None
        self._last_selfplay_preprocessed: Optional[CachedTensorDataset] = None
        # Event signalled by the background thread when new data is ready.
        # Replaces time.sleep(0.5) polling in the stale-data wait loop with
        # zero-latency wakeup (~0ms vs up to 500ms).
        self._data_ready_event = threading.Event()

        # Background checkpoint write thread — tracked to avoid concurrent
        # disk I/O from overlapping checkpoints.
        self._checkpoint_thread: Optional[threading.Thread] = None

        # Padded training path flag (set when CachedTensorDataset is used)
        self._use_padded = False

        # Thermal protection state
        self._last_thermal_check: float = 0.0  # time.time() of last check

        # Training statistics
        self.stats = TrainingStats()
        self._load_stats()
        
        # Timing for step tracking
        self._last_step_time = None

        # Enhanced statistics collector
        self.stats_collector: Optional[StatsCollector] = None
        if config.stats_enabled:
            self.stats_collector = StatsCollector(
                output_dir=config.stats_output_dir,
                buffer_size=config.stats_buffer_size,
                flush_every=config.stats_flush_every,
            )
            # Snapshot the config for the session report
            self.stats_collector.set_config_snapshot({
                'device': config.device,
                'amp': config.amp,
                'amp_dtype': config.amp_dtype,
                'compile_model': config.compile_model,
                'compile_mode': config.compile_mode,
                'model_channels': config.model_channels,
                'model_blocks': config.model_blocks,
                'model_embedding': config.model_embedding,
                'model_hidden': config.model_hidden,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'grad_clip_norm': config.grad_clip_norm,
                'train_steps': config.train_steps,
                'checkpoint_every': config.checkpoint_every,
                'cpu_workers': config.cpu_workers,
                'selfplay_games': config.selfplay_games,
                'selfplay_difficulties': config.selfplay_difficulties,
                'selfplay_noise_prob': config.selfplay_noise_prob,
                'selfplay_max_moves': config.selfplay_max_moves,
                'pipeline_mode': config.pipeline_mode,
                'algo_vs_algo_enabled': config.algo_vs_algo_enabled,
                'algo_vs_algo_games': config.algo_vs_algo_games,
                'algo_vs_algo_difficulties': config.algo_vs_algo_difficulties,
                'dataloader_workers': config.dataloader_workers,
                'ram_cache_enabled': config.ram_cache_enabled,
                'ram_cache_threshold_gb': config.ram_cache_threshold_gb,
                'test_vs_algo': config.test_vs_algo,
                'test_every': config.test_every,
                'test_games': config.test_games,
                'test_difficulty': config.test_difficulty,
                'lr_scheduler_enabled': config.lr_scheduler_enabled,
                'lr_scheduler_type': config.lr_scheduler_type,
                'lr_scheduler_T0': config.lr_scheduler_T0,
                'lr_scheduler_T_mult': config.lr_scheduler_T_mult,
                'lr_scheduler_eta_min': config.lr_scheduler_eta_min,
                'value_head_enabled': config.value_head_enabled,
                'value_head_hidden': config.value_head_hidden,
                'value_weight': config.value_weight,
            })

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(config.log_dir) / f"train_{timestamp}.jsonl"

        # Resume if specified
        if config.resume:
            self._load_checkpoint(config.resume)

        # Compile a fused forward+loss function for maximum performance.
        # Fusing forward and loss in one compiled graph lets the inductor fuse
        # kernels and capture everything in a single CUDAGraph.
        # Falls back to model-only compilation if fused compile fails.
        self._compiled_fwd_loss = None
        if self.config.compile_model and hasattr(torch, "compile"):
            # Guard: Triton's autotuner segfaults on small GPUs (e.g. RTX 5050)
            # when reduce-overhead/max-autotune modes trigger CUDAGraph recording.
            # Fall back to 'default' mode which still provides inductor operator
            # fusion without CUDAGraphs — significant speedup vs eager mode.
            _MIN_SM_COUNT = 64  # safe threshold; MI210=104, A100=108, RTX 4090=128
            if self.device.type == 'cuda':
                _props = torch.cuda.get_device_properties(self.device)
                _sm_count = _props.multi_processor_count
                if _sm_count < _MIN_SM_COUNT and self.config.compile_mode in (
                        'reduce-overhead', 'max-autotune'):
                    print(f"GPU has {_sm_count} SMs (< {_MIN_SM_COUNT}): "
                          f"using torch.compile mode='default' instead of "
                          f"'{self.config.compile_mode}' "
                          f"(avoids Triton autotuner crash)")
                    sys.stdout.flush()
                    self.config.compile_mode = 'default'
                _warmup_bs = self.config.batch_size
                _warmup_max_moves = self.config.max_moves_per_sample
                _wb = torch.randn(_warmup_bs, 5, 8, 8, device=self.device)
                _wm = torch.randn(_warmup_bs, _warmup_max_moves, 8, device=self.device)
                _wc = torch.full((_warmup_bs,), 4, dtype=torch.int32, device=self.device)

                # Stage 1: try fused forward+loss compilation (only useful
                # without value head — value head needs a different fwd path)
                if not config.value_head_enabled:
                    try:
                        print(f"Enabling torch.compile for fused forward+loss "
                              f"(mode={self.config.compile_mode}, fullgraph=True)...")
                        sys.stdout.flush()
                        self._compiled_fwd_loss = _make_compiled_fwd_loss(
                            self.model, self.config.compile_mode)
                        print("  Running compile warmup (fused forward+loss)...")
                        sys.stdout.flush()
                        _wt = torch.zeros(_warmup_bs, dtype=torch.int32, device=self.device)
                        _wr = torch.ones(_warmup_bs, dtype=torch.float32, device=self.device)
                        with torch.no_grad():
                            if self.config.amp:
                                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                                    self._compiled_fwd_loss(_wb, _wm, _wc, _wt, _wr)
                            else:
                                self._compiled_fwd_loss(_wb, _wm, _wc, _wt, _wr)
                        del _wt, _wr
                        print("torch.compile warmup OK — compiled fused forward+loss active")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"Fused compile failed ({e}), trying model-only compile...")
                        sys.stdout.flush()
                        self._compiled_fwd_loss = None

                # Stage 2: model-only compilation — used as fallback when
                # fused compile fails, or as primary when value_head is
                # enabled (fused compile doesn't support the value path).
                if self._compiled_fwd_loss is None:
                    try:
                        _stage2_label = ("value_head enabled"
                                         if config.value_head_enabled
                                         else "fused compile unavailable")
                        print(f"Enabling torch.compile for model forward "
                              f"(mode={self.config.compile_mode}, {_stage2_label})...")
                        sys.stdout.flush()
                        compiled_model = torch.compile(
                            self.model, mode=self.config.compile_mode, fullgraph=True)
                        print("  Running compile warmup (model forward only)...")
                        sys.stdout.flush()
                        _total_moves = int(_wc.sum().item())
                        _wm_flat = torch.randn(_total_moves, _wm.shape[-1], device=self.device)
                        with torch.no_grad():
                            if self.config.amp:
                                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                                    compiled_model.forward_padded(_wb, _wm, _wc)
                                    compiled_model(_wb, _wm_flat, _wc)
                                    if config.value_head_enabled:
                                        compiled_model.forward_padded_with_value(
                                            _wb, _wm, _wc)
                            else:
                                compiled_model.forward_padded(_wb, _wm, _wc)
                                compiled_model(_wb, _wm_flat, _wc)
                                if config.value_head_enabled:
                                    compiled_model.forward_padded_with_value(
                                        _wb, _wm, _wc)
                        del _wm_flat
                        self.model = compiled_model
                        print("torch.compile warmup OK — compiled model active (loss uncompiled)")
                        sys.stdout.flush()
                    except Exception as e2:
                        print(f"torch.compile failed ({e2}), falling back to eager mode")
                        sys.stdout.flush()

                del _wb, _wm, _wc
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Thermal protection
    # ------------------------------------------------------------------

    # Lazily-initialised NVML handle — avoids subprocess per temp check
    _nvml_handle = None
    _nvml_failed = False

    @classmethod
    def _ensure_nvml(cls) -> bool:
        """One-time NVML init; returns True if handle is available."""
        if cls._nvml_handle is not None:
            return True
        if cls._nvml_failed:
            return False
        try:
            import pynvml
            pynvml.nvmlInit()
            cls._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except Exception:
            cls._nvml_failed = True
            return False

    @classmethod
    def _get_gpu_temperature(cls) -> Optional[float]:
        """Read GPU temperature in Celsius. Prefers NVML (~0.1 ms) over
        subprocess nvidia-smi (~50 ms)."""
        # Fast path: NVML
        if cls._ensure_nvml():
            try:
                import pynvml
                return float(pynvml.nvmlDeviceGetTemperature(
                    cls._nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass

        # Fallback: subprocess
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split('\n')[0])
        except Exception:
            pass
        # AMD ROCm
        try:
            result = subprocess.run(
                ['rocm-smi', '--showtemp', '--json'],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                import json as _json
                data = _json.loads(result.stdout)
                for card in data.values():
                    if isinstance(card, dict):
                        for key in ('Temperature (Sensor edge) (C)',
                                    'edge temperature', 'temperature'):
                            if key in card:
                                return float(str(card[key]).rstrip('Cc °'))
        except Exception:
            pass
        return None

    @staticmethod
    def _get_cpu_temperature() -> Optional[float]:
        """Read highest CPU core temperature via psutil (Linux)."""
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            # Check common sensor groups: coretemp (Intel), k10temp (AMD), etc.
            for group_name in ('coretemp', 'k10temp', 'zenpower', 'cpu_thermal',
                               'acpitz', 'thinkpad'):
                if group_name in temps:
                    return max(s.current for s in temps[group_name])
            # Fallback: return max across all sensors
            all_temps = [s.current for entries in temps.values() for s in entries
                         if s.current > 0]
            return max(all_temps) if all_temps else None
        except Exception:
            return None

    def _check_thermal_and_rest(self) -> None:
        """If thermal protection is enabled, check temps and sleep if too hot.

        Called periodically from the training loop. Uses wall-clock gating
        so we don't shell out to nvidia-smi on every batch.
        """
        if not self.config.thermal_enabled:
            return

        now = time.time()
        if now - self._last_thermal_check < self.config.thermal_check_every:
            return
        self._last_thermal_check = now

        limit = self.config.thermal_temp_limit_c
        gpu_temp = self._get_gpu_temperature()
        cpu_temp = self._get_cpu_temperature()

        hot_source = None
        hot_temp = None
        if gpu_temp is not None and gpu_temp >= limit:
            hot_source, hot_temp = 'GPU', gpu_temp
        elif cpu_temp is not None and cpu_temp >= limit:
            hot_source, hot_temp = 'CPU', cpu_temp

        if hot_source is None:
            return

        rest_secs = self.config.thermal_rest_seconds
        rest_min = rest_secs / 60
        print(f"\n{'=' * 50}")
        print(f"THERMAL PROTECTION: {hot_source} temperature is {hot_temp:.0f}°C "
              f"(limit: {limit}°C)")
        print(f"Pausing training for {rest_min:.1f} minutes to cool down...")
        print(f"{'=' * 50}")
        sys.stdout.flush()

        # Sleep in small increments so we can still respond to stop signals
        end_time = time.time() + rest_secs
        while time.time() < end_time:
            if self._stopped:
                break
            time.sleep(min(5.0, end_time - time.time()))

        # Log temps after resting
        gpu_after = self._get_gpu_temperature()
        cpu_after = self._get_cpu_temperature()
        print(f"Thermal rest complete. Temps now — "
              f"GPU: {gpu_after or 'N/A'}°C, CPU: {cpu_after or 'N/A'}°C")
        print("Resuming training...")
        sys.stdout.flush()
        self._last_thermal_check = time.time()

    def _has_non_finite_tensors(self) -> bool:
        """Check if model parameters or buffers contain NaN/Inf.

        Uses a single batched check (flatten + cat + isfinite) to avoid
        N separate CUDA syncs (one per parameter). One sync for the whole
        model instead of ~40.
        """
        all_tensors = [p.data.flatten() for p in self.model.parameters()]
        all_tensors.extend(b.flatten() for b in self.model.buffers()
                           if b is not None and b.numel() > 0)
        if not all_tensors:
            return False
        combined = torch.cat(all_tensors)
        if torch.isfinite(combined).all():
            return False
        # Detailed report: identify which parameter is bad (only on failure)
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"  Non-finite parameter detected: {name}")
                return True
        for name, buf in self.model.named_buffers():
            if buf is not None and buf.numel() > 0 and not torch.isfinite(buf).all():
                print(f"  Non-finite buffer detected: {name}")
                return True
        # Combined check already found non-finite — individual checks may miss
        # due to GPU state changes between reads.  Trust the batched result.
        return True

    def _repair_batchnorm_stats(self) -> bool:
        """Check and repair corrupted BatchNorm running stats.

        FP16 overflow during a forward pass can produce NaN activations that
        corrupt BatchNorm running_mean / running_var.  The GradScaler catches
        NaN *gradients* and skips the optimizer step, but running stats are
        updated in the forward pass — before any loss/gradient check.  Once
        corrupted, every subsequent forward pass outputs NaN, creating a
        cascade that no amount of batch-skipping can break.

        This method detects corrupted stats and resets them to defaults
        (mean=0, var=1), allowing the next forward pass to re-estimate
        clean statistics from the batch.
        """
        repaired = False
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and module.track_running_stats:
                rm = module.running_mean
                rv = module.running_var
                if (rm is not None and not torch.isfinite(rm).all()) or \
                   (rv is not None and not torch.isfinite(rv).all()):
                    module.running_mean.zero_()
                    module.running_var.fill_(1.0)
                    module.num_batches_tracked.zero_()
                    repaired = True
        return repaired

    def _reset_model_state(self, reason: str) -> None:
        """Reset model/optimizer if checkpoint is corrupt."""
        print(f"WARNING: {reason}. Resetting model and optimizer state.")
        self.model = create_model(
            embedding_size=self.config.model_embedding,
            num_blocks=self.config.model_blocks,
            hidden_size=self.config.model_hidden,
            channels=self.config.model_channels,
            value_head_enabled=self.config.value_head_enabled,
            value_head_hidden=self.config.value_head_hidden,
        )
        self.model.to(self.device)
        if self.device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
        # Invalidate compiled function — it captured the old model's parameters.
        # Next train_epoch() will use eager mode (recompilation would require
        # re-running the full compile+warmup sequence).
        self._compiled_fwd_loss = None
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = GradScaler(init_scale=2**10) if (self.config.amp and self.amp_dtype == torch.float16) else None
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')

    def _load_checkpoint(self, path: str) -> None:
        """Load training state from checkpoint."""
        print(f"Resuming from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Handle state_dict from compiled models (torch.compile adds "_orig_mod." prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Load with strict=False to handle old checkpoints that lack value_head keys.
        # New value_head parameters will keep their random initialization.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  New parameters (randomly initialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")

        # Try to restore optimizer state; skip if model architecture changed
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (ValueError, KeyError) as e:
            print(f"  Warning: Could not restore optimizer state ({e}). "
                  f"Using fresh optimizer — momentum/variance buffers will be re-estimated.")

        # Reset LR to the config value after loading optimizer state.
        # load_state_dict() overwrites param_groups['lr'] with the checkpoint's saved LR,
        # which ignores any change made in the config file (e.g. batch-size scaling).
        # We always want the config's learning_rate to win on resume.
        old_lr = self.optimizer.param_groups[0]['lr']
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.config.learning_rate
        print(f"  LR reset: {old_lr:.2e} (checkpoint) → {self.config.learning_rate:.2e} (config)")

        self.step = checkpoint.get('step', 0)
        self.best_loss = checkpoint.get('loss', float('inf'))

        # Restore scheduler state if available
        if (self.scheduler is not None and
                'scheduler_state_dict' in checkpoint):
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Restored LR scheduler state")
            except Exception as e:
                print(f"Warning: Could not restore scheduler state: {e}")

        # Restore GradScaler state if available (prevents float16 overflow on resume)
        if self.scaler is not None:
            if 'scaler_state_dict' in checkpoint:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print(f"Restored GradScaler state (scale={self.scaler.get_scale():.0f})")
                except Exception as e:
                    print(f"Warning: Could not restore GradScaler state: {e}. "
                          f"Using conservative scale=1024.")
                    self.scaler = GradScaler(init_scale=2**10)
            else:
                # Old checkpoint without scaler state: use conservative scale.
                # Default init_scale=65536 causes float16 overflow on trained models
                # whose parameter magnitudes have grown beyond the initial range.
                self.scaler = GradScaler(init_scale=2**10)
                print(f"GradScaler: no saved state in checkpoint, using conservative "
                      f"scale={self.scaler.get_scale():.0f} (will auto-adjust)")

        # Restore RNG state if available
        if 'rng_state' in checkpoint:
            rng = checkpoint['rng_state']
            if 'torch' in rng:
                # Ensure RNG state is a ByteTensor
                rng_state = rng['torch']
                if not isinstance(rng_state, torch.ByteTensor):
                    rng_state = rng_state.to(torch.uint8)
                torch.set_rng_state(rng_state.cpu())
            if 'cuda' in rng and torch.cuda.is_available():
                cuda_rng = rng['cuda']
                if not isinstance(cuda_rng, torch.ByteTensor):
                    cuda_rng = cuda_rng.to(torch.uint8)
                torch.cuda.set_rng_state(cuda_rng.cpu())

        print(f"Resumed at step {self.step}")

        if self._has_non_finite_tensors():
            self._reset_model_state("Loaded checkpoint contains NaN/Inf")

    def _load_stats(self) -> None:
        """Load training stats from file if exists, and merge in any missing test history from log files."""
        stats_path = Path(self.config.stats_file)
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    data = json.load(f)
                self.stats = TrainingStats.from_dict(data)
                print(f"Loaded training stats: {self.stats.total_steps} previous steps")
            except Exception as e:
                print(f"Could not load stats: {e}")
                self.stats = TrainingStats()
        
        # Merge any missing test results from JSONL log files
        self._merge_test_history_from_logs()

    def _merge_test_history_from_logs(self) -> None:
        """Merge test_vs_algo entries from JSONL log files into stats.test_history."""
        log_dir = Path(self.config.log_dir)
        if not log_dir.exists():
            return
        
        # Build a set of existing steps to avoid duplicates
        existing_steps = {entry.get('step') for entry in self.stats.test_history}
        
        # Find all training log files
        log_files = sorted(log_dir.glob("train_*.jsonl"))
        merged_count = 0
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get('type') == 'test_vs_algo':
                                step = entry.get('step')
                                if step is not None and step not in existing_steps:
                                    # Remove the 'type' field before adding to test_history
                                    entry_copy = {k: v for k, v in entry.items() if k != 'type'}
                                    self.stats.test_history.append(entry_copy)
                                    existing_steps.add(step)
                                    merged_count += 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        
        if merged_count > 0:
            # Sort by step
            self.stats.test_history.sort(key=lambda x: x.get('step', 0))
            print(f"Merged {merged_count} test entries from log files into stats")

    def _save_stats(self, *, _snapshot: Optional[dict] = None) -> None:
        """Save training stats to file.

        When called from a background thread, pass a pre-computed ``_snapshot``
        (from ``_snapshot_stats()``) to avoid reading ``self.stats`` while the
        main thread mutates it.
        """
        stats_path = Path(self.config.stats_file)
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        if _snapshot is None:
            # Main-thread path: safe to touch self.stats directly
            self.stats.total_steps = self.step
            self.stats.end_time = datetime.now().isoformat()
            _snapshot = self.stats.to_dict()

        # Atomic write: temp file + os.replace() prevents corruption from
        # concurrent writes (checkpoint bg thread vs main thread) and from
        # process crashes mid-write.
        _tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', dir=stats_path.parent, suffix='.tmp',
                    delete=False) as tmp:
                _tmp_path = tmp.name
                json.dump(_snapshot, tmp, indent=2)
            os.replace(_tmp_path, stats_path)
        except Exception as e:
            print(f"Warning: Failed to save stats: {e}")
            # Clean up temp file on failure
            if _tmp_path is not None:
                try:
                    os.unlink(_tmp_path)
                except Exception:
                    pass

    def _snapshot_stats(self) -> dict:
        """Take a consistent snapshot of training stats for background I/O.

        Must be called on the main thread (where self.stats is mutated).
        The returned dict is a deep copy safe for use in a background thread.
        """
        self.stats.total_steps = self.step
        self.stats.end_time = datetime.now().isoformat()
        return self.stats.to_dict()

    def _record_step_stats(self, loss: float, lr: float) -> None:
        """Record statistics for a training step."""
        current_time = time.time()
        
        # Record loss
        self.stats.loss_history.append({
            'step': self.step,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Record learning rate
        self.stats.lr_history.append({
            'step': self.step,
            'lr': lr
        })
        
        # Record GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e6
            self.stats.gpu_mem_history.append({
                'step': self.step,
                'gpu_mem_mb': gpu_mem
            })
        
        # Record step time
        if self._last_step_time is not None:
            step_time = current_time - self._last_step_time
            self.stats.step_times.append({
                'step': self.step,
                'time_sec': step_time
            })
        self._last_step_time = current_time

        # Trim history lists to prevent unbounded memory growth.
        # At stats_record_every=500 and cap=10000, this retains the last 5M steps.
        _cap = _STATS_HISTORY_CAP
        if len(self.stats.loss_history) > _cap:
            _trim = len(self.stats.loss_history) - _cap
            self.stats.loss_history = self.stats.loss_history[_trim:]
            self.stats.lr_history = self.stats.lr_history[_trim:]
            if self.stats.gpu_mem_history:
                self.stats.gpu_mem_history = self.stats.gpu_mem_history[-_cap:]
            if self.stats.step_times:
                self.stats.step_times = self.stats.step_times[-_cap:]

        # Update best loss
        if loss < self.stats.best_loss:
            self.stats.best_loss = loss

    def _record_validation_stats(self, val_loss: float) -> None:
        """Record validation statistics."""
        self.stats.val_loss_history.append({
            'step': self.step,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat(),
        })
        if len(self.stats.val_loss_history) > _STATS_HISTORY_CAP:
            self.stats.val_loss_history = self.stats.val_loss_history[-_STATS_HISTORY_CAP:]

        if val_loss < self.stats.best_val_loss:
            self.stats.best_val_loss = val_loss

    def _save_checkpoint(self, loss: float) -> str:
        """Save a checkpoint atomically.

        The state_dict copy happens on the main thread (requires CUDA sync),
        but the actual disk I/O is offloaded to a background thread so the
        GPU can continue training while the file is being written.
        """
        # Copy state dicts to CPU (requires CUDA sync — must be on main thread).
        # Both model AND optimizer states must be moved to CPU here, not in the
        # background thread.  optimizer.state_dict() returns references to live
        # CUDA tensors (exp_avg, exp_avg_sq); if the background torch.save()
        # accesses them while the main thread runs optimizer.step(), the tensors
        # may be mid-update.  Copying to CPU on the main thread gives a consistent
        # snapshot and eliminates CUDA sync contention in the background thread.
        #
        # Batch all D2H copies with non_blocking=True and sync once at the end.
        # Each individual .cpu() call forces a stream sync (~50μs); with ~120
        # parameter tensors, batching saves ~6ms per checkpoint (120 × 50μs).
        _opt_sd = self.optimizer.state_dict()
        if self.device.type == 'cuda':
            _cpu_state = {}
            for k, v in _opt_sd['state'].items():
                _cpu_state[k] = {
                    sk: sv.to('cpu', non_blocking=True) if isinstance(sv, torch.Tensor) else sv
                    for sk, sv in v.items()
                }
            _opt_sd = {'state': _cpu_state, 'param_groups': _opt_sd['param_groups']}
            _model_sd = {k: v.to('cpu', non_blocking=True) for k, v in self.model.state_dict().items()}
            # Single sync: all D2H copies are on the default stream; wait for all.
            torch.cuda.current_stream().synchronize()
        else:
            _model_sd = {k: v.cpu() for k, v in self.model.state_dict().items()}
        checkpoint = {
            'model_state_dict': _model_sd,
            'optimizer_state_dict': _opt_sd,
            'step': self.step,
            'loss': loss,
            'epoch': self.epoch,
            'arch_params': getattr(self.model, 'arch_params', {
                'embedding_size': self.config.model_embedding,
                'num_blocks': self.config.model_blocks,
                'hidden_size': self.config.model_hidden,
                'channels': self.config.model_channels,
                'value_head_enabled': self.config.value_head_enabled,
                'value_head_hidden': self.config.value_head_hidden,
            }),
            'rng_state': {
                'torch': torch.get_rng_state(),
            }
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state()

        checkpoint_path = Path(self.config.checkpoint_dir) / f"model_step_{self.step:06d}.pt"
        latest_path = Path(self.config.latest_path)
        _step = self.step
        _stats_collector = self.stats_collector
        # Snapshot stats on main thread to avoid reading self.stats from the
        # background thread while the main thread appends to history lists.
        _stats_snapshot = self._snapshot_stats()

        # Offload disk I/O to background thread — GPU resumes training immediately.
        def _write_checkpoint():
            try:
                # Save to temp file then rename (atomic)
                with tempfile.NamedTemporaryFile(delete=False, dir=self.config.checkpoint_dir) as tmp:
                    torch.save(checkpoint, tmp.name)
                    tmp_path = tmp.name
                os.replace(tmp_path, checkpoint_path)

                # latest.pt = hardlink to checkpoint (instant, avoids second
                # torch.save serialization + disk write of ~55MB).  Falls back
                # to shutil.copy2 if hardlink fails (cross-device, permissions).
                latest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    _lp = str(latest_path)
                    if os.path.exists(_lp):
                        os.unlink(_lp)
                    os.link(str(checkpoint_path), _lp)
                except OSError:
                    import shutil
                    shutil.copy2(str(checkpoint_path), _lp)

                self._save_stats(_snapshot=_stats_snapshot)

                if _stats_collector:
                    ckpt_size = checkpoint_path.stat().st_size / 1e6 if checkpoint_path.exists() else 0
                    _stats_collector.record_checkpoint(
                        step=_step, loss=loss, path=str(checkpoint_path),
                        file_size_mb=ckpt_size,
                    )
                print(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"Checkpoint save error: {e}")

        # Wait for previous checkpoint write to finish before starting a new one.
        # Prevents concurrent disk I/O to the same stats file and avoids
        # accumulating background threads on slow storage.
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            self._checkpoint_thread.join(timeout=30)

        self._checkpoint_thread = threading.Thread(target=_write_checkpoint, daemon=True)
        self._checkpoint_thread.start()

        return str(checkpoint_path)

    def _log(self, data: Dict[str, Any]) -> None:
        """Log training metrics."""
        data['timestamp'] = datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def run_selfplay(self, num_games: int, callback=None,
                     collect_dicts: bool = False,
                     skip_replay: bool = False,
                     preprocess_inline: bool = False) -> int:
        """Run self-play to generate training data.

        Args:
            num_games: Number of games to generate.
            callback: Progress callback.
            collect_dicts: If True, store all raw dicts in self._last_selfplay_dicts
                for incremental dataset updates (avoids re-loading from replay).
            skip_replay: If True, skip JSONL replay I/O entirely. The dicts are
                kept in memory via collect_dicts. Avoids json.dumps + disk writes
                when the data won't be re-read from replay (background self-play
                after the first cycle).
            preprocess_inline: If True, preprocess each completed batch into
                tensors immediately using Cython (or Python fallback) in the
                as_completed() loop.  Eliminates the separate from_dicts()
                preprocessing phase after self-play.  Result stored in
                self._last_selfplay_preprocessed (a CachedTensorDataset).

        Returns:
            Total number of training entries generated.
        """
        if skip_replay and not collect_dicts and not preprocess_inline:
            raise ValueError(
                "skip_replay=True requires collect_dicts=True or "
                "preprocess_inline=True, otherwise generated data is "
                "silently discarded (not saved to disk or memory)."
            )

        _selfplay_start = time.time()
        total_games = max(0, num_games)
        opponent_focus = self.config.selfplay_opponent_focus
        side_focus = self.config.selfplay_focus_side
        _collected = [] if collect_dicts else None
        self._last_selfplay_dicts = None  # Reset previous collection
        self._last_selfplay_preprocessed = None  # Reset previous preprocessed data

        # Inline preprocessing: preprocess each completed batch immediately
        # using Cython/Python encoding in the as_completed() loop.
        # Eliminates the separate CachedTensorDataset.from_dicts() call that
        # spawns a ProcessPoolExecutor after all games finish.
        # Each batch has ~420-1050 entries; Cython processes them in <2ms.
        _preprocess_chunks = [] if preprocess_inline else None
        _pp_max_moves = self.config.max_moves_per_sample if preprocess_inline else 0
        if preprocess_inline:
            from .dataset import _preprocess_chunk as _pp_chunk
        else:
            _pp_chunk = None

        # Algo-vs-algo games are additional (on top of regular self-play)
        algo_vs_algo_games = (
            self.config.algo_vs_algo_games if self.config.algo_vs_algo_enabled else 0
        )

        if opponent_focus == 'ml':
            ml_self_games = total_games
            vs_algo_games = 0
        elif opponent_focus == 'algorithm':
            ml_self_games = 0
            vs_algo_games = total_games
        else:
            ml_self_games = total_games // 2
            vs_algo_games = total_games - ml_self_games

        grand_total = total_games + algo_vs_algo_games

        if opponent_focus == 'ml':
            focus_desc = "ML self-play"
        elif opponent_focus == 'algorithm':
            focus_desc = "ML vs algorithm"
        else:
            focus_desc = "half ML self-play, half vs algorithm"
        if algo_vs_algo_games > 0:
            focus_desc += f" + {algo_vs_algo_games} algo-vs-algo"

        print(
            f"\nGenerating {grand_total} self-play games "
            f"({focus_desc}; focus side: {side_focus})..."
        )

        temp_model_path = Path(self.config.checkpoint_dir) / "temp_selfplay_model.pt"

        entries = 0
        completed_total = 0
        difficulties = self.config.selfplay_difficulties or ['medium']

        # Log difficulties being used
        print(f"  Cycling through difficulties: {difficulties}")

        # ==================================================================
        # Build ALL task arguments upfront, then submit to ONE executor.
        # Previous approach created 4-6+ sequential ProcessPoolExecutors
        # (one per SelfPlayRunner + one for algo-vs-algo). Unified pool:
        #   - eliminates repeated process creation/teardown overhead
        #   - allows ML and algo games to run concurrently
        #   - better CPU utilization (no idle gaps between phases)
        # ==================================================================

        from itertools import combinations_with_replacement
        from concurrent.futures import ProcessPoolExecutor, as_completed as _as_completed

        _max_moves = self.config.selfplay_max_moves
        _noise_prob = self.config.selfplay_noise_prob
        _model_path_str = str(temp_model_path)

        # --- ML task args: (diff, max_moves, noise, start, p1_pol, p2_pol, model_path, device) ---
        all_ml_tasks = []

        if ml_self_games > 0:
            for g in range(ml_self_games):
                start = 1 if g % 2 == 0 else 2
                all_ml_tasks.append((
                    'medium', _max_moves, _noise_prob, start,
                    'ml', 'ml', _model_path_str, self.device,
                ))

        if vs_algo_games > 0:
            if side_focus == 'white':
                ml_as_p1, ml_as_p2 = vs_algo_games, 0
            elif side_focus == 'black':
                ml_as_p1, ml_as_p2 = 0, vs_algo_games
            else:
                ml_as_p1 = vs_algo_games // 2
                ml_as_p2 = vs_algo_games - ml_as_p1

            algo_difficulties = [d for d in difficulties if d != 'self'] or ['medium']

            for side_count, p1_pol, p2_pol in [
                (ml_as_p1, 'ml', 'algorithmic'),
                (ml_as_p2, 'algorithmic', 'ml'),
            ]:
                if side_count <= 0:
                    continue
                gpd = side_count // len(algo_difficulties)
                rem = side_count % len(algo_difficulties)
                for i, diff in enumerate(algo_difficulties):
                    n = gpd + (1 if i < rem else 0)
                    for g in range(n):
                        start = 1 if g % 2 == 0 else 2
                        all_ml_tasks.append((
                            diff, _max_moves, _noise_prob, start,
                            p1_pol, p2_pol, _model_path_str, self.device,
                        ))

        random.shuffle(all_ml_tasks)

        # --- Algo-vs-algo task args: (p1_diff, p2_diff, max_moves, noise, start) ---
        all_algo_tasks = []

        if algo_vs_algo_games > 0:
            ava_diffs = self.config.algo_vs_algo_difficulties or ['medium']
            matchups = list(combinations_with_replacement(ava_diffs, 2))
            gpmu = algo_vs_algo_games // len(matchups)
            rem = algo_vs_algo_games % len(matchups)

            print(f"  Algo-vs-algo: {algo_vs_algo_games} games across matchups {matchups}")

            for idx, (d1, d2) in enumerate(matchups):
                n = gpmu + (1 if idx < rem else 0)
                for g in range(n):
                    start = 1 if g % 2 == 0 else 2
                    all_algo_tasks.append((d1, d2, _max_moves, _noise_prob, start))

            random.shuffle(all_algo_tasks)

        # [Pass 70] Build a CPU model copy for fork-inherited self-play.
        # On Linux (fork start method), workers inherit this global via
        # copy-on-write — zero torch.save/load disk I/O per cycle.
        # Falls back to disk path on Windows (spawn mode) or if anything fails.
        # Still save to disk as fallback for get_model() cache miss.
        import dama.ai.ml.selfplay as _sp_mod
        if all_ml_tasks:
            temp_model_path.parent.mkdir(parents=True, exist_ok=True)
            if self.device.type == 'cuda':
                _sd = {k: v.to('cpu', non_blocking=True)
                       for k, v in self.model.state_dict().items()}
                torch.cuda.current_stream().synchronize()
            else:
                _sd = {k: v.cpu() for k, v in self.model.state_dict().items()}
            # Build CPU model for fork inheritance (avoids N × torch.load)
            try:
                _fork_model = create_model(
                    **getattr(self.model, 'arch_params', {
                        'embedding_size': self.config.model_embedding,
                        'num_blocks': self.config.model_blocks,
                        'hidden_size': self.config.model_hidden,
                        'channels': self.config.model_channels,
                    }))
                _fork_model.load_state_dict(_sd)
                _fork_model.eval()
                _sp_mod._FORK_MODEL = _fork_model
            except Exception:
                _sp_mod._FORK_MODEL = None
            # Save to disk as fallback (Windows spawn, or cache miss)
            torch.save({
                'model_state_dict': _sd,
                'arch_params': getattr(self.model, 'arch_params', {}),
                'step': self.step,
            }, temp_model_path)

        # --- Batch tasks for the unified pool ---
        effective_workers = (
            min(self.config.cpu_workers, 8)
            if platform.system() == 'Windows'
            else self.config.cpu_workers
        )

        # Cap per-batch game count for finer-grained load balancing.
        # Without a cap, 3600 algo games / 10 workers = 360 per batch — one slow
        # batch (e.g., heavy on hard matchups) starves other workers.  Cap at 25
        # games gives 144 batches; workers pick up the next batch as each finishes,
        # naturally balancing fast and slow matchups.  Previous cap of 50 caused
        # hard-hard batches to take ~19s each, creating straggler workers while
        # others sat idle.  25 halves max straggler time to ~9.5s.
        _ALGO_BATCH_CAP = 25
        _ML_BATCH_CAP = 20  # [Pass 67] Increased from 10. Interleaved play batches
                            # all active ML positions per step — more games per batch
                            # = more positions per forward_padded() call = better CPU
                            # BLAS amortization. batch=10 gave 3-5x; 20 should be better.

        def _make_batches(tasks, cap):
            # Target ~1 batch per worker, but cap per batch for load balancing
            if not tasks:
                return []
            bs = max(1, min(cap, (len(tasks) + effective_workers - 1) // effective_workers))
            return [tasks[i:i + bs] for i in range(0, len(tasks), bs)]

        ml_batches = _make_batches(all_ml_tasks, _ML_BATCH_CAP)
        algo_batches = _make_batches(all_algo_tasks, _ALGO_BATCH_CAP)

        total_batches = len(ml_batches) + len(algo_batches)
        if ml_batches:
            print(f"  ML games: {len(all_ml_tasks)} in {len(ml_batches)} batches")
        if algo_batches:
            print(f"  Algo games: {len(all_algo_tasks)} in {len(algo_batches)} batches")
        print(f"  Unified pool: {effective_workers} workers, {total_batches} batches")
        sys.stdout.flush()

        # --- Open one replay file for the entire self-play cycle ---
        if not skip_replay:
            self.replay_buffer.start_new_file()

        # --- Submit ALL to one ProcessPoolExecutor ---
        try:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                # future → (task_type, num_games_in_batch)
                future_meta = {}
                for batch in ml_batches:
                    f = executor.submit(_play_games_batch_worker_full, batch)
                    future_meta[f] = ('ml', len(batch))
                for batch in algo_batches:
                    f = executor.submit(_play_games_batch_worker_algo, batch)
                    future_meta[f] = ('algo', len(batch))

                for future in _as_completed(future_meta):
                    task_type, batch_game_count = future_meta[future]
                    try:
                        entries_data = future.result(timeout=600)
                        if not skip_replay:
                            self.replay_buffer.add_entry_dicts(entries_data)
                        entries += len(entries_data)
                        if _collected is not None:
                            _collected.extend(entries_data)
                        # Inline preprocessing: convert dicts to tensors now,
                        # while the thread is otherwise idle waiting for the
                        # next game batch.  Cython processes ~1000 entries in
                        # <2ms — negligible vs the seconds between batches.
                        if _preprocess_chunks is not None and entries_data:
                            _preprocess_chunks.append(
                                _pp_chunk((entries_data, _pp_max_moves)))
                        completed_total += batch_game_count
                        if callback:
                            callback(completed_total, grand_total)
                        if completed_total % 100 == 0 or completed_total == grand_total:
                            print(f"  Games: {completed_total}/{grand_total}")
                    except Exception as e:
                        print(f"Self-play error ({task_type}): {e}")
        except Exception as e:
            print(f"Unified pool failed ({e}), falling back to sequential")
            # Sequential fallback for algo-vs-algo only (ML fallback is rarely needed)
            for task_a in all_algo_tasks:
                try:
                    entries_data = _play_game_worker_algo_vs_algo(task_a)
                    if not skip_replay:
                        self.replay_buffer.add_entry_dicts(entries_data)
                    entries += len(entries_data)
                    if _collected is not None:
                        _collected.extend(entries_data)
                    if _preprocess_chunks is not None and entries_data:
                        _preprocess_chunks.append(
                            _pp_chunk((entries_data, _pp_max_moves)))
                    completed_total += 1
                    if callback:
                        callback(completed_total, grand_total)
                except Exception as e:
                    print(f"Sequential fallback error: {e}")

        # [Pass 70] Clean up fork-inherited model to free CPU memory.
        _sp_mod._FORK_MODEL = None

        if not skip_replay:
            self.replay_buffer.close()
        print(f"Generated {entries} training entries")

        # Store collected dicts for incremental preprocessing
        if _collected is not None:
            self._last_selfplay_dicts = _collected

        # Assemble inline-preprocessed chunks into a CachedTensorDataset.
        # All chunks were preprocessed during the as_completed() loop, so
        # this is just numpy concatenation + torch.from_numpy (~1ms total).
        if _preprocess_chunks is not None and _preprocess_chunks:
            import numpy as _np
            boards = _np.concatenate([c[0] for c in _preprocess_chunks])
            mf = _np.concatenate([c[1] for c in _preprocess_chunks])
            mc = _np.concatenate([c[2] for c in _preprocess_chunks])
            tgt = _np.concatenate([c[3] for c in _preprocess_chunks])
            rw = _np.concatenate([c[4] for c in _preprocess_chunks])
            vt = _np.concatenate([c[5] for c in _preprocess_chunks])
            self._last_selfplay_preprocessed = CachedTensorDataset(
                torch.from_numpy(boards), torch.from_numpy(mf),
                torch.from_numpy(mc), torch.from_numpy(tgt),
                torch.from_numpy(rw), torch.from_numpy(vt),
            )
            print(f"  Inline preprocessing: {len(self._last_selfplay_preprocessed)} entries "
                  f"({len(_preprocess_chunks)} chunks, zero extra latency)")

        # Record self-play stats
        if self.stats_collector:
            _selfplay_elapsed = time.time() - _selfplay_start
            self.stats_collector.record_selfplay_epoch(
                step=self.step,
                epoch=self.epoch,
                num_games=grand_total,
                num_entries=entries,
                elapsed_sec=_selfplay_elapsed,
            )

            # Record replay buffer state (skip when replay I/O was bypassed —
            # file counts would be stale and count_entries() does unnecessary I/O)
            if not skip_replay:
                try:
                    buf_entries = self.replay_buffer.count_entries()
                    replay_dir = Path(self.config.replay_dir)
                    # Single glob pass for both file count and total size
                    _files = list(replay_dir.glob("*.jsonl")) if replay_dir.exists() else []
                    num_files = len(_files)
                    total_bytes = sum(f.stat().st_size for f in _files)
                    self.stats_collector.record_replay_buffer_state(
                        step=self.step,
                        total_entries=buf_entries,
                        num_files=num_files,
                        total_size_bytes=total_bytes,
                    )
                except Exception:
                    pass  # Non-critical

        return entries

    # ------------------------------------------------------------------
    # Background self-play: overlap CPU game generation with GPU training
    # ------------------------------------------------------------------

    def _start_background_selfplay(self, num_games: int) -> None:
        """Launch continuous self-play + data preparation in a background thread.

        The thread runs a continuous loop: generate games → preprocess →
        store dataset → immediately start next cycle.  This eliminates the
        CPU idle gap between self-play cycles that existed when the thread
        was one-shot and the main thread had to restart it.

        The main thread picks up completed datasets via _collect_background_selfplay()
        at epoch boundaries.  If multiple cycles complete between collections,
        only the latest dataset is kept.

        Uses incremental preprocessing when possible: only preprocesses the
        new self-play entries to tensors, then concatenates with the existing
        dataset.  Falls back to full rebuild when no existing dataset exists.
        """
        if self._bg_selfplay_thread is not None and self._bg_selfplay_thread.is_alive():
            return  # already running

        def _worker():
            _existing = getattr(self, '_current_dataset', None)
            _max_entries = self.config.replay_max_entries

            while not self._stopped:
                try:
                    # Skip JSONL replay I/O when we have an existing dataset to
                    # build on incrementally.  Data flows directly from self-play
                    # workers → memory dicts → CachedTensorDataset → GPU, avoiding
                    # json serialization + disk writes + ReplayEntry conversion.
                    _skip = _existing is not None and len(_existing) > 0

                    # Always use inline preprocessing: preprocess entries as each
                    # game batch completes in the as_completed() loop.  Eliminates
                    # the separate from_dicts() / from_entries() call that spawns
                    # a ProcessPoolExecutor after all games finish.  Saves 5-30s
                    # per cycle (entire preprocessing phase overlapped with games).
                    # collect_dicts is still needed on first cycle for fallback.
                    self.run_selfplay(
                        num_games, collect_dicts=not _skip,
                        skip_replay=_skip, preprocess_inline=True)

                    # Check for inline-preprocessed data first (fast path).
                    incremental = getattr(self, '_last_selfplay_preprocessed', None)

                    if incremental is not None and _existing is not None and len(_existing) > 0:
                        # Inline preprocessing produced the incremental dataset
                        # directly — no from_dicts() call needed.
                        dataset = _existing.concat(incremental, max_entries=_max_entries)
                        print(f"  Merged dataset: {len(dataset)} entries "
                              f"(+{len(incremental)} inline-preprocessed)")
                    elif incremental is not None:
                        # First cycle with inline preprocessing, or no existing
                        # data to concat with.  On first cycle, also load any
                        # existing replay data from disk and merge.
                        if _existing is None or len(_existing) == 0:
                            # Check for existing replay data on disk
                            train_entries, _ = prepare_training_data(
                                self.replay_buffer, max_entries=_max_entries, val_split=0.0)
                            if train_entries:
                                existing_ds = CachedTensorDataset.from_entries(
                                    train_entries, max_moves_per_sample=self.config.max_moves_per_sample, show_progress=True)
                                dataset = existing_ds.concat(incremental, max_entries=_max_entries)
                                print(f"  Loaded {len(existing_ds)} existing + "
                                      f"{len(incremental)} inline-preprocessed = "
                                      f"{len(dataset)} total entries")
                            else:
                                dataset = incremental
                                print(f"  New dataset: {len(dataset)} entries (inline-preprocessed)")
                        else:
                            dataset = incremental
                            print(f"  New dataset: {len(dataset)} entries (inline-preprocessed)")
                    else:
                        # Fallback: no inline preprocessing available.
                        # Use collected dicts or load from replay.
                        new_dicts = getattr(self, '_last_selfplay_dicts', None)

                        if new_dicts and _existing is not None and len(_existing) > 0:
                            print(f"  Incremental preprocessing: {len(new_dicts)} new entries "
                                  f"(existing: {len(_existing)})")
                            incremental = CachedTensorDataset.from_dicts(
                                new_dicts, max_moves_per_sample=self.config.max_moves_per_sample, show_progress=True)
                            dataset = _existing.concat(incremental, max_entries=_max_entries)
                            print(f"  Merged dataset: {len(dataset)} entries")
                        else:
                            incremental = None
                            train_entries, _ = prepare_training_data(
                                self.replay_buffer, max_entries=_max_entries, val_split=0.0)
                            if self.config.clear_replay_after_load:
                                deleted = self.replay_buffer.clear_files()
                                if deleted:
                                    print(f"Cleared {deleted} replay files after loading")
                            dataset = CachedTensorDataset.from_entries(
                                train_entries, max_moves_per_sample=self.config.max_moves_per_sample, show_progress=True)

                    with self._bg_selfplay_lock:
                        self._bg_selfplay_entries = None
                        self._bg_selfplay_dataset = dataset
                        self._bg_selfplay_incremental = incremental
                    # Wake the main thread immediately if it's waiting for data.
                    self._data_ready_event.set()

                    # Use the just-produced dataset as base for next incremental cycle
                    _existing = dataset

                except Exception as e:
                    import traceback
                    print(f"Background self-play error: {e}")
                    traceback.print_exc()
                    # Don't crash the loop — sleep briefly and retry
                    if not self._stopped:
                        time.sleep(2.0)

        self._bg_selfplay_thread = threading.Thread(target=_worker, daemon=True)
        self._bg_selfplay_thread.start()

    def _collect_background_selfplay(self):
        """Check if background self-play produced a dataset. Non-blocking.

        Returns (dataset, incremental) where:
        - dataset: Full merged dataset (old + new), used for _current_dataset tracking
        - incremental: Only the new entries from this cycle, used for GPU update_data
          to avoid re-uploading the entire dataset via PCIe.  None on first cycle
          (full rebuild).
        The background thread keeps running — no need to restart.
        """
        with self._bg_selfplay_lock:
            dataset = self._bg_selfplay_dataset
            if dataset is None:
                return None, None
            incremental = self._bg_selfplay_incremental
            self._bg_selfplay_dataset = None
            self._bg_selfplay_incremental = None
            self._bg_selfplay_entries = None
        return dataset, incremental

    def _refresh_dataloader(self, dataloader, bg_dataset, bg_incremental,
                            effective_workers):
        """Apply new self-play data to the dataloader.

        Handles both FastBatchIterator (in-place GPU update) and standard
        DataLoader (full recreate) paths.  Returns the (possibly new)
        dataloader and whether data is GPU-resident.
        """
        self._current_dataset = bg_dataset
        _is_fast = isinstance(dataloader, FastBatchIterator)
        if _is_fast:
            _update_src = bg_incremental if bg_incremental is not None else bg_dataset
            dataloader.update_data(
                _update_src, max_entries=self.config.replay_max_entries)
            dataloader.dataset = bg_dataset
        else:
            _was_gpu = getattr(dataloader, 'on_gpu', False)
            if _was_gpu:
                dataloader = None
                torch.cuda.empty_cache()
            dataloader = create_dataloader_from_dataset(
                bg_dataset,
                batch_size=self.config.batch_size,
                num_workers=effective_workers,
                pin_memory=self.config.pin_memory,
                device=self.device,
                amp_enabled=self.config.amp,
            )
        self._use_padded = True
        _gpu_resident = getattr(dataloader, 'on_gpu', False)
        return dataloader, _gpu_resident

    def run_test_vs_algo(self, num_games: int = None) -> Dict[str, Any]:
        """
        Run test games between current model and algorithmic AI.

        Args:
            num_games: Number of test games (defaults to config)

        Returns:
            Test statistics dictionary
        """
        if num_games is None:
            num_games = self.config.test_games

        print(f"\nRunning {num_games} test games vs algorithm ({self.config.test_difficulty})...")

        from .model_vs_algo import ModelVsAlgoTester

        # Save current model temporarily for testing.
        # Non_blocking D2H copies + single sync (same as _save_checkpoint).
        temp_path = Path(self.config.checkpoint_dir) / "temp_test_model.pt"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        if self.device.type == 'cuda':
            _sd = {k: v.to('cpu', non_blocking=True)
                   for k, v in self.model.state_dict().items()}
            torch.cuda.current_stream().synchronize()
        else:
            _sd = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save({
            'model_state_dict': _sd,
            'arch_params': getattr(self.model, 'arch_params', {}),
            'step': self.step,
        }, temp_path)

        try:
            tester = ModelVsAlgoTester(
                model_path=str(temp_path),
                algo_difficulty=self.config.test_difficulty,
                num_workers=min(self.config.cpu_workers, 4),
                max_moves=self.config.selfplay_max_moves,
            )

            def progress(completed, total, stats):
                if completed % 10 == 0:
                    print(f"  Test games: {completed}/{total} (ML: {stats.ml_win_rate*100:.1f}%)")

            stats = tester.run_tests(num_games=num_games, callback=progress)

            # Record in training stats
            test_record = {
                'step': self.step,
                'epoch': self.epoch,
                'total_games': stats.total_games,
                'ml_wins': stats.ml_wins,
                'algo_wins': stats.algo_wins,
                'draws': stats.draws,
                'ml_win_rate': stats.ml_win_rate,
                'draw_rate': stats.draw_rate,
                'ml_as_p1_win_rate': stats.ml_as_p1_win_rate,
                'ml_as_p2_win_rate': stats.ml_as_p2_win_rate,
                'avg_game_length': stats.avg_game_length,
                'timestamp': datetime.now().isoformat(),
            }
            self.stats.test_history.append(test_record)

            # Record in enhanced stats collector
            if self.stats_collector:
                self.stats_collector.record_evaluation(
                    step=self.step, epoch=self.epoch, test_record=test_record,
                )

            print(f"  ML Win Rate: {stats.ml_win_rate*100:.1f}%")
            print(f"    As P1 (White): {stats.ml_as_p1_win_rate*100:.1f}%")
            print(f"    As P2 (Black): {stats.ml_as_p2_win_rate*100:.1f}%")
            if self.stats_collector and self.stats_collector.eval_records:
                latest = self.stats_collector.eval_records[-1]
                elo = latest.get('estimated_elo_diff', 0)
                print(f"    Est. ELO diff:  {elo:+.0f}")

            # Log to JSONL
            self._log({
                'type': 'test_vs_algo',
                **test_record
            })

            # Save stats to JSON file immediately so plot_training.py can read them
            self._save_stats()

            return test_record

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def _run_test_cpu_only(self, model_path: str, num_games: int,
                           difficulty: str, max_moves: int,
                           num_workers: int) -> Dict[str, Any]:
        """Run test games using a pre-saved model path. CPU-only, thread-safe.

        Does NOT access self.model or modify self.stats — safe to call from
        a background thread while the main thread continues GPU training.
        """
        from .model_vs_algo import ModelVsAlgoTester

        tester = ModelVsAlgoTester(
            model_path=model_path,
            algo_difficulty=difficulty,
            num_workers=num_workers,
            max_moves=max_moves,
        )

        def progress(completed, total, stats):
            if completed % 10 == 0:
                print(f"  [async test] {completed}/{total} "
                      f"(ML: {stats.ml_win_rate*100:.1f}%)")

        stats = tester.run_tests(num_games=num_games, callback=progress)

        return {
            'total_games': stats.total_games,
            'ml_wins': stats.ml_wins,
            'algo_wins': stats.algo_wins,
            'draws': stats.draws,
            'ml_win_rate': stats.ml_win_rate,
            'draw_rate': stats.draw_rate,
            'ml_as_p1_win_rate': stats.ml_as_p1_win_rate,
            'ml_as_p2_win_rate': stats.ml_as_p2_win_rate,
            'avg_game_length': stats.avg_game_length,
        }

    def _record_test_result(self, test_result: Dict[str, Any],
                            at_step: int, at_epoch: int) -> None:
        """Record a completed test result into stats. Main-thread only."""
        test_record = {
            'step': at_step,
            'epoch': at_epoch,
            'timestamp': datetime.now().isoformat(),
            **test_result,
        }
        self.stats.test_history.append(test_record)

        if self.stats_collector:
            self.stats_collector.record_evaluation(
                step=at_step, epoch=at_epoch, test_record=test_record,
            )

        wr = test_result.get('ml_win_rate', 0)
        p1wr = test_result.get('ml_as_p1_win_rate', 0)
        p2wr = test_result.get('ml_as_p2_win_rate', 0)
        print(f"  [async test @ step {at_step}] ML Win Rate: {wr*100:.1f}%"
              f"  (P1: {p1wr*100:.1f}%, P2: {p2wr*100:.1f}%)")

        self._log({'type': 'test_vs_algo', **test_record})
        self._save_stats()

    def _should_use_scoring_for_epoch(self, epoch: int) -> bool:
        """Determine whether to use the scoring system for a given epoch number."""
        mode = self.config.reward_mode.lower()
        if mode == 'scoring':
            return True
        elif mode == 'none':
            return False
        else:  # 'cycle'
            # Odd epochs (1, 3, 5...) use scoring; even epochs (2, 4, 6...) don't
            return (epoch % 2) == 1

    def _should_use_scoring(self) -> bool:
        """Determine whether to use the scoring system for the upcoming epoch."""
        return self._should_use_scoring_for_epoch(self.epoch + 1)

    # Interval for expensive sanity checks (isfinite on large tensors).
    # Each check forces a CUDA sync (~20-100μs). Checking every step wastes
    # GPU pipeline throughput; periodic checks still catch numerical issues.
    # Numerical instability evolves gradually (over hundreds of steps), so
    # checking every 2000 steps catches problems well before they cascade
    # while minimizing GPU pipeline stalls from forced CUDA syncs.
    _SANITY_CHECK_INTERVAL = 2000

    def train_epoch(self, dataloader, use_scoring: bool = True) -> float:
        """Train for one epoch, returns average loss.

        Args:
            dataloader: Training data loader
            use_scoring: If True, apply reward weights from the scoring system.
                         If False, use uniform weights (classic behavior).
        """
        self.model.train()
        # Accumulate loss on GPU to avoid per-step CUDA sync from .item()
        total_loss_acc = torch.tensor(0.0, device=self.device)
        num_batches = 0
        total_batches = len(dataloader)
        first_batch = True
        epoch_start_time = time.time()
        _step_start = 0.0  # lazily set only when stats recording needs it
        accum_steps = self.config.gradient_accumulation_steps
        _micro_step = 0  # counts mini-batches within an accumulation window

        # Cache frequently accessed config/state as locals to eliminate
        # attribute lookup overhead on every step (~50ns each, adds up at 10+ steps/sec).
        _cfg = self.config
        _device = self.device
        _stats_collector = self.stats_collector
        _stats_record_every = _cfg.stats_record_every
        _stats_score_dist_every = _cfg.stats_score_dist_every
        _stats_system_every = _cfg.stats_system_every
        _stats_model_health_every = _cfg.stats_model_health_every
        _checkpoint_every = _cfg.checkpoint_every
        _train_steps = _cfg.train_steps
        _grad_clip_norm = _cfg.grad_clip_norm
        _use_amp = _cfg.amp
        _amp_dtype = self.amp_dtype
        _value_head_enabled = _cfg.value_head_enabled
        _value_weight = _cfg.value_weight
        _scaler = self.scaler
        _optimizer = self.optimizer
        _scheduler = self.scheduler
        _model = self.model
        _use_padded = self._use_padded
        _sanity_interval = self._SANITY_CHECK_INTERVAL
        _thermal_enabled = _cfg.thermal_enabled

        # Compiled fwd+loss: use when available, padded, and no value head
        _compiled_fwd_loss = self._compiled_fwd_loss
        _use_compiled = (_compiled_fwd_loss is not None and _use_padded
                         and not _value_head_enabled)

        # Pre-allocate uniform weights for non-scoring epochs so the compiled
        # path always receives a tensor (never None).
        if _use_compiled and not use_scoring:
            _uniform_rw = torch.ones(_cfg.batch_size, dtype=torch.float32, device=_device)
        else:
            _uniform_rw = None

        # Use CUDA prefetcher for overlapped H2D transfer — but skip when data
        # is already GPU-resident (no transfer to overlap, prefetcher just adds
        # stream-sync overhead).
        _data_on_gpu = getattr(dataloader, 'on_gpu', False)
        _use_prefetcher = _device.type == 'cuda' and not _data_on_gpu
        if _use_prefetcher:
            iter_loader = CUDAPrefetcher(dataloader, _device)
        else:
            iter_loader = dataloader

        for boards, move_features, move_counts, targets, reward_weights, value_targets in iter_loader:
            if first_batch:
                print(f"  First batch loaded. Processing {total_batches} batches...")
                sys.stdout.flush()

            if self._stopped:
                break

            while self._paused:
                time.sleep(0.1)
                if self._stopped:
                    break

            # Thermal protection: only check when enabled (avoid method call overhead)
            if _thermal_enabled:
                self._check_thermal_and_rest()

            # Move to device — skip when CUDAPrefetcher already transferred
            # or when data is GPU-resident (already on device)
            if not _use_prefetcher and not _data_on_gpu:
                boards = boards.to(_device, non_blocking=True)
                move_features = move_features.to(_device, non_blocking=True)
                move_counts = move_counts.to(_device, non_blocking=True)
                targets = targets.to(_device, non_blocking=True)
            # Conditionally load reward weights / value targets
            if not use_scoring:
                # Compiled path needs a tensor (never None); reuse pre-allocated ones.
                # Slice to actual batch size in case drop_last=False yields a smaller batch.
                if _uniform_rw is not None:
                    _bs = boards.shape[0]
                    reward_weights = _uniform_rw[:_bs] if _bs < _uniform_rw.shape[0] else _uniform_rw
                else:
                    reward_weights = None
            elif not _use_prefetcher and not _data_on_gpu:
                reward_weights = reward_weights.to(_device, non_blocking=True)
            if not _value_head_enabled:
                value_targets = None
            elif not _use_prefetcher and not _data_on_gpu:
                value_targets = value_targets.to(_device, non_blocking=True)

            # Periodic sanity check — avoids CUDA sync on every step
            _do_sanity = (self.step % _sanity_interval == 0)
            if _do_sanity:
                if not torch.isfinite(boards).all() or not torch.isfinite(move_features).all():
                    print("  Warning: non-finite inputs detected; skipping batch")
                    if _stats_collector:
                        _stats_collector.record_non_finite_event(
                            self.step, 'input_data', 'Non-finite board or move features')
                    continue

            # Only zero gradients at the start of an accumulation window
            if _micro_step == 0:
                # set_to_none=True avoids a memset, letting PyTorch deallocate instead
                _optimizer.zero_grad(set_to_none=True)

            # --- Forward + backward ---
            _grad_norm = None
            _grad_norms_per_layer = None
            _current_scores = None

            if _use_compiled:
                # ── Compiled path: fused forward+loss in single CUDAGraph ──
                # nan_to_num inside the compiled function sanitizes NaN/Inf loss
                # without CUDA sync.  No per-step torch.isfinite() needed.
                if _use_amp:
                    with autocast(device_type='cuda', dtype=_amp_dtype):
                        loss, _current_scores = _compiled_fwd_loss(
                            boards, move_features, move_counts, targets, reward_weights)
                else:
                    loss, _current_scores = _compiled_fwd_loss(
                        boards, move_features, move_counts, targets, reward_weights)

                # Score-level NaN check at sanity interval only (expensive array-wide check)
                if _do_sanity and torch.isnan(_current_scores).any():
                    print("  Warning: NaN scores detected; skipping batch")
                    if self._repair_batchnorm_stats():
                        print("  Repaired corrupted BatchNorm running stats")
                    if _stats_collector:
                        _stats_collector.record_non_finite_event(
                            self.step, 'compiled_fwd_loss', 'NaN scores')
                    continue

                if accum_steps > 1:
                    loss = loss / accum_steps

                if _scaler is not None:
                    _scaler.scale(loss).backward()
                else:
                    loss.backward()

            elif _use_amp:
                # ── AMP path (fallback: value head or non-padded) ──
                try:
                    with autocast(device_type='cuda', dtype=_amp_dtype):
                        if _use_padded:
                            if _value_head_enabled:
                                scores, value_preds = _model.forward_padded_with_value(boards, move_features, move_counts)
                            else:
                                scores = _model.forward_padded(boards, move_features, move_counts)
                                value_preds = None
                        else:
                            if _value_head_enabled:
                                scores, value_preds = _model.forward_with_value(boards, move_features, move_counts)
                            else:
                                scores = _model(boards, move_features, move_counts)
                                value_preds = None
                except RuntimeError as _compile_err:
                    if "device kernel image is invalid" in str(_compile_err) and first_batch:
                        # torch.compile generated incompatible CUDA kernels —
                        # unwrap to eager model and retry this batch
                        print(f"torch.compile runtime failure: {_compile_err}")
                        print("Falling back to eager mode for remaining training...")
                        sys.stdout.flush()
                        _orig = getattr(_model, '_orig_mod', None)
                        if _orig is not None:
                            self.model = _orig
                            _model = _orig
                        torch._dynamo.reset()
                        with autocast(device_type='cuda', dtype=_amp_dtype):
                            if _use_padded:
                                if _value_head_enabled:
                                    scores, value_preds = _model.forward_padded_with_value(boards, move_features, move_counts)
                                else:
                                    scores = _model.forward_padded(boards, move_features, move_counts)
                                    value_preds = None
                            else:
                                if _value_head_enabled:
                                    scores, value_preds = _model.forward_with_value(boards, move_features, move_counts)
                                else:
                                    scores = _model(boards, move_features, move_counts)
                                    value_preds = None
                    else:
                        raise
                if _do_sanity:
                    if _use_padded:
                        # Padded path uses -inf for padding slots — only NaN is a real problem
                        _bad = torch.isnan(scores).any()
                    else:
                        _bad = not torch.isfinite(scores).all()
                    if _bad:
                        print("  Warning: non-finite scores detected; skipping batch")
                        if self._repair_batchnorm_stats():
                            print("  Repaired corrupted BatchNorm running stats")
                        if _stats_collector:
                            _stats_collector.record_non_finite_event(
                                self.step, 'model_scores', 'Non-finite output scores')
                        continue
                _current_scores = scores.detach()
                if _use_padded:
                    policy_loss = self._compute_loss_padded(scores, move_counts, targets, reward_weights)
                else:
                    policy_loss = self._compute_loss(scores, move_counts, targets, reward_weights)

                if value_preds is not None and value_targets is not None:
                    value_loss = nn.functional.mse_loss(value_preds, value_targets)
                    loss = policy_loss + _value_weight * value_loss
                else:
                    loss = policy_loss

                if accum_steps > 1:
                    loss = loss / accum_steps

                # NaN-safe: replace non-finite loss with 0 to avoid CUDA sync.
                # GradScaler path: scaler.step() already detects NaN grads and
                # skips the optimizer step, so this is just belt-and-suspenders.
                # No-scaler (bfloat16) path: prevents NaN gradient corruption.
                # Zero loss → zero gradients → optimizer step is a near-no-op.
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                if _scaler is not None:
                    _scaler.scale(loss).backward()
                else:
                    loss.backward()

            else:
                # ── No-AMP path ──
                if _use_padded:
                    if _value_head_enabled:
                        scores, value_preds = _model.forward_padded_with_value(boards, move_features, move_counts)
                    else:
                        scores = _model.forward_padded(boards, move_features, move_counts)
                        value_preds = None
                else:
                    if _value_head_enabled:
                        scores, value_preds = _model.forward_with_value(boards, move_features, move_counts)
                    else:
                        scores = _model(boards, move_features, move_counts)
                        value_preds = None
                if _do_sanity:
                    if _use_padded:
                        # Padded path uses -inf for padding slots — only NaN is a real problem
                        _bad = torch.isnan(scores).any()
                    else:
                        _bad = not torch.isfinite(scores).all()
                    if _bad:
                        print("  Warning: non-finite scores detected; skipping batch")
                        if self._repair_batchnorm_stats():
                            print("  Repaired corrupted BatchNorm running stats")
                        if _stats_collector:
                            _stats_collector.record_non_finite_event(
                                self.step, 'model_scores', 'Non-finite output scores (FP32)')
                        continue
                _current_scores = scores.detach()
                if _use_padded:
                    policy_loss = self._compute_loss_padded(scores, move_counts, targets, reward_weights)
                else:
                    policy_loss = self._compute_loss(scores, move_counts, targets, reward_weights)
                if value_preds is not None and value_targets is not None:
                    value_loss = nn.functional.mse_loss(value_preds, value_targets)
                    loss = policy_loss + _value_weight * value_loss
                else:
                    loss = policy_loss

                if accum_steps > 1:
                    loss = loss / accum_steps

                # NaN-safe: replace non-finite loss with 0 (no CUDA sync needed).
                # No-AMP path typically runs on CPU where sync isn't a concern,
                # but consistency with AMP path and no behavioral change.
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                loss.backward()

            _micro_step += 1
            first_batch = False

            # --- Optimizer step: only after accumulating enough gradients ---
            if _micro_step < accum_steps:
                # Accumulate loss for reporting (unscaled)
                total_loss_acc += loss.detach() * accum_steps
                num_batches += 1
                continue

            # Accumulated enough — clip, step, and reset
            _micro_step = 0

            # Time the optimizer step (placed after accumulation continue so
            # _step_start is always fresh for the step that actually records).
            _will_record = (_stats_collector and
                            (self.step + 1) % _stats_record_every == 0)
            if _will_record:
                _step_start = time.time()

            # Gradient clipping + stats.  clip_grad_norm_ returns the total
            # (unclipped) grad norm, so we capture it instead of iterating all
            # parameters a second time in compute_gradient_stats.  Per-layer
            # norms are only collected at model_health frequency (much lower)
            # to avoid ~40 .item() CUDA syncs per stats step.
            _want_grad_stats = (_stats_collector and
                                self.step % _stats_record_every == 0)

            if _use_amp and _scaler is not None:
                if _grad_clip_norm is not None:
                    _scaler.unscale_(_optimizer)
                    _clip_norm = torch.nn.utils.clip_grad_norm_(
                        _model.parameters(), _grad_clip_norm)
                    if _want_grad_stats:
                        _grad_norm = _clip_norm.item()
                _scaler.step(_optimizer)
                _scaler.update()
            else:
                if _grad_clip_norm is not None:
                    _clip_norm = torch.nn.utils.clip_grad_norm_(
                        _model.parameters(), _grad_clip_norm)
                    if _want_grad_stats:
                        _grad_norm = _clip_norm.item()
                _optimizer.step()

            # Accumulate on GPU — no sync. Only .item() when needed for logging.
            # Undo the /accum_steps scaling so total_loss_acc reflects true loss.
            total_loss_acc += loss.detach() * accum_steps
            num_batches += 1
            self.step += 1
            _step_elapsed = (time.time() - _step_start) if _will_record else 0.0

            # Step the LR scheduler (per-step, not per-epoch)
            if _scheduler is not None:
                _scheduler.step()

            # Get current LR (from scheduler if active, else from config)
            current_lr = (_scheduler.get_last_lr()[0]
                          if _scheduler is not None
                          else _cfg.learning_rate)

            # Only call loss.item() (CUDA sync) when we actually need the scalar.
            # Avoid hardcoded intervals — piggyback on stats_record_every to
            # eliminate extra CUDA sync points on the critical path.
            _step = self.step  # cache for repeated modulo checks below
            _need_loss_val = (
                _step % _stats_record_every == 0
                or _step % _checkpoint_every == 0
            )
            # `loss` was divided by accum_steps for gradient scaling; undo that
            # for human-readable reporting (no extra sync — same .item() call).
            _loss_val = (loss.item() * accum_steps) if _need_loss_val else None

            # Periodic NaN monitoring: piggyback on the stats sync to check
            # for NaN losses without an extra CUDA sync.  nan_to_num converts
            # NaN → 0.0, so a zero loss at a stats interval signals a bad batch.
            if _loss_val is not None and _loss_val == 0.0 and _step > 0:
                # loss == 0.0 is extremely unlikely in normal training (cross-entropy
                # is always > 0 for non-degenerate data).  Likely a nan_to_num replacement.
                if self._repair_batchnorm_stats():
                    print("  Repaired corrupted BatchNorm running stats (NaN loss detected)")
                if _stats_collector:
                    _stats_collector.record_non_finite_event(
                        _step, 'nan_to_num', 'NaN/Inf loss replaced with 0 by nan_to_num')

            # Record step stats every N steps (to avoid excessive memory usage)
            if _loss_val is not None and _step % _stats_record_every == 0:
                self._record_step_stats(_loss_val, current_lr)

                # Enhanced stats collection
                if _stats_collector:
                    # Score distribution stats
                    _score_stats = None
                    if (_current_scores is not None and
                            _step % _stats_score_dist_every == 0):
                        if _use_padded:
                            _score_stats = StatsCollector.compute_score_stats_padded(
                                _current_scores, move_counts)
                        else:
                            _score_stats = StatsCollector.compute_score_stats(
                                _current_scores, move_counts)

                    _stats_collector.record_training_step(
                        step=_step,
                        loss=_loss_val,
                        lr=current_lr,
                        batch_size=boards.shape[0],
                        step_time=_step_elapsed,
                        grad_norm=_grad_norm,
                        grad_norms_per_layer=_grad_norms_per_layer,
                        score_stats=_score_stats,
                    )

            # System metrics (lower frequency)
            if _stats_collector and _step % _stats_system_every == 0:
                _stats_collector.record_system_metrics(_step)

            # Model health (even lower frequency)
            if _stats_collector and _step % _stats_model_health_every == 0:
                _stats_collector.record_model_health(_model, _step)

            # Checkpoint — reuse _loss_val if already computed at this step
            # (avoids a redundant .item() CUDA sync when checkpoint and stats
            # recording align on the same step).
            if _step % _checkpoint_every == 0:
                if _loss_val is not None:
                    avg_loss = _loss_val
                else:
                    avg_loss = (total_loss_acc / max(num_batches, 1)).item()
                self._save_checkpoint(avg_loss)

                # Log metrics
                gpu_mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                self._log({
                    'step': _step,
                    'loss': avg_loss,
                    'lr': current_lr,
                    'gpu_mem_mb': gpu_mem,
                })

            # Progress — print at the stats interval (which already computed .item()).
            # Avoids extra CUDA syncs from a separate hardcoded interval.
            if _loss_val is not None and _step % _stats_record_every == 0:
                print(f"  Step {_step}, Loss: {_loss_val:.4f}")

            if _step >= _train_steps:
                break

        epoch_time = time.time() - epoch_start_time

        # Expose to caller for dead-epoch detection
        self._last_epoch_batches = num_batches

        # Single .item() CUDA sync for the epoch average — reuse for both
        # stats recording and return value.
        _avg_loss = (total_loss_acc / max(num_batches, 1)).item()
        if _stats_collector:
            _stats_collector.record_epoch(
                epoch=self.epoch + 1,  # will be incremented by caller
                step=self.step,
                avg_loss=_avg_loss,
                num_batches=num_batches,
                epoch_time_sec=epoch_time,
            )

        return _avg_loss

    def _compute_loss(
        self,
        scores: torch.Tensor,
        move_counts: torch.Tensor,
        targets: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward-weighted cross-entropy loss for move selection.

        The model outputs one score per move. For each position,
        we want to maximize the score of the chosen move.

        Fully vectorized — no Python loop over batch positions.
        Uses scatter to build a padded (batch_size, max_moves) matrix,
        then applies log_softmax in one GPU call.
        """
        batch_size = move_counts.shape[0]

        # Use config pad size to avoid counts.max().item() CUDA sync every step.
        # -inf padding slots get zero probability from log_softmax, so over-padding
        # is harmless (just a small allocation overhead).
        max_moves = self.config.max_moves_per_sample

        # Fast check: if total moves is zero, return zero loss (no sync needed —
        # scores.shape[0] is determined before the GPU kernel, by the collate).
        if scores.shape[0] == 0:
            return scores.sum() * 0

        # move_counts/targets are int32; use directly for comparisons.
        # Cast to int64 only where required (cumsum for indexing, gather).
        counts_l = move_counts.long()

        # Build scatter indices without a Python loop.
        # exclusive_cumsum[i] = start offset of position i inside the flat scores tensor.
        exclusive_cumsum = counts_l.cumsum(0) - counts_l      # (batch_size,)
        row_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=scores.device), counts_l
        )                                                      # (total_moves,)
        col_idx = (
            torch.arange(scores.shape[0], device=scores.device, dtype=torch.long)
            - exclusive_cumsum[row_idx]
        )                                                      # (total_moves,)

        # Clamp col_idx to pad size (safety — should never trigger in practice)
        col_idx = col_idx.clamp(max=max_moves - 1)

        # Scatter scores into a padded matrix; -inf for unused (padding) slots.
        # float32 for numerical stability regardless of AMP dtype.
        padded = torch.full(
            (batch_size, max_moves), float('-inf'),
            device=scores.device, dtype=torch.float32,
        )
        padded[row_idx, col_idx] = scores.float()

        # Positions with zero moves have an all-inf row: log_softmax(-inf,...) = nan.
        # Set those rows to 0 so softmax gives uniform output — their weight (w=0)
        # ensures they never contribute to the loss regardless.
        no_moves = move_counts == 0
        padded = torch.where(
            no_moves.unsqueeze(1).expand_as(padded), torch.zeros_like(padded), padded
        )

        # log_softmax handles numerical stability internally; -inf pads → 0 prob.
        log_probs = torch.nn.functional.log_softmax(padded, dim=1)  # (batch_size, max_moves)
        log_probs = torch.clamp(log_probs, min=-100.0)

        # Gather the log prob of the chosen move for each position.
        safe_targets = targets.long().clamp(0, max_moves - 1).unsqueeze(1)  # (batch_size, 1)
        chosen_log_probs = log_probs.gather(1, safe_targets).squeeze(1)     # (batch_size,)

        # Mask entries with zero moves or out-of-range target index (int32 comparisons).
        valid = (move_counts > 0) & (targets >= 0) & (targets < move_counts)

        # Build per-sample weights (zero for invalid entries).
        if reward_weights is not None:
            w = reward_weights.float().squeeze(-1) * valid.float()
        else:
            w = valid.float()

        # Avoid sync from `if total_weight == 0` — use max(sum, 1) instead.
        # When all weights are zero the numerator is also zero, so result = 0.
        total_weight = w.sum().clamp(min=1.0)

        return -(chosen_log_probs * w).sum() / total_weight

    def _compute_loss_padded(
        self,
        scores: torch.Tensor,
        move_counts: torch.Tensor,
        targets: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss from padded score matrix (training-optimized path).

        Scores arrive as (batch, max_moves) with -inf for invalid positions
        from forward_padded — no scatter/gather needed.
        """
        batch_size = scores.shape[0]
        if batch_size == 0:
            return scores.sum() * 0

        # move_counts/targets are int32; use directly for comparisons,
        # cast to int64 only for gather (which requires LongTensor).
        # Zero-move rows are all-inf → replace with 0 for stable softmax
        no_moves = move_counts == 0
        scores = scores.masked_fill(no_moves.unsqueeze(1), 0.0)

        log_probs = torch.nn.functional.log_softmax(scores.float(), dim=1)
        log_probs = torch.clamp(log_probs, min=-100.0)

        max_moves = scores.shape[1]
        safe_targets = targets.long().clamp(0, max_moves - 1).unsqueeze(1)
        chosen_log_probs = log_probs.gather(1, safe_targets).squeeze(1)

        valid = (move_counts > 0) & (targets >= 0) & (targets < move_counts)

        if reward_weights is not None:
            w = reward_weights.float().squeeze(-1) * valid.float()
        else:
            w = valid.float()

        total_weight = w.sum().clamp(min=1.0)
        return -(chosen_log_probs * w).sum() / total_weight

    def train(self) -> None:
        """Run the full training loop."""
        print("\n" + "=" * 50)
        print("Filipino Dama - ML Training")
        print("=" * 50)
        # Print model architecture
        total_params = sum(p.numel() for p in self.model.parameters())
        model_mb = total_params * 4 / 1e6  # FP32 size
        value_str = f", value_head={self.config.value_head_hidden}h" if self.config.value_head_enabled else ""
        print(f"Model: {self.config.model_channels}ch, {self.config.model_blocks} blocks, "
              f"{self.config.model_embedding} emb, {self.config.model_hidden} hidden{value_str} "
              f"({total_params:,} params, ~{model_mb:.1f}MB FP32)")
        print(f"Reward mode: {self.config.reward_mode}")
        if self.config.reward_mode == 'cycle':
            print("  (Odd epochs use scoring, even epochs use uniform weights)")
        accum = self.config.gradient_accumulation_steps
        if accum > 1:
            effective_bs = self.config.batch_size * accum
            print(f"Gradient accumulation: {accum} steps (effective batch size: {effective_bs})")
        if self.config.thermal_enabled:
            rest_min = self.config.thermal_rest_seconds / 60
            print(f"Thermal protection: ON (limit={self.config.thermal_temp_limit_c}°C, "
                  f"rest={rest_min:.0f}m, check every {self.config.thermal_check_every}s)")
        _simultaneous = self.config.pipeline_mode == 'simultaneous'
        print(f"Pipeline mode: {self.config.pipeline_mode} "
              f"({'CPU self-play overlaps GPU training' if _simultaneous else 'generate data first, then train'})")

        # Set start time for stats
        if not self.stats.start_time:
            self.stats.start_time = datetime.now().isoformat()

        # Generate initial self-play data if needed
        entry_count = self.replay_buffer.count_entries()
        if entry_count < self.config.batch_size * 10:
            print(f"\nInsufficient training data ({entry_count} entries)")
            self.run_selfplay(self.config.selfplay_games)

        # Prepare data
        print("\nPreparing training data...")
        train_entries, _ = prepare_training_data(
            self.replay_buffer, max_entries=self.config.replay_max_entries, val_split=0.0)
        print(f"Training entries: {len(train_entries)}")

        if self.config.clear_replay_after_load:
            deleted = self.replay_buffer.clear_files()
            if deleted:
                print(f"Cleared {deleted} replay files after loading")

        if not train_entries:
            print("ERROR: No training data available")
            return

        # Create dataloader
        effective_workers = self.config.dataloader_workers
        
        print(f"Creating DataLoader with {effective_workers} workers...")
        sys.stdout.flush()
        dataloader = create_dataloader(
            train_entries,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=effective_workers,
            pin_memory=self.config.pin_memory,
            use_ram_cache=self.config.ram_cache_enabled,
            ram_threshold_gb=self.config.ram_cache_threshold_gb,
            device=self.device,
            capacity=self.config.replay_max_entries if _simultaneous else 0,
            max_moves_per_sample=self.config.max_moves_per_sample,
            amp_enabled=self.config.amp,
        )
        _is_fast = isinstance(dataloader, FastBatchIterator)
        self._use_padded = _is_fast or isinstance(getattr(dataloader, 'dataset', None), CachedTensorDataset)
        # Track dataset for incremental updates in background self-play
        if _is_fast:
            self._current_dataset = dataloader.dataset
        else:
            self._current_dataset = None
        _gpu_resident = getattr(dataloader, 'on_gpu', False)
        _path_label = " (GPU-resident)" if _gpu_resident else (" (fast tensor indexing)" if _is_fast else (" (padded training path)" if self._use_padded else ""))
        print(f"DataLoader ready with {len(dataloader)} batches{_path_label}.")
        sys.stdout.flush()

        # Training loop
        print(f"\nStarting training from step {self.step}...")
        print("(First batch may take a moment to load...)")
        sys.stdout.flush()
        start_time = time.time()

        # Track last test step (init to current step so resumed runs don't
        # immediately trigger a test before the first test_every interval)
        last_test_step = self.step
        loss = 0.0  # Initialize loss in case loop doesn't run
        _consecutive_dead_epochs = 0  # epochs where all batches were skipped (non-finite)
        _DEAD_EPOCH_RECOVERY_THRESHOLD = 3  # trigger checkpoint rollback after this many
        _stale_epochs = 0  # epochs since last data refresh
        _max_stale = self.config.max_stale_epochs  # 0 = unlimited

        # Async testing: run model evaluation on CPU in a background thread
        # so GPU training continues uninterrupted.  Testing uses CPU workers
        # (ProcessPoolExecutor) and never touches the GPU, so the only sync
        # cost is the ~2ms model save before spawning the thread.
        _async_test_thread = None
        _async_test_result = [None]   # mutable container for thread result
        _async_test_step = [0]
        _async_test_epoch = [0]

        def _start_async_test():
            nonlocal _async_test_thread
            # Save model with non_blocking D2H copies (same pattern as
            # _save_checkpoint) — avoids per-tensor CUDA sync overhead.
            _test_path = Path(self.config.checkpoint_dir) / "temp_async_test.pt"
            _test_path.parent.mkdir(parents=True, exist_ok=True)
            if self.device.type == 'cuda':
                _sd = {k: v.to('cpu', non_blocking=True)
                       for k, v in self.model.state_dict().items()}
                torch.cuda.current_stream().synchronize()
            else:
                _sd = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save({
                'model_state_dict': _sd,
                'arch_params': getattr(self.model, 'arch_params', {}),
                'step': self.step,
            }, _test_path)
            _async_test_step[0] = self.step
            _async_test_epoch[0] = self.epoch
            _async_test_result[0] = None

            _test_path_str = str(_test_path)
            _n_games = self.config.test_games
            _diff = self.config.test_difficulty
            _max_mv = self.config.selfplay_max_moves
            # [Pass 67] Reduced from 4 to 2 for async tests during simultaneous
            # mode.  Background self-play uses cpu_workers (11) processes; adding
            # 4 more test workers = 15 processes on 12 cores = oversubscription.
            # 2 test workers keeps total at 13, near core count. Sync tests
            # (run_test_vs_algo) keep 4 workers since self-play isn't running.
            _n_wk = min(self.config.cpu_workers, 2)

            def _worker():
                try:
                    _async_test_result[0] = self._run_test_cpu_only(
                        _test_path_str, _n_games, _diff, _max_mv, _n_wk)
                except Exception as e:
                    print(f"  [async test] error: {e}")
                finally:
                    try:
                        Path(_test_path_str).unlink(missing_ok=True)
                    except Exception:
                        pass

            _async_test_thread = threading.Thread(target=_worker, daemon=True)
            _async_test_thread.start()
            print(f"  [async test] started ({_n_games} games vs {_diff})")

        def _collect_async_test():
            nonlocal _async_test_thread
            if _async_test_thread is None:
                return
            if _async_test_thread.is_alive():
                return  # still running
            result = _async_test_result[0]
            if result is not None:
                self._record_test_result(
                    result, _async_test_step[0], _async_test_epoch[0])
            _async_test_thread = None
            _async_test_result[0] = None

        # Start continuous background self-play immediately so CPU is never idle.
        # Full game count (not half) — GPU epochs are ~100x faster than self-play,
        # so generating more data per cycle improves data freshness.
        if _simultaneous:
            self._start_background_selfplay(self.config.selfplay_games)

        while self.step < self.config.train_steps:
            if self._stopped:
                break

            # Check if stop time has been reached
            if self.config.stop_time and datetime.now() >= self.config.stop_time:
                print(f"\nStop time reached ({self.config.stop_time.strftime('%Y-%m-%d %H:%M')}). Saving and exiting...")
                break

            loss = self.train_epoch(dataloader, use_scoring=self._should_use_scoring())
            self.epoch += 1
            self.stats.epochs_completed = self.epoch
            # Throttle epoch prints: with ~245 epochs per 60s self-play cycle,
            # per-epoch prints add ~735ms of terminal I/O overhead on WSL2
            # (~3ms per print call with stdout flush).  Print every 50 epochs
            # to give periodic progress while keeping overhead at ~15 prints/cycle.
            if self.epoch % 50 == 0 or self.epoch == 1:
                scoring_label = "scoring" if self._should_use_scoring_for_epoch(self.epoch) else "no-scoring"
                current_lr = (self.scheduler.get_last_lr()[0]
                              if self.scheduler is not None
                              else self.config.learning_rate)
                print(f"\nEpoch {self.epoch} complete. Avg Loss: {loss:.4f}  "
                      f"[reward_mode={self.config.reward_mode}, this_epoch={scoring_label}, lr={current_lr:.2e}]")

            # --- Dead-epoch recovery: detect & recover from stuck non-finite state ---
            if getattr(self, '_last_epoch_batches', -1) == 0:
                _consecutive_dead_epochs += 1
                if _consecutive_dead_epochs >= _DEAD_EPOCH_RECOVERY_THRESHOLD:
                    print(f"\n{'='*60}")
                    print(f"WARNING: {_consecutive_dead_epochs} consecutive epochs with "
                          f"all batches skipped (non-finite scores).")
                    weights_corrupted = self._has_non_finite_tensors()
                    if weights_corrupted:
                        print("  Cause: model weights contain NaN/Inf")
                    else:
                        print("  Cause: FP16 overflow (weights finite in FP32, "
                              "but intermediate values overflow float16)")
                    # Find and load the most recent checkpoint
                    ckpt_dir = Path(self.config.checkpoint_dir)
                    ckpts = sorted(ckpt_dir.glob("model_step_*.pt"))
                    if ckpts:
                        last_ckpt = str(ckpts[-1])
                        print(f"  Rolling back to checkpoint: {last_ckpt}")
                        self._load_checkpoint(last_ckpt)
                        # Reset GradScaler with conservative scale to prevent
                        # re-triggering the same overflow.  Default init_scale=65536
                        # is too aggressive for trained models — use 1024 (same as
                        # the resume-without-scaler-state path).
                        if self.scaler is not None:
                            self.scaler = GradScaler(init_scale=2**10)
                            print(f"  GradScaler reset to conservative scale={self.scaler.get_scale():.0f}")
                    else:
                        print("  No checkpoints found — resetting model from scratch")
                        self._reset_model_state("No checkpoint for recovery")
                    print(f"{'='*60}\n")
                    sys.stdout.flush()
                    _consecutive_dead_epochs = 0
            else:
                _consecutive_dead_epochs = 0

            _stale_epochs += 1

            if _simultaneous:
                # Continuous background self-play: check if new data is ready
                bg_dataset, bg_incremental = self._collect_background_selfplay()
                if bg_dataset is not None:
                    print(f"Background self-play complete — refreshing DataLoader "
                          f"({len(bg_dataset)} entries)...")
                    dataloader, _gpu_resident = self._refresh_dataloader(
                        dataloader, bg_dataset, bg_incremental, effective_workers)
                    _stale_epochs = 0  # Fresh data arrived — reset counter
                    # Background thread is continuous — no need to restart
                elif _max_stale > 0 and _stale_epochs >= _max_stale:
                    # GPU has exhausted the current data — yield until fresh data
                    # arrives.  This saves thermal budget and prevents overfitting
                    # on memorized data (GPU trains ~100-144x faster than self-play).
                    # Use _data_ready_event for zero-latency wakeup instead of
                    # time.sleep(0.5) polling (saves up to 500ms per data arrival).
                    while not self._stopped:
                        # Respect pause commands while waiting
                        while self._paused and not self._stopped:
                            time.sleep(0.1)
                        if self._stopped:
                            break
                        # If background thread died, resume training on stale data
                        # rather than spinning forever.
                        if (self._bg_selfplay_thread is not None
                                and not self._bg_selfplay_thread.is_alive()):
                            print("Warning: background self-play thread died "
                                  "— resuming training on existing data")
                            _stale_epochs = 0
                            break
                        # Clear event BEFORE checking for data so that any
                        # signal set by the bg thread after our check is
                        # preserved for the wait() call.  Previous pattern
                        # (check → clear → wait) lost signals that arrived
                        # between check and clear, adding up to 2s latency.
                        self._data_ready_event.clear()
                        bg_dataset, bg_incremental = self._collect_background_selfplay()
                        if bg_dataset is not None:
                            print(f"Fresh data arrived after {_stale_epochs} stale epochs "
                                  f"— refreshing ({len(bg_dataset)} entries)...")
                            dataloader, _gpu_resident = self._refresh_dataloader(
                                dataloader, bg_dataset, bg_incremental, effective_workers)
                            _stale_epochs = 0
                            break
                        # No data yet — block until the bg thread signals or
                        # 2s timeout for stop/pause/thermal checks.
                        self._data_ready_event.wait(timeout=2.0)
                        # Check stop conditions while waiting
                        if self.config.stop_time and datetime.now() >= self.config.stop_time:
                            break
            else:
                # Alternate mode: generate data synchronously, then rebuild dataloader
                print("Running self-play (alternate mode)...")
                self.run_selfplay(self.config.selfplay_games)
                train_entries, _ = prepare_training_data(
                    self.replay_buffer, max_entries=self.config.replay_max_entries, val_split=0.0)
                if self.config.clear_replay_after_load:
                    deleted = self.replay_buffer.clear_files()
                    if deleted:
                        print(f"Cleared {deleted} replay files after loading")
                if train_entries:
                    dataset = CachedTensorDataset.from_entries(
                        train_entries, max_moves_per_sample=self.config.max_moves_per_sample, show_progress=True,
                    )
                    # Free old GPU tensors before allocating new ones
                    _was_gpu = getattr(dataloader, 'on_gpu', False)
                    if _was_gpu:
                        dataloader = None
                        torch.cuda.empty_cache()
                    dataloader = create_dataloader_from_dataset(
                        dataset,
                        batch_size=self.config.batch_size,
                        num_workers=effective_workers,
                        pin_memory=self.config.pin_memory,
                        device=self.device,
                        amp_enabled=self.config.amp,
                    )
                    self._use_padded = True
                    _gpu_resident = getattr(dataloader, 'on_gpu', False)

            # Collect completed async test (non-blocking)
            _collect_async_test()

            # Start async test if due and no test currently running
            if (self.config.test_vs_algo and
                self.step > 0 and
                self.step - last_test_step >= self.config.test_every and
                (_async_test_thread is None or not _async_test_thread.is_alive())):
                try:
                    _start_async_test()
                    last_test_step = self.step
                except Exception as e:
                    print(f"  [async test] failed to start: {e}")

        # Collect any in-flight async test before exit
        if _async_test_thread is not None and _async_test_thread.is_alive():
            print("Waiting for async test to complete...")
            _async_test_thread.join(timeout=30)
        _collect_async_test()

        # Wait for any background self-play to finish before exit
        if _simultaneous and self._bg_selfplay_thread is not None and self._bg_selfplay_thread.is_alive():
            print("Waiting for background self-play to finish...")
            self._bg_selfplay_thread.join(timeout=60)

        # Final checkpoint
        self._save_checkpoint(loss)

        # Final test vs algorithm (synchronous — training is done)
        if self.config.test_vs_algo:
            try:
                print("\nRunning final model evaluation...")
                self.run_test_vs_algo(num_games=self.config.test_games * 2)
            except Exception as e:
                print(f"Final test failed: {e}")

        elapsed = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Total steps: {self.step}")
        print(f"  Epochs: {self.epoch}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Final model: {self.config.latest_path}")
        
        # Print test summary
        if self.stats.test_history:
            latest_test = self.stats.test_history[-1]
            print(f"  Final ML Win Rate: {latest_test.get('ml_win_rate', 0)*100:.1f}%")

        # Export comprehensive statistics
        if self.stats_collector:
            try:
                self.stats_collector.print_session_summary()
                exports = self.stats_collector.export_all()
                print(f"\n  Statistics exported to: {self.config.stats_output_dir}/")
                for name, path in exports.items():
                    print(f"    {name}: {path}")
            except Exception as e:
                print(f"  Warning: Failed to export statistics: {e}")

    def pause(self) -> None:
        """Pause training."""
        self._paused = True

    def resume(self) -> None:
        """Resume training."""
        self._paused = False

    def stop(self) -> None:
        """Stop training."""
        self._stopped = True
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        gpu_mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        # Get recent loss from history
        recent_loss = None
        if self.stats.loss_history:
            recent_loss = self.stats.loss_history[-1].get('loss')
        
        return {
            'step': self.step,
            'epoch': self.epoch,
            'paused': self._paused,
            'device': str(self.device),
            'gpu_mem_mb': gpu_mem,
            'recent_loss': recent_loss,
            'best_loss': self.stats.best_loss if self.stats.best_loss != float('inf') else None,
        }

    def get_stats(self) -> TrainingStats:
        """Get the full training statistics."""
        return self.stats


def list_checkpoints(checkpoint_dir: str = 'models/checkpoints') -> list:
    """List available checkpoints sorted by step."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []
    
    checkpoints = []
    for f in checkpoint_path.glob('model_step_*.pt'):
        try:
            step = int(f.stem.split('_')[-1])
            checkpoints.append({
                'path': str(f),
                'step': step,
                'name': f.name,
            })
        except ValueError:
            continue
    
    # Sort by step
    checkpoints.sort(key=lambda x: x['step'])
    return checkpoints


def load_training_stats(stats_file: str = 'models/training_stats.json') -> Optional[TrainingStats]:
    """Load training statistics from file."""
    stats_path = Path(stats_file)
    if not stats_path.exists():
        return None
    
    try:
        with open(stats_path, 'r') as f:
            data = json.load(f)
        return TrainingStats.from_dict(data)
    except Exception:
        return None


def load_config_from_yaml(config_path: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Load training configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file
        profile: Optional profile name to apply (e.g., 'server', 'local', 'cpu')
    
    Returns:
        Dictionary of configuration values
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply profile if specified
    if profile and 'profiles' in config:
        if profile in config['profiles']:
            profile_config = config['profiles'][profile]
            # Deep merge profile into config
            def deep_merge(base: dict, override: dict) -> dict:
                result = base.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            config = deep_merge(config, profile_config)
            print(f"Applied profile: {profile}")
        else:
            print(f"Warning: Profile '{profile}' not found. Available: {list(config['profiles'].keys())}")
    
    return config


def _auto_detect_resume(resume_cfg: dict, paths_cfg: dict) -> Optional[str]:
    """Resolve the resume checkpoint path from YAML config.

    When ``resume.enabled`` is True and ``checkpoint_path`` is null (None),
    auto-detects the latest valid checkpoint in the checkpoint directory.
    Skips checkpoints whose model weights contain NaN/Inf when
    ``skip_corrupted`` is True (default).
    """
    if not resume_cfg.get('enabled', False):
        return None

    explicit = resume_cfg.get('checkpoint_path')
    if explicit:
        return str(explicit)

    # Auto-detect: find the latest checkpoint by step number.
    ckpt_dir = Path(paths_cfg.get('checkpoint_dir', 'models/checkpoints'))
    if not ckpt_dir.exists():
        return None

    import re as _re
    checkpoints = sorted(
        ckpt_dir.glob('model_step_*.pt'),
        key=lambda p: int(_re.search(r'(\d+)', p.stem).group(1))
        if _re.search(r'(\d+)', p.stem) else 0,
        reverse=True,
    )
    if not checkpoints:
        return None

    skip_corrupted = resume_cfg.get('skip_corrupted', True)
    if not skip_corrupted:
        # Take the latest without validation
        print(f"Auto-detected checkpoint: {checkpoints[0]}")
        return str(checkpoints[0])

    # Validate checkpoints (newest first), skip corrupted ones
    for ckpt in checkpoints:
        try:
            c = torch.load(ckpt, map_location='cpu', weights_only=True)
            sd = c.get('model_state_dict', c)
            if all(torch.isfinite(v).all() for v in sd.values()
                   if isinstance(v, torch.Tensor)):
                print(f"Auto-detected checkpoint: {ckpt}")
                return str(ckpt)
            print(f"  Skipping corrupted checkpoint: {ckpt}")
        except Exception as e:
            print(f"  Skipping unreadable checkpoint {ckpt}: {e}")
    return None


def config_from_yaml(yaml_config: Dict[str, Any]) -> TrainingConfig:
    """
    Convert YAML config dictionary to TrainingConfig dataclass.
    
    Args:
        yaml_config: Dictionary loaded from YAML file
    
    Returns:
        TrainingConfig instance
    """
    # Evaluate arithmetic expressions in all scalar values (e.g. 8192*2 → 16384).
    def _resolve(d: dict) -> dict:
        return {k: (_resolve(v) if isinstance(v, dict) else
                    [_eval_expr(i) for i in v] if isinstance(v, list) else
                    _eval_expr(v))
                for k, v in d.items()}
    yaml_config = _resolve(yaml_config)

    device_cfg = yaml_config.get('device', {})
    selfplay_cfg = yaml_config.get('selfplay', {})
    algo_vs_algo_cfg = selfplay_cfg.get('algo_vs_algo', {})
    training_cfg = yaml_config.get('training', {})
    dataloader_cfg = yaml_config.get('dataloader', {})
    testing_cfg = yaml_config.get('testing', {})
    paths_cfg = yaml_config.get('paths', {})
    resume_cfg = yaml_config.get('resume', {})
    time_cfg = yaml_config.get('time_limit', {})
    stats_cfg = yaml_config.get('statistics', {})
    model_cfg = yaml_config.get('model', {})
    lr_sched_cfg = training_cfg.get('lr_scheduler', {})
    value_head_cfg = yaml_config.get('value_head', {})
    thermal_cfg = yaml_config.get('thermal_protection', {})
    
    # Parse stop time if duration is set
    stop_time = None
    if time_cfg.get('enabled') and time_cfg.get('duration'):
        stop_time = parse_duration(time_cfg['duration'])
    
    return TrainingConfig(
        # Device settings
        device=device_cfg.get('type', 'cuda'),
        amp=device_cfg.get('amp', {}).get('enabled', True),
        amp_dtype=device_cfg.get('amp', {}).get('dtype', 'float16'),
        compile_model=device_cfg.get('compile', {}).get('enabled', True),
        compile_mode=device_cfg.get('compile', {}).get('mode', 'reduce-overhead'),
        matmul_precision=device_cfg.get('matmul_precision', 'medium'),
        # Model architecture
        model_channels=model_cfg.get('channels', 64),
        model_blocks=model_cfg.get('num_blocks', 4),
        model_embedding=model_cfg.get('embedding_size', 128),
        model_hidden=model_cfg.get('hidden_size', 64),
        # Self-play settings
        cpu_workers=selfplay_cfg.get('cpu_workers', max(2, os.cpu_count() or 2)),
        selfplay_games=selfplay_cfg.get('games_per_epoch', 500),
        selfplay_focus_side=selfplay_cfg.get('focus_side', 'both'),
        selfplay_opponent_focus=selfplay_cfg.get('opponent_focus', 'both'),
        selfplay_difficulties=selfplay_cfg.get('difficulties', ['medium']),
        selfplay_noise_prob=selfplay_cfg.get('noise_prob', 0.1),
        selfplay_max_moves=selfplay_cfg.get('max_moves_per_game', 200),
        pipeline_mode=selfplay_cfg.get('pipeline_mode', 'simultaneous'),
        max_stale_epochs=selfplay_cfg.get('max_stale_epochs', 0),
        # Algo-vs-algo settings
        algo_vs_algo_enabled=algo_vs_algo_cfg.get('enabled', False),
        algo_vs_algo_games=algo_vs_algo_cfg.get('games_per_epoch', 100),
        algo_vs_algo_difficulties=algo_vs_algo_cfg.get('difficulties', ['easy', 'medium', 'hard']),
        # Training settings
        batch_size=training_cfg.get('batch_size', 256),
        learning_rate=training_cfg.get('learning_rate', 3e-4),
        weight_decay=training_cfg.get('weight_decay', 1e-5),
        grad_clip_norm=training_cfg.get('grad_clip_norm'),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
        train_steps=training_cfg.get('train_steps', 999999999),
        checkpoint_every=training_cfg.get('checkpoint_every', 1000),
        reward_mode=training_cfg.get('reward_mode', 'cycle'),
        # LR scheduler settings
        lr_scheduler_enabled=lr_sched_cfg.get('enabled', False),
        lr_scheduler_type=lr_sched_cfg.get('type', 'cosine_warm_restarts'),
        lr_scheduler_T0=lr_sched_cfg.get('T_0', 500),
        lr_scheduler_T_mult=lr_sched_cfg.get('T_mult', 2),
        lr_scheduler_eta_min=lr_sched_cfg.get('eta_min', 1e-5),
        lr_warmup_steps=lr_sched_cfg.get('warmup_steps', 0),
        # Value head / TD learning
        value_head_enabled=value_head_cfg.get('enabled', False),
        value_head_hidden=value_head_cfg.get('hidden_size', 128),
        value_weight=value_head_cfg.get('value_weight', 0.5),
        # DataLoader settings
        dataloader_workers=dataloader_cfg.get('num_workers', 0),
        pin_memory=dataloader_cfg.get('pin_memory', True),
        ram_cache_enabled=dataloader_cfg.get('ram_cache', {}).get('enabled', True),
        ram_cache_threshold_gb=dataloader_cfg.get('ram_cache', {}).get('threshold_gb', 8.0),
        replay_max_entries=dataloader_cfg.get('replay_max_entries', 100000),
        clear_replay_after_load=dataloader_cfg.get('clear_replay_after_load', False),
        max_moves_per_sample=dataloader_cfg.get('max_moves_per_sample', 32),
        # Testing settings
        test_vs_algo=testing_cfg.get('enabled', False),
        test_every=testing_cfg.get('every_n_steps', 5000),
        test_games=testing_cfg.get('num_games', 50),
        test_difficulty=testing_cfg.get('difficulty', 'medium'),
        # Statistics collection settings
        stats_enabled=stats_cfg.get('enabled', True),
        stats_record_every=stats_cfg.get('record_every', 10),
        stats_system_every=stats_cfg.get('system_every', 500),
        stats_model_health_every=stats_cfg.get('model_health_every', 2000),
        stats_score_dist_every=stats_cfg.get('score_dist_every', 50),
        stats_buffer_size=stats_cfg.get('buffer_size', 50000),
        stats_flush_every=stats_cfg.get('flush_every', 5000),
        stats_output_dir=stats_cfg.get('output_dir', paths_cfg.get('log_dir', 'logs') + '/stats'),
        # Paths
        checkpoint_dir=paths_cfg.get('checkpoint_dir', 'models/checkpoints'),
        latest_path=paths_cfg.get('latest_model', 'models/latest.pt'),
        replay_dir=paths_cfg.get('replay_dir', 'data/replay'),
        log_dir=paths_cfg.get('log_dir', 'logs'),
        stats_file=paths_cfg.get('stats_file', 'models/training_stats.json'),
        # Thermal protection
        thermal_enabled=thermal_cfg.get('enabled', False),
        thermal_temp_limit_c=thermal_cfg.get('temp_limit_c', 90),
        thermal_rest_seconds=_parse_rest_duration(thermal_cfg.get('rest_duration', '5m')),
        thermal_check_every=thermal_cfg.get('check_every', 30),
        # Resume: when enabled with no specific path, auto-detect the latest checkpoint.
        # This matches the documented behavior ("null = auto-detect latest") in all
        # YAML configs.  Without this, resume.enabled=True + checkpoint_path=null
        # silently starts fresh instead of resuming.
        resume=_auto_detect_resume(resume_cfg, paths_cfg),
        stop_time=stop_time,
    )


def main():
    """Main entry point for command-line training."""
    parser = argparse.ArgumentParser(description='Train Filipino Dama ML model')

    # Config file support
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (e.g., config/training_config.yaml)')
    parser.add_argument('--profile', type=str, default=None,
                       help='Config profile to use (e.g., server, local, cpu)')

    # Device settings
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--amp-dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16'],
                       help='AMP dtype: float16 (default) or bfloat16 (recommended for AMD MI210)')
    parser.add_argument('--compile-model', action='store_true',
                       help='Use torch.compile for faster training')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead', 
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='Compilation mode: default (fast compile), reduce-overhead (fast run)')

    # Self-play settings
    parser.add_argument('--cpu-workers', type=int, default=10,
                       help='Number of parallel self-play workers')
    parser.add_argument('--selfplay-games', type=int, default=500,
                       help='Number of self-play games per iteration')
    parser.add_argument('--focus-side', type=str, default='both',
                       choices=['white', 'black', 'both'],
                       help='Which side to focus on during self-play vs algorithm')
    parser.add_argument('--opponent-focus', type=str, default='both',
                       choices=['ml', 'algorithm', 'both'],
                       help='Opponent type to focus on during self-play')
    parser.add_argument('--selfplay-difficulties', type=str, default='medium',
                       help='Comma-separated difficulties to cycle: easy,medium,hard,self')
    parser.add_argument('--noise-prob', type=float, default=0.1,
                       help='Probability of random move for exploration (0.0 to 1.0)')
    parser.add_argument('--max-moves', type=int, default=200,
                       help='Maximum moves per game before declaring draw')
    parser.add_argument('--pipeline-mode', type=str, default='simultaneous',
                       choices=['simultaneous', 'alternate'],
                       help='Pipeline mode: simultaneous (overlap CPU/GPU) or alternate (sequential)')

    # Algo-vs-algo settings
    parser.add_argument('--algo-vs-algo', action='store_true', default=False,
                       help='Enable algo-vs-algo games as additional training data')
    parser.add_argument('--algo-vs-algo-games', type=int, default=100,
                       help='Number of algo-vs-algo games per self-play epoch')
    parser.add_argument('--algo-vs-algo-difficulties', type=str, default='easy,medium,hard',
                       help='Comma-separated difficulties for algo-vs-algo matchups')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for regularization (0 to disable)')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--train-steps', type=int, default=10000,
                       help='Total training steps')
    parser.add_argument('--checkpoint-every', type=int, default=1000,
                       help='Steps between checkpoints')
    parser.add_argument('--reward-mode', type=str, default='cycle',
                       choices=['scoring', 'none', 'cycle'],
                       help='Reward scoring mode: scoring (always use), none (never use), cycle (alternate epochs)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps (effective batch = batch_size * N)')

    # DataLoader settings
    parser.add_argument('--dataloader-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--pin-memory', action='store_true',
                       help='Pin memory for faster GPU transfer')

    # Model testing settings
    parser.add_argument('--test-vs-algo', action='store_true',
                       help='Enable periodic testing against algorithm')
    parser.add_argument('--test-every', type=int, default=5000,
                       help='Steps between model tests')
    parser.add_argument('--test-games', type=int, default=50,
                       help='Number of test games per evaluation')
    parser.add_argument('--test-difficulty', type=str, default='medium',
                       choices=['easy', 'medium', 'hard', 'super_hard'],
                       help='Algorithm difficulty for testing')

    # Resume settings
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--resume-latest', action='store_true',
                       help='Resume from the latest checkpoint in models/checkpoints/')
    parser.add_argument('--train-duration', type=str, default=None,
                       help='Train for this duration (e.g., 2d, 4h, 30m, 1d12h)')

    args = parser.parse_args()

    # If config file provided, load it and use as base
    if args.config:
        print(f"Loading config from: {args.config}")
        yaml_config = load_config_from_yaml(args.config, args.profile)
        config = config_from_yaml(yaml_config)
        
        # Override with any CLI arguments that were explicitly provided
        # (detect via comparison with parser defaults)
        defaults = parser.parse_args([])
        explicit = {k for k, v in vars(args).items()
                     if v != getattr(defaults, k, None)}

        cli_map = {
            'device': 'device',
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate',
            'weight_decay': 'weight_decay',
            'grad_clip_norm': 'grad_clip_norm',
            'gradient_accumulation_steps': 'gradient_accumulation_steps',
            'train_steps': 'train_steps',
            'checkpoint_every': 'checkpoint_every',
            'reward_mode': 'reward_mode',
            'cpu_workers': 'cpu_workers',
            'selfplay_games': 'selfplay_games',
            'focus_side': 'selfplay_focus_side',
            'opponent_focus': 'selfplay_opponent_focus',
            'noise_prob': 'selfplay_noise_prob',
            'max_moves': 'selfplay_max_moves',
            'pipeline_mode': 'pipeline_mode',
            'dataloader_workers': 'dataloader_workers',
            'test_every': 'test_every',
            'test_games': 'test_games',
            'test_difficulty': 'test_difficulty',
            'amp_dtype': 'amp_dtype',
            'compile_mode': 'compile_mode',
        }
        for arg_name, config_attr in cli_map.items():
            if arg_name in explicit:
                setattr(config, config_attr, getattr(args, arg_name))
        # Handle special cases
        if 'no_amp' in explicit:
            config.amp = not args.no_amp
        if 'compile_model' in explicit:
            config.compile_model = args.compile_model
        if 'pin_memory' in explicit:
            config.pin_memory = args.pin_memory
        if 'test_vs_algo' in explicit:
            config.test_vs_algo = args.test_vs_algo
        if 'selfplay_difficulties' in explicit:
            config.selfplay_difficulties = [d.strip() for d in args.selfplay_difficulties.split(',')]
        if 'algo_vs_algo' in explicit:
            config.algo_vs_algo_enabled = args.algo_vs_algo
        if 'algo_vs_algo_games' in explicit:
            config.algo_vs_algo_games = args.algo_vs_algo_games
        if 'algo_vs_algo_difficulties' in explicit:
            config.algo_vs_algo_difficulties = [d.strip() for d in args.algo_vs_algo_difficulties.split(',')]
        if 'grad_clip_norm' in explicit:
            config.grad_clip_norm = args.grad_clip_norm if args.grad_clip_norm > 0 else None

        # Resume handling
        if args.resume:
            config.resume = args.resume
        elif args.resume_latest:
            import glob
            import re
            pattern = os.path.join(config.checkpoint_dir, 'model_step_*.pt')
            checkpoints = glob.glob(pattern)
            if checkpoints:
                def get_step(path):
                    match = re.search(r'model_step_(\d+)\.pt$', path)
                    return int(match.group(1)) if match else 0
                checkpoints.sort(key=get_step)
                config.resume = checkpoints[-1]
                print(f'Resuming from latest checkpoint: {config.resume}')

        if args.train_duration:
            config.stop_time = parse_duration(args.train_duration)

        if explicit:
            print(f"CLI overrides: {', '.join(sorted(explicit - {'config', 'profile', 'resume', 'resume_latest', 'train_duration'}))}")

        # Print loaded config summary
        print(f"Config loaded: batch_size={config.batch_size}, lr={config.learning_rate}, "
              f"workers={config.dataloader_workers}, amp={config.amp}")
    else:
        # Use command line arguments only
        # Handle --resume-latest
        resume_path = args.resume
        if args.resume_latest:
            import glob
            import re
            checkpoint_dir = 'models/checkpoints'
            pattern = os.path.join(checkpoint_dir, 'model_step_*.pt')
            checkpoints = glob.glob(pattern)
            if checkpoints:
                # Sort by step number to find the latest
                def get_step(path):
                    match = re.search(r'model_step_(\d+)\.pt$', path)
                    return int(match.group(1)) if match else 0
                checkpoints.sort(key=get_step)
                resume_path = checkpoints[-1]
                print(f'Resuming from latest checkpoint: {resume_path}')
            else:
                print('No checkpoints found in models/checkpoints/, starting fresh.')
                resume_path = None

        # Parse train duration
        stop_time = parse_duration(args.train_duration) if args.train_duration else None
        if stop_time:
            print(f'Training duration: {args.train_duration}')
            print(f'Training will stop at: {stop_time.strftime("%Y-%m-%d %H:%M:%S")}')

        config = TrainingConfig(
            # Device settings
            compile_mode=args.compile_mode,
            device=args.device,
            amp=not args.no_amp,
            amp_dtype=args.amp_dtype,
            compile_model=args.compile_model,
            # Self-play settings
            cpu_workers=args.cpu_workers,
            selfplay_games=args.selfplay_games,
            selfplay_focus_side=args.focus_side,
            selfplay_opponent_focus=args.opponent_focus,
            selfplay_difficulties=[d.strip() for d in args.selfplay_difficulties.split(',')],
            selfplay_noise_prob=args.noise_prob,
            selfplay_max_moves=args.max_moves,
            # Algo-vs-algo settings
            algo_vs_algo_enabled=args.algo_vs_algo,
            algo_vs_algo_games=args.algo_vs_algo_games,
            algo_vs_algo_difficulties=[d.strip() for d in args.algo_vs_algo_difficulties.split(',')],
            # Training settings
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            train_steps=args.train_steps,
            checkpoint_every=args.checkpoint_every,
            reward_mode=args.reward_mode,
            # DataLoader settings
            dataloader_workers=args.dataloader_workers,
            pin_memory=args.pin_memory,
            # Model testing settings
            test_vs_algo=args.test_vs_algo,
            test_every=args.test_every,
            test_games=args.test_games,
            test_difficulty=args.test_difficulty,
            # Resume settings
            resume=resume_path,
            stop_time=stop_time,
        )

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
