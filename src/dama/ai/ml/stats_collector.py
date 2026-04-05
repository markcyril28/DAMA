"""
Comprehensive training statistics collector for optimization analysis.

Collects granular metrics across all phases of the training pipeline:
- Training loop: loss, gradients, throughput, convergence
- Self-play: game generation throughput, result distribution, move diversity
- Model health: parameter norms, gradient flow, score distributions
- System: GPU memory, utilization, throughput efficiency
- Evaluation: win rates over time, ELO estimation, confidence metrics

All metrics are stored in-memory with periodic flush to disk, and exported
as JSON + CSV for easy analysis. Designed to be non-intrusive to the
training loop with minimal overhead.

Usage:
    collector = StatsCollector(config)
    collector.record_training_step(step, loss, lr, grad_norm=..., ...)
    collector.record_selfplay_epoch(...)
    collector.record_model_health(model, step)
    report = collector.generate_session_report()
    collector.export_all(output_dir)
"""

import os
import csv
import json
import math
import time
import platform
import statistics
from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list) -> float:
    """Mean that handles empty lists and non-finite values."""
    finite = [v for v in values if math.isfinite(v)]
    return statistics.mean(finite) if finite else 0.0


def _safe_stdev(values: list) -> float:
    """Stdev that handles short lists."""
    finite = [v for v in values if math.isfinite(v)]
    return statistics.stdev(finite) if len(finite) >= 2 else 0.0


def _safe_median(values: list) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.median(finite) if finite else 0.0


def _percentile(sorted_values: list, pct: float) -> float:
    """Simple percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def _elo_from_win_rate(win_rate: float, draw_rate: float = 0.0) -> float:
    """Estimate ELO difference from win rate (vs 50% baseline).
    
    Uses the standard logistic model: ELO_diff = -400 * log10(1/score - 1)
    where score = win_rate + 0.5 * draw_rate.
    """
    score = win_rate + 0.5 * draw_rate
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


# ---------------------------------------------------------------------------
# Ring buffer for high-frequency time series
# ---------------------------------------------------------------------------

class MetricBuffer:
    """Fixed-size ring buffer for time-series metrics with summary statistics."""

    def __init__(self, maxlen: int = 50000):
        self._data: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._total_count: int = 0
        self._finite_count: int = 0
        self._running_sum: float = 0.0
        self._running_min: float = float('inf')
        self._running_max: float = float('-inf')

    def append(self, value: float, step: int, **extra):
        entry = {'step': step, 'value': value, **extra}
        self._data.append(entry)
        self._total_count += 1
        if math.isfinite(value):
            self._finite_count += 1
            self._running_sum += value
            self._running_min = min(self._running_min, value)
            self._running_max = max(self._running_max, value)

    @property
    def count(self) -> int:
        return self._total_count

    @property
    def running_mean(self) -> float:
        return self._running_sum / self._finite_count if self._finite_count > 0 else 0.0

    @property
    def running_min(self) -> float:
        return self._running_min if self._running_min != float('inf') else 0.0

    @property
    def running_max(self) -> float:
        return self._running_max if self._running_max != float('-inf') else 0.0

    def last_n(self, n: int = 100) -> List[Dict[str, Any]]:
        return list(self._data)[-n:]

    def last_n_values(self, n: int = 100) -> List[float]:
        return [e['value'] for e in list(self._data)[-n:]]

    def all_entries(self) -> List[Dict[str, Any]]:
        return list(self._data)

    def summary(self, window: int = 100) -> Dict[str, Any]:
        """Compute summary stats over the last `window` entries."""
        recent = self.last_n_values(window)
        return {
            'total_count': self._total_count,
            'buffered_count': len(self._data),
            'running_mean': self.running_mean,
            'running_min': self.running_min,
            'running_max': self.running_max,
            'recent_mean': _safe_mean(recent),
            'recent_stdev': _safe_stdev(recent),
            'recent_min': min(recent) if recent else 0.0,
            'recent_max': max(recent) if recent else 0.0,
        }


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class StatsCollector:
    """Centralized statistics collector for ML training optimization.
    
    Captures granular metrics across all pipeline phases and generates
    comprehensive reports for configuration optimization.
    """

    def __init__(
        self,
        output_dir: str = "logs/stats",
        session_id: Optional[str] = None,
        buffer_size: int = 50000,
        flush_every: int = 5000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        self.flush_every = flush_every

        # --- Training step metrics (high-frequency) ---
        self.loss = MetricBuffer(buffer_size)
        self.learning_rate = MetricBuffer(buffer_size)
        self.grad_norm_global = MetricBuffer(buffer_size)
        self.grad_norm_per_layer: Dict[str, MetricBuffer] = {}
        self.step_time_sec = MetricBuffer(buffer_size)
        self.throughput_samples_sec = MetricBuffer(buffer_size)
        self.batch_size_actual = MetricBuffer(buffer_size)

        # Score distribution
        self.score_mean = MetricBuffer(buffer_size)
        self.score_std = MetricBuffer(buffer_size)
        self.score_entropy = MetricBuffer(buffer_size)
        self.top1_margin = MetricBuffer(buffer_size)  # score[0] - score[1]

        # Loss decomposition (if result info is available)
        self.loss_on_wins = MetricBuffer(buffer_size)
        self.loss_on_losses = MetricBuffer(buffer_size)
        self.loss_on_draws = MetricBuffer(buffer_size)

        # --- GPU / system metrics ---
        self.gpu_mem_allocated_mb = MetricBuffer(buffer_size)
        self.gpu_mem_reserved_mb = MetricBuffer(buffer_size)
        self.gpu_utilization_pct = MetricBuffer(buffer_size)
        self.cpu_percent = MetricBuffer(buffer_size)
        self.ram_used_gb = MetricBuffer(buffer_size)

        # --- Model health (lower frequency) ---
        self.param_norms: Dict[str, MetricBuffer] = {}  # per-layer L2 norms
        self.param_means: Dict[str, MetricBuffer] = {}
        self.param_stds: Dict[str, MetricBuffer] = {}
        self.weight_update_ratios: Dict[str, MetricBuffer] = {}  # |Δw| / |w|
        self._prev_param_snapshot: Dict[str, torch.Tensor] = {}

        # BatchNorm running stats
        self.bn_running_mean_norms: Dict[str, MetricBuffer] = {}
        self.bn_running_var_means: Dict[str, MetricBuffer] = {}

        # --- Self-play metrics ---
        self.selfplay_records: List[Dict[str, Any]] = []

        # --- Evaluation metrics ---
        self.eval_records: List[Dict[str, Any]] = []

        # --- Epoch metrics ---
        self.epoch_records: List[Dict[str, Any]] = []

        # --- Replay buffer metrics ---
        self.replay_records: List[Dict[str, Any]] = []

        # --- Convergence tracking ---
        self._loss_ema = None
        self._loss_ema_alpha = 0.01  # Slow EMA for convergence detection
        self._loss_plateau_steps = 0
        self._loss_plateau_threshold = 1e-4
        self._step_since_last_flush = 0

        # --- Configuration snapshot (set by caller) ---
        self.config_snapshot: Dict[str, Any] = {}

        # --- Non-finite event counter ---
        self.nan_inf_events: List[Dict[str, Any]] = []

        # --- Checkpoint metadata ---
        self.checkpoint_records: List[Dict[str, Any]] = []

    # ===================================================================
    # Configuration
    # ===================================================================

    def set_config_snapshot(self, config: Dict[str, Any]) -> None:
        """Store a snapshot of the training config for the session report."""
        self.config_snapshot = config

    # ===================================================================
    # Training step recording
    # ===================================================================

    def record_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        batch_size: int = 0,
        step_time: Optional[float] = None,
        grad_norm: Optional[float] = None,
        grad_norms_per_layer: Optional[Dict[str, float]] = None,
        score_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record metrics for a single training step.
        
        Args:
            step: Global step number.
            loss: Training loss for this step.
            lr: Current learning rate.
            batch_size: Actual batch size for this step.
            step_time: Wall-clock time for this step in seconds.
            grad_norm: Global gradient norm (post-clipping).
            grad_norms_per_layer: Per-layer gradient norms.
            score_stats: Dict with keys 'mean', 'std', 'entropy', 'top1_margin'.
        """
        ts = datetime.now().isoformat()

        self.loss.append(loss, step, timestamp=ts)
        self.learning_rate.append(lr, step)

        if batch_size > 0:
            self.batch_size_actual.append(batch_size, step)

        if step_time is not None and step_time > 0:
            self.step_time_sec.append(step_time, step)
            if batch_size > 0:
                self.throughput_samples_sec.append(batch_size / step_time, step)

        if grad_norm is not None:
            self.grad_norm_global.append(grad_norm, step)

        if grad_norms_per_layer:
            for name, norm in grad_norms_per_layer.items():
                if name not in self.grad_norm_per_layer:
                    self.grad_norm_per_layer[name] = MetricBuffer(10000)
                self.grad_norm_per_layer[name].append(norm, step)

        if score_stats:
            if 'mean' in score_stats:
                self.score_mean.append(score_stats['mean'], step)
            if 'std' in score_stats:
                self.score_std.append(score_stats['std'], step)
            if 'entropy' in score_stats:
                self.score_entropy.append(score_stats['entropy'], step)
            if 'top1_margin' in score_stats:
                self.top1_margin.append(score_stats['top1_margin'], step)

        # Convergence tracking (EMA of loss)
        if math.isfinite(loss):
            if self._loss_ema is None:
                self._loss_ema = loss
            else:
                prev_ema = self._loss_ema
                self._loss_ema = self._loss_ema_alpha * loss + (1 - self._loss_ema_alpha) * self._loss_ema
                if abs(self._loss_ema - prev_ema) < self._loss_plateau_threshold:
                    self._loss_plateau_steps += 1
                else:
                    self._loss_plateau_steps = 0

        # Auto-flush
        self._step_since_last_flush += 1
        if self._step_since_last_flush >= self.flush_every:
            self.flush_incremental()
            self._step_since_last_flush = 0

    def record_non_finite_event(self, step: int, location: str, details: str = "") -> None:
        """Record a NaN/Inf event for stability analysis."""
        self.nan_inf_events.append({
            'step': step,
            'location': location,
            'details': details,
            'timestamp': datetime.now().isoformat(),
        })

    # ===================================================================
    # GPU / System metrics
    # ===================================================================

    def record_system_metrics(self, step: int) -> Dict[str, float]:
        """Capture current GPU, CPU, and RAM metrics.
        
        Returns a dict of the captured values for convenience.
        """
        metrics: Dict[str, float] = {}

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            self.gpu_mem_allocated_mb.append(alloc, step)
            self.gpu_mem_reserved_mb.append(reserved, step)
            metrics['gpu_mem_allocated_mb'] = alloc
            metrics['gpu_mem_reserved_mb'] = reserved

            # Try to get GPU utilization via rocm-smi / nvidia-smi parsing
            util = self._get_gpu_utilization()
            if util is not None:
                self.gpu_utilization_pct.append(util, step)
                metrics['gpu_utilization_pct'] = util

        if HAS_PSUTIL:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / 1e9
            self.cpu_percent.append(cpu_pct, step)
            self.ram_used_gb.append(ram_used_gb, step)
            metrics['cpu_percent'] = cpu_pct
            metrics['ram_used_gb'] = ram_used_gb

        return metrics

    # Lazily-initialised NVML handle shared across instances
    _nvml_handle = None
    _nvml_failed = False

    @classmethod
    def _ensure_nvml(cls) -> bool:
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
    def _get_gpu_utilization(cls) -> Optional[float]:
        """Read GPU utilization %. Prefers NVML (~0.1 ms) over subprocess."""
        # Fast path: NVML
        if cls._ensure_nvml():
            try:
                import pynvml
                rates = pynvml.nvmlDeviceGetUtilizationRates(cls._nvml_handle)
                return float(rates.gpu)
            except Exception:
                pass

        # Fallback: subprocess
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split('\n')[0])
        except Exception:
            pass

        try:
            import subprocess
            result = subprocess.run(
                ['rocm-smi', '--showuse', '--json'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for card in data.values():
                    if isinstance(card, dict):
                        for key in ('GPU use (%)', 'GPU Usage (%)', 'gpu_use_percent'):
                            if key in card:
                                return float(str(card[key]).rstrip('%'))
        except Exception:
            pass

        return None

    # ===================================================================
    # Model health
    # ===================================================================

    def record_model_health(self, model: torch.nn.Module, step: int) -> Dict[str, Any]:
        """Snapshot parameter norms, means, stds, and BatchNorm stats.

        Batches GPU→CPU transfers: collects all per-param tensors on device,
        then does a single .cpu() transfer for norms/means/stds and deltas.

        Also computes weight update ratios if a previous snapshot exists.
        Returns a summary dict.
        """
        summary: Dict[str, Any] = {}
        layer_summaries: Dict[str, Dict[str, float]] = {}

        # --- Phase 1: compute all stats on GPU without .item() ---
        names: List[str] = []
        gpu_norms: List[torch.Tensor] = []
        gpu_means: List[torch.Tensor] = []
        gpu_stds: List[torch.Tensor] = []
        gpu_deltas: List[torch.Tensor] = []
        has_prev: List[bool] = []

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                names.append(name)
                p = param.float()
                gpu_norms.append(p.norm())
                gpu_means.append(p.mean())
                gpu_stds.append(p.std() if p.numel() > 1
                                else torch.tensor(0.0, device=param.device))

                if name in self._prev_param_snapshot:
                    gpu_deltas.append(
                        (param - self._prev_param_snapshot[name]).float().norm())
                    has_prev.append(True)
                else:
                    has_prev.append(False)

                self._prev_param_snapshot[name] = param.detach().clone()

        if not names:
            summary['layer_count'] = 0
            summary['total_params'] = sum(p.numel() for p in model.parameters())
            summary['trainable_params'] = 0
            summary['layers'] = {}
            return summary

        # --- Phase 2: single CPU transfer ---
        all_tensors = gpu_norms + gpu_means + gpu_stds + gpu_deltas
        all_cpu = torch.stack(all_tensors).cpu().tolist()

        n = len(names)
        norms_cpu = all_cpu[:n]
        means_cpu = all_cpu[n:2 * n]
        stds_cpu = all_cpu[2 * n:3 * n]
        deltas_cpu = all_cpu[3 * n:]

        # --- Phase 3: populate buffers (pure Python, no GPU) ---
        delta_idx = 0
        for i, name in enumerate(names):
            norm, mean, std = norms_cpu[i], means_cpu[i], stds_cpu[i]

            if name not in self.param_norms:
                self.param_norms[name] = MetricBuffer(5000)
                self.param_means[name] = MetricBuffer(5000)
                self.param_stds[name] = MetricBuffer(5000)
                self.weight_update_ratios[name] = MetricBuffer(5000)

            self.param_norms[name].append(norm, step)
            self.param_means[name].append(mean, step)
            self.param_stds[name].append(std, step)

            if has_prev[i]:
                delta = deltas_cpu[delta_idx]
                delta_idx += 1
                ratio = delta / max(norm, 1e-8)
                self.weight_update_ratios[name].append(ratio, step)
                layer_summaries[name] = {
                    'norm': norm, 'mean': mean, 'std': std,
                    'update_ratio': ratio,
                }
            else:
                layer_summaries[name] = {
                    'norm': norm, 'mean': mean, 'std': std,
                }

        # BatchNorm running statistics — batch transfer
        bn_names: List[str] = []
        bn_tensors: List[torch.Tensor] = []
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    if module.running_mean is not None:
                        bn_names.append(f"bn_{name}")
                        bn_tensors.append(module.running_mean.norm())
                        bn_tensors.append(module.running_var.mean())

        if bn_tensors:
            bn_cpu = torch.stack(bn_tensors).cpu().tolist()
            for i, bn_key in enumerate(bn_names):
                if bn_key not in self.bn_running_mean_norms:
                    self.bn_running_mean_norms[bn_key] = MetricBuffer(5000)
                    self.bn_running_var_means[bn_key] = MetricBuffer(5000)
                self.bn_running_mean_norms[bn_key].append(bn_cpu[2 * i], step)
                self.bn_running_var_means[bn_key].append(bn_cpu[2 * i + 1], step)

        summary['layer_count'] = len(layer_summaries)
        summary['total_params'] = sum(p.numel() for p in model.parameters())
        summary['trainable_params'] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        summary['layers'] = layer_summaries
        return summary

    # ===================================================================
    # Score distribution (called from training loop)
    # ===================================================================

    @staticmethod
    def compute_score_stats_padded(
        scores: torch.Tensor, move_counts: torch.Tensor
    ) -> Dict[str, float]:
        """Vectorized score stats from padded (batch, max_moves) scores.

        Padded positions must be -inf.  Only 4 CUDA syncs total
        (mean, std, entropy, margin) vs O(batch_size) syncs in the
        per-position loop.

        Args:
            scores: (batch, max_moves) padded output scores (-inf for padding).
            move_counts: (batch,) number of valid moves per position.
        """
        result: Dict[str, float] = {}
        with torch.no_grad():
            max_moves = scores.shape[1]
            arange = torch.arange(max_moves, device=scores.device)
            valid_mask = arange.unsqueeze(0) < move_counts.unsqueeze(1)

            valid_scores = scores[valid_mask]
            finite_mask = torch.isfinite(valid_scores)
            finite_scores = valid_scores[finite_mask]
            if finite_scores.numel() == 0:
                return result

            result['mean'] = finite_scores.mean().item()
            result['std'] = (finite_scores.std().item()
                             if finite_scores.numel() > 1 else 0.0)

            # Filter to positions with >1 move for entropy / margin
            multi = move_counts > 1
            if not multi.any():
                return result

            ms = scores[multi].float()          # (N, max_moves)
            mc = move_counts[multi]              # (N,)
            mm = arange.unsqueeze(0) < mc.unsqueeze(1)  # (N, max_moves)

            # softmax: -inf slots → prob 0 naturally
            probs = torch.softmax(ms, dim=1)
            log_probs = torch.log_softmax(ms, dim=1)
            ent = -(probs * log_probs)
            ent[~mm] = 0.0
            result['entropy'] = ent.sum(dim=1).mean().item()

            # top-1 margin
            ss = ms.clone()
            ss[~mm] = -1e9
            sorted_s, _ = ss.sort(dim=1, descending=True)
            result['top1_margin'] = (sorted_s[:, 0] - sorted_s[:, 1]).mean().item()

        return result

    @staticmethod
    def compute_score_stats(
        scores: torch.Tensor, move_counts: torch.Tensor
    ) -> Dict[str, float]:
        """Compute distribution statistics over model output scores.

        Args:
            scores: (total_moves,) flat raw output scores.
            move_counts: (batch_size,) number of moves per position.

        Returns:
            Dict with mean, std, entropy, top1_margin.
        """
        result: Dict[str, float] = {}
        with torch.no_grad():
            finite_scores = scores[torch.isfinite(scores)]
            if finite_scores.numel() == 0:
                return result

            result['mean'] = finite_scores.mean().item()
            result['std'] = (finite_scores.std().item()
                             if finite_scores.numel() > 1 else 0.0)

            batch_size = move_counts.shape[0]
            max_moves = move_counts.max().item()

            multi = move_counts > 1
            if not multi.any():
                return result

            # Pad flat scores into (batch, max_moves) matrix
            padded = scores.new_full((batch_size, max_moves), float('-inf'))
            arange = torch.arange(max_moves, device=scores.device)
            mask = arange.unsqueeze(0) < move_counts.unsqueeze(1)

            offsets = torch.zeros(batch_size, dtype=torch.long,
                                 device=scores.device)
            offsets[1:] = move_counts[:-1].cumsum(0)
            flat_idx = (offsets.unsqueeze(1) + arange.unsqueeze(0)).clamp(
                max=scores.shape[0] - 1)
            padded[mask] = scores[flat_idx[mask]]

            ms = padded[multi].float()
            mc = move_counts[multi]
            mm = arange.unsqueeze(0) < mc.unsqueeze(1)

            probs = torch.softmax(ms, dim=1)
            log_probs = torch.log_softmax(ms, dim=1)
            ent = -(probs * log_probs)
            ent[~mm] = 0.0
            result['entropy'] = ent.sum(dim=1).mean().item()

            ss = ms.clone()
            ss[~mm] = -1e9
            sorted_s, _ = ss.sort(dim=1, descending=True)
            result['top1_margin'] = (sorted_s[:, 0] - sorted_s[:, 1]).mean().item()

        return result

    # ===================================================================
    # Gradient analysis (called from training loop)
    # ===================================================================

    @staticmethod
    def compute_gradient_stats(
        model: torch.nn.Module,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute global and per-layer gradient norms.

        Should be called AFTER loss.backward() but BEFORE optimizer.step()
        or grad clipping.  Uses a single CPU transfer for all parameter
        norms instead of per-parameter .item() syncs.

        Returns:
            (global_grad_norm, per_layer_norms)
        """
        names: List[str] = []
        norms: List[torch.Tensor] = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                names.append(name)
                norms.append(param.grad.float().norm())

        if not norms:
            return 0.0, {}

        # Single CUDA→CPU sync for all norms
        norms_cpu = torch.stack(norms).cpu().tolist()
        per_layer = dict(zip(names, norms_cpu))
        global_norm = math.sqrt(sum(n * n for n in norms_cpu))
        return global_norm, per_layer

    # ===================================================================
    # Self-play recording
    # ===================================================================

    def record_selfplay_epoch(
        self,
        step: int,
        epoch: int,
        num_games: int,
        num_entries: int,
        elapsed_sec: float,
        difficulty_distribution: Optional[Dict[str, int]] = None,
        result_distribution: Optional[Dict[str, int]] = None,
        game_lengths: Optional[List[int]] = None,
        avg_moves_per_position: Optional[float] = None,
    ) -> None:
        """Record statistics for a self-play data generation epoch."""
        record: Dict[str, Any] = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'num_games': num_games,
            'num_entries': num_entries,
            'elapsed_sec': elapsed_sec,
            'games_per_sec': num_games / max(elapsed_sec, 1e-6),
            'entries_per_sec': num_entries / max(elapsed_sec, 1e-6),
        }
        if difficulty_distribution:
            record['difficulty_distribution'] = difficulty_distribution
        if result_distribution:
            record['result_distribution'] = result_distribution
        if game_lengths:
            sorted_lengths = sorted(game_lengths)
            record['game_length_stats'] = {
                'mean': _safe_mean(game_lengths),
                'median': _safe_median(game_lengths),
                'stdev': _safe_stdev(game_lengths),
                'min': min(game_lengths),
                'max': max(game_lengths),
                'p25': _percentile(sorted_lengths, 25),
                'p75': _percentile(sorted_lengths, 75),
                'p95': _percentile(sorted_lengths, 95),
            }
        if avg_moves_per_position is not None:
            record['avg_legal_moves_per_position'] = avg_moves_per_position

        self.selfplay_records.append(record)

    # ===================================================================
    # Replay buffer recording
    # ===================================================================

    def record_replay_buffer_state(
        self,
        step: int,
        total_entries: int,
        num_files: int,
        total_size_bytes: Optional[int] = None,
        oldest_file_age_hours: Optional[float] = None,
        result_balance: Optional[Dict[str, float]] = None,
    ) -> None:
        """Snapshot the state of the replay buffer."""
        record: Dict[str, Any] = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'total_entries': total_entries,
            'num_files': num_files,
        }
        if total_size_bytes is not None:
            record['total_size_mb'] = total_size_bytes / 1e6
        if oldest_file_age_hours is not None:
            record['oldest_file_age_hours'] = oldest_file_age_hours
        if result_balance:
            record['result_balance'] = result_balance

        self.replay_records.append(record)

    # ===================================================================
    # Evaluation recording
    # ===================================================================

    def record_evaluation(
        self,
        step: int,
        epoch: int,
        test_record: Dict[str, Any],
    ) -> None:
        """Record a model-vs-algorithm evaluation result with derived metrics."""
        record = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **test_record,
        }

        # Compute ELO estimate
        wr = test_record.get('ml_win_rate', 0.0)
        dr = test_record.get('draw_rate', 0.0)
        record['estimated_elo_diff'] = _elo_from_win_rate(wr, dr)

        # Win rate trend (rolling over last 5 evaluations)
        recent_wrs = [r.get('ml_win_rate', 0.0) for r in self.eval_records[-4:]]
        recent_wrs.append(wr)
        if len(recent_wrs) >= 2:
            record['win_rate_trend'] = recent_wrs[-1] - recent_wrs[0]
            record['win_rate_rolling_mean'] = _safe_mean(recent_wrs)
        
        self.eval_records.append(record)

    # ===================================================================
    # Epoch recording
    # ===================================================================

    def record_epoch(
        self,
        epoch: int,
        step: int,
        avg_loss: float,
        num_batches: int,
        epoch_time_sec: float,
        data_refresh: bool = False,
    ) -> None:
        """Record epoch-level summary."""
        self.epoch_records.append({
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'epoch_time_sec': epoch_time_sec,
            'batches_per_sec': num_batches / max(epoch_time_sec, 1e-6),
            'data_refresh': data_refresh,
        })

    # ===================================================================
    # Checkpoint recording
    # ===================================================================

    def record_checkpoint(
        self,
        step: int,
        loss: float,
        path: str,
        save_time_sec: float = 0.0,
        file_size_mb: float = 0.0,
    ) -> None:
        """Record checkpoint save event."""
        self.checkpoint_records.append({
            'step': step,
            'loss': loss,
            'path': path,
            'timestamp': datetime.now().isoformat(),
            'save_time_sec': save_time_sec,
            'file_size_mb': file_size_mb,
        })

    # ===================================================================
    # Convergence / optimization signals
    # ===================================================================

    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Compute derived convergence and optimization signals."""
        metrics: Dict[str, Any] = {}

        # Loss EMA and plateau detection
        metrics['loss_ema'] = self._loss_ema
        metrics['loss_plateau_steps'] = self._loss_plateau_steps
        metrics['loss_is_plateauing'] = self._loss_plateau_steps > 500

        # Loss improvement rate (over different windows)
        for window_name, n in [('100', 100), ('1000', 1000), ('5000', 5000)]:
            recent = self.loss.last_n_values(n)
            if len(recent) >= 10:
                first_half = _safe_mean(recent[:len(recent) // 2])
                second_half = _safe_mean(recent[len(recent) // 2:])
                improvement = first_half - second_half  # positive = improving
                metrics[f'loss_improvement_{window_name}'] = improvement
                # Relative improvement
                if first_half > 0:
                    metrics[f'loss_improvement_pct_{window_name}'] = improvement / first_half * 100
            else:
                metrics[f'loss_improvement_{window_name}'] = None

        # Gradient health
        grad_recent = self.grad_norm_global.last_n_values(100)
        if grad_recent:
            metrics['grad_norm_mean'] = _safe_mean(grad_recent)
            metrics['grad_norm_stdev'] = _safe_stdev(grad_recent)
            metrics['grad_norm_max'] = max(grad_recent) if grad_recent else 0
            # Gradient explosion indicator
            metrics['grad_exploding'] = any(g > 100.0 for g in grad_recent)
            # Gradient vanishing indicator
            metrics['grad_vanishing'] = _safe_mean(grad_recent) < 1e-6

        # Throughput statistics
        throughput_recent = self.throughput_samples_sec.last_n_values(100)
        if throughput_recent:
            metrics['throughput_mean'] = _safe_mean(throughput_recent)
            metrics['throughput_stdev'] = _safe_stdev(throughput_recent)
            # Throughput stability (lower = more stable)
            if _safe_mean(throughput_recent) > 0:
                metrics['throughput_cv'] = (
                    _safe_stdev(throughput_recent) / _safe_mean(throughput_recent)
                )
            else:
                metrics['throughput_cv'] = 0

        # Step time statistics
        step_times = self.step_time_sec.last_n_values(100)
        if step_times:
            metrics['step_time_mean_sec'] = _safe_mean(step_times)
            metrics['step_time_stdev_sec'] = _safe_stdev(step_times)

        # ELO progress from evaluations
        if len(self.eval_records) >= 2:
            elos = [r.get('estimated_elo_diff', 0) for r in self.eval_records]
            metrics['elo_first'] = elos[0]
            metrics['elo_latest'] = elos[-1]
            metrics['elo_improvement'] = elos[-1] - elos[0]
            metrics['elo_max'] = max(elos)

        # Training efficiency
        total_elapsed = (datetime.now() - self.session_start).total_seconds()
        total_steps = self.loss.count
        if total_elapsed > 0 and total_steps > 0:
            metrics['overall_steps_per_sec'] = total_steps / total_elapsed
            metrics['overall_steps_per_hour'] = total_steps / total_elapsed * 3600

        # NaN/Inf stability
        metrics['nan_inf_event_count'] = len(self.nan_inf_events)
        if self.nan_inf_events:
            metrics['last_nan_inf_step'] = self.nan_inf_events[-1].get('step')

        return metrics

    # ===================================================================
    # Optimization recommendations
    # ===================================================================

    def generate_optimization_hints(self) -> List[Dict[str, str]]:
        """Generate actionable hints based on collected metrics.
        
        Returns a list of dicts with 'area', 'severity', 'hint' keys.
        Severity: 'info', 'warning', 'critical'.
        """
        hints: List[Dict[str, str]] = []
        conv = self.get_convergence_metrics()

        # --- Loss plateau ---
        if conv.get('loss_is_plateauing'):
            hints.append({
                'area': 'convergence',
                'severity': 'warning',
                'hint': (
                    f"Loss has been plateauing for {conv['loss_plateau_steps']} steps. "
                    "Consider: (1) reducing learning rate, (2) increasing self-play "
                    "diversity (noise_prob), (3) adding data from harder difficulties."
                ),
            })

        # --- Gradient issues ---
        if conv.get('grad_exploding'):
            hints.append({
                'area': 'stability',
                'severity': 'critical',
                'hint': (
                    "Gradient norms exceeding 100.0 detected. "
                    "Consider: (1) reducing learning rate, (2) reducing grad_clip_norm, "
                    "(3) checking for data quality issues."
                ),
            })
        if conv.get('grad_vanishing'):
            hints.append({
                'area': 'stability',
                'severity': 'warning',
                'hint': (
                    "Very small gradient norms detected (< 1e-6). "
                    "Consider: (1) increasing learning rate, (2) checking model architecture "
                    "for dead layers, (3) reviewing activation functions."
                ),
            })

        # --- Throughput ---
        cv = conv.get('throughput_cv', 0)
        if cv > 0.5:
            hints.append({
                'area': 'performance',
                'severity': 'warning',
                'hint': (
                    f"Throughput is highly variable (CV={cv:.2f}). "
                    "Possible causes: (1) dataloader bottleneck — increase num_workers or "
                    "prefetch_factor, (2) GPU thermal throttling, (3) competing processes."
                ),
            })

        # --- GPU memory ---
        gpu_recent = self.gpu_mem_allocated_mb.last_n_values(10)
        if gpu_recent and torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e6
            utilization = max(gpu_recent) / total_vram if total_vram > 0 else 0
            if utilization < 0.3:
                hints.append({
                    'area': 'performance',
                    'severity': 'info',
                    'hint': (
                        f"GPU VRAM utilization is only {utilization*100:.0f}%. "
                        "You could increase batch_size to better utilize the GPU."
                    ),
                })
            elif utilization > 0.95:
                hints.append({
                    'area': 'performance',
                    'severity': 'warning',
                    'hint': (
                        f"GPU VRAM utilization is {utilization*100:.0f}%, near capacity. "
                        "Risk of OOM. Consider reducing batch_size slightly."
                    ),
                })

        # --- NaN/Inf frequency ---
        if len(self.nan_inf_events) > 10:
            hints.append({
                'area': 'stability',
                'severity': 'critical',
                'hint': (
                    f"{len(self.nan_inf_events)} NaN/Inf events recorded. "
                    "This indicates numerical instability. Consider: (1) enabling AMP with "
                    "bfloat16 instead of float16, (2) reducing learning rate, (3) increasing "
                    "grad_clip_norm, (4) checking data for outliers."
                ),
            })

        # --- Self-play balance ---
        if self.selfplay_records:
            latest_sp = self.selfplay_records[-1]
            rd = latest_sp.get('result_distribution', {})
            total = sum(rd.values()) if rd else 0
            if total > 0:
                wins = rd.get('ml_win', 0) + rd.get('p1_win', 0) + rd.get('p2_win', 0)
                draws_count = rd.get('draw', 0)
                draw_pct = draws_count / total
                if draw_pct > 0.5:
                    hints.append({
                        'area': 'data_quality',
                        'severity': 'warning',
                        'hint': (
                            f"High draw rate ({draw_pct*100:.0f}%) in self-play. "
                            "This may limit learning signal. Consider: (1) increasing "
                            "max_moves_per_game, (2) using more aggressive difficulties."
                        ),
                    })

        # --- Score entropy ---
        entropy_recent = self.score_entropy.last_n_values(100)
        if entropy_recent:
            avg_entropy = _safe_mean(entropy_recent)
            if avg_entropy < 0.1:
                hints.append({
                    'area': 'model_behavior',
                    'severity': 'warning',
                    'hint': (
                        f"Average score entropy is very low ({avg_entropy:.3f}). "
                        "The model is very confident — may be overfitting or not exploring. "
                        "Consider: (1) increasing noise_prob, (2) adding temperature to scores."
                    ),
                })

        return hints

    # ===================================================================
    # Incremental flush (periodic save during training)
    # ===================================================================

    def flush_incremental(self) -> None:
        """Save incremental stats to a JSONL file for crash recovery."""
        path = self.output_dir / f"incremental_{self.session_id}.jsonl"
        try:
            # Write just the latest metrics snapshot
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'loss_summary': self.loss.summary(100),
                'grad_norm_summary': self.grad_norm_global.summary(100),
                'throughput_summary': self.throughput_samples_sec.summary(100),
                'step_time_summary': self.step_time_sec.summary(100),
                'convergence': self.get_convergence_metrics(),
            }
            with open(path, 'a') as f:
                f.write(json.dumps(snapshot) + '\n')
        except Exception:
            pass  # Non-critical — don't disrupt training

    # ===================================================================
    # Session report generation
    # ===================================================================

    def generate_session_report(self) -> Dict[str, Any]:
        """Generate a comprehensive session report for analysis.
        
        This is the main output — a single JSON object capturing the
        full training session with all metrics, summaries, and
        optimization hints.
        """
        session_end = datetime.now()
        elapsed = (session_end - self.session_start).total_seconds()

        report: Dict[str, Any] = {
            'meta': {
                'session_id': self.session_id,
                'start_time': self.session_start.isoformat(),
                'end_time': session_end.isoformat(),
                'elapsed_seconds': elapsed,
                'elapsed_human': str(timedelta(seconds=int(elapsed))),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'gpu_name': (
                    torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'
                ),
                'gpu_vram_gb': (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                    if torch.cuda.is_available() else 0
                ),
            },
            'config': self.config_snapshot,
            'summary': {
                'total_steps': self.loss.count,
                'total_epochs': len(self.epoch_records),
                'total_selfplay_epochs': len(self.selfplay_records),
                'total_evaluations': len(self.eval_records),
                'total_checkpoints': len(self.checkpoint_records),
                'nan_inf_events': len(self.nan_inf_events),
            },
            'loss': {
                'summary': self.loss.summary(1000),
                'first_100': self.loss.last_n(100) if self.loss.count <= 100 else [],
                'last_100': self.loss.last_n(100),
            },
            'learning_rate': {
                'summary': self.learning_rate.summary(100),
                'last_10': self.learning_rate.last_n(10),
            },
            'gradient_norms': {
                'global_summary': self.grad_norm_global.summary(1000),
                'per_layer_summaries': {
                    name: buf.summary(100)
                    for name, buf in self.grad_norm_per_layer.items()
                },
            },
            'throughput': {
                'samples_per_sec': self.throughput_samples_sec.summary(1000),
                'step_time_sec': self.step_time_sec.summary(1000),
            },
            'score_distribution': {
                'mean_summary': self.score_mean.summary(1000),
                'std_summary': self.score_std.summary(1000),
                'entropy_summary': self.score_entropy.summary(1000),
                'top1_margin_summary': self.top1_margin.summary(1000),
            },
            'gpu_memory': {
                'allocated_mb': self.gpu_mem_allocated_mb.summary(100),
                'reserved_mb': self.gpu_mem_reserved_mb.summary(100),
            },
            'system': {
                'gpu_utilization': self.gpu_utilization_pct.summary(100),
                'cpu_percent': self.cpu_percent.summary(100),
                'ram_used_gb': self.ram_used_gb.summary(100),
            },
            'model_health': {
                'param_norm_summaries': {
                    name: buf.summary(50)
                    for name, buf in self.param_norms.items()
                },
                'weight_update_ratio_summaries': {
                    name: buf.summary(50)
                    for name, buf in self.weight_update_ratios.items()
                },
                'bn_running_mean_norms': {
                    name: buf.summary(50)
                    for name, buf in self.bn_running_mean_norms.items()
                },
                'bn_running_var_means': {
                    name: buf.summary(50)
                    for name, buf in self.bn_running_var_means.items()
                },
            },
            'selfplay': self.selfplay_records,
            'evaluations': self.eval_records,
            'epochs': self.epoch_records,
            'replay_buffer': self.replay_records,
            'checkpoints': self.checkpoint_records,
            'nan_inf_events': self.nan_inf_events,
            'convergence': self.get_convergence_metrics(),
            'optimization_hints': self.generate_optimization_hints(),
        }

        return report

    # ===================================================================
    # Export methods
    # ===================================================================

    def export_session_report(self) -> str:
        """Generate and save the full session report as JSON.
        
        Returns the path to the saved report.
        """
        report = self.generate_session_report()
        path = self.output_dir / f"session_report_{self.session_id}.json"
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return str(path)

    def export_loss_csv(self) -> str:
        """Export loss history to CSV for external analysis."""
        path = self.output_dir / f"loss_history_{self.session_id}.csv"
        entries = self.loss.all_entries()
        if not entries:
            return str(path)

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'value', 'timestamp'])
            writer.writeheader()
            for entry in entries:
                writer.writerow({
                    'step': entry.get('step', ''),
                    'value': entry.get('value', ''),
                    'timestamp': entry.get('timestamp', ''),
                })
        return str(path)

    def export_gradient_csv(self) -> str:
        """Export global gradient norm history to CSV."""
        path = self.output_dir / f"gradient_norms_{self.session_id}.csv"
        entries = self.grad_norm_global.all_entries()
        if not entries:
            return str(path)

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'value'])
            writer.writeheader()
            for entry in entries:
                writer.writerow({
                    'step': entry.get('step', ''),
                    'value': entry.get('value', ''),
                })
        return str(path)

    def export_throughput_csv(self) -> str:
        """Export throughput history to CSV."""
        path = self.output_dir / f"throughput_{self.session_id}.csv"
        entries = self.throughput_samples_sec.all_entries()
        if not entries:
            return str(path)

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'value'])
            writer.writeheader()
            for entry in entries:
                writer.writerow({
                    'step': entry.get('step', ''),
                    'value': entry.get('value', ''),
                })
        return str(path)

    def export_evaluations_csv(self) -> str:
        """Export evaluation results to CSV."""
        path = self.output_dir / f"evaluations_{self.session_id}.csv"
        if not self.eval_records:
            return str(path)

        fieldnames = [
            'step', 'epoch', 'timestamp', 'total_games',
            'ml_wins', 'algo_wins', 'draws',
            'ml_win_rate', 'ml_as_p1_win_rate', 'ml_as_p2_win_rate',
            'avg_game_length', 'estimated_elo_diff',
            'win_rate_trend', 'win_rate_rolling_mean',
        ]
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for record in self.eval_records:
                writer.writerow(record)
        return str(path)

    def export_epochs_csv(self) -> str:
        """Export epoch-level summary to CSV."""
        path = self.output_dir / f"epochs_{self.session_id}.csv"
        if not self.epoch_records:
            return str(path)

        fieldnames = [
            'epoch', 'step', 'timestamp', 'avg_loss', 'num_batches',
            'epoch_time_sec', 'batches_per_sec', 'data_refresh',
        ]
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for record in self.epoch_records:
                writer.writerow(record)
        return str(path)

    def export_selfplay_csv(self) -> str:
        """Export self-play generation stats to CSV."""
        path = self.output_dir / f"selfplay_{self.session_id}.csv"
        if not self.selfplay_records:
            return str(path)

        fieldnames = [
            'step', 'epoch', 'timestamp', 'num_games', 'num_entries',
            'elapsed_sec', 'games_per_sec', 'entries_per_sec',
        ]
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for record in self.selfplay_records:
                writer.writerow(record)
        return str(path)

    def export_system_csv(self) -> str:
        """Export system metrics time series to CSV."""
        path = self.output_dir / f"system_metrics_{self.session_id}.csv"
        
        # Merge GPU and system metrics by step
        gpu_entries = {e['step']: e for e in self.gpu_mem_allocated_mb.all_entries()}
        cpu_entries = {e['step']: e for e in self.cpu_percent.all_entries()}
        ram_entries = {e['step']: e for e in self.ram_used_gb.all_entries()}
        gpu_util_entries = {e['step']: e for e in self.gpu_utilization_pct.all_entries()}
        gpu_reserved = {e['step']: e for e in self.gpu_mem_reserved_mb.all_entries()}

        all_steps = sorted(set(
            list(gpu_entries.keys()) + list(cpu_entries.keys()) +
            list(ram_entries.keys()) + list(gpu_util_entries.keys()) +
            list(gpu_reserved.keys())
        ))

        if not all_steps:
            return str(path)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'gpu_mem_allocated_mb', 'gpu_mem_reserved_mb',
                'gpu_utilization_pct', 'cpu_percent', 'ram_used_gb',
            ])
            for step in all_steps:
                writer.writerow([
                    step,
                    gpu_entries.get(step, {}).get('value', ''),
                    gpu_reserved.get(step, {}).get('value', ''),
                    gpu_util_entries.get(step, {}).get('value', ''),
                    cpu_entries.get(step, {}).get('value', ''),
                    ram_entries.get(step, {}).get('value', ''),
                ])
        return str(path)

    def export_all(self) -> Dict[str, str]:
        """Export all statistics to multiple files.
        
        Returns a dict mapping export name to file path.
        """
        exports = {}
        exports['session_report'] = self.export_session_report()
        exports['loss_csv'] = self.export_loss_csv()
        exports['gradient_csv'] = self.export_gradient_csv()
        exports['throughput_csv'] = self.export_throughput_csv()
        exports['evaluations_csv'] = self.export_evaluations_csv()
        exports['epochs_csv'] = self.export_epochs_csv()
        exports['selfplay_csv'] = self.export_selfplay_csv()
        exports['system_csv'] = self.export_system_csv()
        return exports

    # ===================================================================
    # Summary for console output
    # ===================================================================

    def print_session_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        elapsed = (datetime.now() - self.session_start).total_seconds()
        conv = self.get_convergence_metrics()

        print("\n" + "=" * 60)
        print("  TRAINING SESSION STATISTICS")
        print("=" * 60)
        print(f"  Session ID:     {self.session_id}")
        print(f"  Duration:       {timedelta(seconds=int(elapsed))}")
        print(f"  Total Steps:    {self.loss.count:,}")
        print(f"  Total Epochs:   {len(self.epoch_records)}")

        # Loss
        loss_s = self.loss.summary(100)
        print(f"\n  Loss:")
        print(f"    Latest (avg 100):   {loss_s['recent_mean']:.6f}")
        print(f"    Best:               {loss_s['running_min']:.6f}")
        print(f"    Std (last 100):     {loss_s['recent_stdev']:.6f}")
        if self._loss_ema is not None:
            print(f"    EMA:                {self._loss_ema:.6f}")
        imp = conv.get('loss_improvement_1000')
        if imp is not None:
            print(f"    Improvement (1K):   {imp:+.6f}")

        # Throughput
        tp = self.throughput_samples_sec.summary(100)
        if tp['total_count'] > 0:
            print(f"\n  Throughput:")
            print(f"    Samples/sec:        {tp['recent_mean']:.1f} (std={tp['recent_stdev']:.1f})")
            sps = conv.get('overall_steps_per_hour', 0)
            if sps:
                print(f"    Steps/hour:         {sps:,.0f}")

        # Gradients
        gs = self.grad_norm_global.summary(100)
        if gs['total_count'] > 0:
            print(f"\n  Gradient Norms:")
            print(f"    Mean (last 100):    {gs['recent_mean']:.4f}")
            print(f"    Max (last 100):     {gs['recent_max']:.4f}")

        # GPU
        gpu_s = self.gpu_mem_allocated_mb.summary(10)
        if gpu_s['total_count'] > 0:
            print(f"\n  GPU Memory:")
            print(f"    Allocated:          {gpu_s['recent_mean']:.0f} MB")

        # Evaluations
        if self.eval_records:
            latest = self.eval_records[-1]
            print(f"\n  Evaluation (latest):")
            print(f"    ML Win Rate:        {latest.get('ml_win_rate', 0)*100:.1f}%")
            print(f"    Est. ELO diff:      {latest.get('estimated_elo_diff', 0):+.0f}")
            if len(self.eval_records) >= 2:
                first_wr = self.eval_records[0].get('ml_win_rate', 0)
                last_wr = latest.get('ml_win_rate', 0)
                print(f"    WR Progress:        {first_wr*100:.1f}% → {last_wr*100:.1f}%")

        # Stability
        if self.nan_inf_events:
            print(f"\n  Stability:")
            print(f"    NaN/Inf Events:     {len(self.nan_inf_events)}")

        # Optimization hints
        hints = self.generate_optimization_hints()
        if hints:
            print(f"\n  Optimization Hints ({len(hints)}):")
            for h in hints:
                severity_icon = {'info': 'ℹ', 'warning': '⚠', 'critical': '‼'}
                icon = severity_icon.get(h['severity'], '•')
                print(f"    {icon} [{h['area']}] {h['hint'][:120]}")

        print("=" * 60)
