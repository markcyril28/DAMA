"""Deterministic teacher targets for the gated enhanced training stage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from ...game_state import GameState
from ...types import Move
from ..algorithmic.eval import evaluate
from .dataset import CachedTensorDataset, _batch_count
from .replay import ReplayEntry


@dataclass(frozen=True)
class TeacherTarget:
    """Per-move teacher values plus normalized policy and state value."""

    move_values: tuple[float, ...]
    policy: tuple[float, ...]
    state_value: float


def fixed_depth_move_values(state: GameState, depth: int = 3) -> tuple[float, ...]:
    """Evaluate every legal move with deterministic fixed-depth negamax."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    moves = state.legal_moves()
    values = []
    for move in moves:
        child = state.apply_move(move)
        values.append(-_negamax(child, depth - 1, float("-inf"), float("inf")))
    return tuple(values)


def soft_teacher_policy(
    move_values: Sequence[float],
    *,
    temperature: float = 1.0,
    hard_index: int | None = None,
    hard_label_blend: float = 0.25,
) -> tuple[float, ...]:
    """Convert teacher values to a soft policy with an optional hard-label blend."""
    if not move_values:
        return ()
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0.0 <= hard_label_blend < 1.0:
        raise ValueError("hard_label_blend must be within [0, 1)")
    if hard_index is not None and not 0 <= hard_index < len(move_values):
        raise ValueError("hard_index is outside the legal-move range")

    scaled = torch.tensor(move_values, dtype=torch.float64) / temperature
    policy = torch.softmax(scaled, dim=0)
    if hard_index is not None and hard_label_blend:
        policy *= 1.0 - hard_label_blend
        policy[hard_index] += hard_label_blend
    return tuple(float(value) for value in policy.tolist())


def build_teacher_target(
    entry: ReplayEntry,
    *,
    depth: int,
    temperature: float,
    value_scale: float,
    hard_label_blend: float,
) -> TeacherTarget:
    """Build reproducible enhanced-stage targets for one replay entry."""
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be finite and positive")
    state = (
        entry.state
        if isinstance(entry.state, GameState)
        else GameState.from_compact(entry.state)
    )
    values = fixed_depth_move_values(state, depth=depth)
    if len(values) != len(entry.legal_moves):
        raise RuntimeError("teacher value count does not match legal move count")
    replay_moves = [
        move if isinstance(move, Move) else Move.from_dict(move)
        for move in entry.legal_moves
    ]
    generated_moves = state.legal_moves()
    if replay_moves != generated_moves:
        raise RuntimeError(
            "replay legal-move order differs from the teacher search order")
    policy = soft_teacher_policy(
        tuple(value / value_scale for value in values),
        temperature=temperature,
        hard_index=entry.chosen_index,
        hard_label_blend=hard_label_blend,
    )
    state_value = math.tanh(max(values, default=0.0) / value_scale)
    return TeacherTarget(values, policy, state_value)


class EnhancedTensorDataset:
    """Padded tensors for soft teacher policy and teacher-evaluation value loss."""

    def __init__(
        self,
        entries: Sequence[ReplayEntry],
        *,
        max_moves_per_sample: int,
        teacher_depth: int,
        temperature: float,
        value_scale: float,
        hard_label_blend: float,
        show_progress: bool = True,
    ) -> None:
        self.base = CachedTensorDataset.from_entries(
            list(entries),
            max_moves_per_sample=max_moves_per_sample,
            show_progress=show_progress,
        )
        self.boards = self.base.boards
        self.move_features = self.base.move_features
        self.move_counts = self.base.move_counts
        self.targets = self.base.targets
        self.reward_weights = self.base.reward_weights
        self.teacher_probabilities = torch.zeros(
            (len(entries), max_moves_per_sample), dtype=torch.float32)
        teacher_values = torch.zeros(len(entries), dtype=torch.float32)

        for index, entry in enumerate(entries):
            target = build_teacher_target(
                entry,
                depth=teacher_depth,
                temperature=temperature,
                value_scale=value_scale,
                hard_label_blend=hard_label_blend,
            )
            if len(target.policy) > max_moves_per_sample:
                # Proofread 2026-08-25 C4: a sliced soft policy no longer sums
                # to one and the chosen-index clip disagrees with it, so any
                # sample that hits the cap is silently corrupted training data.
                # Latent at cap 32, but raise now so a lowered cap can never
                # poison labels quietly.
                raise RuntimeError(
                    f"entry {index} has {len(target.policy)} legal moves, "
                    f"exceeding max_moves_per_sample={max_moves_per_sample}; "
                    "teacher policy would be truncated and denormalized"
                )
            count = min(len(target.policy), max_moves_per_sample)
            if count:
                self.teacher_probabilities[index, :count] = torch.tensor(
                    target.policy[:count], dtype=torch.float32)
            teacher_values[index] = target.state_value
        self.value_targets = teacher_values

    def __len__(self) -> int:
        return len(self.boards)


class EnhancedBatchIterator:
    """Shuffle and batch an EnhancedTensorDataset without per-record collation."""

    on_gpu = False

    def __init__(
        self,
        dataset: EnhancedTensorDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.n = len(dataset)

    def __len__(self) -> int:
        return _batch_count(self.n, self.batch_size, self.drop_last)

    def __iter__(self) -> Iterable[tuple[torch.Tensor, ...]]:
        order = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        data = self.dataset
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            if self.drop_last and end - start < self.batch_size:
                break
            index = order[start:end]
            yield (
                data.boards[index].contiguous(memory_format=torch.channels_last),
                data.move_features[index],
                data.move_counts[index],
                data.targets[index],
                data.reward_weights[index],
                data.value_targets[index],
                data.teacher_probabilities[index],
            )


def create_enhanced_dataloader(
    entries: Sequence[ReplayEntry],
    *,
    batch_size: int,
    max_moves_per_sample: int,
    teacher_depth: int,
    temperature: float,
    value_scale: float,
    hard_label_blend: float,
    shuffle: bool = True,
    show_progress: bool = True,
) -> EnhancedBatchIterator:
    """Create the gated soft-policy and teacher-value training iterator."""
    dataset = EnhancedTensorDataset(
        entries,
        max_moves_per_sample=max_moves_per_sample,
        teacher_depth=teacher_depth,
        temperature=temperature,
        value_scale=value_scale,
        hard_label_blend=hard_label_blend,
        show_progress=show_progress,
    )
    return EnhancedBatchIterator(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=len(dataset) > batch_size,
    )


def soft_target_cross_entropy(
    scores: torch.Tensor,
    move_counts: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked soft-target cross entropy for padded move scores."""
    if scores.ndim != 2 or teacher_probabilities.shape != scores.shape:
        raise ValueError("scores and teacher_probabilities must have equal 2D shapes")
    width = scores.shape[1]
    valid_moves = torch.arange(width, device=scores.device).unsqueeze(0)
    valid_moves = valid_moves < move_counts.long().unsqueeze(1)
    stable_scores = scores.masked_fill(~valid_moves, float("-inf"))
    stable_scores = stable_scores.masked_fill((move_counts == 0).unsqueeze(1), 0.0)
    log_probs = torch.log_softmax(stable_scores, dim=1, dtype=torch.float32)
    targets = teacher_probabilities.float() * valid_moves.float()
    targets = targets / targets.sum(dim=1, keepdim=True).clamp(min=1.0e-12)
    per_sample = -(targets * torch.clamp(log_probs, min=-100.0)).sum(dim=1)
    valid_samples = move_counts > 0
    if sample_weights is None:
        weights = valid_samples.float()
    else:
        weights = sample_weights.float() * valid_samples.float()
    return (per_sample * weights).sum() / weights.sum().clamp(min=1.0)


def _negamax(state: GameState, depth: int, alpha: float, beta: float) -> float:
    if depth <= 0 or state.is_terminal():
        return float(evaluate(state))
    moves = state.legal_moves()
    if not moves:
        return -10000.0
    best = float("-inf")
    for move in moves:
        value = -_negamax(state.apply_move(move), depth - 1, -beta, -alpha)
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best
