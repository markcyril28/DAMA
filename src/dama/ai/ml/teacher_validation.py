"""Frozen teacher-suite creation, evaluation, and checkpoint promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from ...game_state import GameState
from ...types import Move
from ..algorithmic.search import get_best_move
from .corpus import canonical_state_key
from .dataset import CachedTensorDataset
from .replay import ReplayEntry


SUITE_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_move_index(legal_moves: Sequence[Move], chosen: Optional[Move]) -> int:
    if not legal_moves:
        raise ValueError("Cannot label a state without legal moves")
    if chosen is None:
        raise RuntimeError("Teacher returned no move for a non-terminal state")
    try:
        return legal_moves.index(chosen)
    except ValueError:
        for index, move in enumerate(legal_moves):
            if move.path == chosen.path:
                return index
    raise RuntimeError("Teacher move is not present in the legal move list")


def _apply_opening(state: GameState, plies: int, rng: random.Random) -> GameState:
    for _ in range(max(0, plies)):
        legal_moves = state.legal_moves()
        if not legal_moves:
            break
        state = state.apply_move(rng.choice(legal_moves))
    return state


def _manifest_path(suite_path: Path) -> Path:
    return suite_path.with_suffix(suite_path.suffix + ".manifest.json")


def create_frozen_teacher_suite(
    suite_path: str,
    target_states: int = 5000,
    seed: int = 20260819,
    teacher_difficulty: str = "hard",
    opening_plies: Sequence[int] = (0, 2, 4, 6, 8),
    played_action_noise: float = 0.10,
    max_moves_per_game: int = 200,
    max_games: int = 20000,
    exclude_state_keys: Optional[set[str]] = None,
) -> dict:
    """Create one immutable, exact-size, fixed-seed teacher validation suite.

    Existing suites are validated and returned, never overwritten. Generation
    cost is O(s * teacher_search), where ``s`` is ``target_states``. Memory use
    is O(s) for exact canonical-state deduplication.
    """

    if target_states <= 0:
        raise ValueError("target_states must be positive")
    if not 0.0 <= played_action_noise <= 1.0:
        raise ValueError("played_action_noise must be between 0 and 1")
    opening_plies = tuple(int(value) for value in opening_plies) or (0,)
    path = Path(suite_path)
    manifest_path = _manifest_path(path)
    if path.exists() or manifest_path.exists():
        entries, manifest = load_frozen_teacher_suite(
            suite_path, expected_count=target_states
        )
        expected = {
            "seed": seed,
            "teacher_difficulty": teacher_difficulty,
            "opening_plies": list(opening_plies),
            "played_action_noise": played_action_noise,
            "max_moves_per_game": max_moves_per_game,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"Existing frozen suite uses {key}={manifest.get(key)!r}, expected {value!r}"
                )
        if len(entries) != target_states:
            raise RuntimeError("Existing frozen suite has the wrong state count")
        excluded = set(exclude_state_keys or ())
        overlap = {
            canonical_state_key(entry.state) for entry in entries
        }.intersection(excluded)
        if overlap:
            raise RuntimeError(
                f"Existing frozen suite overlaps training replay in {len(overlap)} states"
            )
        return manifest

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries: List[dict] = []
    seen = set()
    excluded = set(exclude_state_keys or ())
    game_index = 0

    while len(entries) < target_states and game_index < max_games:
        opening_length = opening_plies[game_index % len(opening_plies)]
        state = _apply_opening(GameState.initial(), opening_length, rng)
        move_index = 0
        while move_index < max_moves_per_game and len(entries) < target_states:
            legal_moves = state.legal_moves()
            if not legal_moves:
                break
            state_dict = state.to_compact()
            state_key = canonical_state_key(state_dict)
            if len(legal_moves) == 1:
                teacher_index = 0
            else:
                teacher_move = get_best_move(
                    state, teacher_difficulty, use_parallel=False
                )
                teacher_index = _matching_move_index(legal_moves, teacher_move)

            if state_key not in seen and state_key not in excluded:
                seen.add(state_key)
                entries.append({
                    "state": state_dict,
                    "legal_moves": [move.to_dict() for move in legal_moves],
                    "chosen_index": teacher_index,
                    "result": 0,
                    "teacher_difficulty": teacher_difficulty,
                    "trajectory_source": "frozen_teacher_suite",
                    "game_id": f"suite-{seed}-{game_index:06d}",
                    "opening_plies": opening_length,
                })

            played_index = teacher_index
            if len(legal_moves) > 1 and rng.random() < played_action_noise:
                played_index = rng.randrange(len(legal_moves))
            state = state.apply_move(legal_moves[played_index])
            move_index += 1
        game_index += 1

    if len(entries) != target_states:
        raise RuntimeError(
            f"Could only generate {len(entries)} unique states after {game_index} games"
        )

    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    manifest = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "kind": "frozen_hard_teacher_suite",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_file": path.name,
        "suite_sha256": _file_sha256(path),
        "state_count": target_states,
        "unique_state_count": len(seen),
        "seed": seed,
        "teacher_difficulty": teacher_difficulty,
        "opening_plies": list(opening_plies),
        "played_action_noise": played_action_noise,
        "max_moves_per_game": max_moves_per_game,
        "excluded_state_count": len(excluded),
        "excluded_state_set_sha256": hashlib.sha256(
            "\n".join(sorted(excluded)).encode("ascii")).hexdigest(),
    }
    if manifest_path.exists():
        raise RuntimeError(f"Frozen suite manifest unexpectedly exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def load_frozen_teacher_suite(
    suite_path: str,
    expected_count: int = 5000,
) -> tuple[List[ReplayEntry], dict]:
    """Load and verify an immutable suite against its manifest and exact count."""

    path = Path(suite_path)
    manifest_path = _manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen teacher suite is incomplete: {path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    actual_hash = _file_sha256(path)
    if actual_hash != manifest.get("suite_sha256"):
        raise RuntimeError("Frozen teacher suite fingerprint does not match its manifest")

    entries = []
    state_keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                entry = ReplayEntry.from_dict(value)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid suite entry at {path}:{line_number}") from exc
            entries.append(entry)
            state_keys.add(canonical_state_key(entry.state))

    required = int(expected_count or manifest.get("state_count", 0))
    if len(entries) != required:
        raise RuntimeError(f"Frozen suite contains {len(entries)} states, expected {required}")
    if len(state_keys) != len(entries):
        raise RuntimeError("Frozen teacher suite contains duplicate canonical states")
    if manifest.get("state_count") != len(entries):
        raise RuntimeError("Frozen teacher suite manifest state count is inconsistent")
    return entries, manifest


def evaluate_teacher_agreement(
    model: torch.nn.Module,
    entries: Sequence[ReplayEntry],
    max_moves_per_sample: int = 32,
    batch_size: int = 1024,
) -> Dict[str, Any]:
    """Evaluate top-1 agreement on a pre-labeled frozen teacher suite."""

    if not entries:
        raise ValueError("Teacher suite is empty")
    dataset = CachedTensorDataset.from_entries(
        list(entries),
        max_moves_per_sample=max_moves_per_sample,
        show_progress=False,
    )
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    correct = 0
    decision_correct = 0
    decision_total = 0
    model_was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            boards = dataset.boards[start:end].to(device)
            move_features = dataset.move_features[start:end].to(device)
            move_counts = dataset.move_counts[start:end].to(device)
            targets = dataset.targets[start:end].to(device)
            scores = model.forward_padded(boards, move_features, move_counts)
            predictions = scores.argmax(dim=1)
            matches = predictions.eq(targets.long())
            correct += int(matches.sum().item())
            decision_mask = move_counts > 1
            decision_total += int(decision_mask.sum().item())
            decision_correct += int((matches & decision_mask).sum().item())
    if model_was_training:
        model.train()

    total = len(dataset)
    return {
        "total_states": total,
        "correct_states": correct,
        "top1_teacher_agreement": correct / total,
        "decision_states": decision_total,
        "decision_correct_states": decision_correct,
        "decision_top1_teacher_agreement": (
            decision_correct / decision_total if decision_total else 1.0
        ),
        "forced_move_fraction": 1.0 - (decision_total / total),
    }


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    record: Mapping[str, Any]


class PromotionRegistry:
    """Append-only checkpoint selection driven only by held-out agreement."""

    def __init__(self, path: str, agreement_threshold: float = 0.50) -> None:
        self.path = Path(path)
        self.agreement_threshold = float(agreement_threshold)

    def _records(self) -> List[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def records(self) -> tuple[dict, ...]:
        """Return the durable promotion history for startup reconciliation."""
        return tuple(self._records())

    def consider(
        self,
        checkpoint_path: str,
        step: int,
        agreement: float,
        suite_fingerprint: str,
        dataset_fingerprint: str,
        training_stage: str = "policy_only",
        comparison_context: Optional[Mapping[str, Any]] = None,
        persist: bool = True,
    ) -> PromotionDecision:
        records = self._records()
        context = dict(comparison_context or {})
        comparable = [
            record for record in records
            if record.get("suite_fingerprint") == suite_fingerprint
            and record.get("dataset_fingerprint") == dataset_fingerprint
            and dict(record.get("comparison_context", {})) == context
            and record.get("promoted")
        ]
        previous_best = max(
            (float(record["teacher_agreement"]) for record in comparable),
            default=float("-inf"),
        )
        if agreement < self.agreement_threshold:
            promoted = False
            reason = "below_teacher_agreement_gate"
        elif agreement <= previous_best:
            promoted = False
            reason = "not_better_than_current_promoted_checkpoint"
        else:
            promoted = True
            reason = "best_held_out_teacher_agreement"

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(checkpoint_path),
            "step": int(step),
            "teacher_agreement": float(agreement),
            "teacher_agreement_threshold": self.agreement_threshold,
            "suite_fingerprint": suite_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "training_stage": training_stage,
            "comparison_context": context,
            "promoted": promoted,
            "reason": reason,
        }
        decision = PromotionDecision(promoted, reason, record)
        if persist:
            self.persist(decision)
        return decision

    def persist(self, decision: PromotionDecision) -> None:
        """Append a decision after its referenced checkpoint is durable."""
        record = dict(decision.record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
