import hashlib
import json
from pathlib import Path

import pytest
import torch

from dama.ai.ml.replay import ReplayEntry
from dama.ai.ml.teacher_validation import (
    PromotionRegistry,
    evaluate_teacher_agreement,
    load_frozen_teacher_suite,
)


def _entry(chosen_index: int = 0, forced: bool = False) -> ReplayEntry:
    moves = [{"path": [[2, 1], [3, 0]], "captures": [], "promotion": False}]
    if not forced:
        moves.append({"path": [[2, 1], [3, 2]], "captures": [], "promotion": False})
    return ReplayEntry(
        state={
            "p1_men": [[2, 1]],
            "p1_kings": [],
            "p2_men": [[5, 0]],
            "p2_kings": [],
            "turn": 1,
            "move_count": 0,
        },
        legal_moves=moves,
        chosen_index=chosen_index,
        result=0,
    )


class _FirstMoveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward_padded(self, boards, move_features, move_counts):
        scores = torch.zeros(
            boards.shape[0], move_features.shape[1], device=boards.device
        )
        scores[:, 0] = 1.0 + self.anchor
        invalid = torch.arange(move_features.shape[1], device=boards.device).unsqueeze(0)
        scores = scores.masked_fill(invalid >= move_counts.unsqueeze(1), float("-inf"))
        return scores


def test_teacher_agreement_reports_forced_fraction() -> None:
    entries = [_entry(0), _entry(1), _entry(0, forced=True)]
    result = evaluate_teacher_agreement(
        _FirstMoveModel(), entries, max_moves_per_sample=4, batch_size=2
    )

    assert result["correct_states"] == 2
    assert result["top1_teacher_agreement"] == 2 / 3
    assert result["decision_top1_teacher_agreement"] == 0.5
    assert result["forced_move_fraction"] == pytest.approx(1 / 3)


def test_frozen_suite_loader_checks_hash_count_and_uniqueness(tmp_path: Path) -> None:
    suite = tmp_path / "suite.jsonl"
    entry = _entry().to_dict()
    second = _entry().to_dict()
    second["state"] = dict(second["state"], p1_kings=[[4, 3]], p1_men=[])
    payload = json.dumps(entry) + "\n" + json.dumps(second) + "\n"
    suite.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(suite.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "state_count": 2,
        "suite_sha256": digest,
        "seed": 1,
        "teacher_difficulty": "hard",
        "opening_plies": [0, 2],
    }
    suite.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    entries, loaded = load_frozen_teacher_suite(str(suite), expected_count=2)
    assert len(entries) == 2
    assert loaded["suite_sha256"] == digest


def test_promotion_uses_agreement_not_training_loss(tmp_path: Path) -> None:
    registry = PromotionRegistry(str(tmp_path / "promotions.jsonl"), 0.50)
    below = registry.consider("step_1.pt", 1, 0.49, "suite", "data")
    first = registry.consider("step_2.pt", 2, 0.51, "suite", "data")
    worse = registry.consider("step_3.pt", 3, 0.505, "suite", "data")
    better = registry.consider("step_4.pt", 4, 0.55, "suite", "data")

    assert not below.promoted
    assert first.promoted
    assert not worse.promoted
    assert better.promoted
    assert "loss" not in better.record


def test_promotion_can_be_persisted_only_after_checkpoint_write(tmp_path: Path) -> None:
    path = tmp_path / "promotions.jsonl"
    registry = PromotionRegistry(str(path), 0.50)
    decision = registry.consider(
        "step_2.pt", 2, 0.51, "suite", "data", persist=False
    )

    assert decision.promoted
    assert not path.exists()

    registry.persist(decision)
    saved = json.loads(path.read_text(encoding="utf-8").strip())
    assert saved["checkpoint_path"] == "step_2.pt"
    assert saved["promoted"] is True
