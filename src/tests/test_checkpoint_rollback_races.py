"""Checkpoint pruning must not abort or silently bypass dead-epoch recovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from dama.ai.ml import trainer as trainer_module


def _holder(tmp_path, monkeypatch, *, enforced):
    checkpoints = [tmp_path / f"model_step_{step:06d}.pt" for step in (2000, 4000)]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    holder = SimpleNamespace(
        config=SimpleNamespace(
            checkpoint_dir=str(tmp_path), recovery_enforced=enforced,
            recovery_baseline_sha256="a" * 64, policy_stage="policy_only",
        ),
        scaler=None,
        loaded=[], resets=[],
        _rollback_checkpoint_candidates=lambda pattern: list(checkpoints),
    )
    holder._reset_model_state = lambda reason: holder.resets.append(reason)
    monkeypatch.setattr(trainer_module, "recovery_checkpoint_continues_lineage",
                        lambda *args: True)

    def remaining_verified_pick():
        for checkpoint in reversed(checkpoints):
            if checkpoint.is_file():
                return checkpoint
        raise RuntimeError("No verified checkpoint remains")

    holder._verified_recovery_rollback_checkpoint = remaining_verified_pick
    return holder, checkpoints


@pytest.mark.parametrize("enforced", [False, True])
def test_rollback_retries_when_pick_vanishes_after_existence_check(
    tmp_path, monkeypatch, enforced,
):
    holder, checkpoints = _holder(tmp_path, monkeypatch, enforced=enforced)
    attempted = []

    def load(path):
        candidate = Path(path)
        attempted.append(candidate)
        if candidate == checkpoints[-1]:
            candidate.unlink()
        # Model loading starts with this same open boundary; checking is_file
        # before entering the helper cannot prevent a deletion here.
        candidate.read_bytes()
        holder.loaded.append(candidate)

    holder._load_checkpoint = load
    trainer_module.Trainer._rollback_after_dead_epoch(holder, reason="test")

    assert attempted == list(reversed(checkpoints))
    assert holder.loaded == checkpoints[:1]
    assert holder.resets == []


def test_enforced_rollback_fails_when_verified_pick_and_fallback_are_gone(
    tmp_path, monkeypatch,
):
    holder, checkpoints = _holder(tmp_path, monkeypatch, enforced=True)
    for checkpoint in checkpoints:
        checkpoint.unlink()
    holder._load_checkpoint = lambda path: holder.loaded.append(path)

    with pytest.raises(RuntimeError, match="No verified checkpoint remains"):
        trainer_module.Trainer._rollback_after_dead_epoch(holder, reason="test")

    assert holder.loaded == holder.resets == []


@pytest.mark.parametrize("enforced", [False, True])
def test_rollback_exhaustion_after_every_open_races_with_pruning(
    tmp_path, monkeypatch, enforced,
):
    holder, _ = _holder(tmp_path, monkeypatch, enforced=enforced)

    def load(path):
        Path(path).unlink()
        Path(path).read_bytes()

    holder._load_checkpoint = load
    if enforced:
        with pytest.raises(RuntimeError, match="No verified checkpoint remains"):
            trainer_module.Trainer._rollback_after_dead_epoch(holder, reason="test")
        assert holder.resets == []
    else:
        trainer_module.Trainer._rollback_after_dead_epoch(holder, reason="test")
        assert holder.resets == ["No checkpoint for recovery"]


@pytest.mark.parametrize("enforced", [False, True])
def test_rollback_preserves_other_loading_failures(tmp_path, monkeypatch, enforced):
    holder, _ = _holder(tmp_path, monkeypatch, enforced=enforced)

    def load(path):
        raise PermissionError("checkpoint access denied")

    holder._load_checkpoint = load
    with pytest.raises(PermissionError, match="checkpoint access denied"):
        trainer_module.Trainer._rollback_after_dead_epoch(holder, reason="test")
    assert holder.resets == []


@pytest.mark.parametrize("enforced", [False, True])
def test_rollback_preserves_unrelated_missing_file_errors(
    tmp_path, monkeypatch, enforced,
):
    holder, checkpoints = _holder(tmp_path, monkeypatch, enforced=enforced)

    def load(path):
        # A later load failure must not masquerade as a missing checkpoint.
        (tmp_path / "missing_metadata.json").read_bytes()

    holder._load_checkpoint = load
    with pytest.raises(FileNotFoundError, match="missing_metadata.json"):
        trainer_module.Trainer._rollback_after_dead_epoch(holder, reason="test")
    assert all(checkpoint.is_file() for checkpoint in checkpoints)
    assert holder.resets == []
