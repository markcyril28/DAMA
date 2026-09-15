"""A pruned verified pick must preserve the full recovery selection contract."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dama.ai.ml.trainer import Trainer


def _holder(tmp_path):
    anchor = tmp_path / "model_step_002000.pt"
    anchor.write_bytes(b"approved anchor from the preceding lineage")
    baseline = hashlib.sha256(anchor.read_bytes()).hexdigest().upper()
    directory = tmp_path / "continuation"
    directory.mkdir()
    holder = object.__new__(Trainer)
    holder.config = SimpleNamespace(
        resume=str(anchor), checkpoint_dir=str(directory),
        recovery_enforced=True, recovery_baseline_sha256=baseline,
        policy_stage="policy_only",
    )
    holder.step = 6000
    holder.scaler = None
    holder.loaded = []

    def write_checkpoint(step):
        path = directory / f"model_step_{step:06d}.pt"
        torch.save({
            "model_state_dict": {"weight": torch.ones(1)},
            "step": step,
            "recovery_experiment": {
                "enabled": True, "baseline_sha256": baseline,
                "training_stage": "policy_only",
            },
        }, path)
        return path

    return holder, anchor, write_checkpoint


@pytest.mark.parametrize("future_checkpoint", [False, True])
@pytest.mark.parametrize("vanishing_count", [1, 2])
def test_verified_retry_reaches_external_anchor_and_excludes_future_steps(
    tmp_path, future_checkpoint, vanishing_count,
):
    holder, anchor, write_checkpoint = _holder(tmp_path)
    vanishing = {write_checkpoint(step)
                 for step in (4000, 6000)[:vanishing_count]}
    if future_checkpoint:
        write_checkpoint(8000)
    attempted = []

    def load(path):
        candidate = Path(path)
        attempted.append(candidate)
        if candidate in vanishing:
            candidate.unlink()
        candidate.read_bytes()
        holder.loaded.append(candidate)

    holder._load_checkpoint = load
    holder._rollback_after_dead_epoch("test")

    assert attempted == sorted(vanishing, reverse=True) + [anchor]
    assert holder.loaded == [anchor]


def test_verified_retry_preserves_auxiliary_failure_with_selected_anchor_present(
    tmp_path,
):
    holder, anchor, write_checkpoint = _holder(tmp_path)
    # The resolver properly excludes this checkpoint, even though it is in
    # the same lineage. A load error must not send control to a weaker scan.
    future = write_checkpoint(8000)
    attempted = []

    def load(path):
        candidate = Path(path)
        attempted.append(candidate)
        if candidate == anchor:
            (tmp_path / "required_metadata.json").read_bytes()
        holder.loaded.append(candidate)

    holder._load_checkpoint = load
    with pytest.raises(FileNotFoundError, match="required_metadata.json"):
        holder._rollback_after_dead_epoch("test")

    assert attempted == [anchor]
    assert holder.loaded == []
    assert anchor.is_file() and future.is_file()


def test_verified_retry_is_bounded_when_picks_keep_disappearing(tmp_path):
    holder, anchor, write_checkpoint = _holder(tmp_path)
    descendant = write_checkpoint(4000)
    attempted = []

    def select():
        # Simulate a concurrently republished pick disappearing again before
        # every load. Recovery must fail visibly instead of spinning forever.
        descendant.write_bytes(b"concurrently republished")
        return descendant

    def load(path):
        candidate = Path(path)
        attempted.append(candidate)
        candidate.unlink()
        candidate.read_bytes()

    holder._verified_recovery_rollback_checkpoint = select
    holder._load_checkpoint = load
    with pytest.raises(RuntimeError, match="recovery retry limit"):
        holder._rollback_after_dead_epoch("test")

    assert attempted == [descendant, descendant]
    assert anchor.is_file()


def test_anchor_disappearing_during_hash_does_not_enable_future_checkpoint(tmp_path):
    holder, anchor, write_checkpoint = _holder(tmp_path)
    future = write_checkpoint(8000)
    hashes = []

    def hash_checkpoint(path):
        hashes.append(path)
        path.unlink()
        return Trainer._checkpoint_file_sha256(path)

    holder._checkpoint_file_sha256 = hash_checkpoint
    holder._load_checkpoint = lambda path: holder.loaded.append(path)
    with pytest.raises(RuntimeError, match="No verified checkpoint remains"):
        holder._rollback_after_dead_epoch("test")

    assert hashes == [anchor]
    assert holder.loaded == []
    assert future.is_file()


def test_permanent_verification_error_has_bounded_failure(tmp_path):
    holder, anchor, _ = _holder(tmp_path)
    attempts = []

    def select():
        attempts.append(True)
        raise FileNotFoundError("verification input unavailable")

    holder._verified_recovery_rollback_checkpoint = select
    holder._load_checkpoint = lambda path: holder.loaded.append(path)
    with pytest.raises(RuntimeError, match="recovery retry limit"):
        holder._rollback_after_dead_epoch("test")

    assert attempts == [True]
    assert holder.loaded == []
    assert anchor.is_file()
