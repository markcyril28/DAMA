"""Snapshot startup must retain only the storage its consumers use."""

import pytest
import torch

from .test_startup_dataset_lifetime import (
    _entries, _exercise_startup_caches, _FIELDS,
)
from dama.ai.ml.dataset import CachedTensorDataset


def _check_startup(monkeypatch, storage, cached_training, snapshots):
    ownership = []

    def observe(holder, loader, sources):
        ownership.append({
            "current_dataset": holder._current_dataset is not None,
            "loader_dataset": loader.dataset is not None,
            "training_alive": (
                sources["training"]["dataset"]() is not None
                if cached_training else None),
            "training_tensors_alive": (
                sum(ref() is not None for ref in sources["training"]["tensors"])
                if cached_training else None),
        })

    observations = _exercise_startup_caches(
        monkeypatch, storage=storage, cached_training=cached_training,
        snapshots=snapshots, observe=observe)
    copied = storage in ("copied_cpu", "cuda")
    release = snapshots and copied
    assert ownership[0]["current_dataset"] == (
        storage != "standard_cpu" and not release)
    assert ownership[0]["loader_dataset"] == (not release)
    if cached_training:
        assert ownership[0]["training_alive"] == (not release)
        assert ownership[0]["training_tensors_alive"] == (0 if release else 6)
    # The initial validation data still has a real consumer until refresh.
    assert observations[0]["alive_datasets"] == (
        1 + int(cached_training and not release))
    assert observations[1]["alive_datasets"] == 0
    # Read batches both before and after the complete snapshot replacement.
    for index, result in enumerate(observations):
        for key, version in (("batch", index * 2), ("validation", index * 2 + 1)):
            expected = CachedTensorDataset.from_entries(
                _entries(version), max_moves_per_sample=32, show_progress=False)
            for actual, field in zip(result[key], _FIELDS):
                assert torch.equal(actual, getattr(expected, field))


@pytest.mark.parametrize("storage", ["copied_cpu", "fast_cpu", "standard_cpu"])
@pytest.mark.parametrize("cached_training", [True, False])
@pytest.mark.parametrize("snapshots", [True, False])
def test_startup_releases_only_redundant_snapshot_sources(
    monkeypatch, storage, cached_training, snapshots,
):
    _check_startup(monkeypatch, storage, cached_training, snapshots)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("cached_training", [True, False])
def test_snapshot_startup_release_preserves_cuda_batches(monkeypatch, cached_training):
    _check_startup(monkeypatch, "cuda", cached_training, snapshots=True)
