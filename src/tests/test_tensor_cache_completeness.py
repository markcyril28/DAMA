"""Source-verified caches must retain the original training-loss tensors."""

import gzip

import pytest
import torch

from dama.ai.ml.dataset import CachedTensorDataset, load_matching_cached_tensor_dataset
from dama.ai.ml.move_encoder import BOARD_PLANES, MOVE_FEATURE_SIZE


@pytest.mark.parametrize("field", ["reward_weights", "value_targets"])
@pytest.mark.parametrize("compressed", [False, True])
def test_matching_cache_rejects_null_loss_tensor(tmp_path, field, compressed):
    expected_key = {"snapshot_fingerprint": "a" * 64, "max_moves_per_sample": 4}
    payload = {
        "boards": torch.zeros(2, BOARD_PLANES, 8, 8),
        "move_features": torch.zeros(2, 4, MOVE_FEATURE_SIZE),
        "move_counts": torch.full((2,), 4, dtype=torch.int32),
        "targets": torch.tensor([1, 2], dtype=torch.int32),
        "reward_weights": torch.tensor([0.25, 1.75]),
        "value_targets": torch.tensor([-1.0, 1.0]),
        "metadata": {**expected_key, "entry_count": 2},
    }
    path = tmp_path / "cache.pt"

    def write_payload():
        if compressed:
            with gzip.open(path, "wb") as stream:
                torch.save(payload, stream)
        else:
            torch.save(payload, path)

    write_payload()
    restored = load_matching_cached_tensor_dataset(str(path), expected_key)
    assert restored is not None
    assert torch.equal(getattr(restored, field), payload[field])

    payload[field] = None
    write_payload()
    # Ordinary legacy loading retains its documented optional-field defaults.
    legacy = CachedTensorDataset.load(str(path))
    default = torch.ones(2) if field == "reward_weights" else torch.zeros(2)
    assert torch.equal(getattr(legacy, field), default)
    # The manifest-keyed path cannot invent weights or outcomes for this hit.
    assert load_matching_cached_tensor_dataset(str(path), expected_key) is None
