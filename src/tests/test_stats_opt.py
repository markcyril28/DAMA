"""Quick verification of batched stats_collector methods."""
import torch
from dama.ai.ml.stats_collector import StatsCollector
from dama.ai.ml.model import MoveScorerNet


def _make_model_with_grads():
    model = MoveScorerNet(channels=32, num_blocks=2, embedding_size=64, hidden_size=32)
    boards = torch.randn(4, 5, 8, 8)
    # Use forward_padded which takes (batch, max_moves, feat_dim) move_features
    move_features = torch.randn(4, 10, 8)
    move_counts = torch.tensor([3, 5, 2, 4])
    scores = model.forward_padded(boards, move_features, move_counts)
    loss = scores.sum()
    loss.backward()
    return model


def test_compute_gradient_stats_batched():
    model = _make_model_with_grads()
    global_norm, per_layer = StatsCollector.compute_gradient_stats(model)
    assert global_norm > 0
    assert len(per_layer) > 0
    for v in per_layer.values():
        assert isinstance(v, float)


def test_record_model_health_batched():
    model = _make_model_with_grads()
    collector = StatsCollector()

    summary = collector.record_model_health(model, step=100)
    assert summary['layer_count'] > 0
    assert summary['total_params'] > 0
    for stats in summary['layers'].values():
        assert 'norm' in stats
        assert 'mean' in stats
        assert 'std' in stats

    # Second call: should have update_ratio
    summary2 = collector.record_model_health(model, step=200)
    has_ratio = any('update_ratio' in s for s in summary2['layers'].values())
    assert has_ratio, "Expected weight update ratios on second call"


def test_compute_score_stats_padded_matches_flat():
    torch.manual_seed(42)
    batch_size, max_moves = 8, 10
    move_counts = torch.randint(1, max_moves + 1, (batch_size,))

    padded = torch.randn(batch_size, max_moves)
    arange = torch.arange(max_moves)
    valid_mask = arange.unsqueeze(0) < move_counts.unsqueeze(1)
    padded[~valid_mask] = float('-inf')
    flat = padded[valid_mask]

    r_p = StatsCollector.compute_score_stats_padded(padded, move_counts)
    r_f = StatsCollector.compute_score_stats(flat, move_counts)

    for key in r_p:
        assert key in r_f, f"Missing key {key}"
        assert abs(r_p[key] - r_f[key]) < 1e-4, f"{key}: {r_p[key]} != {r_f[key]}"
