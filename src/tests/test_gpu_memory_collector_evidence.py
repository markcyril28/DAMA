"""Collector GPU memory summaries need valid samples before batch-size advice."""

from types import SimpleNamespace

import pytest
import torch

from dama.ai.ml.stats_collector import StatsCollector


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Controlled GPU")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=8_000_000_000),
    )
    return StatsCollector(str(tmp_path), session_id="gpu_memory_evidence")


def _memory_hints(collector):
    return [hint["hint"] for hint in collector.generate_session_report()["optimization_hints"]
            if "GPU" in hint["hint"]]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_only_allocation_is_not_zero_or_batch_size_advice(collector, capsys, invalid):
    collector.gpu_mem_allocated_mb.append(invalid, 10)
    hints = _memory_hints(collector)
    assert any("nonfinite" in hint.lower() for hint in hints)
    assert not any("batch_size" in hint for hint in hints)

    collector.print_session_summary()
    memory_section = capsys.readouterr().out.split("  GPU Memory:", 1)[1]
    assert "Allocated:          Unavailable" in memory_section
    assert "Nonfinite (last 10): 1" in memory_section
    assert "Allocated:          0 MB" not in memory_section


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("invalid_first", [True, False])
def test_mixed_allocation_retains_finite_mean_without_batch_size_advice(
    collector, capsys, invalid, invalid_first,
):
    values = [invalid, 512.0] if invalid_first else [512.0, invalid]
    for step, value in enumerate(values):
        collector.gpu_mem_allocated_mb.append(value, step)
    hints = _memory_hints(collector)
    assert any("nonfinite" in hint.lower() for hint in hints)
    assert not any("batch_size" in hint for hint in hints)
    collector.print_session_summary()
    memory_section = capsys.readouterr().out.split("  GPU Memory:", 1)[1]
    assert "Allocated:          512 MB" in memory_section
    assert "Nonfinite (last 10): 1" in memory_section


@pytest.mark.parametrize("allocated, expected", [
    (0.0, "only 0%"), (512.0, "only 6%"), (7900.0, "99%, near capacity"),
])
def test_healthy_allocation_preserves_existing_hints(collector, capsys, allocated, expected):
    collector.gpu_mem_allocated_mb.append(allocated, 10)
    hints = _memory_hints(collector)
    assert any(expected in hint for hint in hints)
    assert not any("nonfinite" in hint.lower() for hint in hints)
    collector.print_session_summary()
    assert f"Allocated:          {allocated:.0f} MB" in capsys.readouterr().out


def test_recent_allocation_recovers_after_invalid_sample_leaves_window(collector):
    collector.gpu_mem_allocated_mb.append(float("inf"), 0)
    for step in range(1, 11):
        collector.gpu_mem_allocated_mb.append(512.0, step)
    hints = _memory_hints(collector)
    assert any("only 6%" in hint for hint in hints)
    assert not any("nonfinite" in hint.lower() for hint in hints)


def test_empty_allocation_has_no_memory_claim(collector, capsys):
    assert not _memory_hints(collector)
    collector.print_session_summary()
    assert "  GPU Memory:" not in capsys.readouterr().out


@pytest.mark.parametrize("capacity", [0, float("inf"), float("nan")])
def test_unknown_capacity_cannot_support_batch_size_advice(collector, monkeypatch, capacity):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=capacity),
    )
    collector.gpu_mem_allocated_mb.append(512.0, 10)
    assert not any("batch_size" in hint for hint in _memory_hints(collector))
