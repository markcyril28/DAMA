"""Quick verification of batched stats_collector methods."""
import json
import os
import stat
from pathlib import Path

import pytest
import torch
import dama.ai.ml.stats_collector as stats_collector_module
from dama.ai.ml.stats_collector import StatsCollector
from dama.ai.ml.model import MoveScorerNet
from scripts import analyze_training_stats


def test_stats_collector_commits_output_directory_before_collecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collector must not accept writes under an uncommitted namespace."""
    parent = tmp_path / "session"
    parent.mkdir()
    output_dir = parent / "stats"
    output_dir.mkdir()
    commits = []

    def record_commit(path):
        assert output_dir.is_dir()
        commits.append(Path(path))

    monkeypatch.setattr(
        stats_collector_module, "_fsync_directory", record_commit)

    collector = StatsCollector(output_dir=str(output_dir))
    retry = StatsCollector(output_dir=str(output_dir))

    assert collector.output_dir == output_dir
    assert retry.output_dir == output_dir
    assert commits == [parent, parent]


def test_stats_collector_stops_when_output_directory_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not begin a telemetry session past a failed namespace boundary."""
    parent = tmp_path / "session"
    parent.mkdir()
    output_dir = parent / "stats"

    def fail_commit(path):
        assert Path(path) == parent
        raise OSError(5, "simulated stats namespace sync failure")

    monkeypatch.setattr(
        stats_collector_module, "_fsync_directory", fail_commit)

    with pytest.raises(OSError, match="stats namespace sync failure"):
        StatsCollector(output_dir=str(output_dir))

    assert output_dir.is_dir()
    assert not list(output_dir.iterdir())


def test_atomic_stats_write_fsyncs_file_and_directory_around_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete bytes and then their public pathname become crash-durable."""
    path = tmp_path / "session_report.json"
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        mode = os.fstat(fd).st_mode
        events.append("directory_fsync" if stat.S_ISDIR(mode) else "file_fsync")
        return real_fsync(fd)

    def tracking_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(stats_collector_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(stats_collector_module.os, "replace", tracking_replace)

    with stats_collector_module._atomic_text_writer(path) as handle:
        json.dump({"status": "complete"}, handle)

    assert events == ["file_fsync", "replace", "directory_fsync"]
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "complete",
    }


def test_atomic_stats_write_reports_directory_fsync_failure_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed directory commit is visible and leaves no staging file."""
    path = tmp_path / "session_report.json"
    path.write_text('{"status": "old"}', encoding="utf-8")
    real_fsync = os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(5, "simulated stats directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(stats_collector_module.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="stats directory fsync failure"):
        with stats_collector_module._atomic_text_writer(path) as handle:
            json.dump({"status": "new"}, handle)

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "new"}
    assert not list(tmp_path.glob("*.tmp"))


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


def test_record_model_health_batched(tmp_path):
    model = _make_model_with_grads()
    collector = StatsCollector(output_dir=str(tmp_path))

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


def test_resumed_session_counts_optimizer_steps_not_metric_rows(tmp_path):
    collector = StatsCollector(output_dir=str(tmp_path), flush_every=1000)
    collector.set_training_start_step(246_000)

    collector.record_training_step(
        step=246_200,
        loss=1.5,
        lr=2e-4,
        batch_size=2048,
        step_time=0.16,
    )
    collector.record_training_step(
        step=246_400,
        loss=1.4,
        lr=2e-4,
        batch_size=2048,
        step_time=0.16,
    )

    assert collector.loss.count == 2
    report = collector.generate_session_report()
    assert report["summary"]["total_steps"] == 400
    assert report["summary"]["training_start_step"] == 246_000
    assert report["summary"]["training_end_step"] == 246_400

    collector.set_training_end_step(246_450)
    assert collector.generate_session_report()["summary"]["total_steps"] == 450


def test_incremental_flush_preserves_partial_run_diagnostics(tmp_path):
    collector = StatsCollector(
        output_dir=str(tmp_path), session_id="partial", flush_every=1000)
    collector.set_config_snapshot({"learning_rate": 2e-4, "batch_size": 2048})
    collector.set_training_start_step(100)
    collector.record_training_step(
        step=140,
        loss=0.75,
        lr=2e-4,
        batch_size=2048,
        step_time=0.25,
        grad_norm=3.5,
    )
    collector.gpu_utilization_pct.append(91.0, 140)
    collector.gpu_power_w.append(42.0, 140)
    collector.process_rss_gb.append(7.25, 140)
    collector.record_gpu_idle_wait(1.5, stale_epochs=4)
    collector.record_selfplay_epoch(
        step=140,
        epoch=3,
        num_games=240,
        num_entries=14_000,
        elapsed_sec=12.0,
    )
    collector.record_epoch(
        epoch=3,
        step=140,
        avg_loss=0.8,
        num_batches=20,
        epoch_time_sec=4.0,
    )
    collector.record_replay_buffer_state(
        step=140, total_entries=650_000, num_files=60)
    collector.record_checkpoint(
        step=140, loss=0.75, path="models/checkpoint.pt")

    collector.flush_incremental()

    output = tmp_path / "incremental_partial.jsonl"
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[-1])
    assert row["session_summary"]["training_start_step"] == 100
    assert row["session_summary"]["training_end_step"] == 140
    assert row["session_summary"]["completed_training_steps"] == 40
    assert row["session_summary"]["gpu_idle_wait_seconds"] == 1.5
    assert row["config"] == {"learning_rate": 2e-4, "batch_size": 2048}
    assert row["learning_rate_summary"]["recent_mean"] == 2e-4
    assert row["system_summary"]["gpu_utilization_pct"]["latest"] == {
        "step": 140,
        "value": 91.0,
    }
    assert row["system_summary"]["gpu_power_w"]["recent_mean"] == 42.0
    assert row["system_summary"]["process_rss_gb"]["recent_max"] == 7.25
    assert row["latest_records"]["selfplay"]["num_games"] == 240
    assert row["latest_records"]["epoch"]["num_batches"] == 20
    assert row["latest_records"]["replay_buffer"]["num_files"] == 60
    assert row["latest_records"]["checkpoint"]["step"] == 140


def test_incremental_flush_failure_preserves_history_and_later_retry(
    tmp_path,
    monkeypatch,
):
    collector = StatsCollector(
        output_dir=str(tmp_path), session_id="fault", flush_every=1000)
    target = tmp_path / "incremental_fault.jsonl"
    prior = b'{"prior": true}\n'
    target.write_bytes(prior)
    real_dump = stats_collector_module.json.dump

    def fail_dump(payload, handle, *args, **kwargs):
        handle.write('{"partial":')
        handle.flush()
        raise OSError(28, "simulated incremental stats disk full")

    monkeypatch.setattr(stats_collector_module.json, "dump", fail_dump)
    collector.flush_incremental()

    assert target.read_bytes() == prior
    assert list(Path(tmp_path).glob("incremental_fault.jsonl.*.tmp")) == []

    monkeypatch.setattr(stats_collector_module.json, "dump", real_dump)
    collector.set_training_start_step(100)
    collector.record_training_step(
        step=101,
        loss=0.5,
        lr=2e-4,
        batch_size=2048,
        step_time=0.25,
    )
    collector.flush_incremental()

    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0] == {"prior": True}
    assert rows[1]["session_summary"]["training_end_step"] == 101
    assert list(Path(tmp_path).glob("incremental_fault.jsonl.*.tmp")) == []


def test_terminal_report_preserves_prior_file_on_write_failure(
    tmp_path,
    monkeypatch,
):
    collector = StatsCollector(
        output_dir=str(tmp_path), session_id="fault", flush_every=1000)
    target = tmp_path / "session_report_fault.json"
    prior = b'{"prior": true}\n'
    target.write_bytes(prior)

    def fail_dump(payload, handle, *args, **kwargs):
        handle.write('{"partial":')
        handle.flush()
        raise OSError(28, "simulated terminal stats disk full")

    monkeypatch.setattr(stats_collector_module.json, "dump", fail_dump)

    with pytest.raises(OSError, match="simulated terminal stats disk full"):
        collector.export_session_report()

    assert target.read_bytes() == prior
    assert list(Path(tmp_path).glob("session_report_fault.json.*.tmp")) == []


def test_terminal_csv_preserves_prior_file_on_write_failure(
    tmp_path,
    monkeypatch,
):
    collector = StatsCollector(
        output_dir=str(tmp_path), session_id="fault", flush_every=1000)
    collector.loss.append(0.5, step=1)
    target = tmp_path / "loss_history_fault.csv"
    prior = b"prior,report\n"
    target.write_bytes(prior)

    class FailingDictWriter:
        def __init__(self, handle, **kwargs):
            self.handle = handle

        def writeheader(self):
            self.handle.write("partial,")
            self.handle.flush()
            raise OSError(28, "simulated terminal stats disk full")

    monkeypatch.setattr(
        stats_collector_module.csv, "DictWriter", FailingDictWriter)

    with pytest.raises(OSError, match="simulated terminal stats disk full"):
        collector.export_loss_csv()

    assert target.read_bytes() == prior
    assert list(Path(tmp_path).glob("loss_history_fault.csv.*.tmp")) == []


def test_zero_sample_report_is_missing_not_healthy():
    empty_summary = {
        "total_count": 0,
        "running_mean": 0.0,
        "running_min": 0.0,
        "recent_mean": 0.0,
        "recent_stdev": 0.0,
        "recent_max": 0.0,
    }
    report = {
        "meta": {},
        "config": {"selfplay_noise_prob": 0.1},
        "summary": {"total_steps": 0, "nan_inf_events": 0},
        "loss": {"summary": dict(empty_summary)},
        "gradient_norms": {"global_summary": dict(empty_summary)},
        "throughput": {
            "samples_per_sec": dict(empty_summary),
            "step_time_sec": dict(empty_summary),
        },
        "score_distribution": {
            "entropy_summary": dict(empty_summary),
        },
    }

    loss = analyze_training_stats.analyze_loss(report)
    throughput = analyze_training_stats.analyze_throughput(report)
    gradients = analyze_training_stats.analyze_gradients(report)
    model = analyze_training_stats.analyze_model_health(report)
    evaluations = analyze_training_stats.analyze_evaluations(report)
    selfplay = analyze_training_stats.analyze_selfplay(report)
    system = analyze_training_stats.analyze_system(report)
    recommendations = analyze_training_stats.generate_recommendations(
        report,
        loss,
        throughput,
        gradients,
        model,
        evaluations,
        selfplay,
        system,
    )
    markdown = analyze_training_stats.format_markdown_report(
        report,
        loss,
        throughput,
        gradients,
        model,
        evaluations,
        selfplay,
        system,
        recommendations,
    )

    assert not loss["has_data"]
    assert not throughput["has_data"]
    assert not gradients["has_data"]
    assert [item["category"] for item in recommendations] == ["Data Quality"]
    assert "MISSING (0 recorded loss samples)" in markdown
    assert "MISSING (0 recorded gradient samples)" in markdown
    assert "No issues detected" not in markdown
    assert "Score entropy is low" not in markdown


def test_unknown_gpu_capacity_does_not_block_partial_system_analysis():
    report = {
        "meta": {"gpu_name": "N/A", "gpu_vram_gb": None},
        "gpu_memory": {
            "allocated_mb": {
                "total_count": 2,
                "recent_mean": 4096.0,
                "recent_max": 5120.0,
            },
        },
        "system": {
            "gpu_utilization": {"total_count": 1, "recent_mean": 91.0},
        },
    }

    analysis = analyze_training_stats.analyze_system(report)

    assert analysis["gpu_vram_gb"] is None
    assert analysis["gpu_mem_mean_mb"] == 4096.0
    assert analysis["gpu_mem_max_mb"] == 5120.0
    assert "gpu_mem_utilization" not in analysis
    assert analysis["gpu_compute_utilization"] == 91.0


def test_newer_incremental_stream_marks_terminal_report_stale(tmp_path):
    report = tmp_path / "session_report_20260903_120000.json"
    incremental = tmp_path / "incremental_20260903_120500.jsonl"
    report.write_text("{}", encoding="utf-8")
    incremental.write_text("{}\n", encoding="utf-8")
    os.utime(report, ns=(1_000_000_000, 1_000_000_000))
    os.utime(incremental, ns=(2_000_000_000, 2_000_000_000))

    assert analyze_training_stats.find_newer_incremental(
        str(tmp_path), str(report)) == str(incremental)


def test_incremental_stream_builds_partial_report(tmp_path):
    incremental = tmp_path / "incremental_partial.jsonl"
    rows = [
        {
            "timestamp": "2026-09-03T12:00:00",
            "loss_summary": {"total_count": 0},
        },
        {
            "timestamp": "2026-09-03T12:05:00",
            "session_summary": {
                "session_id": "partial",
                "start_time": "2026-09-03T12:00:00",
                "elapsed_seconds": 300.0,
                "training_start_step": 100,
                "training_end_step": 140,
                "completed_training_steps": 40,
                "epochs_recorded": 2,
                "selfplay_epochs_recorded": 3,
                "evaluations_recorded": 1,
                "checkpoints_recorded": 1,
                "gpu_idle_wait_pct": 2.5,
            },
            "config": {"learning_rate": 2e-4},
            "loss_summary": {"total_count": 4, "recent_mean": 0.7},
            "learning_rate_summary": {
                "total_count": 4, "recent_mean": 2e-4},
            "grad_norm_summary": {"total_count": 4, "recent_mean": 3.5},
            "throughput_summary": {
                "total_count": 4, "recent_mean": 5000.0},
            "step_time_summary": {"total_count": 4, "recent_mean": 0.4},
            "system_summary": {
                "gpu_utilization_pct": {
                    "total_count": 1, "recent_mean": 91.0},
            },
            "latest_records": {
                "selfplay": {"num_games": 240},
                "epoch": {"epoch": 2},
                "evaluation": {"ml_win_rate": 0.5},
                "checkpoint": {"step": 140},
            },
            "convergence": {"nan_inf_event_count": 0},
        },
    ]
    incremental.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    report, warning = analyze_training_stats.load_incremental_report(
        str(incremental))

    assert warning is None
    assert report["meta"]["data_source"] == "incremental"
    assert report["summary"]["total_steps"] == 40
    assert report["summary"]["gpu_idle_wait_pct"] == 2.5
    assert report["config"]["learning_rate"] == 2e-4
    assert report["loss"]["summary"]["recent_mean"] == 0.7
    assert report["learning_rate"]["summary"]["recent_mean"] == 2e-4
    assert report["gradient_norms"]["global_summary"]["recent_mean"] == 3.5
    assert report["system"]["gpu_utilization"]["recent_mean"] == 91.0
    assert report["selfplay"] == [{"num_games": 240}]

    analyses = {
        "loss_analysis": analyze_training_stats.analyze_loss(report),
        "throughput_analysis": analyze_training_stats.analyze_throughput(report),
        "grad_analysis": analyze_training_stats.analyze_gradients(report),
        "model_analysis": analyze_training_stats.analyze_model_health(report),
        "eval_analysis": analyze_training_stats.analyze_evaluations(report),
        "selfplay_analysis": analyze_training_stats.analyze_selfplay(report),
        "system_analysis": analyze_training_stats.analyze_system(report),
    }
    markdown = analyze_training_stats.format_markdown_report(
        report,
        **analyses,
        recommendations=[],
        data_warning="partial session",
    )
    assert "Total Steps | 40" in markdown
    assert "N/A (unknown)" in markdown
    assert "terminal-only metrics remain unavailable" in markdown


def test_incremental_stream_uses_last_complete_row_after_truncated_append(tmp_path):
    incremental = tmp_path / "incremental_partial.jsonl"
    incremental.write_text(
        json.dumps({
            "timestamp": "2026-09-03T12:00:00",
            "loss_summary": {"total_count": 2, "recent_mean": 0.8},
        }) + "\n{\"timestamp\":",
        encoding="utf-8",
    )

    report, warning = analyze_training_stats.load_incremental_report(
        str(incremental))

    assert report["loss"]["summary"]["recent_mean"] == 0.8
    assert warning is not None
    assert "truncated final row" in warning
