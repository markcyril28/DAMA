"""Lost incremental telemetry is visible without making training fail."""

import errno
import json

from dama.ai.ml import stats_collector


def test_failed_flush_reports_path_and_error_once_then_reports_recovery(
    tmp_path, monkeypatch, capsys,
):
    collector = stats_collector.StatsCollector(
        output_dir=str(tmp_path), session_id="diagnostic", flush_every=1000,
    )
    target = tmp_path / "incremental_diagnostic.jsonl"
    prior = b'{"prior": true}\n'
    target.write_bytes(prior)
    real_dump = stats_collector.json.dump

    def disk_full(payload, handle, *args, **kwargs):
        handle.write('{"partial":')
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(stats_collector.json, "dump", disk_full)
    collector.flush_incremental()
    first = capsys.readouterr().out
    assert str(target) in first
    assert "OSError" in first and "injected disk full" in first
    assert "retry" in first.lower()
    collector.flush_incremental()
    assert capsys.readouterr().out == ""
    assert target.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))

    monkeypatch.setattr(stats_collector.json, "dump", real_dump)
    collector.record_training_step(step=101, loss=0.5, lr=2e-4)
    collector.flush_incremental()
    recovery = capsys.readouterr().out
    assert str(target) in recovery
    assert "recovered" in recovery.lower() and "2 failed" in recovery
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert rows[0] == {"prior": True}
    assert rows[1]["session_summary"]["training_end_step"] == 101
    collector.flush_incremental()
    assert capsys.readouterr().out == ""

    # A new failure after recovery must be visible again.
    monkeypatch.setattr(stats_collector.json, "dump", disk_full)
    collector.flush_incremental()
    assert "injected disk full" in capsys.readouterr().out


def test_incomplete_stream_is_reported_and_preserved(tmp_path, capsys):
    collector = stats_collector.StatsCollector(
        output_dir=str(tmp_path), session_id="truncated",
    )
    target = tmp_path / "incremental_truncated.jsonl"
    prior = b'{"partial":'
    target.write_bytes(prior)
    collector.flush_incremental()
    output = capsys.readouterr().out
    assert str(target) in output
    assert "RuntimeError" in output and "incomplete final row" in output
    assert target.read_bytes() == prior
    assert not list(tmp_path.glob("*.tmp"))


def test_unavailable_console_does_not_turn_flush_failure_into_training_error(
    tmp_path, monkeypatch,
):
    collector = stats_collector.StatsCollector(
        output_dir=str(tmp_path), session_id="closed_console",
    )

    def disk_full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected disk full")

    def closed_console(*args, **kwargs):
        raise BrokenPipeError("injected closed console")

    real_append = stats_collector._append_jsonl_atomic
    monkeypatch.setattr(stats_collector, "_append_jsonl_atomic", disk_full)
    monkeypatch.setattr("builtins.print", closed_console)
    collector.flush_incremental()
    monkeypatch.setattr(stats_collector, "_append_jsonl_atomic", real_append)
    collector.flush_incremental()
    target = tmp_path / "incremental_closed_console.jsonl"
    assert json.loads(target.read_text())["session_summary"]["session_id"] == "closed_console"
