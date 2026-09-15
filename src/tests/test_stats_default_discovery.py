"""The analyzer must follow the launcher's run instead of legacy artifacts."""

import json
import sys

import pytest
import yaml

from scripts import analyze_training_stats as analysis


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "config").mkdir()
    monkeypatch.setattr(analysis, "__file__", str(root / "scripts" / "analyze_training_stats.py"))
    # Discovery is relative to the script's project, even from another cwd.
    monkeypatch.chdir(tmp_path)
    return root


def _select(project, config):
    (project / "config" / "selected.yaml").write_text(yaml.safe_dump(config))
    (project / "local_train.sh").write_text(
        'TRAINING_CONFIG="config/retired.yaml"\n'
        '# TRAINING_CONFIG="config/commented.yaml"\n'
        'TRAINING_CONFIG="config/selected.yaml" # active run\n')


def _report(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "session_report_20260915_120000.json"
    path.write_text("{}")
    return path


def _main(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["analyze_training_stats.py", "--format", "json", *args])
    analysis.main()
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("config,relative", [
    ({"statistics": {"output_dir": "logs/active/stats"},
      "paths": {"log_dir": "logs/unused", "policy_output_namespace": "active"}},
     "logs/active/stats"),
    ({"paths": {"log_dir": "logs/active"}}, "logs/active/stats"),
    ({}, "logs/stats"),
])
def test_default_follows_selected_config(project, monkeypatch, capsys, config, relative):
    _select(project, config)
    expected = _report(project / relative)
    _report(project / "logs" / "retired" / "stats")
    result = _main(monkeypatch, capsys)
    assert result["report_path"] == str(expected)


def test_default_preserves_absolute_stats_path(project, tmp_path, monkeypatch, capsys):
    directory = tmp_path / "external stats"
    _select(project, {"statistics": {"output_dir": str(directory)}})
    expected = _report(directory)
    assert _main(monkeypatch, capsys)["report_path"] == str(expected)


def test_default_uses_incremental_when_no_terminal_report(project, monkeypatch, capsys):
    _select(project, {"statistics": {"output_dir": "logs/active/stats"}})
    directory = project / "logs" / "active" / "stats"
    directory.mkdir(parents=True)
    stream = directory / "incremental_20260915_120000.jsonl"
    stream.write_text(json.dumps({
        "timestamp": "2026-09-15T12:00:00",
        "session_summary": {"completed_training_steps": 20},
    }) + "\n")
    result = _main(monkeypatch, capsys)
    assert result["report_path"] == str(stream)
    assert "terminal-only histories are unknown" in result["data_warning"]


@pytest.mark.parametrize("config", [None, [], {"statistics": []},
                                         {"statistics": {"output_dir": None}},
                                         {"paths": {"log_dir": 123}}])
def test_invalid_config_does_not_fall_back_to_legacy(
        project, monkeypatch, capsys, config):
    _select(project, config)
    _report(project / "logs" / "stats")
    monkeypatch.setattr(sys, "argv", ["analyze_training_stats.py"])
    with pytest.raises(SystemExit) as exc:
        analysis.main()
    output = capsys.readouterr()
    assert exc.value.code == 2
    assert "--stats-dir" in output.err
    assert not output.out


def test_missing_launcher_reports_discovery_error(project, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["analyze_training_stats.py"])
    with pytest.raises(SystemExit) as exc:
        analysis.main()
    output = capsys.readouterr()
    assert exc.value.code == 2
    assert "local_train.sh" in output.err
    assert "--stats-dir" in output.err


def test_nonliteral_selection_is_not_executed_or_ignored(project, monkeypatch, capsys):
    _select(project, {"statistics": {"output_dir": "logs/active/stats"}})
    with (project / "local_train.sh").open("a") as stream:
        stream.write('TRAINING_CONFIG="$(touch should_not_exist)"\n')
    monkeypatch.setattr(sys, "argv", ["analyze_training_stats.py"])
    with pytest.raises(SystemExit) as exc:
        analysis.main()
    assert exc.value.code == 2
    assert "TRAINING_CONFIG" in capsys.readouterr().err
    assert not (project.parent / "should_not_exist").exists()


def test_enhanced_launcher_does_not_silently_read_policy_stats(project, monkeypatch, capsys):
    _select(project, {"statistics": {"output_dir": "logs/policy/stats"}})
    _report(project / "logs" / "policy" / "stats")
    with (project / "local_train.sh").open("a") as stream:
        stream.write("ENHANCED_STAGE=true\n")
    monkeypatch.setattr(sys, "argv", ["analyze_training_stats.py"])
    with pytest.raises(SystemExit) as exc:
        analysis.main()
    output = capsys.readouterr()
    assert exc.value.code == 2
    assert "ENHANCED_STAGE" in output.err
    assert "--stats-dir" in output.err
    assert not output.out


@pytest.mark.parametrize("option", ["--session", "--stats-dir"])
def test_explicit_source_does_not_require_launcher(project, monkeypatch, capsys, option):
    report = _report(project / "custom")
    source = report if option == "--session" else report.parent
    assert _main(monkeypatch, capsys, option, str(source))["report_path"] == str(report)


def test_comparison_does_not_require_launcher(project, monkeypatch, capsys):
    report = _report(project / "custom")
    monkeypatch.setattr(sys, "argv", ["analyze_training_stats.py", "--compare", str(report), str(report)])
    analysis.main()
    assert "# Session Comparison" in capsys.readouterr().out
