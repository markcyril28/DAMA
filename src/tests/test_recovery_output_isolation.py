"""Recovery output paths must not reuse the preserved model lineage."""

from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from dama.ai.ml.trainer import (
    Trainer,
    TrainingConfig,
    config_from_yaml,
    load_config_from_yaml,
    validate_recovery_experiment_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(
    "models/checkpoints_policy_distillation/model_step_134000.pt"
)
# The prior recovery namespace contains the preserved 136k-144k artifacts.
# Corrected 1e-4 runs must never reuse or overwrite it.
PRESERVED_PRIOR_RECOVERY_NAMESPACE = "policy_distillation_recovery"
RECOVERY_NAMESPACE = "policy_distillation_recovery_wd1e4"
RECOVERY_CHECKPOINT_DIR = Path(
    f"models/checkpoints_{RECOVERY_NAMESPACE}"
)


@pytest.mark.parametrize(
    ("relative_config", "profile"),
    [
        ("config/training_config_policy_distillation.yaml", None),
        ("config/training_config_policy_distillation.yaml", "server"),
        ("config/training_config_server_policy_distillation.yaml", None),
    ],
)
def test_recovery_configs_isolate_all_mutable_model_outputs(
    relative_config: str,
    profile: str | None,
) -> None:
    path = PROJECT_ROOT / relative_config
    config = config_from_yaml(load_config_from_yaml(str(path), profile))

    validate_recovery_experiment_config(config)
    assert Path(config.recovery_baseline_path or "") == BASELINE
    assert Path(config.resume or "") == BASELINE
    assert Path(config.checkpoint_dir) == RECOVERY_CHECKPOINT_DIR
    assert Path(config.runtime_model_root) == Path(
        f"logs/{RECOVERY_NAMESPACE}/runtime_models"
    )
    assert Path(config.replay_dir) == Path(
        f"data/replay_{RECOVERY_NAMESPACE}"
    )
    assert Path(config.log_dir) == Path(f"logs/{RECOVERY_NAMESPACE}")
    assert Path(config.stats_file) == Path(
        f"models/training_stats_{RECOVERY_NAMESPACE}.json"
    )
    assert Path(config.promotion_registry) == Path(
        f"logs/{RECOVERY_NAMESPACE}/promotions.jsonl"
    )
    assert Path(config.acceptance_dir) == Path(
        f"logs/{RECOVERY_NAMESPACE}/acceptance"
    )
    assert Path(config.stats_output_dir) == Path(
        f"logs/{RECOVERY_NAMESPACE}/stats"
    )
    assert Path(config.snapshot_root) == Path(
        f"data/corpus_snapshots/{RECOVERY_NAMESPACE}"
    )
    server_cache = profile == "server" or relative_config.endswith(
        "server_policy_distillation.yaml"
    )
    # Keep the server-policy filename distinction while retaining the corrected
    # namespace in both cache variants.
    expected_cache = (
        f"data/replay_cache/{RECOVERY_NAMESPACE}_server_cache.pt"
        if server_cache
        else f"data/replay_cache/{RECOVERY_NAMESPACE}_cache.pt"
    )
    assert Path(config.ram_cache_file) == Path(expected_cache)

    aliases = {
        Path(config.latest_path),
        Path(config.promoted_path),
        Path(config.accepted_path),
    }
    assert len(aliases) == 3
    assert all(RECOVERY_NAMESPACE in alias.stem for alias in aliases)
    assert all(alias.parent == Path("models") for alias in aliases)
    assert BASELINE.parent != Path(config.checkpoint_dir)
    assert Path(config.checkpoint_dir) != Path(
        f"models/checkpoints_{PRESERVED_PRIOR_RECOVERY_NAMESPACE}"
    )


@pytest.mark.parametrize(
    "launcher",
    ["local_train.sh", "train_server.sh", "local_train.ps1"],
)
def test_recovery_launchers_seed_isolated_stats_without_overwriting_legacy(
    launcher: str,
) -> None:
    text = (PROJECT_ROOT / launcher).read_text(encoding="utf-8")

    assert "training_stats_policy_distillation.json" in text
    assert "training_stats_policy_distillation_recovery_wd1e4.json" in text
    assert "without modifying the legacy stats file" in text


def test_recovery_updates_progress_report_to_recovery_log_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        stats_file=str(tmp_path / "models" / "training_stats.json"),
        log_dir="logs/policy_distillation_recovery",
        recovery_enforced=True,
    )
    fake_outputs = []

    def _fake_write_progress_outputs(
            *,
            stats_path,
            logs_dir,
            html_output_path,
            image_output_path=None,
            dpi=300,
    ):
        fake_outputs.append((stats_path, logs_dir, html_output_path, image_output_path))
        return {
            'html': Path(html_output_path),
            'image': (
                Path(image_output_path)
                if image_output_path is not None
                else Path(html_output_path).with_suffix('.png')
            ),
        }

    fake_plot_training = ModuleType("plot_training")
    fake_plot_training.write_progress_outputs = _fake_write_progress_outputs
    monkeypatch.setitem(sys.modules, "plot_training", fake_plot_training)

    trainer_obj = SimpleNamespace(config=config)
    path = Path(config.stats_file)
    Trainer._update_training_progress_report(trainer_obj, path)

    assert fake_outputs, "Expected progress report writer to be invoked"
    _, _, html_output_path, _ = fake_outputs[0]
    assert Path(html_output_path) == Path("logs/policy_distillation_recovery") / "training_progress.html"


def test_non_recovery_updates_progress_report_to_stats_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        stats_file=str(tmp_path / "models" / "training_stats.json"),
        log_dir="logs/training",
        recovery_enforced=False,
    )
    fake_outputs = []

    def _fake_write_progress_outputs(
            *,
            stats_path,
            logs_dir,
            html_output_path,
            image_output_path=None,
            dpi=300,
    ):
        fake_outputs.append((stats_path, logs_dir, html_output_path, image_output_path))
        return {
            'html': Path(html_output_path),
            'image': (
                Path(image_output_path)
                if image_output_path is not None
                else Path(html_output_path).with_suffix('.png')
            ),
        }

    fake_plot_training = ModuleType("plot_training")
    fake_plot_training.write_progress_outputs = _fake_write_progress_outputs
    monkeypatch.setitem(sys.modules, "plot_training", fake_plot_training)

    trainer_obj = SimpleNamespace(config=config)
    path = Path(config.stats_file)
    Trainer._update_training_progress_report(trainer_obj, path)

    assert fake_outputs, "Expected progress report writer to be invoked"
    _, _, html_output_path, _ = fake_outputs[0]
    assert Path(html_output_path) == path.parent / "training_progress.html"
