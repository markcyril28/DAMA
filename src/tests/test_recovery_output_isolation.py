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
# The approved 2026-08-24 continuation resumes the best durable checkpoint --
# step 174000 at 49.30% frozen-suite teacher agreement -- in a namespace of its
# own, so the occupied 136000-196000 range is never rewritten.
CONTINUATION_BASELINE = Path(
    "models/checkpoints_policy_distillation_recovery_wd1e4/model_step_174000.pt"
)
# The prior recovery namespace contains the preserved 136k-144k artifacts.
# Corrected 1e-4 runs must never reuse or overwrite it.
PRESERVED_PRIOR_RECOVERY_NAMESPACE = "policy_distillation_recovery"
RECOVERY_NAMESPACE = "policy_distillation_recovery_wd1e4"
CONTINUATION_NAMESPACE = "policy_distillation_recovery_c174k"
RECOVERY_CHECKPOINT_DIR = Path(
    f"models/checkpoints_{RECOVERY_NAMESPACE}"
)


@pytest.mark.parametrize(
    ("relative_config", "profile", "expected_baseline", "expected_namespace"),
    [
        ("config/training_config_policy_distillation.yaml", None,
         BASELINE, RECOVERY_NAMESPACE),
        ("config/training_config_policy_distillation.yaml", "server",
         BASELINE, RECOVERY_NAMESPACE),
        ("config/training_config_server_policy_distillation.yaml", None,
         BASELINE, RECOVERY_NAMESPACE),
        ("config/training_config_policy_distillation_c174k.yaml", None,
         CONTINUATION_BASELINE, CONTINUATION_NAMESPACE),
        ("config/training_config_policy_distillation_c174k.yaml", "server",
         CONTINUATION_BASELINE, CONTINUATION_NAMESPACE),
    ],
)
def test_recovery_configs_isolate_all_mutable_model_outputs(
    relative_config: str,
    profile: str | None,
    expected_baseline: Path,
    expected_namespace: str,
) -> None:
    path = PROJECT_ROOT / relative_config
    config = config_from_yaml(load_config_from_yaml(str(path), profile))
    RECOVERY_NAMESPACE = expected_namespace

    validate_recovery_experiment_config(config)
    assert Path(config.recovery_baseline_path or "") == expected_baseline
    assert Path(config.resume or "") == expected_baseline
    assert Path(config.checkpoint_dir) == Path(
        f"models/checkpoints_{RECOVERY_NAMESPACE}"
    )
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
    assert expected_baseline.parent != Path(config.checkpoint_dir)
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
    """The seed is config-driven now, so no launcher may hardcode an anchor.

    All three used to carry their own copy of the step-134000 path, digest, and
    stats filenames.  Moving the anchor to step 174000 would then require four
    edits (three launchers plus the config) with nothing detecting a miss.
    """
    text = (PROJECT_ROOT / launcher).read_text(encoding="utf-8")

    assert "seed_stats_from" in text
    assert "without modifying the legacy stats file" in text
    assert "model_step_134000.pt" not in text, (
        f"{launcher} still hardcodes the superseded step-134000 anchor")
    assert "7238CD80F2EF6DC9D8487D2579DE4BDF35AF4B85DCB2B3BD271659E795B14D27" not in text, (
        f"{launcher} still hardcodes an anchor digest")


@pytest.mark.parametrize(
    "launcher",
    ["local_train.sh", "train_server.sh", "local_train.ps1"],
)
def test_recovery_launchers_select_the_continuation_config(launcher: str) -> None:
    text = (PROJECT_ROOT / launcher).read_text(encoding="utf-8")
    assert "training_config_policy_distillation_c174k.yaml" in text


def test_every_recovery_config_seeds_its_stats_from_a_readable_source() -> None:
    import yaml

    for relative in (
        "config/training_config_policy_distillation.yaml",
        "config/training_config_policy_distillation_c174k.yaml",
    ):
        raw = yaml.safe_load(
            (PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        paths = raw["paths"]
        assert paths["seed_stats_from"]
        assert paths["seed_stats_from"] != paths["stats_file"], (
            f"{relative} would seed its stats file from itself")


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


def test_gui_offers_the_continuation_preset_before_the_superseded_one() -> None:
    """The GUI preset list is a fourth place the active config is chosen.

    Its preference used to require checkpoints in the preset's own checkpoint
    directory. The continuation namespace is empty until its first run, so that
    rule would have quietly offered the superseded step-134000 preset instead.
    """
    from dama.ui.training_panel import (
        _PREFERRED_TRAINING_PRESETS,
        _training_preset_rank,
    )

    assert _PREFERRED_TRAINING_PRESETS[0] == (
        "training_config_policy_distillation_c174k.yaml")
    assert _training_preset_rank(
        "training_config_policy_distillation_c174k.yaml"
    ) < _training_preset_rank("training_config_policy_distillation.yaml")
    assert _training_preset_rank(
        "training_config_policy_distillation.yaml"
    ) < _training_preset_rank("training_config_local.yaml")
