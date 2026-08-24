"""Focused checks for recovery launcher pins and presentation fallbacks."""

from pathlib import Path

import plot_training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The active anchor is the approved 2026-08-24 continuation: the best durable
# checkpoint at 49.30% frozen-suite teacher agreement.
ACTIVE_CONFIG = "config/training_config_policy_distillation_c174k.yaml"
EXPECTED_BASELINE = "model_step_174000.pt"
EXPECTED_SHA256 = (
    "6230777F09927C0794FD97116AB1C6B54887BBE432187EE5961BD86507C908B3"
)


def test_active_recovery_config_pins_the_anchor_path_and_hash():
    """The pin moved out of the launchers and into the config they select.

    It used to be copied into local_train.sh, train_server.sh, and
    local_train.ps1 as well as the config, so moving the anchor from step
    134000 to step 174000 needed four edits with nothing detecting a miss.
    """
    import yaml

    raw = yaml.safe_load(
        (PROJECT_ROOT / ACTIVE_CONFIG).read_text(encoding="utf-8"))
    recovery = raw["recovery_experiment"]
    assert recovery["enabled"] is True
    assert recovery["baseline_checkpoint"].endswith(EXPECTED_BASELINE)
    assert recovery["baseline_sha256"].upper() == EXPECTED_SHA256
    assert raw["resume"]["checkpoint_path"] == recovery["baseline_checkpoint"]


def test_policy_recovery_launchers_read_the_pin_from_the_selected_config():
    powershell = (PROJECT_ROOT / "local_train.ps1").read_text(encoding="utf-8")
    shell = (PROJECT_ROOT / "local_train.sh").read_text(encoding="utf-8")
    server = (PROJECT_ROOT / "train_server.sh").read_text(encoding="utf-8")

    for launcher in (powershell, shell, server):
        assert "recovery_experiment" in launcher
        assert "baseline_sha256" in launcher
        assert "baseline_checkpoint" in launcher
        assert ACTIVE_CONFIG.replace("/", "\\") in launcher or ACTIVE_CONFIG in launcher

    assert "FreshStart is disabled" in powershell
    assert "--resume-latest is disabled" in shell
    assert "EnhancedStage" in powershell
    assert "ENHANCED_STAGE" in shell
    assert "--enhanced-stage" in powershell
    assert "--enhanced-stage" in shell
    assert "recorded promoted policy-only checkpoint" in powershell
    assert "recorded promoted policy-only checkpoint" in shell


def test_recovery_summary_prefers_explicit_metrics():
    stats = {
        "current_train_loss": 0.88,
        "current_dataset_best_train_loss": 0.72,
        "historical_best_train_loss": 0.03,
        "validation_teacher_agreement": 0.51,
        "promotion_state": "eligible",
        "acceptance_state": "pending",
        "match_score": 0.60,
        "match_score_ci_lower": 0.50,
        "match_score_ci_upper": 0.70,
    }

    metrics = plot_training._recovery_summary_metrics(stats, [])

    assert metrics == {
        "current_train_loss": "0.8800",
        "current_dataset_best_train_loss": "0.7200",
        "historical_best_train_loss": "0.0300",
        "validation_teacher_agreement": "51.0%",
        "promotion_state": "eligible",
        "acceptance_state": "pending",
        "match_score": "60.0%",
        "confidence_interval": "50.0% to 70.0%",
    }


def test_recovery_summary_has_safe_legacy_fallbacks():
    stats = {
        "best_loss": 0.0345,
        "loss_history": [{"step": 1, "loss": 0.9}],
    }
    tests = [{
        "total_games": 10,
        "ml_wins": 4,
        "draws": 2,
        "algo_wins": 4,
    }]

    metrics = plot_training._recovery_summary_metrics(stats, tests)

    assert metrics["current_train_loss"] == "0.9000"
    assert metrics["current_dataset_best_train_loss"] == "N/A"
    assert metrics["historical_best_train_loss"] == "0.0345"
    assert metrics["validation_teacher_agreement"] == "N/A"
    assert metrics["promotion_state"] == "N/A"
    assert metrics["acceptance_state"] == "N/A"
    assert metrics["match_score"] == "50.0%"
    assert metrics["confidence_interval"] == "N/A"


def test_recovery_summary_reads_runtime_history_shapes():
    stats = {
        "teacher_agreement_history": [{"top1_teacher_agreement": 0.54}],
        "promotion_history": [{"promoted": True}],
        "acceptance_history": [{"passed": False}],
    }
    tests = [{
        "match_score": 0.62,
        "match_score_ci_95": {
            "lower": 0.52,
            "upper": 0.71,
            "method": "wilson_score_interval_95_match_points",
        },
    }]

    metrics = plot_training._recovery_summary_metrics(stats, tests)

    assert metrics["validation_teacher_agreement"] == "54.0%"
    assert metrics["promotion_state"] == "Promoted"
    assert metrics["acceptance_state"] == "Not accepted"
    assert metrics["match_score"] == "62.0%"
    assert metrics["confidence_interval"] == "52.0% to 71.0%"


def test_recovery_summary_labels_acceptance_opponents_and_ignores_stale_score():
    accepted = {
        "acceptance_state": "accepted",
        "match_score": 0.99,
        "acceptance": {
            "passed": True,
            "metrics": {
                "random": {"match_score": 0.84},
                "easy": {"match_score": 0.63},
            },
        },
    }
    metrics = plot_training._recovery_summary_metrics(accepted, [])
    assert metrics["match_score"] == "Random 84.0%; Easy 63.0%"

    # A legacy score without protocol records must not be shown beside an
    # Accepted state, since its opponent/opening provenance is unknown.
    stale = {"acceptance_state": "Accepted", "match_score": 0.99}
    stale_metrics = plot_training._recovery_summary_metrics(stale, [])
    assert stale_metrics["match_score"] == "N/A"
    assert stale_metrics["confidence_interval"] == "N/A"
