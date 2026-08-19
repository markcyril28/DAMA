"""Focused checks for recovery launcher pins and presentation fallbacks."""

from pathlib import Path

import plot_training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BASELINE = "model_step_134000.pt"
EXPECTED_SHA256 = (
    "7238CD80F2EF6DC9D8487D2579DE4BDF35AF4B85DCB2B3BD271659E795B14D27"
)


def test_policy_recovery_launchers_pin_path_and_hash():
    powershell = (PROJECT_ROOT / "local_train.ps1").read_text(encoding="utf-8")
    shell = (PROJECT_ROOT / "local_train.sh").read_text(encoding="utf-8")

    for launcher in (powershell, shell):
        assert EXPECTED_BASELINE in launcher
        assert EXPECTED_SHA256 in launcher

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
