#!/usr/bin/env bash
# =============================================================================
# eval_checkpoints.sh - Evaluate checkpoints against the algorithmic AI.
#
# Resolves the config selected by local_train.sh, watches that run's checkpoint
# directory, evaluates each untested checkpoint, appends namespaced JSONL
# results, and regenerates a namespaced PNG plot.
#
# Usage:
#   bash eval_checkpoints.sh              # watch mode (default)
#   bash eval_checkpoints.sh --once       # evaluate untested checkpoints and exit
#   bash eval_checkpoints.sh --replot     # regenerate the plot only
#   bash eval_checkpoints.sh --all        # re-evaluate all checkpoints
#
# Optional environment overrides:
#   EVAL_CONFIG=config/training_config.yaml NUM_GAMES=50 bash eval_checkpoints.sh --once
#   DAMA_PYTHON=/path/to/dama/bin/python CHECKPOINT_DIR=/path bash eval_checkpoints.sh --once
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================== CONFIGURATION ====================================

CONDA_ENV="${CONDA_ENV:-dama}"

# Follow the launcher's active CONFIG SELECTION by default. EVAL_CONFIG remains
# an explicit escape hatch for evaluating another run without editing a script.
if [[ -n "${EVAL_CONFIG:-}" ]]; then
    CONFIG_FILE="$EVAL_CONFIG"
else
    _selected_config="$({
        awk -F'"' '/^TRAINING_CONFIG="/ { selected = $2 } END { print selected }' \
            "$PROJECT_DIR/local_train.sh"
    } 2>/dev/null)"
    if [[ -z "$_selected_config" ]]; then
        echo "ERROR: Could not resolve the active TRAINING_CONFIG from local_train.sh." >&2
        exit 1
    fi
    CONFIG_FILE="$_selected_config"
fi
[[ "$CONFIG_FILE" = /* ]] || CONFIG_FILE="$PROJECT_DIR/$CONFIG_FILE"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Evaluation config not found: $CONFIG_FILE" >&2
    exit 1
fi
CONFIG_FILE="$(readlink -f "$CONFIG_FILE")"

_yaml_scalar() {
    # _yaml_scalar <file> <top-level-section> <key>
    awk -v section="$2" -v key="$3" '
        /^[^[:space:]#]/ { in_section = ($0 ~ "^"section":[[:space:]]*$"); next }
        in_section && $0 ~ "^[[:space:]]+"key":[[:space:]]*" {
            line = $0
            sub("^[[:space:]]+"key":[[:space:]]*", "", line)
            sub(/[[:space:]]*#.*$/, "", line)
            gsub(/^"|"$|^'"'"'|'"'"'$/, "", line)
            print line
            exit
        }
    ' "$1"
}
_abs_project_path() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *) printf '%s/%s' "$PROJECT_DIR" "$1" ;;
    esac
}

_checkpoint_rel="$(_yaml_scalar "$CONFIG_FILE" paths checkpoint_dir)"
if [[ -z "$_checkpoint_rel" ]]; then
    echo "ERROR: paths.checkpoint_dir is missing from $CONFIG_FILE" >&2
    exit 1
fi
OUTPUT_NAMESPACE="$(_yaml_scalar "$CONFIG_FILE" paths policy_output_namespace)"
if [[ -n "$OUTPUT_NAMESPACE" && "$OUTPUT_NAMESPACE" == *[!A-Za-z0-9._-]* ]]; then
    echo "ERROR: Unsafe paths.policy_output_namespace in $CONFIG_FILE: $OUTPUT_NAMESPACE" >&2
    exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(_abs_project_path "$_checkpoint_rel")}"
if [[ -n "$OUTPUT_NAMESPACE" ]]; then
    TEST_STATS_DIR="${TEST_STATS_DIR:-$PROJECT_DIR/models/test_stats/$OUTPUT_NAMESPACE}"
    RESULTS_FILE="${RESULTS_FILE:-$PROJECT_DIR/models/eval_results_$OUTPUT_NAMESPACE.jsonl}"
    PLOT_OUTPUT="${PLOT_OUTPUT:-$PROJECT_DIR/models/eval_progress_$OUTPUT_NAMESPACE.png}"
    LOCK_DIR="${LOCK_DIR:-$PROJECT_DIR/models/.eval_checkpoints_$OUTPUT_NAMESPACE.lock}"
else
    # Legacy configs do not declare a namespace, so retain their established
    # generic output locations while still taking checkpoint_dir from YAML.
    TEST_STATS_DIR="${TEST_STATS_DIR:-$PROJECT_DIR/models/test_stats}"
    RESULTS_FILE="${RESULTS_FILE:-$PROJECT_DIR/models/eval_results.jsonl}"
    PLOT_OUTPUT="${PLOT_OUTPUT:-$PROJECT_DIR/models/eval_progress.png}"
    LOCK_DIR="${LOCK_DIR:-$PROJECT_DIR/models/.eval_checkpoints.lock}"
fi
_configured_test_games="$(_yaml_scalar "$CONFIG_FILE" testing num_games)"
_configured_test_difficulty="$(_yaml_scalar "$CONFIG_FILE" testing difficulty)"
_configured_max_moves="$(_yaml_scalar "$CONFIG_FILE" selfplay max_moves_per_game)"
NUM_GAMES="${NUM_GAMES:-${_configured_test_games:-100}}"          # Games per checkpoint evaluation
ALGO_DIFFICULTY="${ALGO_DIFFICULTY:-${_configured_test_difficulty:-easy}}"
NUM_WORKERS="${NUM_WORKERS:-4}"               # Parallel test workers
MAX_MOVES="${MAX_MOVES:-${_configured_max_moves:-200}}"           # Max moves per game before draw
POLL_INTERVAL="${POLL_INTERVAL:-60}"          # Seconds between scans in watch mode

# ========================== END CONFIGURATION ================================

# Never let this diagnostic fall back to conda base. DAMA_PYTHON can pin a
# nonstandard install; otherwise use the active dama env or `conda run`.
PYTHON_CMD=()
if [[ -n "${DAMA_PYTHON:-}" ]]; then
    if [[ ! -x "$DAMA_PYTHON" ]]; then
        echo "ERROR: DAMA_PYTHON is not executable: $DAMA_PYTHON" >&2
        exit 1
    fi
    PYTHON_CMD=("$DAMA_PYTHON")
elif [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python)")
else
    _conda_bin="${CONDA_EXE:-}"
    [[ -x "$_conda_bin" ]] || _conda_bin="$(command -v conda 2>/dev/null || true)"
    if [[ ! -x "$_conda_bin" ]]; then
        for _candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
                          "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" \
                          "/opt/conda/bin/conda"; do
            if [[ -x "$_candidate" ]]; then
                _conda_bin="$_candidate"
                break
            fi
        done
    fi
    if [[ ! -x "$_conda_bin" ]]; then
        echo "ERROR: The '$CONDA_ENV' interpreter is required. Run bash setup_conda.sh." >&2
        exit 1
    fi
    if ! _dama_python="$("$_conda_bin" run -n "$CONDA_ENV" python -c \
        'import sys; print(sys.executable)' 2>/dev/null)" || [[ ! -x "$_dama_python" ]]; then
        echo "ERROR: Could not resolve the '$CONDA_ENV' interpreter. Run bash setup_conda.sh." >&2
        exit 1
    fi
    PYTHON_CMD=("$_dama_python")
fi

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if ! _python_info="$("${PYTHON_CMD[@]}" -c '
import sys
from dama.ai.algorithmic.search import _HAS_FAST_SEARCH
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"expected CPython 3.11, got {sys.version.split()[0]}")
if not _HAS_FAST_SEARCH:
    raise SystemExit("the compiled fast-search extension did not load")
print(sys.executable)
' 2>&1)"; then
    echo "ERROR: The dama interpreter preflight failed: $_python_info" >&2
    exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: Configured checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi
mkdir -p "$TEST_STATS_DIR" "$(dirname "$RESULTS_FILE")" "$(dirname "$PLOT_OUTPUT")"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
    cat <<EOF
Usage: $0 [--once|--all|--replot|--watch]

Modes:
  --watch   (default) Poll for new checkpoints and evaluate them
  --once    Evaluate untested checkpoints and exit
  --all     Re-evaluate all checkpoints from scratch
  --replot  Regenerate plot from existing results
EOF
}

cleanup_lock() {
    if [[ -d "$LOCK_DIR" ]]; then
        rm -f "$LOCK_DIR/pid"
        rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
}

acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        printf '%s\n' "$$" > "$LOCK_DIR/pid"
        trap cleanup_lock EXIT INT TERM
        return 0
    fi

    local pid=""
    if [[ -f "$LOCK_DIR/pid" ]]; then
        pid="$(<"$LOCK_DIR/pid")"
    fi

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Another evaluation process is already running (pid $pid)."
        exit 1
    fi

    log "Removing stale lock: $LOCK_DIR"
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || {
        log "Could not remove stale lock: $LOCK_DIR"
        exit 1
    }
    acquire_lock
}

extract_step() {
    local name="$1"
    if [[ "$name" =~ ^model_step_0*([0-9]+)\.pt$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    fi
}

already_tested() {
    local ckpt_name="$1"
    [[ -f "$RESULTS_FILE" ]] || return 1

    CHECKPOINT_NAME="$ckpt_name" RESULTS_FILE="$RESULTS_FILE" EXPECTED_GAMES="$NUM_GAMES" \
        "${PYTHON_CMD[@]}" - <<'PY'
import json
import os
import sys
from pathlib import Path

checkpoint_name = os.environ["CHECKPOINT_NAME"]
results_file = Path(os.environ["RESULTS_FILE"])
expected_games = int(os.environ["EXPECTED_GAMES"])

try:
    with results_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if (
                    record.get("checkpoint") == checkpoint_name
                    and int(record.get("total_games") or 0) >= expected_games
                ):
                    sys.exit(0)
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    pass

sys.exit(1)
PY
}

# ---------------------------------------------------------------------------
# Evaluate a single checkpoint
# ---------------------------------------------------------------------------

evaluate_checkpoint() {
    local ckpt_path="$1"
    local ckpt_name step json_line wr

    ckpt_name="$(basename "$ckpt_path")"
    step="$(extract_step "$ckpt_name")"

    if [[ -z "$step" ]]; then
        log "SKIP: cannot parse step from $ckpt_name"
        return 1
    fi

    log "EVAL: $ckpt_name (step $step) - $NUM_GAMES games vs $ALGO_DIFFICULTY algo"

    if ! json_line="$(
        cd "$PROJECT_DIR"
        "${PYTHON_CMD[@]}" -m dama.ai.ml.eval_checkpoint_once \
            --checkpoint "$ckpt_path" \
            --checkpoint-name "$ckpt_name" \
            --step "$step" \
            --num-games "$NUM_GAMES" \
            --difficulty "$ALGO_DIFFICULTY" \
            --num-workers "$NUM_WORKERS" \
            --max-moves "$MAX_MOVES" \
            --stats-dir "$TEST_STATS_DIR"
    )"; then
        log "ERROR: evaluation failed for $ckpt_name"
        return 1
    fi

    if ! printf '%s\n' "$json_line" | \
         "${PYTHON_CMD[@]}" -c 'import json, sys; json.load(sys.stdin)' 2>/dev/null; then
        log "ERROR: invalid JSON from evaluation of $ckpt_name"
        log "Output: $json_line"
        return 1
    fi

    printf '%s\n' "$json_line" >> "$RESULTS_FILE"
    wr="$(printf '%s\n' "$json_line" | \
        "${PYTHON_CMD[@]}" -c 'import json, sys; d=json.load(sys.stdin); print("{:.1%}".format(d["ml_win_rate"]))')"
    log "DONE: $ckpt_name - win rate: $wr"
}

# ---------------------------------------------------------------------------
# Generate the plot
# ---------------------------------------------------------------------------

generate_plot() {
    if [[ ! -f "$RESULTS_FILE" ]]; then
        log "No results file yet, skipping plot."
        return 0
    fi

    local count
    count="$(grep -cve '^[[:space:]]*$' "$RESULTS_FILE" || true)"
    if [[ "$count" -eq 0 ]]; then
        log "No results to plot."
        return 0
    fi

    log "PLOT: generating $PLOT_OUTPUT ($count evaluations)"

    cd "$PROJECT_DIR"
    RESULTS_FILE="$RESULTS_FILE" PLOT_OUTPUT="$PLOT_OUTPUT" \
        "${PYTHON_CMD[@]}" - <<'PY'
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_file = Path(os.environ["RESULTS_FILE"])
plot_output = Path(os.environ["PLOT_OUTPUT"])

entries = []
with results_file.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping malformed JSONL line: {line[:120]}")
            continue
        if int(entry.get("total_games") or 0) <= 0:
            print(f"Skipping incomplete evaluation: {entry.get('checkpoint', 'unknown')}")
            continue
        entries.append(entry)

if not entries:
    raise SystemExit("No valid results to plot.")

entries.sort(key=lambda item: (item.get("step", 0), item.get("timestamp", "")))

# Deduplicate by checkpoint name when available, otherwise by step. Keep latest.
by_checkpoint = {}
for entry in entries:
    key = entry.get("checkpoint") or f"step:{entry.get('step')}"
    by_checkpoint[key] = entry
entries = sorted(by_checkpoint.values(), key=lambda item: item["step"])

steps = [entry["step"] for entry in entries]
wr = [entry["ml_win_rate"] * 100 for entry in entries]
p1_wr = [entry["ml_as_p1_win_rate"] * 100 for entry in entries]
p2_wr = [entry["ml_as_p2_win_rate"] * 100 for entry in entries]
game_len = [entry["avg_game_length"] for entry in entries]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Checkpoint Evaluation Progress", fontsize=16, fontweight="bold")

ax = axes[0, 0]
ax.plot(steps, wr, "b-o", lw=2, ms=5, label="ML Win Rate")
ax.axhline(50, color="r", ls="--", alpha=0.6, label="50%")
ax.fill_between(steps, wr, alpha=0.2)
if len(steps) > 2 and min(steps) != max(steps):
    degree = min(2, len(steps) - 1)
    z = np.polyfit(steps, wr, degree)
    xs = np.linspace(min(steps), max(steps), 200)
    ax.plot(xs, np.clip(np.polyval(z, xs), 0, 100), "g--", alpha=0.7, label="Trend")
ax.set_xlabel("Training Step")
ax.set_ylabel("Win Rate (%)")
ax.set_title("ML vs Algorithm - Overall")
ax.set_ylim(0, 100)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(steps, p1_wr, "g-s", lw=2, ms=4, label="ML as P1 (White)")
ax.plot(steps, p2_wr, "m-^", lw=2, ms=4, label="ML as P2 (Black)")
ax.axhline(50, color="r", ls="--", alpha=0.6)
ax.set_xlabel("Training Step")
ax.set_ylabel("Win Rate (%)")
ax.set_title("Win Rate by Position")
ax.set_ylim(0, 100)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(steps, game_len, "c-d", lw=2, ms=4)
ax.set_xlabel("Training Step")
ax.set_ylabel("Moves")
ax.set_title("Average Game Length")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.axis("off")

total_games = sum(entry["total_games"] for entry in entries)
total_ml = sum(entry["ml_wins"] for entry in entries)
total_algo = sum(entry["algo_wins"] for entry in entries)
total_draws = sum(entry["draws"] for entry in entries)
overall_wr = total_ml / total_games * 100 if total_games else 0
algo_wr = total_algo / total_games * 100 if total_games else 0
best_wr = max(wr)
best_step = steps[wr.index(best_wr)]
latest_wr = wr[-1]

summary = f"""
Evaluation Summary
==================

Checkpoints Tested:   {len(entries)}
Difficulty:           {entries[-1].get("algo_difficulty", "N/A")}
Games per Checkpoint: {entries[-1].get("total_games", "N/A")}

Aggregate Results
=================
Total Games:          {total_games:,}
ML Wins:              {total_ml:,} ({overall_wr:.1f}%)
Algo Wins:            {total_algo:,} ({algo_wr:.1f}%)
Draws:                {total_draws:,}

Best Win Rate:        {best_wr:.1f}% (step {best_step:,})
Latest Win Rate:      {latest_wr:.1f}% (step {steps[-1]:,})
Step Range:           {steps[0]:,} - {steps[-1]:,}
"""

ax.text(
    0.05,
    0.95,
    summary,
    transform=ax.transAxes,
    fontsize=10,
    va="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plot_output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_output, dpi=150, bbox_inches="tight")
print(f"Plot saved to {plot_output}")
PY

    log "PLOT: done"
}

# ---------------------------------------------------------------------------
# Main modes
# ---------------------------------------------------------------------------

run_pending() {
    local mode="${1:-}"
    local tested=0
    local checkpoints=()

    mapfile -t checkpoints < <(
        find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'model_step_*.pt' -print | sort -V
    )

    if [[ "${#checkpoints[@]}" -eq 0 ]]; then
        log "No checkpoints found in $CHECKPOINT_DIR."
        return 0
    fi

    local ckpt name
    for ckpt in "${checkpoints[@]}"; do
        name="$(basename "$ckpt")"
        if [[ "$mode" == "--all" ]] || ! already_tested "$name"; then
            if evaluate_checkpoint "$ckpt"; then
                tested=$((tested + 1))
                generate_plot
            else
                log "WARN: evaluation failed for $name; continuing."
            fi
        fi
    done

    if [[ "$tested" -eq 0 ]]; then
        log "No new checkpoints to evaluate."
    else
        log "Evaluated $tested checkpoint(s)."
    fi
}

mode="${1:---watch}"
if [[ "$mode" != "-h" && "$mode" != "--help" ]]; then
    log "Config: ${CONFIG_FILE#"$PROJECT_DIR"/}"
    log "Namespace: ${OUTPUT_NAMESPACE:-legacy}"
    log "Python: $_python_info"
fi
case "$mode" in
    --once)
        acquire_lock
        log "Running single pass."
        run_pending
        ;;
    --all)
        acquire_lock
        log "Re-evaluating all checkpoints."
        : > "$RESULTS_FILE"
        run_pending "--all"
        ;;
    --replot)
        acquire_lock
        generate_plot
        ;;
    --watch|watch)
        acquire_lock
        log "Watching $CHECKPOINT_DIR for new checkpoints (poll every ${POLL_INTERVAL}s)"
        log "Config: $NUM_GAMES games, $ALGO_DIFFICULTY difficulty, $NUM_WORKERS workers"
        log "Results: $RESULTS_FILE"
        log "Plot:    $PLOT_OUTPUT"
        log "Press Ctrl+C to stop."
        printf '\n'

        run_pending
        while true; do
            sleep "$POLL_INTERVAL"
            run_pending
        done
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
