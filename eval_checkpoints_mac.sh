#!/usr/bin/env bash
# =============================================================================
# eval_checkpoints_mac.sh - Evaluate checkpoints against the algorithmic AI
# (macOS / Apple Silicon).
#
# macOS counterpart of eval_checkpoints.sh. Watches the checkpoint directory of
# config/training_config_mac.yaml (models/checkpoints_mac/) for new
# model_step_*.pt files, evaluates each untested checkpoint, appends compact
# JSONL results, and regenerates a PNG plot. The games run on the CPU, as they
# do on every machine.
#
# The results, plot and lock are named after the watched directory
# (checkpoints_mac/ -> models/eval_results_mac.jsonl, eval_progress_mac.png,
# .eval_checkpoints_mac.lock). "Already tested" is decided by checkpoint file
# name alone, and every machine names its checkpoints model_step_NNNNNN.pt, so
# a results file shared with another directory's evaluations -- such as
# eval_results.jsonl copied over with models/ -- would skip these as done.
#
# Usage:
#   bash eval_checkpoints_mac.sh              # watch mode (default)
#   bash eval_checkpoints_mac.sh --once       # evaluate untested checkpoints and exit
#   bash eval_checkpoints_mac.sh --replot     # regenerate the plot only
#   bash eval_checkpoints_mac.sh --all        # re-evaluate all checkpoints
#
# Optional environment overrides (relative paths are taken from the project folder):
#   NUM_GAMES=50 ALGO_DIFFICULTY=hard NUM_WORKERS=2 bash eval_checkpoints_mac.sh --once
#   TRAINING_CONFIG=config/other.yaml bash eval_checkpoints_mac.sh --once   # watch another config's checkpoints
#   CHECKPOINT_DIR=models/checkpoints bash eval_checkpoints_mac.sh --once   # watch a directory directly
# =============================================================================

set -euo pipefail

# sysctl lives in /usr/sbin, which cron's default PATH (/usr/bin:/bin) leaves
# out; without it the checks below would misread this Mac.
PATH="$PATH:/usr/sbin"
# Platform guard. The Linux/WSL version is eval_checkpoints.sh.
if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: eval_checkpoints_mac.sh is for macOS. Use eval_checkpoints.sh on Linux/WSL." >&2
    exit 1
fi
# A terminal running under Rosetta makes every universal binary it starts
# (bash itself, Apple's /usr/bin/python3) run as x86_64 too. Re-run natively.
# _DAMA_ARM64_REEXEC stops a loop if the re-exec is still translated somehow.
if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
    if [ -z "${_DAMA_ARM64_REEXEC:-}" ]; then
        echo "Terminal is running under Rosetta (x86_64); re-running natively as arm64..."
        export _DAMA_ARM64_REEXEC=1
        exec arch -arm64 /bin/bash "$0" "$@"
    fi
    echo "[warn] Still running under Rosetta after re-exec; continuing as x86_64."
fi

# Get the directory where this script is located
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Config and override paths are project-relative, as the trainer reads them.
cd "$PROJECT_DIR"

# ========================== CONFIGURATION ====================================

TRAINING_CONFIG="${TRAINING_CONFIG:-config/training_config_mac.yaml}"  # Its paths.checkpoint_dir is watched
CONDA_ENV="${CONDA_ENV:-dama}"                # Conda env activated if not already active (see setup_conda_mac.sh)
NUM_GAMES="${NUM_GAMES:-100}"                 # Games per checkpoint evaluation
ALGO_DIFFICULTY="${ALGO_DIFFICULTY:-easy}"    # Algorithmic opponent difficulty
NUM_WORKERS="${NUM_WORKERS:-4}"               # Parallel test workers
MAX_MOVES="${MAX_MOVES:-200}"                 # Max moves per game before draw
POLL_INTERVAL="${POLL_INTERVAL:-60}"          # Seconds between scans in watch mode

# ========================== END CONFIGURATION ================================

# Same reader as local_train_mac.sh: one scalar from a top-level YAML section.
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
_abs_project_path() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PROJECT_DIR" "$1" ;; esac; }

# Watch the directory the training config writes to, so a Mac run is found
# without an override. An explicit CHECKPOINT_DIR still wins.
if [ -z "${CHECKPOINT_DIR:-}" ]; then
    _config_path="$(_abs_project_path "$TRAINING_CONFIG")"
    if [ ! -f "$_config_path" ]; then
        echo "ERROR: Training config not found: $_config_path" >&2
        echo "       Set TRAINING_CONFIG to the config you train with, or CHECKPOINT_DIR to a directory." >&2
        exit 1
    fi
    _ckpt_rel="$(_yaml_scalar "$_config_path" paths checkpoint_dir)"
    # A config without the key trains into config_from_yaml()'s default.
    CHECKPOINT_DIR="${_ckpt_rel:-models/checkpoints}"
fi
CHECKPOINT_DIR="$(_abs_project_path "$CHECKPOINT_DIR")"

# checkpoints_mac -> _mac; any other directory name is used whole, so the
# names below never fall back to eval_checkpoints.sh's unsuffixed ones.
_ckpt_tag="$(basename "$CHECKPOINT_DIR")"
_ckpt_tag="_${_ckpt_tag#checkpoints_}"

TEST_STATS_DIR="${TEST_STATS_DIR:-$PROJECT_DIR/models/test_stats${_ckpt_tag}}"
RESULTS_FILE="${RESULTS_FILE:-$PROJECT_DIR/models/eval_results${_ckpt_tag}.jsonl}"
PLOT_OUTPUT="${PLOT_OUTPUT:-$PROJECT_DIR/models/eval_progress${_ckpt_tag}.png}"
LOCK_DIR="${LOCK_DIR:-$PROJECT_DIR/models/.eval_checkpoints${_ckpt_tag}.lock}"

# Everything below runs Python. Without the conda env that is Apple's
# /usr/bin/python3 -- no torch, and on a Mac without the Xcode Command Line
# Tools it opens an installer dialog instead of running. Activate the env the
# same way local_train_mac.sh does when it isn't already active.
#
# `python`, not `python3`, as in local_train_mac.sh: when the base env was
# already active with Homebrew ahead of it on PATH, `conda activate` puts the
# env's bin/ where base's was -- still behind Homebrew -- so `python3` stays
# Homebrew's, which has neither torch nor yaml. Conda envs ship `python`;
# Homebrew and Apple ship only `python3`.
_find_python() { command -v python 2>/dev/null || command -v python3 2>/dev/null || echo python3; }
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    _conda_bin="${CONDA_EXE:-}"
    [ -x "$_conda_bin" ] || _conda_bin="$(command -v conda 2>/dev/null || true)"
    CONDA_BASE=""
    [ -x "$_conda_bin" ] && CONDA_BASE="$(dirname "$(dirname "$_conda_bin")")"
    if [ -z "$CONDA_BASE" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        CONDA_BASE=""
        for _b in "/opt/homebrew/Caskroom/miniforge/base" "/opt/homebrew/Caskroom/miniconda/base" \
                  "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" \
                  "/opt/miniconda3" "/opt/anaconda3"; do
            if [ -f "$_b/etc/profile.d/conda.sh" ]; then CONDA_BASE="$_b"; break; fi
        done
    fi
    if [ -n "$CONDA_BASE" ]; then
        set +eu                      # conda.sh / activate are not -eu clean
        . "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" \
            || echo "[warn] Could not activate conda env '${CONDA_ENV}' (create it with: bash setup_conda_mac.sh); using $(_find_python)"
        set -euo pipefail
    else
        echo "[warn] conda installation not found; using $(_find_python)"
    fi
fi
# Prefer the activated env's own interpreter: a pyenv shim can sit ahead of
# conda on PATH and ship its own `python`, which `command -v` would pick.
if [ "$(basename "${CONDA_PREFIX:-/}")" = "$CONDA_ENV" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="$(_find_python)"
fi

mkdir -p "$CHECKPOINT_DIR" "$(dirname "$RESULTS_FILE")" "$(dirname "$PLOT_OUTPUT")" "$(dirname "$LOCK_DIR")"

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
        trap cleanup_lock EXIT
        # Exit on Ctrl+C / TERM rather than only releasing the lock: a trap
        # that returns lets bash carry on, and a Ctrl+C during an evaluation
        # (which that evaluation reports as failed) then went on to the next
        # checkpoint with the lock already gone. The exit runs the EXIT trap.
        trap 'exit 130' INT
        trap 'exit 143' TERM
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

# Checked once up front: an interpreter without the project's packages fails
# every evaluation, which the loop below only logs as a warning per checkpoint.
require_python_packages() {
    if ! "$PYTHON_BIN" -c 'import torch, yaml' >/dev/null 2>&1; then
        echo "ERROR: $PYTHON_BIN cannot import torch and yaml, so no checkpoint can be evaluated." >&2
        echo "       Create the '${CONDA_ENV}' env with: bash setup_conda_mac.sh (or set CONDA_ENV to yours)." >&2
        exit 1
    fi
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

    CHECKPOINT_NAME="$ckpt_name" RESULTS_FILE="$RESULTS_FILE" EXPECTED_GAMES="$NUM_GAMES" "$PYTHON_BIN" - <<'PY'
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
        PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" -m dama.ai.ml.eval_checkpoint_once \
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

    if ! printf '%s\n' "$json_line" | "$PYTHON_BIN" -c 'import json, sys; json.load(sys.stdin)' 2>/dev/null; then
        log "ERROR: invalid JSON from evaluation of $ckpt_name"
        log "Output: $json_line"
        return 1
    fi

    printf '%s\n' "$json_line" >> "$RESULTS_FILE"
    wr="$(printf '%s\n' "$json_line" | "$PYTHON_BIN" -c 'import json, sys; d=json.load(sys.stdin); print("{:.1%}".format(d["ml_win_rate"]))')"
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
    RESULTS_FILE="$RESULTS_FILE" PLOT_OUTPUT="$PLOT_OUTPUT" "$PYTHON_BIN" - <<'PY'
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

# Failed evaluations over the whole run; --once and --all exit non-zero on any.
EVAL_FAILURES=0

run_pending() {
    local mode="${1:-}"
    local tested=0
    local failed=0
    local checkpoints=()
    local ckpt name

    # A read loop rather than mapfile, which is bash 4+: macOS ships bash 3.2
    # as /bin/bash. Same result -- one element per line, newline stripped.
    while IFS= read -r ckpt; do
        checkpoints+=("$ckpt")
    done < <(
        find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'model_step_*.pt' -print | sort -V
    )

    # Also load-bearing on bash 3.2: under set -u it rejects "${arr[@]}" for
    # an empty array, so the loop below must never see one.
    if [[ "${#checkpoints[@]}" -eq 0 ]]; then
        log "No checkpoints found in $CHECKPOINT_DIR."
        return 0
    fi

    for ckpt in "${checkpoints[@]}"; do
        name="$(basename "$ckpt")"
        if [[ "$mode" == "--all" ]] || ! already_tested "$name"; then
            if evaluate_checkpoint "$ckpt"; then
                tested=$((tested + 1))
                generate_plot
            else
                failed=$((failed + 1))
                log "WARN: evaluation failed for $name; continuing."
            fi
        fi
    done
    EVAL_FAILURES=$((EVAL_FAILURES + failed))

    if [[ "$failed" -gt 0 ]]; then
        log "Evaluated $tested checkpoint(s); $failed evaluation(s) failed."
    elif [[ "$tested" -eq 0 ]]; then
        log "No new checkpoints to evaluate."
    else
        log "Evaluated $tested checkpoint(s)."
    fi
}

mode="${1:---watch}"
case "$mode" in
    --once)
        require_python_packages
        acquire_lock
        log "Running single pass."
        run_pending
        [[ "$EVAL_FAILURES" -eq 0 ]] || exit 1
        ;;
    --all)
        require_python_packages
        acquire_lock
        log "Re-evaluating all checkpoints."
        : > "$RESULTS_FILE"
        run_pending "--all"
        [[ "$EVAL_FAILURES" -eq 0 ]] || exit 1
        ;;
    --replot)
        acquire_lock
        generate_plot
        ;;
    --watch|watch)
        require_python_packages
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
