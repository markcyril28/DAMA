#!/usr/bin/env bash
# =============================================================================
# eval_checkpoints.sh — Evaluate each checkpoint against algorithmic AI
#
# Watches models/checkpoints/ for new .pt files, runs ML-vs-algo tests on
# each one that hasn't been tested yet, logs results to a JSONL file, and
# regenerates a progress plot after every evaluation.
#
# Usage:
#   bash eval_checkpoints.sh              # watch mode (default)
#   bash eval_checkpoints.sh --once       # evaluate untested & exit
#   bash eval_checkpoints.sh --replot     # just regenerate the plot
#   bash eval_checkpoints.sh --all        # re-evaluate ALL checkpoints
# =============================================================================

set -euo pipefail

# ========================== CONFIGURATION ====================================

NUM_GAMES=100              # Games per checkpoint evaluation
ALGO_DIFFICULTY="medium"   # Algorithmic opponent difficulty
NUM_WORKERS=4              # Parallel test workers
MAX_MOVES=200              # Max moves per game before draw
POLL_INTERVAL=60           # Seconds between checkpoint scans (watch mode)

# ========================== END CONFIGURATION ================================

# Resolve project root (script lives in project root)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT_DIR="$PROJECT_DIR/models/checkpoints"
RESULTS_FILE="$PROJECT_DIR/models/eval_results.jsonl"
PLOT_OUTPUT="$PROJECT_DIR/models/eval_progress.png"
LOCK_FILE="$PROJECT_DIR/models/.eval_lock"

# Ensure directories exist
mkdir -p "$CHECKPOINT_DIR" "$(dirname "$RESULTS_FILE")"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

already_tested() {
    local ckpt_name="$1"
    [[ -f "$RESULTS_FILE" ]] && grep -q "\"checkpoint\": \"$ckpt_name\"" "$RESULTS_FILE"
}

extract_step() {
    # model_step_002500.pt -> 2500
    local name="$1"
    echo "$name" | sed -n 's/model_step_0*\([0-9]*\)\.pt/\1/p'
}

cleanup_lock() { rm -f "$LOCK_FILE"; }
trap cleanup_lock EXIT

# ---------------------------------------------------------------------------
# Evaluate a single checkpoint
# ---------------------------------------------------------------------------

evaluate_checkpoint() {
    local ckpt_path="$1"
    local ckpt_name
    ckpt_name="$(basename "$ckpt_path")"
    local step
    step="$(extract_step "$ckpt_name")"

    if [[ -z "$step" ]]; then
        log "SKIP: cannot parse step from $ckpt_name"
        return 1
    fi

    log "EVAL: $ckpt_name (step $step) — $NUM_GAMES games vs $ALGO_DIFFICULTY algo"

    # Run the evaluation via Python
    local result
    result=$(cd "$PROJECT_DIR" && PYTHONPATH=src python -c "
import json, sys
from dama.ai.ml.model_vs_algo import ModelVsAlgoTester

tester = ModelVsAlgoTester(
    model_path='$ckpt_path',
    algo_difficulty='$ALGO_DIFFICULTY',
    num_workers=$NUM_WORKERS,
    max_moves=$MAX_MOVES,
    stats_dir='models/test_stats',
)

def progress(done, total, stats):
    if done % 20 == 0 or done == total:
        print(f'  [{done}/{total}] win_rate={stats.ml_win_rate:.2%}', file=sys.stderr)

stats = tester.run_tests(num_games=$NUM_GAMES, callback=progress)

# Emit compact JSONL record
entry = {
    'type': 'test_vs_algo',
    'checkpoint': '$ckpt_name',
    'step': $step,
    'total_games': stats.total_games,
    'ml_wins': stats.ml_wins,
    'algo_wins': stats.algo_wins,
    'draws': stats.draws,
    'ml_win_rate': round(stats.ml_win_rate, 4),
    'ml_as_p1_win_rate': round(stats.ml_as_p1_win_rate, 4),
    'ml_as_p2_win_rate': round(stats.ml_as_p2_win_rate, 4),
    'avg_game_length': round(stats.avg_game_length, 2),
    'algo_difficulty': '$ALGO_DIFFICULTY',
    'timestamp': stats.end_time,
}
print(json.dumps(entry))
" 2>&1)

    # Last line is the JSON; everything before is progress output
    local json_line
    json_line="$(echo "$result" | tail -1)"
    local progress_output
    progress_output="$(echo "$result" | head -n -1)"

    # Show progress
    if [[ -n "$progress_output" ]]; then
        echo "$progress_output"
    fi

    # Validate and append
    if echo "$json_line" | python -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
        echo "$json_line" >> "$RESULTS_FILE"
        local wr
        wr=$(echo "$json_line" | python -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['ml_win_rate']:.1%}\")")
        log "DONE: $ckpt_name — win rate: $wr"
    else
        log "ERROR: invalid JSON from evaluation of $ckpt_name"
        log "Output: $json_line"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Generate the plot
# ---------------------------------------------------------------------------

generate_plot() {
    if [[ ! -f "$RESULTS_FILE" ]]; then
        log "No results file yet, skipping plot."
        return
    fi

    local count
    count=$(wc -l < "$RESULTS_FILE")
    if [[ "$count" -eq 0 ]]; then
        log "No results to plot."
        return
    fi

    log "PLOT: generating $PLOT_OUTPUT ($count evaluations)"

    cd "$PROJECT_DIR" && PYTHONPATH=src python -c "
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
entries = []
with open('$RESULTS_FILE') as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

# Sort by step
entries.sort(key=lambda e: e['step'])

# Deduplicate by step (keep latest)
by_step = {e['step']: e for e in entries}
entries = [by_step[s] for s in sorted(by_step)]

steps      = [e['step'] for e in entries]
wr         = [e['ml_win_rate'] * 100 for e in entries]
p1_wr      = [e['ml_as_p1_win_rate'] * 100 for e in entries]
p2_wr      = [e['ml_as_p2_win_rate'] * 100 for e in entries]
game_len   = [e['avg_game_length'] for e in entries]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Checkpoint Evaluation Progress', fontsize=16, fontweight='bold')

# --- Plot 1: Overall win rate ---
ax = axes[0, 0]
ax.plot(steps, wr, 'b-o', lw=2, ms=5, label='ML Win Rate')
ax.axhline(50, color='r', ls='--', alpha=0.6, label='50%')
ax.fill_between(steps, wr, alpha=0.2)
if len(steps) > 2:
    z = np.polyfit(steps, wr, min(2, len(steps)-1))
    xs = np.linspace(min(steps), max(steps), 200)
    ax.plot(xs, np.clip(np.polyval(z, xs), 0, 100), 'g--', alpha=0.7, label='Trend')
ax.set_xlabel('Training Step')
ax.set_ylabel('Win Rate (%)')
ax.set_title('ML vs Algorithm — Overall')
ax.set_ylim(0, 100)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# --- Plot 2: By position ---
ax = axes[0, 1]
ax.plot(steps, p1_wr, 'g-s', lw=2, ms=4, label='ML as P1 (White)')
ax.plot(steps, p2_wr, 'm-^', lw=2, ms=4, label='ML as P2 (Black)')
ax.axhline(50, color='r', ls='--', alpha=0.6)
ax.set_xlabel('Training Step')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate by Position')
ax.set_ylim(0, 100)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# --- Plot 3: Average game length ---
ax = axes[1, 0]
ax.plot(steps, game_len, 'c-d', lw=2, ms=4)
ax.set_xlabel('Training Step')
ax.set_ylabel('Moves')
ax.set_title('Average Game Length')
ax.grid(True, alpha=0.3)

# --- Plot 4: Summary ---
ax = axes[1, 1]
ax.axis('off')

total_games = sum(e['total_games'] for e in entries)
total_ml    = sum(e['ml_wins'] for e in entries)
total_algo  = sum(e['algo_wins'] for e in entries)
total_draws = sum(e['draws'] for e in entries)
overall_wr  = total_ml / total_games * 100 if total_games else 0
best_wr     = max(wr)
best_step   = steps[wr.index(best_wr)]
latest_wr   = wr[-1]

summary = f'''
Evaluation Summary
==================

Checkpoints Tested:  {len(entries)}
Difficulty:          {entries[0].get('algo_difficulty', 'N/A')}
Games per Checkpoint:{entries[0].get('total_games', 'N/A')}

Aggregate Results
=================
Total Games:         {total_games:,}
ML Wins:             {total_ml:,} ({overall_wr:.1f}%)
Algo Wins:           {total_algo:,} ({100-overall_wr:.1f}%)
Draws:               {total_draws:,}

Best Win Rate:       {best_wr:.1f}% (step {best_step:,})
Latest Win Rate:     {latest_wr:.1f}% (step {steps[-1]:,})
Step Range:          {steps[0]:,} — {steps[-1]:,}
'''

ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('$PLOT_OUTPUT', dpi=150, bbox_inches='tight')
print('Plot saved to $PLOT_OUTPUT')
"
    log "PLOT: done"
}

# ---------------------------------------------------------------------------
# Main modes
# ---------------------------------------------------------------------------

run_pending() {
    # Evaluate all untested checkpoints in order
    local tested=0
    for ckpt in $(ls "$CHECKPOINT_DIR"/model_step_*.pt 2>/dev/null | sort); do
        local name
        name="$(basename "$ckpt")"
        if [[ "$1" == "--all" ]] || ! already_tested "$name"; then
            evaluate_checkpoint "$ckpt" && tested=$((tested + 1))
            generate_plot
        fi
    done
    if [[ "$tested" -eq 0 ]]; then
        log "No new checkpoints to evaluate."
    else
        log "Evaluated $tested checkpoint(s)."
    fi
}

case "${1:-watch}" in
    --once)
        log "Running single pass..."
        run_pending ""
        ;;
    --all)
        log "Re-evaluating ALL checkpoints..."
        # Clear previous results
        > "$RESULTS_FILE"
        run_pending "--all"
        ;;
    --replot)
        generate_plot
        ;;
    watch|--watch)
        log "Watching $CHECKPOINT_DIR for new checkpoints (poll every ${POLL_INTERVAL}s)"
        log "Config: $NUM_GAMES games, $ALGO_DIFFICULTY difficulty, $NUM_WORKERS workers"
        log "Results: $RESULTS_FILE"
        log "Plot:    $PLOT_OUTPUT"
        log "Press Ctrl+C to stop."
        echo ""

        # First pass: evaluate anything pending
        run_pending ""

        # Watch loop
        while true; do
            sleep "$POLL_INTERVAL"
            run_pending ""
        done
        ;;
    *)
        echo "Usage: $0 [--once|--all|--replot|--watch]"
        echo ""
        echo "Modes:"
        echo "  --watch   (default) Poll for new checkpoints and evaluate them"
        echo "  --once    Evaluate untested checkpoints and exit"
        echo "  --all     Re-evaluate ALL checkpoints from scratch"
        echo "  --replot  Regenerate plot from existing results"
        exit 1
        ;;
esac
