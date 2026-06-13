#!/usr/bin/env python3
"""Plot training progress and ML model improvement against algorithm."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

# Output settings
DEFAULT_DPI = 150                # DPI for output image
FIGURE_SIZE = (14, 10)           # Figure size in inches (width, height)

# Plot styling
MOVING_AVG_WINDOW = 50           # Window size for loss moving average
TREND_LINE_DEGREE = 2            # Polynomial degree for trend line
SAMPLE_RATE_TARGET = 500         # Target number of loss points to display

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

def load_stats(stats_path: str = "models/training_stats.json"):
    """Load training statistics from JSON file."""
    with open(stats_path, 'r') as f:
        return json.load(f)


def load_jsonl_logs(logs_dir: str = "logs") -> list[dict]:
    """Load test results from training log JSONL files.
    
    Args:
        logs_dir: Directory containing train_*.jsonl files
        
    Returns:
        List of test_vs_algo entries from all log files
    """
    logs_path = Path(logs_dir)
    test_entries = []
    
    # Find all training log files
    log_files = sorted(logs_path.glob("train_*.jsonl"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get('type') == 'test_vs_algo':
                            test_entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Could not read {log_file}: {e}")
    
    return test_entries


def merge_test_history(stats: dict, log_entries: list[dict]) -> list[dict]:
    """Merge test history from stats file and log files, removing duplicates.
    
    Args:
        stats: Training statistics dictionary (with test_history)
        log_entries: Test entries from JSONL log files
        
    Returns:
        Combined and deduplicated test history, sorted by step
    """
    # Start with test_history from stats (skip malformed entries with no step)
    combined = {
        entry['step']: entry
        for entry in stats.get('test_history', [])
        if entry.get('step') is not None
    }
    
    # Add entries from log files (will overwrite duplicates)
    for entry in log_entries:
        step = entry.get('step')
        if step is not None:
            combined[step] = entry
    
    # Sort by step and return as list
    return [combined[step] for step in sorted(combined.keys())]


def plot_win_rate(stats: dict, test_history: list[dict], output_path: str = "models/training_progress.png", dpi: int = DEFAULT_DPI, show: bool = True):
    """Plot ML model win rate against algorithm over training steps.
    
    Args:
        stats: Training statistics dictionary
        test_history: Merged test history from stats and log files
        output_path: Path to save the output image
        dpi: DPI for output image
        show: Whether to display the plot
    """
    if not test_history:
        print("No test history found.")
        return

    # Extract data
    steps = [t['step'] for t in test_history]
    win_rates = [t['ml_win_rate'] * 100 for t in test_history]  # Convert to percentage
    p1_win_rates = [t['ml_as_p1_win_rate'] * 100 for t in test_history]
    p2_win_rates = [t['ml_as_p2_win_rate'] * 100 for t in test_history]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.suptitle('Filipino Micro ML Model Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall Win Rate
    ax1 = axes[0, 0]
    ax1.plot(steps, win_rates, 'b-o', linewidth=2, markersize=6, label='ML Win Rate')
    ax1.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% (Equal)')
    ax1.fill_between(steps, win_rates, alpha=0.3)
    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Win Rate (%)', fontsize=11)
    ax1.set_title('ML Model vs Algorithm - Overall Win Rate', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(steps) > 2:
        z = np.polyfit(steps, win_rates, TREND_LINE_DEGREE)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(steps), max(steps), 100)
        ax1.plot(x_smooth, np.clip(p(x_smooth), 0, 100), 'g--', alpha=0.7, label='Trend')
    
    ax1.legend(loc='upper left')
    
    # Plot 2: Win Rate by Player Position
    ax2 = axes[0, 1]
    ax2.plot(steps, p1_win_rates, 'g-s', linewidth=2, markersize=5, label='ML as Player 1 (White)')
    ax2.plot(steps, p2_win_rates, 'm-^', linewidth=2, markersize=5, label='ML as Player 2 (Black)')
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Training Steps', fontsize=11)
    ax2.set_ylabel('Win Rate (%)', fontsize=11)
    ax2.set_title('Win Rate by Player Position', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss History
    ax3 = axes[1, 0]
    loss_history = stats.get('loss_history', [])
    if loss_history:
        # Sample every N points to avoid overcrowding
        sample_rate = max(1, len(loss_history) // SAMPLE_RATE_TARGET)
        loss_steps = [l['step'] for l in loss_history[::sample_rate]]
        losses = [l['loss'] for l in loss_history[::sample_rate]]
        ax3.plot(loss_steps, losses, 'b-', linewidth=1, alpha=0.7)

        # Add moving average
        window = min(MOVING_AVG_WINDOW, len(losses) // 5) if len(losses) > 10 else 1
        if window > 1:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            ma_steps = loss_steps[window-1:]
            ax3.plot(ma_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            ax3.legend()
    ax3.set_xlabel('Training Steps', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Training Loss', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    total_games = sum(t['total_games'] for t in test_history)
    total_ml_wins = sum(t['ml_wins'] for t in test_history)
    total_algo_wins = sum(t['algo_wins'] for t in test_history)
    total_draws = sum(t['draws'] for t in test_history)
    overall_win_rate = (total_ml_wins / total_games * 100) if total_games > 0 else 0
    algo_win_rate = (total_algo_wins / total_games * 100) if total_games > 0 else 0
    
    best_win_rate = max(win_rates) if win_rates else 0
    best_step = steps[win_rates.index(best_win_rate)] if win_rates else 0
    latest_win_rate = win_rates[-1] if win_rates else 0
    
    # Format values safely
    # Use max step from test history if it's higher than stats
    total_steps_from_stats = stats.get('total_steps')
    max_step_from_tests = max(steps) if steps else 0
    total_steps = max(total_steps_from_stats or 0, max_step_from_tests)
    total_steps_str = f"{total_steps:,}" if total_steps > 0 else 'N/A'
    epochs = stats.get('epochs_completed')
    epochs_str = f"{epochs:,}" if isinstance(epochs, (int, float)) else 'N/A'
    best_loss = stats.get('best_loss')
    best_loss_str = f"{best_loss:.4f}" if isinstance(best_loss, (int, float)) else 'N/A'
    start_time = stats.get('start_time', 'N/A')
    start_time_str = start_time[:19] if isinstance(start_time, str) and len(start_time) >= 19 else str(start_time)
    end_time = stats.get('end_time', 'N/A')
    end_time_str = end_time[:19] if isinstance(end_time, str) and len(end_time) >= 19 else str(end_time)
    
    summary_text = f"""
    Training Summary
    ================
    
    Total Training Steps:     {total_steps_str}
    Epochs Completed:         {epochs_str}
    Best Loss:                {best_loss_str}
    
    Model vs Algorithm Tests (cumulative)
    =====================================
    
    Cumulative Test Games:    {total_games:,}
    ML Wins:                  {total_ml_wins:,} ({overall_win_rate:.1f}%)
    Algorithm Wins:           {total_algo_wins:,} ({algo_win_rate:.1f}%)
    Draws:                    {total_draws:,}
    
    Best Win Rate:            {best_win_rate:.1f}% (at step {best_step:,})
    Latest Win Rate:          {latest_win_rate:.1f}%
    
    Training Period
    ===============
    Start: {start_time_str}
    End:   {end_time_str}
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Training progress plot saved to: {output_path}")

    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def resolve_stats_path(project_dir: Path, explicit: str | None) -> Path:
    """Choose which training-stats JSON to plot.

    With --stats, honor the given path verbatim. Otherwise pick the most
    recently modified non-legacy stats file (training_stats.json,
    training_stats_local.json, training_stats_server.json) so the no-arg
    command tracks whichever run last wrote stats. Fall back to
    training_stats_legacy.json only when none of those exist, then to the
    conventional default path so the caller reports a consistent message.
    """
    if explicit:
        return Path(explicit)

    models_dir = project_dir / "models"
    candidates = [
        models_dir / "training_stats.json",
        models_dir / "training_stats_local.json",
        models_dir / "training_stats_server.json",
    ]
    existing = [p for p in candidates if p.exists()]
    if existing:
        return max(existing, key=lambda p: p.stat().st_mtime)

    legacy = models_dir / "training_stats_legacy.json"
    if legacy.exists():
        return legacy

    return models_dir / "training_stats.json"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Plot Filipino Micro training progress')
    parser.add_argument('--stats', type=str, default=None,
                       help='Path to training_stats.json file')
    parser.add_argument('--logs', type=str, default=None,
                       help='Path to logs directory containing train_*.jsonl files')
    parser.add_argument('--output', type=str, default=None,
                       help='Path for output PNG file')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI,
                       help=f'DPI for output image (default: {DEFAULT_DPI})')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot, only save to file')
    args = parser.parse_args()

    # Find project root (handle script being in root or scripts/)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir if (script_dir / "models").exists() else script_dir.parent

    stats_path = resolve_stats_path(project_dir, args.stats)

    logs_dir = Path(args.logs) if args.logs else project_dir / "logs"
    output_path = Path(args.output) if args.output else project_dir / "models" / "training_progress.png"

    if not stats_path.exists():
        print(f"Stats file not found: {stats_path}")
        sys.exit(1)

    print(f"Loading stats from: {stats_path}")
    stats = load_stats(str(stats_path))
    
    # Load additional test results from log files
    log_entries = []
    if logs_dir.exists():
        print(f"Loading logs from: {logs_dir}")
        log_entries = load_jsonl_logs(str(logs_dir))
        print(f"  Found {len(log_entries)} test entries in log files")
    
    # Merge test history
    test_history = merge_test_history(stats, log_entries)
    stats_count = len(stats.get('test_history', []))
    print(f"  Combined {stats_count} from stats and {len(log_entries)} from logs -> {len(test_history)} unique entries (deduplicated by step)")

    print("Generating training progress plot...")
    plot_win_rate(stats, test_history, str(output_path), dpi=args.dpi, show=not args.no_show)


if __name__ == '__main__':
    main()
