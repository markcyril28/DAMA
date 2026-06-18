#!/usr/bin/env python3
"""Plot training progress and ML model improvement against algorithm.

By default produces both:
- a self-contained interactive HTML (Plotly)
- a static matplotlib PNG snapshot

The HTML supports pan/zoom/hover and auto-refresh; the PNG is convenient for
quick image viewing or sharing. Pass --static to generate only the PNG.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


def _is_wsl() -> bool:
    """True when running under WSL (where the Linux browser fails to render)."""
    return "microsoft" in platform.uname().release.lower()


def open_file(output_path: str) -> None:
    """Open the saved file with the default app (browser for HTML, viewer for PNG).

    On WSL, hand the file to Windows (via the translated \\wsl$ path) instead of
    letting xdg-open launch a Linux GUI app through WSLg, which fails to render
    (libEGL/DRI3 errors) and shows a blank window.
    """
    if _is_wsl():
        try:
            win_path = subprocess.check_output(
                ["wslpath", "-w", str(Path(output_path).resolve())],
                text=True,
            ).strip()
            # WSL interop may not put the Windows tools on PATH, so try the bare
            # name first and fall back to the absolute path. explorer.exe opens
            # the file with its Windows default app (it returns a non-zero exit
            # even on success, so we don't gate on the return code).
            for launcher in ("explorer.exe", "/mnt/c/Windows/explorer.exe"):
                try:
                    subprocess.run([launcher, win_path], check=False)
                    return
                except FileNotFoundError:
                    continue
            raise FileNotFoundError("explorer.exe not found")
        except Exception as e:  # noqa: BLE001 - fall back to the default opener
            print(f"Warning: could not open via Windows browser ({e}); falling back.")
    webbrowser.open(f"file://{Path(output_path).resolve()}")


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

# Plot styling
MOVING_AVG_WINDOW = 50           # Window size for loss moving average
TREND_LINE_DEGREE = 2            # Polynomial degree for trend line
SAMPLE_RATE_TARGET = 500         # Target number of loss points to display

# Static (matplotlib) mode, used only with --static
DEFAULT_DPI = 150                # DPI for the static PNG
FIGURE_SIZE = (14, 10)           # Static figure size in inches (width, height)

# Plotly interactivity, applied to every figure in the interactive dashboard.
# dragmode='pan' is set per-figure in layout; the rest live here because they are
# config (not layout) and must be passed to each to_html() call explicitly.
INTERACTIVE_CONFIG = {
    'responsive': True,
    'scrollZoom': True,        # two-finger / wheel zoom; over a single axis = that axis only
    'displaylogo': False,
    'doubleClick': 'reset',    # double-click restores the default view
}

PAGE_TITLE = 'Filipino Micro ML Model Training Progress'
AUTO_REFRESH_MS = 15000

# --- Interactive dashboard HTML template ---------------------------------------
# Built with token replacement (not str.format/f-strings) because the CSS/JS below
# is full of literal { } braces. Tokens are %%UPPER%% so they can't collide with
# the embedded plotly.js or the figure HTML.
_DASHBOARD_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
       padding: 14px 18px; color: #222; background: #fafafa; }
h1 { text-align: center; font-weight: 600; font-size: 22px; margin: 4px 0 16px; }
.page-meta { text-align: center; margin: -8px 0 16px; color: #666; font-size: 12px; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.card { border: 1px solid #e3e3e3; border-radius: 10px; padding: 8px 10px; background: #fff;
        display: flex; flex-direction: column; height: 440px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.card-head { display: flex; justify-content: space-between; align-items: center;
             margin-bottom: 6px; }
.card-head .name { font-weight: 600; font-size: 14px; }
.card-head button { cursor: pointer; border: 1px solid #ccc; background: #f7f7f7;
                    border-radius: 6px; padding: 3px 9px; font-size: 12px; }
.card-head button:hover { background: #ececec; }
.plotwrap { position: relative; flex: 1 1 auto; min-height: 0; }
.plotwrap > div { position: absolute; inset: 0; }
.plotwrap .plotly-graph-div { width: 100% !important; height: 100% !important; }
.summary { font-family: ui-monospace, Consolas, monospace; white-space: pre-wrap;
           font-size: 13px; line-height: 1.5; background: wheat; border-radius: 8px;
           padding: 14px 16px; overflow: auto; flex: 1 1 auto; min-height: 0; }
/* Fullscreen one card: fill the screen and let the plot grow to match. */
.card:fullscreen { height: 100vh; width: 100vw; padding: 16px 20px; background: #fff; }
.card:-webkit-full-screen { height: 100vh; width: 100vw; padding: 16px 20px; background: #fff; }
@media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
"""

# Resize every Plotly graph whenever fullscreen toggles or the window resizes, so a
# graph promoted to fullscreen actually grows to fill the screen (Plotly does not
# re-layout on container size changes on its own for fixed-size renders).
_DASHBOARD_JS = """
var AUTO_REFRESH_MS = %%REFRESH_MS%%;
var GENERATED_AT = "%%GENERATED_AT%%";
var SCROLL_KEY = "training-progress-scroll-y";

function toggleFs(id) {
  var el = document.getElementById(id);
  if (document.fullscreenElement || document.webkitFullscreenElement) {
    (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  } else if (el.requestFullscreen) {
    el.requestFullscreen();
  } else if (el.webkitRequestFullscreen) {
    el.webkitRequestFullscreen();
  } else {
    alert('Fullscreen is not supported by this browser.');
  }
}
function resizeAllPlots() {
  if (!window.Plotly) return;
  document.querySelectorAll('.plotly-graph-div').forEach(function (gd) {
    try { Plotly.Plots.resize(gd); } catch (e) {}
  });
}
document.addEventListener('fullscreenchange', resizeAllPlots);
document.addEventListener('webkitfullscreenchange', resizeAllPlots);
window.addEventListener('resize', resizeAllPlots);

window.addEventListener('load', function () {
  try {
    var scrollY = sessionStorage.getItem(SCROLL_KEY);
    if (scrollY !== null) {
      window.scrollTo(0, parseInt(scrollY, 10) || 0);
    }
  } catch (e) {}
});

window.addEventListener('beforeunload', function () {
  try {
    sessionStorage.setItem(SCROLL_KEY, String(window.scrollY));
  } catch (e) {}
});

function scheduleAutoRefresh() {
  if (!AUTO_REFRESH_MS || AUTO_REFRESH_MS <= 0) return;

  if (window.location.protocol === 'file:') {
    window.setTimeout(function () {
      window.location.reload();
    }, AUTO_REFRESH_MS);
    return;
  }

  window.setInterval(async function () {
    try {
      var response = await fetch(window.location.href, { cache: 'no-store' });
      var html = await response.text();
      var match = html.match(/<meta name="report-generated-at" content="([^"]+)"/i);
      if (match && match[1] !== GENERATED_AT) {
        window.location.reload();
      }
    } catch (e) {}
  }, AUTO_REFRESH_MS);
}

scheduleAutoRefresh();
"""

_CARD_TEMPLATE = """  <div class="card" id="card-%%KEY%%">
    <div class="card-head"><span class="name">%%NAME%%</span>
      <button type="button" onclick="toggleFs('card-%%KEY%%')">&#9974; Fullscreen</button></div>
    <div class="plotwrap">%%PLOT%%</div>
  </div>"""

_SUMMARY_CARD_TEMPLATE = """  <div class="card" id="card-summary">
    <div class="card-head"><span class="name">Summary</span>
      <button type="button" onclick="toggleFs('card-summary')">&#9974; Fullscreen</button></div>
    <div class="summary">%%SUMMARY%%</div>
  </div>"""

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="report-generated-at" content="%%GENERATED_AT%%"/>
<title>%%TITLE%%</title>
<script type="text/javascript">%%PLOTLYJS%%</script>
<style>%%CSS%%</style>
</head>
<body>
<h1>%%TITLE%%</h1>
<div class="page-meta">Updated %%GENERATED_AT%% • Auto-refresh every %%REFRESH_SECONDS%%s</div>
<div class="grid">
%%CARDS%%
</div>
<script>%%JS%%</script>
</body>
</html>
"""

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


def _build_summary_text(stats: dict, test_history: list[dict], steps, win_rates) -> str:
    """Build the monospace summary block (uses <br> for Plotly line breaks)."""
    total_games = sum(t['total_games'] for t in test_history)
    total_ml_wins = sum(t['ml_wins'] for t in test_history)
    total_algo_wins = sum(t['algo_wins'] for t in test_history)
    total_draws = sum(t['draws'] for t in test_history)
    overall_win_rate = (total_ml_wins / total_games * 100) if total_games > 0 else 0
    algo_win_rate = (total_algo_wins / total_games * 100) if total_games > 0 else 0

    best_win_rate = max(win_rates) if win_rates else 0
    best_step = steps[win_rates.index(best_win_rate)] if win_rates else 0
    latest_win_rate = win_rates[-1] if win_rates else 0

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

    lines = [
        "<b>Training Summary</b>",
        "================",
        f"Total Training Steps:  {total_steps_str}",
        f"Epochs Completed:      {epochs_str}",
        f"Best Loss:             {best_loss_str}",
        "",
        "<b>Model vs Algorithm (cumulative)</b>",
        "================================",
        f"Cumulative Test Games: {total_games:,}",
        f"ML Wins:               {total_ml_wins:,} ({overall_win_rate:.1f}%)",
        f"Algorithm Wins:        {total_algo_wins:,} ({algo_win_rate:.1f}%)",
        f"Draws:                 {total_draws:,}",
        "",
        f"Best Win Rate:         {best_win_rate:.1f}% (at step {best_step:,})",
        f"Latest Win Rate:       {latest_win_rate:.1f}%",
        "",
        "<b>Training Period</b>",
        "===============",
        f"Start: {start_time_str}",
        f"End:   {end_time_str}",
    ]
    return "<br>".join(lines)


def _apply_interactive_layout(fig: go.Figure, y_title: str, y_range=None) -> go.Figure:
    """Shared layout for every dashboard figure: pan-on-drag, compact margins,
    a horizontal legend in the top margin, and axis titles."""
    fig.update_layout(
        template='plotly_white', hovermode='closest',
        # dragmode='pan' makes click-drag move (pan) the figure instead of
        # box-zooming. With scrollZoom (in INTERACTIVE_CONFIG) the figure pans
        # freely and zooms: scroll over the plot zooms both axes, scroll over a
        # single axis zooms just that axis (independent x/y zoom — handy for the
        # Training Loss figure, whose early spikes dominate the y scale).
        dragmode='pan',
        margin=dict(l=58, r=18, t=38, b=46),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0),
    )
    fig.update_xaxes(title_text='Training Steps')
    fig.update_yaxes(title_text=y_title)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def _fig_overall(steps, win_rates) -> go.Figure:
    """Overall ML win-rate vs the algorithm, with a polynomial trend line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=win_rates, mode='lines+markers', name='ML Win Rate',
        line=dict(color='royalblue', width=2), marker=dict(size=7),
        fill='tozeroy', fillcolor='rgba(65,105,225,0.2)',
        hovertemplate='Step %{x:,}<br>Win Rate %{y:.1f}%<extra></extra>',
    ))
    if len(steps) > 2:
        z = np.polyfit(steps, win_rates, TREND_LINE_DEGREE)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(steps), max(steps), 100)
        fig.add_trace(go.Scatter(
            x=x_smooth, y=np.clip(p(x_smooth), 0, 100), mode='lines', name='Trend',
            line=dict(color='green', width=2, dash='dash'),
            hovertemplate='Step %{x:,.0f}<br>Trend %{y:.1f}%<extra></extra>',
        ))
    fig.add_hline(y=50, line_dash='dash', line_color='red', opacity=0.7,
                  annotation_text='50% (Equal)', annotation_position='top left')
    return _apply_interactive_layout(fig, 'Win Rate (%)', y_range=[0, 100])


def _fig_position(steps, p1_win_rates, p2_win_rates) -> go.Figure:
    """Win rate split by which side the ML model played (Player 1 vs Player 2)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=p1_win_rates, mode='lines+markers', name='ML as Player 1 (White)',
        line=dict(color='green', width=2), marker=dict(size=6, symbol='square'),
        hovertemplate='Step %{x:,}<br>P1 Win Rate %{y:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=p2_win_rates, mode='lines+markers', name='ML as Player 2 (Black)',
        line=dict(color='magenta', width=2), marker=dict(size=6, symbol='triangle-up'),
        hovertemplate='Step %{x:,}<br>P2 Win Rate %{y:.1f}%<extra></extra>',
    ))
    fig.add_hline(y=50, line_dash='dash', line_color='red', opacity=0.7)
    return _apply_interactive_layout(fig, 'Win Rate (%)', y_range=[0, 100])


def _fig_loss(stats: dict) -> go.Figure:
    """Training loss (downsampled) plus a moving average."""
    fig = go.Figure()
    loss_history = stats.get('loss_history', [])
    if loss_history:
        # Sample every N points to avoid overcrowding
        sample_rate = max(1, len(loss_history) // SAMPLE_RATE_TARGET)
        loss_steps = [l['step'] for l in loss_history[::sample_rate]]
        losses = [l['loss'] for l in loss_history[::sample_rate]]
        fig.add_trace(go.Scatter(
            x=loss_steps, y=losses, mode='lines', name='Loss',
            line=dict(color='royalblue', width=1), opacity=0.7,
            hovertemplate='Step %{x:,}<br>Loss %{y:.4f}<extra></extra>',
        ))

        # Moving average
        window = min(MOVING_AVG_WINDOW, len(losses) // 5) if len(losses) > 10 else 1
        if window > 1:
            moving_avg = np.convolve(losses, np.ones(window) / window, mode='valid')
            ma_steps = loss_steps[window - 1:]
            fig.add_trace(go.Scatter(
                x=ma_steps, y=moving_avg, mode='lines', name=f'Moving Avg ({window})',
                line=dict(color='red', width=2),
                hovertemplate='Step %{x:,}<br>Avg Loss %{y:.4f}<extra></extra>',
            ))
    return _apply_interactive_layout(fig, 'Loss')


def _atomic_write_text(path: Path, content: str) -> None:
    """Write a text file atomically so the browser never reads partial HTML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', dir=path.parent, suffix='.tmp',
                delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
            tmp.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise


def _render_dashboard(panels: list, summary_html: str, title: str = PAGE_TITLE) -> str:
    """Assemble the figures + summary into one self-contained HTML page.

    plotly.js is embedded exactly once in the <head>; every figure is emitted
    with include_plotlyjs=False so it reuses that single copy. INTERACTIVE_CONFIG
    is passed to each to_html() because config does not travel inside the figure
    layout. Token replacement (not str.format) is used so the CSS/JS braces and
    the embedded library are left untouched.
    """
    cards = []
    for key, name, fig in panels:
        plot_div = fig.to_html(full_html=False, include_plotlyjs=False,
                               div_id=f'gd-{key}', config=INTERACTIVE_CONFIG)
        cards.append(_CARD_TEMPLATE
                     .replace('%%KEY%%', key)
                     .replace('%%NAME%%', name)
                     .replace('%%PLOT%%', plot_div))
    cards.append(_SUMMARY_CARD_TEMPLATE.replace('%%SUMMARY%%', summary_html))
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return (_PAGE_TEMPLATE
            .replace('%%TITLE%%', title)
            .replace('%%CSS%%', _DASHBOARD_CSS)
            .replace('%%JS%%', _DASHBOARD_JS
                     .replace('%%REFRESH_MS%%', str(AUTO_REFRESH_MS))
                     .replace('%%GENERATED_AT%%', generated_at))
            .replace('%%CARDS%%', '\n'.join(cards))
            .replace('%%GENERATED_AT%%', generated_at)
            .replace('%%REFRESH_SECONDS%%', str(AUTO_REFRESH_MS // 1000))
            .replace('%%PLOTLYJS%%', get_plotlyjs()))  # last: library may be large


def plot_win_rate(stats: dict, test_history: list[dict], output_path: str = "models/training_progress.html", show: bool = True):
    """Render the interactive Plotly dashboard (default mode).

    Three independent figures (overall win rate, win rate by player position,
    training loss) plus a text Summary card, laid out in a 2x2 grid. Each figure
    pans on click-drag, zooms on scroll (per-axis when scrolling over an axis),
    and has its own Fullscreen button. See plot_win_rate_static() for the older
    static matplotlib PNG.

    Args:
        stats: Training statistics dictionary
        test_history: Merged test history from stats and log files
        output_path: Path to save the output HTML file
        show: Whether to open the plot in a browser

    Returns:
        The list of plotly Figures (overall, position, loss), or None when there
        is no test history.
    """
    # Extract data
    steps = [t['step'] for t in test_history]
    win_rates = [t['ml_win_rate'] * 100 for t in test_history]  # Convert to percentage
    p1_win_rates = [t['ml_as_p1_win_rate'] * 100 for t in test_history]
    p2_win_rates = [t['ml_as_p2_win_rate'] * 100 for t in test_history]

    # (div-id key, card title, figure) — keys must be CSS/HTML-id safe.
    panels = [
        ('overall', 'ML Model vs Algorithm - Overall Win Rate', _fig_overall(steps, win_rates)),
        ('position', 'Win Rate by Player Position', _fig_position(steps, p1_win_rates, p2_win_rates)),
        ('loss', 'Training Loss', _fig_loss(stats)),
    ]
    summary_html = _build_summary_text(stats, test_history, steps, win_rates)

    # Self-contained page (plotly.js embedded inline), so it renders offline and
    # when the CDN is blocked (e.g. opening a WSL2-generated file in a Windows
    # browser). open_file() handles opening it on WSL vs. native.
    _atomic_write_text(Path(output_path), _render_dashboard(panels, summary_html))
    print(f"Interactive training progress plot saved to: {output_path}")
    if show:
        open_file(output_path)
    return [fig for _, _, fig in panels]


def write_progress_report(stats_path: str | Path,
                          logs_dir: str | Path | None = None,
                          output_path: str | Path = "models/training_progress.html") -> Path:
    """Regenerate the interactive HTML report from the latest stats/log files."""
    stats_path = Path(stats_path)
    stats = load_stats(str(stats_path))

    logs_path = Path(logs_dir) if logs_dir is not None else stats_path.parent.parent / "logs"
    log_entries = load_jsonl_logs(str(logs_path)) if logs_path.exists() else []
    test_history = merge_test_history(stats, log_entries)

    plot_win_rate(stats, test_history, str(output_path), show=False)
    return Path(output_path)


def _companion_png_path(output_path: str | Path) -> Path:
    """Return the PNG path paired with an interactive HTML output path."""
    output_path = Path(output_path)
    return output_path.with_suffix('.png') if output_path.suffix else output_path.with_name(output_path.name + '.png')


def write_progress_outputs(stats_path: str | Path,
                           logs_dir: str | Path | None = None,
                           html_output_path: str | Path = "models/training_progress.html",
                           image_output_path: str | Path | None = None,
                           dpi: int = DEFAULT_DPI) -> dict[str, Path]:
    """Regenerate both the interactive HTML report and the static PNG snapshot."""
    stats_path = Path(stats_path)
    stats = load_stats(str(stats_path))

    logs_path = Path(logs_dir) if logs_dir is not None else stats_path.parent.parent / "logs"
    log_entries = load_jsonl_logs(str(logs_path)) if logs_path.exists() else []
    test_history = merge_test_history(stats, log_entries)

    html_output = Path(html_output_path)
    image_output = Path(image_output_path) if image_output_path is not None else _companion_png_path(html_output)

    plot_win_rate(stats, test_history, str(html_output), show=False)
    plot_win_rate_static(stats, test_history, str(image_output), dpi=dpi, show=False)
    return {'html': html_output, 'image': image_output}


def plot_win_rate_static(stats: dict, test_history: list[dict],
                         output_path: str = "models/training_progress.png",
                         dpi: int = DEFAULT_DPI, show: bool = True):
    """Older, static matplotlib rendering: a single 2x2 PNG (--static mode).

    Kept as a lightweight, dependency-light alternative to the interactive HTML
    dashboard. matplotlib is imported lazily (so the interactive path doesn't
    require it) with the non-interactive 'Agg' backend, since a GUI backend's
    window fails to render under WSLg. The saved PNG is then opened with the
    platform default image viewer via open_file().

    Args:
        stats: Training statistics dictionary
        test_history: Merged test history from stats and log files
        output_path: Path to save the output PNG
        dpi: DPI for the output image
        show: Whether to open the saved PNG in the default viewer
    """
    import matplotlib
    matplotlib.use('Agg')  # headless; we open the saved PNG ourselves
    import matplotlib.pyplot as plt

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
            moving_avg = np.convolve(losses, np.ones(window) / window, mode='valid')
            ma_steps = loss_steps[window - 1:]
            ax3.plot(ma_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            ax3.legend()
    ax3.set_xlabel('Training Steps', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Training Loss', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    # Reuse the shared summary builder, converting its HTML line breaks/bold tags
    # (meant for the interactive page) back to plain text for matplotlib.
    summary_text = (_build_summary_text(stats, test_history, steps, win_rates)
                    .replace('<br>', '\n').replace('<b>', '').replace('</b>', ''))
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Static training progress plot saved to: {output_path}")
    if show:
        open_file(output_path)


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
                       help='Output path (default: models/training_progress.html; in default mode a companion .png is also written)')
    parser.add_argument('--static', action='store_true',
                       help='Use the older static matplotlib PNG instead of the interactive HTML dashboard')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI,
                       help=f'DPI for the static PNG (only with --static; default: {DEFAULT_DPI})')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not open the plot, only save the output file')
    args = parser.parse_args()

    # Find project root (handle script being in root or scripts/)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir if (script_dir / "models").exists() else script_dir.parent

    stats_path = resolve_stats_path(project_dir, args.stats)

    logs_dir = Path(args.logs) if args.logs else project_dir / "logs"
    default_name = "training_progress.png" if args.static else "training_progress.html"
    output_path = Path(args.output) if args.output else project_dir / "models" / default_name

    # Fail fast on contradictory args (clean message instead of a deep traceback):
    # a non-positive DPI, or an --output extension that doesn't match the mode
    # (interactive writes HTML; --static writes a matplotlib raster/vector image).
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    suffix = output_path.suffix.lower()
    if args.static:
        if suffix not in {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.webp', '.tif', '.tiff'}:
            parser.error(f"--static writes an image; matplotlib cannot write '{suffix}' "
                         f"(use e.g. .png, or drop --static for interactive HTML)")
    elif suffix not in {'.html', '.htm', ''}:
        parser.error(f"interactive mode writes HTML, so '{suffix}' would hold HTML bytes "
                     f"(use a .html path, or pass --static for an image)")

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

    if args.static:
        print("Generating static training progress plot (matplotlib PNG)...")
        plot_win_rate_static(stats, test_history, str(output_path),
                             dpi=args.dpi, show=not args.no_show)
    else:
        image_output = _companion_png_path(output_path)
        print("Generating interactive training progress report + static PNG snapshot...")
        outputs = write_progress_outputs(
            stats_path=stats_path,
            logs_dir=logs_dir,
            html_output_path=output_path,
            image_output_path=image_output,
            dpi=args.dpi,
        )
        if not args.no_show:
            open_file(str(outputs['html']))


if __name__ == '__main__':
    main()
