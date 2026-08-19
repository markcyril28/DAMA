#!/usr/bin/env python3
"""Plot training progress and ML model improvement against algorithm.

By default produces both:
- a self-contained interactive HTML (Plotly)
- a static matplotlib PNG snapshot

The HTML supports pan/zoom/hover and auto-refresh; the PNG is convenient for
quick image viewing or sharing. Pass --static to generate only the PNG.
"""

import argparse
from bisect import bisect_left
import json
import math
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
THEME_STORAGE_KEY = 'training-progress-theme'

# --- Interactive dashboard HTML template ---------------------------------------
# Built with token replacement (not str.format/f-strings) because the CSS/JS below
# is full of literal { } braces. Tokens are %%UPPER%% so they can't collide with
# the embedded plotly.js or the figure HTML.
_DASHBOARD_CSS = """
:root {
  color-scheme: light;
  --page-bg: #fafafa;
  --page-fg: #222222;
  --muted-fg: #666666;
  --card-bg: #ffffff;
  --card-border: #e3e3e3;
  --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
  --button-bg: #f7f7f7;
  --button-fg: #222222;
  --button-border: #cccccc;
  --button-hover-bg: #ececec;
  --button-active-bg: #e8f5ee;
  --button-active-fg: #0f5f3a;
  --button-active-border: #82c99d;
  --summary-bg: wheat;
  --summary-fg: #222222;
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --page-bg: #0f1117;
  --page-fg: #e5e7eb;
  --muted-fg: #a1a1aa;
  --card-bg: #161b22;
  --card-border: #30363d;
  --card-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  --button-bg: #222938;
  --button-fg: #e5e7eb;
  --button-border: #3b4657;
  --button-hover-bg: #2d3647;
  --button-active-bg: #123225;
  --button-active-fg: #9ce2b8;
  --button-active-border: #2f8f58;
  --summary-bg: #1e2430;
  --summary-fg: #e5e7eb;
}

* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  margin: 0;
  padding: 14px 18px 18px;
  color: var(--page-fg);
  background: var(--page-bg);
  transition: background-color 0.18s ease, color 0.18s ease;
}
.toolbar { display: flex; justify-content: flex-end; align-items: center; margin-bottom: 8px; }
.theme-toggle {
  cursor: pointer;
  border: 1px solid var(--button-border);
  background: var(--button-bg);
  color: var(--button-fg);
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  box-shadow: var(--card-shadow);
}
.theme-toggle:hover { background: var(--button-hover-bg); }
.theme-toggle:focus-visible { outline: 2px solid #4f8cff; outline-offset: 2px; }
h1 { text-align: center; font-weight: 600; font-size: 22px; margin: 4px 0 16px; }
.page-meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 8px;
  text-align: center;
  margin: -8px 0 16px;
  color: var(--muted-fg);
  font-size: 12px;
}
.page-actions {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.page-action {
  cursor: pointer;
  border: 1px solid var(--button-border);
  background: var(--button-bg);
  color: var(--button-fg);
  border-radius: 6px;
  padding: 4px 9px;
  font-size: 12px;
  font-weight: 600;
}
.page-action:hover { background: var(--button-hover-bg); }
.page-action:focus-visible { outline: 2px solid #4f8cff; outline-offset: 2px; }
.page-action[aria-pressed="true"] {
  border-color: var(--button-active-border);
  background: var(--button-active-bg);
  color: var(--button-active-fg);
}
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.card {
  border: 1px solid var(--card-border);
  border-radius: 10px;
  padding: 8px 10px;
  background: var(--card-bg);
  display: flex;
  flex-direction: column;
  height: 440px;
  box-shadow: var(--card-shadow);
  transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  gap: 12px;
}
.card-head .name { font-weight: 600; font-size: 14px; }
.card-head button {
  cursor: pointer;
  border: 1px solid var(--button-border);
  background: var(--button-bg);
  color: var(--button-fg);
  border-radius: 6px;
  padding: 3px 9px;
  font-size: 12px;
}
.card-head button:hover { background: var(--button-hover-bg); }
.plotwrap { position: relative; flex: 1 1 auto; min-height: 0; }
.plotwrap > div { position: absolute; inset: 0; }
.plotwrap .plotly-graph-div { width: 100% !important; height: 100% !important; }
.summary {
  font-family: ui-monospace, Consolas, monospace;
  white-space: pre-wrap;
  font-size: 13px;
  line-height: 1.5;
  background: var(--summary-bg);
  color: var(--summary-fg);
  border-radius: 8px;
  padding: 14px 16px;
  overflow: auto;
  flex: 1 1 auto;
  min-height: 0;
}
/* Fullscreen one card: fill the screen and let the plot grow to match. */
.card:fullscreen { height: 100vh; width: 100vw; padding: 16px 20px; background: var(--card-bg); }
.card:-webkit-full-screen { height: 100vh; width: 100vw; padding: 16px 20px; background: var(--card-bg); }
.modebar { background: transparent !important; }
:root[data-theme="dark"] .modebar { background: rgba(15, 17, 23, 0.85) !important; }
:root[data-theme="dark"] .updatemenu-item-rect {
  fill: var(--button-bg) !important;
  stroke: var(--button-border) !important;
}
:root[data-theme="dark"] .updatemenu-item-text { fill: var(--button-fg) !important; }
:root[data-theme="dark"] .updatemenu-button[data-active="true"] .updatemenu-item-rect {
  fill: var(--button-active-bg) !important;
  stroke: var(--button-active-border) !important;
}
:root[data-theme="dark"] .updatemenu-button[data-active="true"] .updatemenu-item-text {
  fill: var(--button-active-fg) !important;
}
@media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
"""

# Resize every Plotly graph whenever fullscreen toggles or the window resizes, so a
# graph promoted to fullscreen actually grows to fill the screen (Plotly does not
# re-layout on container size changes on its own for fixed-size renders).
_DASHBOARD_JS = """
var AUTO_REFRESH_MS = %%REFRESH_MS%%;
var GENERATED_AT = "%%GENERATED_AT%%";
var SCROLL_KEY = "training-progress-scroll-y";
var THEME_STORAGE_KEY = "%%THEME_STORAGE_KEY%%";
var CURRENT_SOURCE_FILE = "%%CURRENT_SOURCE_FILE%%";
var autoRefreshTimer = null;
var autoRefreshEnabled = true;
var allStatsVisible = true;

function getPreferredTheme() {
  try {
    var saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'dark' || saved === 'light') return saved;
  } catch (e) {}
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark';
  }
  return 'light';
}

function getPlotLayout(theme) {
  var isDark = theme === 'dark';
  return {
    paper_bgcolor: isDark ? '#161b22' : '#ffffff',
    plot_bgcolor: isDark ? '#161b22' : '#ffffff',
    font: { color: isDark ? '#e5e7eb' : '#222222' },
    legend: { font: { color: isDark ? '#e5e7eb' : '#222222' } },
    xaxis: {
      gridcolor: isDark ? 'rgba(148, 163, 184, 0.18)' : 'rgba(0, 0, 0, 0.12)',
      zerolinecolor: isDark ? 'rgba(148, 163, 184, 0.18)' : 'rgba(0, 0, 0, 0.12)',
      linecolor: isDark ? 'rgba(148, 163, 184, 0.38)' : 'rgba(0, 0, 0, 0.3)',
      tickfont: { color: isDark ? '#e5e7eb' : '#222222' },
      title: { font: { color: isDark ? '#e5e7eb' : '#222222' } }
    },
    yaxis: {
      gridcolor: isDark ? 'rgba(148, 163, 184, 0.18)' : 'rgba(0, 0, 0, 0.12)',
      zerolinecolor: isDark ? 'rgba(148, 163, 184, 0.18)' : 'rgba(0, 0, 0, 0.12)',
      linecolor: isDark ? 'rgba(148, 163, 184, 0.38)' : 'rgba(0, 0, 0, 0.3)',
      tickfont: { color: isDark ? '#e5e7eb' : '#222222' },
      title: { font: { color: isDark ? '#e5e7eb' : '#222222' } }
    }
  };
}

function updateThemeButton(theme) {
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;
  var isDark = theme === 'dark';
  btn.textContent = isDark ? 'Light mode' : 'Dark mode';
  btn.setAttribute('aria-pressed', isDark ? 'true' : 'false');
  btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
}

function updatePlotMenuTheme(gd) {
  var menuLayouts = gd._fullLayout && gd._fullLayout.updatemenus || [];
  gd.querySelectorAll('.updatemenu-container').forEach(function (menuEl, menuIndex) {
    var activeIndex = menuLayouts[menuIndex] ? menuLayouts[menuIndex].active : -1;
    menuEl.querySelectorAll('.updatemenu-button').forEach(function (buttonEl, buttonIndex) {
      buttonEl.setAttribute('data-active', buttonIndex === activeIndex ? 'true' : 'false');
    });
  });
}

function updatePlotsForTheme(theme) {
  if (!window.Plotly) return;
  var layout = getPlotLayout(theme);
  document.querySelectorAll('.plotly-graph-div').forEach(function (gd) {
    try {
      Promise.resolve(Plotly.relayout(gd, layout)).then(function () {
        updatePlotMenuTheme(gd);
      });
      if (!gd.__menuThemeListenerAttached && typeof gd.on === 'function') {
        gd.on('plotly_buttonclicked', function () { updatePlotMenuTheme(gd); });
        gd.__menuThemeListenerAttached = true;
      }
    } catch (e) {}
  });
}

function applyTheme(theme, persist) {
  var next = theme === 'dark' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  document.documentElement.style.colorScheme = next;
  updateThemeButton(next);
  updatePlotsForTheme(next);
  if (persist) {
    try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch (e) {}
  }
}

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

window.addEventListener('DOMContentLoaded', function () {
  var themeBtn = document.getElementById('theme-toggle');
  var refreshBtn = document.getElementById('refresh-now');
  var autoRefreshBtn = document.getElementById('auto-refresh-toggle');
  var allStatsBtn = document.getElementById('all-stats-toggle');

  if (themeBtn) themeBtn.addEventListener('click', function () {
    var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next, true);
  });

  if (refreshBtn) refreshBtn.addEventListener('click', refreshPage);
  if (autoRefreshBtn) autoRefreshBtn.addEventListener('click', toggleAutoRefresh);
  if (allStatsBtn) allStatsBtn.addEventListener('click', toggleAllStats);
  updateAutoRefreshButton();
  updateAllStatsButton();
  applyAllStatsVisibility();
});

window.addEventListener('load', function () {
  applyTheme(getPreferredTheme(), false);
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

function refreshPage() {
  window.location.reload();
}

async function checkForUpdatedReport() {
  if (window.location.protocol === 'file:') {
    refreshPage();
    return;
  }

  try {
    var response = await fetch(window.location.href, { cache: 'no-store' });
    var html = await response.text();
    var match = html.match(/<meta name="report-generated-at" content="([^"]+)"/i);
    if (match && match[1] !== GENERATED_AT) {
      refreshPage();
    }
  } catch (e) {}
}

function updateAutoRefreshButton() {
  var btn = document.getElementById('auto-refresh-toggle');
  if (!btn) return;
  btn.textContent = autoRefreshEnabled ? 'Auto refresh: On' : 'Auto refresh: Off';
  btn.setAttribute('aria-pressed', autoRefreshEnabled ? 'true' : 'false');
  btn.setAttribute('aria-label', autoRefreshEnabled ? 'Turn auto refresh off' : 'Turn auto refresh on');
}

function startAutoRefresh() {
  if (!AUTO_REFRESH_MS || AUTO_REFRESH_MS <= 0 || autoRefreshTimer) return;
  autoRefreshEnabled = true;
  updateAutoRefreshButton();
  autoRefreshTimer = window.setInterval(checkForUpdatedReport, AUTO_REFRESH_MS);
}

function stopAutoRefresh() {
  if (autoRefreshTimer) {
    window.clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
  autoRefreshEnabled = false;
  updateAutoRefreshButton();
}

function toggleAutoRefresh() {
  if (autoRefreshTimer) {
    stopAutoRefresh();
  } else {
    startAutoRefresh();
  }
}

function updateAllStatsButton() {
  var btn = document.getElementById('all-stats-toggle');
  if (!btn) return;
  btn.textContent = allStatsVisible ? 'All stats: On' : 'All stats: Off';
  btn.setAttribute('aria-pressed', allStatsVisible ? 'true' : 'false');
  btn.setAttribute('aria-label', allStatsVisible ? 'Show current run only' : 'Show all training stats');
}

function traceIsPreviousRun(trace) {
  if (!trace || !trace.meta || !trace.meta.source_file || !CURRENT_SOURCE_FILE) return false;
  return trace.meta.source_file !== CURRENT_SOURCE_FILE;
}

function applyAllStatsVisibility() {
  if (!window.Plotly) return;
  document.querySelectorAll('.plotly-graph-div').forEach(function (gd) {
    var updates = {};
    var indices = [];
    (gd.data || []).forEach(function (trace, index) {
      if (traceIsPreviousRun(trace)) {
        updates.visible = updates.visible || [];
        updates.visible.push(allStatsVisible ? true : 'legendonly');
        indices.push(index);
      }
    });
    if (indices.length) {
      try { Plotly.restyle(gd, updates, indices); } catch (e) {}
    }
  });
}

function toggleAllStats() {
  allStatsVisible = !allStatsVisible;
  updateAllStatsButton();
  applyAllStatsVisibility();
}

startAutoRefresh();
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

_THEME_INIT_SCRIPT = """
(function () {
  var key = "%%THEME_STORAGE_KEY%%";
  var theme = null;
  try {
    theme = localStorage.getItem(key);
  } catch (e) {}
  if (theme !== 'dark' && theme !== 'light') {
    theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.setAttribute('data-theme', theme);
  document.documentElement.style.colorScheme = theme;
})();
"""

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="report-generated-at" content="%%GENERATED_AT%%"/>
<title>%%TITLE%%</title>
<script type="text/javascript">%%THEME_INIT%%</script>
<script type="text/javascript">%%PLOTLYJS%%</script>
<style>%%CSS%%</style>
</head>
<body>
<div class="toolbar">
  <button id="theme-toggle" class="theme-toggle" type="button" aria-pressed="false">Dark mode</button>
</div>
<h1>%%TITLE%%</h1>
<div class="page-meta">
  <span>Updated %%GENERATED_AT%%</span>
  <span class="page-actions" aria-label="Refresh controls">
    <button id="all-stats-toggle" class="page-action" type="button" aria-pressed="true">All stats: On</button>
    <button id="refresh-now" class="page-action" type="button">Refresh</button>
    <button id="auto-refresh-toggle" class="page-action" type="button" aria-pressed="true">Auto refresh: On</button>
  </span>
</div>
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


def _stats_source_label(path: Path) -> str:
    """Human-readable label for a training stats file."""
    name = path.stem
    if name == 'training_stats':
        return 'Default'
    if name.startswith('training_stats_'):
        return name.removeprefix('training_stats_').replace('_', ' ').title()
    return name.replace('_', ' ').title()


def _candidate_stats_paths(models_dir: Path) -> list[Path]:
    """Return non-legacy stats files that should appear in the dashboard."""
    preferred = [
        models_dir / "training_stats.json",
        models_dir / "training_stats_local.json",
        models_dir / "training_stats_server.json",
        models_dir / "training_stats_policy_distillation.json",
    ]
    extras = sorted(
        p for p in models_dir.glob("training_stats*.json")
        if p.name != "training_stats_legacy.json" and p not in preferred
    )
    return [p for p in preferred if p.exists()] + extras


def _copy_history_with_run(entries, run_label: str, source_file: str) -> list:
    """Copy history entries and tag them with the originating run."""
    copied = []
    if not isinstance(entries, list):
        return copied
    for entry in entries:
        if isinstance(entry, dict):
            tagged = dict(entry)
            tagged.setdefault('run', run_label)
            tagged.setdefault('source_file', source_file)
            copied.append(tagged)
        else:
            copied.append(entry)
    return copied


def _aggregate_stats(stats_by_path: list[tuple[Path, dict]]) -> dict:
    """Merge multiple training stats files while preserving run labels."""
    if not stats_by_path:
        return {}
    if len(stats_by_path) == 1:
        path, stats = stats_by_path[0]
        label = _stats_source_label(path)
        merged = dict(stats)
        for key in ('loss_history', 'val_loss_history', 'lr_history',
                    'gpu_mem_history', 'step_times', 'test_history'):
            merged[key] = _copy_history_with_run(stats.get(key, []), label, path.name)
        merged['source_files'] = [path.name]
        merged['current_source_file'] = path.name
        merged['current_run'] = label
        return merged

    current_path, current_stats = max(
        stats_by_path, key=lambda item: item[0].stat().st_mtime)
    # Start from the current run so newly added scalar/nested metrics remain
    # available even when the dashboard also overlays older stats files.
    merged = dict(current_stats)
    merged.update({
        'start_time': '',
        'end_time': '',
        'total_steps': 0,
        'epochs_completed': 0,
        'best_loss': None,
        'best_val_loss': None,
        'loss_history': [],
        'val_loss_history': [],
        'lr_history': [],
        'gpu_mem_history': [],
        'step_times': [],
        'test_history': [],
        'source_files': [path.name for path, _ in stats_by_path],
        'current_source_file': current_path.name,
        'current_run': _stats_source_label(current_path),
    })

    start_times = []
    end_times = []
    best_losses = []
    best_val_losses = []
    for path, stats in stats_by_path:
        label = _stats_source_label(path)
        source_file = path.name

        start = stats.get('start_time')
        end = stats.get('end_time')
        if isinstance(start, str) and start:
            start_times.append(start)
        if isinstance(end, str) and end:
            end_times.append(end)

        merged['total_steps'] = max(merged['total_steps'], stats.get('total_steps') or 0)
        merged['epochs_completed'] += stats.get('epochs_completed') or 0

        best_loss = _as_finite_float(stats.get('best_loss'))
        best_val_loss = _as_finite_float(stats.get('best_val_loss'))
        if best_loss is not None:
            best_losses.append(best_loss)
        if best_val_loss is not None:
            best_val_losses.append(best_val_loss)

        for key in ('loss_history', 'val_loss_history', 'lr_history',
                    'gpu_mem_history', 'step_times', 'test_history'):
            merged[key].extend(_copy_history_with_run(stats.get(key, []), label, source_file))

    merged['start_time'] = min(start_times) if start_times else ''
    merged['end_time'] = max(end_times) if end_times else ''
    merged['best_loss'] = min(best_losses) if best_losses else None
    merged['best_val_loss'] = min(best_val_losses) if best_val_losses else None
    return merged


def load_stats_bundle(stats_path: str | Path, include_related: bool = True) -> tuple[dict, list[Path]]:
    """Load one stats file plus sibling non-legacy stats files for the dashboard."""
    stats_path = Path(stats_path)
    paths = _candidate_stats_paths(stats_path.parent) if include_related else []
    if stats_path.exists() and stats_path not in paths:
        paths.append(stats_path)
    if not paths:
        return load_stats(str(stats_path)), [stats_path]

    loaded = []
    for path in paths:
        try:
            loaded.append((path, load_stats(str(path))))
        except Exception as e:
            print(f"Warning: Could not read stats file {path}: {e}")

    if not loaded:
        return load_stats(str(stats_path)), [stats_path]
    return _aggregate_stats(loaded), [path for path, _ in loaded]


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


def _tag_log_entries(log_entries: list[dict], stats: dict) -> list[dict]:
    """Tag external JSONL entries with the current report source when absent."""
    source_file = stats.get('current_source_file') or 'logs'
    run_label = stats.get('current_run') or source_file
    tagged = []
    for entry in log_entries:
        copied = dict(entry)
        copied.setdefault('source_file', source_file)
        copied.setdefault('run', run_label)
        tagged.append(copied)
    return tagged


def merge_test_history(stats: dict, log_entries: list[dict]) -> list[dict]:
    """Merge test history from stats file and log files, removing duplicates.

    Args:
        stats: Training statistics dictionary (with test_history)
        log_entries: Test entries from JSONL log files

    Returns:
        Combined and deduplicated test history, sorted by step
    """
    # Start with test_history from stats (skip malformed entries with no step)
    combined = {}
    for idx, entry in enumerate(stats.get('test_history', [])):
        step = entry.get('step')
        if step is not None:
            combined[(entry.get('source_file', 'stats'), step, idx)] = entry

    # Add entries from log files (will overwrite duplicates)
    for idx, entry in enumerate(_tag_log_entries(log_entries, stats)):
        step = entry.get('step')
        if step is not None:
            combined[(entry.get('source_file', 'logs'), step, idx)] = entry

    # Sort by step and return as list
    return [entry for _, entry in sorted(
        combined.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])
    )]


def _summary_value(source, *keys, default=None):
    """Read the first available key from a summary metric mapping."""
    if not isinstance(source, dict):
        return default
    for key in keys:
        if key in source:
            return source[key]
    return default


def _summary_latest(source: dict, *history_keys) -> dict:
    for key in history_keys:
        history = _summary_value(source, key)
        if isinstance(history, list):
            for entry in reversed(history):
                if isinstance(entry, dict):
                    return entry
    return {}


def _summary_number(sources, *keys) -> float | None:
    for source in sources:
        for key in keys:
            value = _as_finite_float(_summary_value(source, key))
            if value is not None:
                return value
    return None


def _summary_state(sources, *keys):
    for source in sources:
        for key in keys:
            value = _summary_value(source, key)
            if value is not None and str(value).strip():
                return value
    return None


def _summary_loss_text(value) -> str:
    metric = _as_finite_float(value)
    return 'N/A' if metric is None else f'{metric:.4f}'


def _summary_percent_text(value) -> str:
    metric = _as_finite_float(value)
    if metric is None:
        return 'N/A'
    if abs(metric) <= 1.0:
        metric *= 100.0
    return f'{metric:.1f}%'


def _summary_state_text(value, true_text: str, false_text: str) -> str:
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return true_text if value else false_text
    return str(value)


def _summary_confidence_interval(sources) -> tuple:
    """Return numeric interval bounds or a preformatted interval string."""
    for source in sources:
        if not isinstance(source, dict):
            continue
        lower = _summary_number(
            [source], 'match_score_ci_lower', 'match_score_ci_low',
            'confidence_interval_lower', 'ci_lower', 'lower_bound')
        upper = _summary_number(
            [source], 'match_score_ci_upper', 'match_score_ci_high',
            'confidence_interval_upper', 'ci_upper', 'upper_bound')
        if lower is not None and upper is not None:
            return lower, upper, None

        container = _summary_value(
            source, 'match_score_confidence_interval', 'match_score_ci',
            'match_score_ci_95', 'confidence_interval_95', 'ci_95',
            'confidence_interval')
        if isinstance(container, dict):
            lower = _summary_number([container], 'lower', 'low', 'lower_bound')
            upper = _summary_number([container], 'upper', 'high', 'upper_bound')
            if lower is not None and upper is not None:
                return lower, upper, None
        elif isinstance(container, (list, tuple)) and len(container) >= 2:
            lower = _as_finite_float(container[0])
            upper = _as_finite_float(container[1])
            if lower is not None and upper is not None:
                return lower, upper, None
        elif isinstance(container, str) and container.strip():
            return None, None, container.strip()
    return None, None, None


def _recovery_summary_metrics(stats: dict, test_history: list[dict]) -> dict:
    """Build explicit P0/P1 recovery metric strings with legacy fallbacks."""
    latest_loss = _summary_latest(stats, 'loss_history')
    latest_validation = _summary_latest(
        stats, 'validation_teacher_agreement_history',
        'teacher_agreement_history', 'validation_history')
    latest_promotion = _summary_latest(stats, 'promotion_history')
    latest_acceptance = _summary_latest(stats, 'acceptance_history')
    latest_test = test_history[-1] if test_history else _summary_latest(
        stats, 'test_history', 'evaluation_history')

    dataset_metrics = _summary_value(
        stats, 'dataset_metrics', 'current_dataset', default={})
    validation_metrics = _summary_value(
        stats, 'validation_metrics', 'validation', default={})
    promotion_metrics = _summary_value(
        stats, 'promotion', 'promotion_metrics', default={})
    acceptance_metrics = _summary_value(
        stats, 'acceptance', 'acceptance_metrics', default={}) or latest_acceptance
    evaluation_metrics = _summary_value(
        stats, 'latest_evaluation', 'evaluation_metrics',
        'game_evaluation', default={})
    acceptance_payload = _summary_value(
        acceptance_metrics, 'metrics', default=acceptance_metrics)
    acceptance_evaluation = _summary_value(
        acceptance_payload, 'easy', 'random', default={})

    current_train_loss = _summary_number(
        [stats, latest_loss], 'current_train_loss', 'recent_loss', 'loss')
    current_dataset_best = _summary_number(
        [stats, dataset_metrics], 'current_dataset_best_train_loss',
        'dataset_best_train_loss', 'best_train_loss')
    historical_best = _summary_number(
        [stats], 'historical_best_train_loss', 'best_loss')
    validation_agreement = _summary_number(
        [stats, validation_metrics, latest_validation],
        'validation_teacher_agreement', 'held_out_teacher_agreement',
        'heldout_teacher_agreement', 'teacher_agreement',
        'top1_teacher_agreement', 'top1_agreement')

    promotion_state = _summary_state(
        [stats, promotion_metrics, latest_promotion],
        'promotion_state', 'promotion_status', 'state', 'status', 'promoted')
    acceptance_state = _summary_state(
        [stats, acceptance_metrics, latest_acceptance, evaluation_metrics, latest_test],
        'acceptance_state', 'acceptance_status', 'state', 'status',
        'accepted', 'passed')

    evaluation_sources = [
        stats, evaluation_metrics, latest_test, acceptance_evaluation]
    match_score = _summary_number(evaluation_sources, 'match_score', 'score')
    if match_score is None:
        wins = _summary_number(evaluation_sources, 'ml_wins', 'wins')
        draws = _summary_number(evaluation_sources, 'draws')
        games = _summary_number(evaluation_sources, 'total_games', 'games')
        if wins is not None and draws is not None and games:
            match_score = (wins + 0.5 * draws) / games

    ci_lower, ci_upper, ci_text = _summary_confidence_interval(evaluation_sources)
    if ci_lower is not None and ci_upper is not None:
        interval_text = (
            f'{_summary_percent_text(ci_lower)} to '
            f'{_summary_percent_text(ci_upper)}'
        )
    elif ci_text:
        interval_text = ci_text
    else:
        interval_text = 'N/A'

    return {
        'current_train_loss': _summary_loss_text(current_train_loss),
        'current_dataset_best_train_loss': _summary_loss_text(current_dataset_best),
        'historical_best_train_loss': _summary_loss_text(historical_best),
        'validation_teacher_agreement': _summary_percent_text(validation_agreement),
        'promotion_state': _summary_state_text(
            promotion_state, 'Promoted', 'Not promoted'),
        'acceptance_state': _summary_state_text(
            acceptance_state, 'Accepted', 'Not accepted'),
        'match_score': _summary_percent_text(match_score),
        'confidence_interval': interval_text,
    }


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
    recovery_metrics = _recovery_summary_metrics(stats, test_history)
    start_time = stats.get('start_time', 'N/A')
    start_time_str = start_time[:19] if isinstance(start_time, str) and len(start_time) >= 19 else str(start_time)
    end_time = stats.get('end_time', 'N/A')
    end_time_str = end_time[:19] if isinstance(end_time, str) and len(end_time) >= 19 else str(end_time)

    lines = [
        "<b>Training Summary</b>",
        "================",
        f"Total Training Steps:  {total_steps_str}",
        f"Epochs Completed:      {epochs_str}",
        f"Current Train Loss:    {recovery_metrics['current_train_loss']}",
        f"Current Dataset Best:  {recovery_metrics['current_dataset_best_train_loss']}",
        f"Historical Best Loss:  {recovery_metrics['historical_best_train_loss']}",
        "",
        "<b>Validation and Promotion</b>",
        "========================",
        f"Teacher Agreement:     {recovery_metrics['validation_teacher_agreement']}",
        f"Promotion State:       {recovery_metrics['promotion_state']}",
        f"Acceptance State:      {recovery_metrics['acceptance_state']}",
        f"Match Score:           {recovery_metrics['match_score']}",
        f"Match Score 95% CI:    {recovery_metrics['confidence_interval']}",
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


def _as_finite_float(value) -> float | None:
    """Return a finite float, or None for missing/non-numeric metric values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_container(stats: dict, path: tuple[str, ...]):
    """Fetch a nested metric container from a stats dict."""
    current = stats
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _normalise_metric_container(container):
    """Return the list-like payload from a metric container when available."""
    if isinstance(container, dict):
        for key in ('history', 'entries', 'all', 'values'):
            values = container.get(key)
            if isinstance(values, list):
                return values
        return None
    return container


def _step_metric_entries(stats: dict, candidates: list[tuple[tuple[str, ...], tuple[str, ...]]]) -> list[tuple[float, float]]:
    """Extract ordered ``(step, value)`` metric entries from any known stats shape."""
    for path, value_keys in candidates:
        entries = _normalise_metric_container(_metric_container(stats, path))
        if not isinstance(entries, list):
            continue

        parsed = []
        for entry in entries:
            step = None
            value = None
            if isinstance(entry, dict):
                step = _as_finite_float(entry.get('step'))
                for key in value_keys:
                    value = _as_finite_float(entry.get(key))
                    if value is not None:
                        break
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                step = _as_finite_float(entry[0])
                value = _as_finite_float(entry[1])

            if step is not None and value is not None:
                parsed.append((step, value))

        if parsed:
            return parsed
    return []


def _step_metric_series(stats: dict, candidates: list[tuple[tuple[str, ...], tuple[str, ...]]]) -> list[tuple[float, float]]:
    """Extract a sorted, de-duplicated ``(step, value)`` series."""
    by_step = {
        step: value
        for step, value in _step_metric_entries(stats, candidates)
    }
    return sorted(by_step.items())


def _nearest_step_value(steps: list[float], values: list[float], step: float) -> float | None:
    """Return the value recorded at the nearest metric step."""
    if not steps:
        return None
    idx = bisect_left(steps, step)
    if idx <= 0:
        return values[0]
    if idx >= len(steps):
        return values[-1]
    before = idx - 1
    if abs(steps[idx] - step) < abs(step - steps[before]):
        return values[idx]
    return values[before]


def _ordered_metric_values_for_steps(steps: list[float], entries: list[tuple[float, float]]) -> list[float | None]:
    """Align an ordered metric stream to ordered loss steps.

    Training stats can contain duplicate step values when a run resumes or
    restarts.  This keeps the original metric order so duplicate steps match the
    corresponding occurrence instead of a later sorted duplicate.
    """
    values = []
    cursor = 0
    last_value = None
    for raw_step in steps:
        step = _as_finite_float(raw_step)
        value = None
        if step is None:
            values.append(last_value)
            continue

        while cursor < len(entries):
            metric_step, metric_value = entries[cursor]
            if metric_step == step:
                value = metric_value
                last_value = metric_value
                cursor += 1
                break
            if metric_step > step:
                break
            last_value = metric_value
            cursor += 1

        values.append(value if value is not None else last_value)
    return values


def _format_metric(value: float | None, precision: int = 6) -> str:
    return 'N/A' if value is None else f'{value:.{precision}g}'


def _learning_rates_for_losses(sampled_losses: list[dict], stats: dict) -> list[float | None]:
    """Return per-loss learning rates from direct loss entries or LR history."""
    lr_candidates = [
        (('lr_history',), ('lr', 'learning_rate', 'value')),
        (('learning_rate_history',), ('learning_rate', 'lr', 'value')),
        (('learning_rate',), ('learning_rate', 'lr', 'value')),
    ]
    lr_entries = _step_metric_entries(stats, lr_candidates)
    ordered_rates = _ordered_metric_values_for_steps(
        [loss_entry.get('step') for loss_entry in sampled_losses if isinstance(loss_entry, dict)],
        lr_entries,
    )
    lr_series = sorted({step: value for step, value in lr_entries}.items())
    lr_steps = [step for step, _ in lr_series]
    lr_values = [value for _, value in lr_series]

    rates: list[float | None] = []
    for idx, loss_entry in enumerate(sampled_losses):
        direct = None
        if isinstance(loss_entry, dict):
            direct = _as_finite_float(loss_entry.get('learning_rate'))
            if direct is None:
                direct = _as_finite_float(loss_entry.get('lr'))
        if direct is not None:
            rates.append(direct)
            continue

        step = _as_finite_float(loss_entry.get('step')) if isinstance(loss_entry, dict) else None
        ordered = ordered_rates[idx] if idx < len(ordered_rates) else None
        if ordered is not None:
            rates.append(ordered)
        else:
            rates.append(_nearest_step_value(lr_steps, lr_values, step) if step is not None else None)
    return rates


def _gpu_hours_for_steps(steps: list[float], stats: dict) -> list[float]:
    """Return cumulative GPU-hours at each step using recorded step durations."""
    step_time_entries = _step_metric_entries(stats, [
        (('step_times',), ('time_sec', 'step_time_sec', 'seconds', 'value')),
        (('step_time_history',), ('time_sec', 'step_time_sec', 'seconds', 'value')),
        (('throughput', 'step_time_sec'), ('time_sec', 'step_time_sec', 'seconds', 'value')),
    ])
    cumulative: list[tuple[float, float]] = []
    total_seconds = 0.0
    for step, seconds in step_time_entries:
        if seconds > 0:
            total_seconds += seconds
        cumulative.append((step, total_seconds / 3600.0))

    if not cumulative:
        return [0.0 for _ in steps]
    return [float(value or 0.0) for value in _ordered_metric_values_for_steps(steps, cumulative)]


def _group_history_by_source(entries: list[dict], default_label: str = 'Run') -> list[tuple[str, str, list[dict]]]:
    """Group history entries by source file while preserving first-seen order."""
    groups: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_file = entry.get('source_file') or default_label
        run_label = entry.get('run') or source_file
        if source_file not in groups:
            groups[source_file] = {'label': run_label, 'entries': []}
        groups[source_file]['entries'].append(entry)
    return [(source, data['label'], data['entries']) for source, data in groups.items()]


def _stats_for_source(stats: dict, source_file: str) -> dict:
    """Return a shallow stats view containing only metrics for one run source."""
    filtered = dict(stats)
    for key in ('lr_history', 'step_times', 'gpu_mem_history'):
        filtered[key] = [
            entry for entry in stats.get(key, [])
            if not isinstance(entry, dict) or entry.get('source_file') == source_file
        ]
    return filtered


def _fig_overall(test_history: list[dict]) -> go.Figure:
    """Overall ML win-rate vs the algorithm, with a polynomial trend line."""
    fig = go.Figure()
    palette = ['royalblue', 'darkorange', 'seagreen', 'crimson', 'mediumpurple', 'sienna']
    for group_idx, (source_file, run_label, entries) in enumerate(
            _group_history_by_source(test_history)):
        steps = [t['step'] for t in entries]
        win_rates = [t['ml_win_rate'] * 100 for t in entries]
        color = palette[group_idx % len(palette)]
        meta = {'source_file': source_file, 'run': run_label}
        fig.add_trace(go.Scatter(
            x=steps, y=win_rates, mode='lines+markers', name=f'{run_label} ML Win Rate',
            line=dict(color=color, width=2), marker=dict(size=7),
            fill='tozeroy' if group_idx == 0 else None,
            fillcolor='rgba(65,105,225,0.12)' if group_idx == 0 else None,
            meta=meta,
            customdata=[[run_label] for _ in steps],
            hovertemplate='Run %{customdata[0]}<br>Step %{x:,}<br>Win Rate %{y:.1f}%<extra></extra>',
        ))
        if len(steps) > 2:
            z = np.polyfit(steps, win_rates, TREND_LINE_DEGREE)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(steps), max(steps), 100)
            fig.add_trace(go.Scatter(
                x=x_smooth, y=np.clip(p(x_smooth), 0, 100), mode='lines',
                name=f'{run_label} Trend',
                line=dict(color=color, width=2, dash='dash'),
                meta=meta,
                customdata=[[run_label] for _ in x_smooth],
                hovertemplate='Run %{customdata[0]}<br>Step %{x:,.0f}<br>Trend %{y:.1f}%<extra></extra>',
            ))
    fig.add_hline(y=50, line_dash='dash', line_color='red', opacity=0.7,
                  annotation_text='50% (Equal)', annotation_position='top left')
    return _apply_interactive_layout(fig, 'Win Rate (%)', y_range=[0, 100])


def _fig_position(test_history: list[dict]) -> go.Figure:
    """Win rate split by which side the ML model played (Player 1 vs Player 2)."""
    fig = go.Figure()
    p1_palette = ['green', 'darkorange', 'seagreen', 'crimson', 'mediumpurple', 'sienna']
    p2_palette = ['magenta', 'firebrick', 'teal', 'goldenrod', 'slateblue', 'gray']
    for group_idx, (source_file, run_label, entries) in enumerate(
            _group_history_by_source(test_history)):
        steps = [t['step'] for t in entries]
        p1_win_rates = [t['ml_as_p1_win_rate'] * 100 for t in entries]
        p2_win_rates = [t['ml_as_p2_win_rate'] * 100 for t in entries]
        meta = {'source_file': source_file, 'run': run_label}
        fig.add_trace(go.Scatter(
            x=steps, y=p1_win_rates, mode='lines+markers', name=f'{run_label} P1 (White)',
            line=dict(color=p1_palette[group_idx % len(p1_palette)], width=2),
            marker=dict(size=6, symbol='square'),
            meta=meta,
            customdata=[[run_label] for _ in steps],
            hovertemplate='Run %{customdata[0]}<br>Step %{x:,}<br>P1 Win Rate %{y:.1f}%<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=steps, y=p2_win_rates, mode='lines+markers', name=f'{run_label} P2 (Black)',
            line=dict(color=p2_palette[group_idx % len(p2_palette)], width=2),
            marker=dict(size=6, symbol='triangle-up'),
            meta=meta,
            customdata=[[run_label] for _ in steps],
            hovertemplate='Run %{customdata[0]}<br>Step %{x:,}<br>P2 Win Rate %{y:.1f}%<extra></extra>',
        ))
    fig.add_hline(y=50, line_dash='dash', line_color='red', opacity=0.7)
    return _apply_interactive_layout(fig, 'Win Rate (%)', y_range=[0, 100])


def _fig_loss(stats: dict) -> go.Figure:
    """Training loss (downsampled) plus a moving average."""
    fig = go.Figure()
    trace_x_steps = []
    trace_x_gpu_hours = []
    loss_history = stats.get('loss_history', [])
    current_source_file = stats.get('current_source_file')
    palette = ['royalblue', 'darkorange', 'seagreen', 'crimson', 'mediumpurple', 'sienna']
    if loss_history:
        for group_idx, (source_file, run_label, entries) in enumerate(
                _group_history_by_source(loss_history)):
            # Sample per run so short recent runs are not skipped by a long history.
            sample_rate = max(1, len(entries) // SAMPLE_RATE_TARGET)
            sampled_losses = entries[::sample_rate]
            loss_steps = [l['step'] for l in sampled_losses]
            losses = [l['loss'] for l in sampled_losses]
            group_stats = _stats_for_source(stats, source_file)
            learning_rates = _learning_rates_for_losses(sampled_losses, group_stats)
            loss_gpu_hours = _gpu_hours_for_steps(loss_steps, group_stats)
            color = palette[group_idx % len(palette)]
            meta = {'source_file': source_file, 'run': run_label}
            loss_customdata = [
                [run_label, step, f'{gpu_hours:.4f}', _format_metric(lr)]
                for step, gpu_hours, lr in zip(loss_steps, loss_gpu_hours, learning_rates)
            ]
            trace_x_steps.append(loss_steps)
            trace_x_gpu_hours.append(loss_gpu_hours)
            fig.add_trace(go.Scatter(
                x=loss_steps, y=losses, mode='lines', name=f'{run_label} Loss',
                line=dict(color=color, width=1), opacity=0.7,
                meta=meta,
                customdata=loss_customdata,
                hovertemplate=(
                    'Run %{customdata[0]}<br>'
                    'Step %{customdata[1]:,}<br>'
                    'GPU Hours %{customdata[2]}<br>'
                    'Loss %{y:.4f}<br>'
                    'learning_rate %{customdata[3]}<extra></extra>'
                ),
            ))

            # Moving average
            window = min(MOVING_AVG_WINDOW, len(losses) // 5) if len(losses) > 10 else 1
            if window > 1:
                moving_avg = np.convolve(losses, np.ones(window) / window, mode='valid')
                ma_steps = loss_steps[window - 1:]
                ma_gpu_hours = loss_gpu_hours[window - 1:]
                ma_learning_rates = learning_rates[window - 1:]
                ma_customdata = [
                    [run_label, step, f'{gpu_hours:.4f}', _format_metric(lr)]
                    for step, gpu_hours, lr in zip(ma_steps, ma_gpu_hours, ma_learning_rates)
                ]
                trace_x_steps.append(ma_steps)
                trace_x_gpu_hours.append(ma_gpu_hours)
                fig.add_trace(go.Scatter(
                    x=ma_steps, y=moving_avg, mode='lines', name=f'{run_label} Moving Avg ({window})',
                    line=dict(color=color, width=2, dash='dash'),
                    meta=meta,
                    customdata=ma_customdata,
                    hovertemplate=(
                        'Run %{customdata[0]}<br>'
                        'Step %{customdata[1]:,}<br>'
                        'GPU Hours %{customdata[2]}<br>'
                        'Avg Loss %{y:.4f}<br>'
                        'learning_rate %{customdata[3]}<extra></extra>'
                    ),
                ))

    fig = _apply_interactive_layout(fig, 'Loss')
    if trace_x_steps and any(any(hours > 0 for hours in trace) for trace in trace_x_gpu_hours):
        fig.update_layout(
            margin=dict(l=58, r=18, t=66, b=46),
            updatemenus=[dict(
                type='buttons',
                direction='right',
                showactive=True,
                active=0,
                x=1.0,
                xanchor='right',
                y=1.24,
                yanchor='top',
                buttons=[
                    dict(
                        label='Steps',
                        method='update',
                        args=[
                            {'x': trace_x_steps},
                            {
                                'xaxis.title.text': 'Training Steps',
                                'xaxis.tickformat': ',d',
                                'xaxis.autorange': True,
                            },
                        ],
                    ),
                    dict(
                        label='GPU hours',
                        method='update',
                        args=[
                            {'x': trace_x_gpu_hours},
                            {
                                'xaxis.title.text': 'GPU Hours',
                                'xaxis.tickformat': '.2f',
                                'xaxis.autorange': True,
                            },
                        ],
                    ),
                ],
            )],
        )
        fig.update_xaxes(tickformat=',d')
    return fig


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


def _render_dashboard(panels: list, summary_html: str, title: str = PAGE_TITLE,
                      current_source_file: str = "") -> str:
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
            .replace('%%THEME_INIT%%', _THEME_INIT_SCRIPT.replace('%%THEME_STORAGE_KEY%%', THEME_STORAGE_KEY))
            .replace('%%JS%%', _DASHBOARD_JS
                     .replace('%%REFRESH_MS%%', str(AUTO_REFRESH_MS))
                     .replace('%%GENERATED_AT%%', generated_at)
                     .replace('%%CURRENT_SOURCE_FILE%%', current_source_file)
                     .replace('%%THEME_STORAGE_KEY%%', THEME_STORAGE_KEY))
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
        ('overall', 'ML Model vs Algorithm - Overall Win Rate', _fig_overall(test_history)),
        ('position', 'Win Rate by Player Position', _fig_position(test_history)),
        ('loss', 'Training Loss', _fig_loss(stats)),
    ]
    summary_html = _build_summary_text(stats, test_history, steps, win_rates)

    # Self-contained page (plotly.js embedded inline), so it renders offline and
    # when the CDN is blocked (e.g. opening a WSL2-generated file in a Windows
    # browser). open_file() handles opening it on WSL vs. native.
    _atomic_write_text(Path(output_path), _render_dashboard(
        panels, summary_html, current_source_file=stats.get('current_source_file', '')
    ))
    print(f"Interactive training progress plot saved to: {output_path}")
    if show:
        open_file(output_path)
    return [fig for _, _, fig in panels]


def write_progress_report(stats_path: str | Path,
                          logs_dir: str | Path | None = None,
                          output_path: str | Path = "models/training_progress.html") -> Path:
    """Regenerate the interactive HTML report from the latest stats/log files."""
    stats_path = Path(stats_path)
    stats, _stats_paths = load_stats_bundle(stats_path)

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
    stats, _stats_paths = load_stats_bundle(stats_path)

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
    training_stats_local.json, training_stats_server.json,
    training_stats_policy_distillation.json) so the no-arg command tracks
    whichever run last wrote stats. Fall back to
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
        models_dir / "training_stats_policy_distillation.json",
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

    stats, stats_paths = load_stats_bundle(stats_path)
    if len(stats_paths) == 1:
        print(f"Loading stats from: {stats_paths[0]}")
    else:
        print("Loading stats from:")
        for path in stats_paths:
            current = " (current)" if path.name == stats.get('current_source_file') else ""
            print(f"  {path}{current}")

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
