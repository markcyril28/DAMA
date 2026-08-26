"""The launcher preflight gates must reject bad session settings before setup.

local_train.sh validates operator edits (TRAIN_DURATION, MIN_FREE_DISK_GB,
recovery_experiment.enabled) before conda activation, the Cython scan, and
CUDA init -- the expensive part of a launch. A regression that let an invalid
value through would surface only as a traceback minutes later (or, for the
disk floor, as a dead volume mid-run), so the blocks are executed here exactly
as they appear in the working-tree script.
"""

from pathlib import Path
import os
import shutil
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "local_train.sh"
# Resolved once: one test deliberately empties the child PATH, and exec must
# still be able to locate the shell itself.
BASH = shutil.which("bash") or "/bin/bash"


def _launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def _extract(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def _run_block(
    block: str,
    env_overrides: dict[str, str],
    project_dir: str = "/tmp",
) -> subprocess.CompletedProcess:
    """Source one launcher block under the launcher's own strict mode.

    A gate that rejects must exit non-zero out of the sourced block; one that
    accepts falls through to the sentinel print. TRAIN_DURATION/PROJECT_DIR
    mirror the launcher header, where they are always defined.
    """
    env = dict(os.environ)
    for name in ("POLICY_RECOVERY_ENABLED", "MIN_FREE_DISK_GB",
                 "TRAIN_DURATION", "PROJECT_DIR"):
        env.pop(name, None)
    env.setdefault("TRAIN_DURATION", "")
    env.setdefault("PROJECT_DIR", project_dir)
    env.update(env_overrides)
    script = (
        "set -euo pipefail\n"
        + block
        + "\nprintf '__ACCEPTED__ recovery=%s free_kb=%s\\n' "
          '"${IS_POLICY_RECOVERY:-unset}" "${_free_kb:-unset}"\n'
    )
    return subprocess.run(
        [BASH, "-c", script],
        capture_output=True, text=True, env=env,
        timeout=60,
    )


def test_launcher_script_is_valid_bash() -> None:
    completed = subprocess.run(
        [BASH, "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_recovery_enabled_accepts_every_yaml_boolean_spelling() -> None:
    """The trainer classifies this key via yaml.safe_load; so must the launcher.

    A textual `= true` comparison silently skipped every recovery guard on a
    config saying `enabled: Yes`, because PyYAML resolves True/yes/on (any
    case) as booleans while the shell saw three different strings.
    """
    text = _launcher_text()
    block = _extract(
        text,
        "IS_POLICY_RECOVERY=false\n",
        'if [ "$IS_POLICY_RECOVERY" = true ]; then',
    )

    truthy = ["true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"]
    falsy = ["false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"]
    for spelling in truthy:
        completed = _run_block(
            block, {"POLICY_RECOVERY_ENABLED": spelling,
                    "MIN_FREE_DISK_GB": "0"})
        assert completed.returncode == 0, (spelling, completed.stderr)
        assert "__ACCEPTED__ recovery=true" in completed.stdout
    for spelling in falsy:
        completed = _run_block(
            block, {"POLICY_RECOVERY_ENABLED": spelling,
                    "MIN_FREE_DISK_GB": "0"})
        assert completed.returncode == 0, (spelling, completed.stderr)
        assert "__ACCEPTED__ recovery=false" in completed.stdout


def test_recovery_enabled_rejects_non_boolean_garbage() -> None:
    """A mistyped value must abort, not degrade into 'recovery off'."""
    text = _launcher_text()
    block = _extract(
        text,
        "IS_POLICY_RECOVERY=false\n",
        'if [ "$IS_POLICY_RECOVERY" = true ]; then',
    )

    completed = _run_block(
        block, {"POLICY_RECOVERY_ENABLED": "ture",
                "MIN_FREE_DISK_GB": "0"})
    assert completed.returncode != 0
    assert "not a YAML boolean" in completed.stderr


def test_train_duration_gate_stays_inside_the_trainer_grammar() -> None:
    """Invalid durations died as post-CUDA-init tracebacks; now they die here.

    The gate is an approximation of parse_duration(), which is deliberately
    lenient: a bare number is hours and unit words are digit-scanned anywhere
    in the string (so "-4h", "1.5d" and "2d4x" all reach the trainer). The
    gate must therefore only reject strings the trainer itself raises on --
    rejecting a supported duration would block launches the trainer accepts,
    which is worse than letting one through to its own ValueError.
    """
    text = _launcher_text()
    block = _extract(
        text,
        'case "$MIN_FREE_DISK_GB" in',
        'if [ "$ENHANCED_STAGE" = true ] '
        '&& [ "$IS_POLICY_RECOVERY" = false ]; then',
    )

    for duration in ("2d", "4h", "30m", "10s", "1d12h", "48h",
                     "2 days", "45m30s", "24", "1.5"):
        completed = _run_block(
            block, {"MIN_FREE_DISK_GB": "0", "TRAIN_DURATION": duration})
        assert completed.returncode == 0, (duration, completed.stderr)

    for duration in ("4x", "d", "half a day", "day"):
        completed = _run_block(
            block, {"MIN_FREE_DISK_GB": "0", "TRAIN_DURATION": duration})
        assert completed.returncode != 0, duration
        assert "TRAIN_DURATION" in completed.stderr


def test_min_free_disk_gb_must_be_a_non_negative_integer() -> None:
    text = _launcher_text()
    block = _extract(
        text,
        'case "$MIN_FREE_DISK_GB" in',
        'if [ "$ENHANCED_STAGE" = true ] '
        '&& [ "$IS_POLICY_RECOVERY" = false ]; then',
    )

    for value in ("abc", "-5", "", "3.5", "10GB"):
        completed = _run_block(
            block, {"MIN_FREE_DISK_GB": value})
        assert completed.returncode != 0, repr(value)
        assert "MIN_FREE_DISK_GB" in completed.stderr


def test_disk_floor_refuses_launch_below_the_floor(tmp_path: Path) -> None:
    """The audited 48h run exhausted its volume near the 24h mark.

    An absurdly large floor stands in for a full disk, so this never depends
    on how much space the host running the tests happens to have.
    """
    text = _launcher_text()
    block = _extract(
        text,
        'case "$MIN_FREE_DISK_GB" in',
        'if [ "$ENHANCED_STAGE" = true ] '
        '&& [ "$IS_POLICY_RECOVERY" = false ]; then',
    )

    completed = _run_block(
        block, {"MIN_FREE_DISK_GB": "999999999"}, project_dir=str(tmp_path))
    assert completed.returncode != 0
    assert "free on" in completed.stderr


def test_disk_floor_zero_disables_and_missing_df_degrades_to_warning(
    tmp_path: Path,
) -> None:
    text = _launcher_text()
    block = _extract(
        text,
        'case "$MIN_FREE_DISK_GB" in',
        'if [ "$ENHANCED_STAGE" = true ] '
        '&& [ "$IS_POLICY_RECOVERY" = false ]; then',
    )

    # Floor disabled: accepts regardless of the measured volume.
    completed = _run_block(block, {"MIN_FREE_DISK_GB": "0"},
                           project_dir=str(tmp_path))
    assert completed.returncode == 0, completed.stderr
    assert "__ACCEPTED__" in completed.stdout

    # df/awk unreachable (empty PATH): the gate warns and stays usable rather
    # than blocking every launch on a measurement failure.
    env = dict(os.environ)
    for name in ("POLICY_RECOVERY_ENABLED", "MIN_FREE_DISK_GB",
                 "TRAIN_DURATION", "PROJECT_DIR"):
        env.pop(name, None)
    env["PATH"] = str(tmp_path)
    env["MIN_FREE_DISK_GB"] = "10"
    env["PROJECT_DIR"] = str(tmp_path)
    env["TRAIN_DURATION"] = ""
    completed = subprocess.run(
        [BASH, "-c",
         "set -euo pipefail\n" + block
         + "\nprintf '__ACCEPTED__ free_kb=%s\\n' \"${_free_kb:-unset}\"\n"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Could not measure free disk space" in completed.stdout
    assert "free_kb=unset" in completed.stdout
