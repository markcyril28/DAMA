"""The Mac scripts: their preflight gates, their launch plumbing, and bash 3.2.

local_train_mac.sh carries local_train.sh's preflight gates line for line, and
they are executed here the way test_local_train_launcher_guards.py executes
the Linux launcher's -- but under /bin/bash, which on a Mac is bash 3.2. The
rest covers what only the Mac side does: the platform and Rosetta guards, the
readlink/sha256 fallbacks, the config-driven accelerator gate, the Mac config,
eval_checkpoints_mac.sh's config-derived paths, and stop_training_mac.sh's
cleanup of spawn-started workers. bash 3.2 also rejects bash 4 syntax that no
Linux run would ever notice.
"""

from pathlib import Path
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import uuid

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINUX_LAUNCHER = PROJECT_ROOT / "local_train.sh"
MAC_LAUNCHER = PROJECT_ROOT / "local_train_mac.sh"
MAC_GAME = PROJECT_ROOT / "run_game_mac.sh"
MAC_SETUP = PROJECT_ROOT / "setup_conda_mac.sh"
MAC_CONFIG = PROJECT_ROOT / "config" / "training_config_mac.yaml"
EVAL_SCRIPT = PROJECT_ROOT / "eval_checkpoints_mac.sh"
STOP_SCRIPT = PROJECT_ROOT / "stop_training_mac.sh"
# What `#!/usr/bin/env bash` gets on a stock Mac: bash 3.2.
SYSTEM_BASH = "/bin/bash" if os.path.exists("/bin/bash") else (
    shutil.which("bash") or "/bin/bash")

# Every script a Mac user runs, with the Linux script each one refers to.
MAC_SCRIPTS = {
    MAC_LAUNCHER: "local_train.sh",
    MAC_GAME: "run_game.sh",
    MAC_SETUP: "setup_conda.sh",
    EVAL_SCRIPT: "eval_checkpoints.sh",
    STOP_SCRIPT: "stop_training.sh",
}
_SCRIPT_IDS = [script.name for script in MAC_SCRIPTS]

on_macos = pytest.mark.skipif(
    sys.platform != "darwin", reason="exercises macOS-only behaviour")


def _extract(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def _function(text: str, name: str) -> str:
    """Source of one top-level shell function (``name() {`` .. ``}``)."""
    return _extract(text, f"{name}() {{\n", "\n}\n") + "\n}\n"


def _shim(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


def _code_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture(scope="module")
def shims(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Stand-in commands, one directory per set, to put first on PATH.

    Made once per module and steered through the environment: macOS vets
    every new executable on its first run, which costs most of a second.
    """
    root = tmp_path_factory.mktemp("shims")
    # This interpreter. With FAKE_EVAL set, eval_checkpoint_once runs that
    # script instead.
    env_python = f"""\
        if [ -n "${{FAKE_EVAL:-}}" ] && [ "$1" = "-m" ] \\
                && [ "$2" = "dama.ai.ml.eval_checkpoint_once" ]; then
            shift 2
            exec "{sys.executable}" "$FAKE_EVAL" "$@"
        fi
        exec "{sys.executable}" "$@"
        """
    sets = {
        # The platform and Rosetta guards. setup_conda_mac.sh also asks
        # sysctl for hw.optional.arm64 (Apple Silicon).
        "platform": {
            "uname": 'echo "$FAKE_UNAME"\n',
            "sysctl": """\
                case "$*" in
                    *proc_translated*) echo "$FAKE_TRANSLATED" ;;
                    *) echo 1 ;;
                esac
                """,
            "arch": 'echo "ARCH-SHIM reexec=${_DAMA_ARM64_REEXEC:-} $*"\n',
        },
        # BSD readlink before macOS 12.3: no -f.
        "readlink_without_f": {
            "readlink": 'echo "readlink: illegal option -- f" >&2\nexit 1\n',
        },
        "broken_python3": {"python3": "exit 1\n"},
        # An interpreter without torch or yaml, under both names.
        "broken_python": {"python": "exit 1\n", "python3": "exit 1\n"},
        # stop_training_mac.sh --all edits the crontab; never the real one.
        "no_crontab": {"crontab": 'echo "crontab: no crontab" >&2\nexit 1\n'},
        # The activated env's interpreter, under both names it ships.
        "python3": {"python": env_python, "python3": env_python},
    }
    dirs = {}
    for name, commands in sets.items():
        dirs[name] = root / name
        dirs[name].mkdir()
        for command, body in commands.items():
            _shim(dirs[name], command, body)
    return dirs


def _path_with(*bin_dirs: Path) -> str:
    return os.pathsep.join([*map(str, bin_dirs), os.environ.get("PATH", "")])


# ---------------------------------------------------------------------------
# bash 3.2 compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", MAC_SCRIPTS, ids=_SCRIPT_IDS)
def test_mac_scripts_parse_under_system_bash(script: Path) -> None:
    completed = subprocess.run(
        [SYSTEM_BASH, "-n", str(script)],
        capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


# bash -n does not catch these: they parse, then fail (or silently misbehave)
# only when the line runs. mapfile in eval_checkpoints.sh was exactly that.
_BASH4_ONLY = {
    "mapfile/readarray (bash 4.0)": re.compile(r"\b(mapfile|readarray)\b"),
    "associative array (bash 4.0)":
        re.compile(r"\b(declare|local|typeset)\s+-[a-zA-Z]*A\b"),
    "nameref (bash 4.3)":
        re.compile(r"\b(declare|local|typeset)\s+-[a-zA-Z]*n\b"),
    "case modification (bash 4.0)":
        re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(\^\^?|,,?)\}"),
    "|& pipe (bash 4.0)": re.compile(r"\|&"),
    "&>> append (bash 4.0)": re.compile(r"&>>"),
    "coproc (bash 4.0)": re.compile(r"\bcoproc\b"),
    "case fallthrough (bash 4.0)": re.compile(r"(;;&|;&)\s*$"),
    "negative array index (bash 4.3)":
        re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\[-[0-9]+\]\}"),
    "wait -n (bash 4.3)": re.compile(r"\bwait\s+-n\b"),
    "[[ -v ]] (bash 4.2)": re.compile(r"\[\[\s+-v\s"),
    "${var@op} (bash 4.4)": re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*@[QEPAa]\}"),
}


@pytest.mark.parametrize("script", MAC_SCRIPTS, ids=_SCRIPT_IDS)
def test_mac_scripts_avoid_bash4_only_syntax(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    found = [
        f"{script.name}:{number}: {label}: {line.strip()}"
        for number, line in enumerate(text.splitlines(), 1)
        if not line.lstrip().startswith("#")
        for label, pattern in _BASH4_ONLY.items()
        if pattern.search(line)
    ]
    assert not found, "bash 4+ syntax in a script macOS runs with bash 3.2:\n" \
        + "\n".join(found)


# ---------------------------------------------------------------------------
# The Mac launcher keeps local_train.sh's gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("start_marker", "end_marker"), [
    ('if [[ "$TRAINING_CONFIG" = /* ]]; then', "# Add src to PYTHONPATH"),
    ('ARGS=(--config "$SELECTED_CONFIG_PATH")',
     "exec python -W ignore::FutureWarning -m dama.ai.ml.trainer"),
], ids=["preflight-gates", "trainer-args"])
def test_mac_launcher_gates_match_local_train(
    start_marker: str, end_marker: str,
) -> None:
    """Every preflight gate and the argument building stay in step.

    The only intended differences are the two portability helpers: BSD
    readlink lacks -f before macOS 12.3, and sha256sum is recent on macOS.
    A gate added to (or changed in) local_train.sh fails this until it is
    carried over to local_train_mac.sh as well.
    """
    linux = _extract(LINUX_LAUNCHER.read_text(encoding="utf-8"),
                     start_marker, end_marker)
    linux = linux.replace("readlink -f ", "_realpath ").replace(
        "sha256sum ", "_sha256 ")
    mac = _extract(MAC_LAUNCHER.read_text(encoding="utf-8"),
                   start_marker, end_marker)
    assert _code_lines(mac) == _code_lines(linux)


def test_mac_launcher_selects_the_mac_config() -> None:
    text = MAC_LAUNCHER.read_text(encoding="utf-8")
    active = re.findall(r'^TRAINING_CONFIG="([^"]*)"', text, re.MULTILINE)
    assert active == ["config/training_config_mac.yaml"]
    assert MAC_CONFIG.is_file()


# ---------------------------------------------------------------------------
# The Mac launcher's gates, executed under bash 3.2
# ---------------------------------------------------------------------------

_RECOVERY_BLOCK = ("IS_POLICY_RECOVERY=false\n",
                   'if [ "$IS_POLICY_RECOVERY" = true ]; then')
_SESSION_BLOCK = ('case "$MIN_FREE_DISK_GB" in',
                  'if [ "$ENHANCED_STAGE" = true ] '
                  '&& [ "$IS_POLICY_RECOVERY" = false ]; then')


def _gate_env(overrides: dict[str, str], project_dir: str) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("POLICY_RECOVERY_ENABLED", "MIN_FREE_DISK_GB",
                 "TRAIN_DURATION", "PROJECT_DIR"):
        env.pop(name, None)
    # Always defined in the launcher header.
    env["TRAIN_DURATION"] = ""
    env["PROJECT_DIR"] = project_dir
    env.update(overrides)
    return env


def _run_gate(
    markers: tuple[str, str],
    overrides: dict[str, str],
    project_dir: str = "/tmp",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run one block of local_train_mac.sh under its own strict mode.

    A gate that rejects exits non-zero out of the block; one that accepts
    falls through to the sentinel print.
    """
    block = _extract(MAC_LAUNCHER.read_text(encoding="utf-8"), *markers)
    script = (
        "set -euo pipefail\n"
        + block
        + "\nprintf '__ACCEPTED__ recovery=%s free_kb=%s\\n' "
          '"${IS_POLICY_RECOVERY:-unset}" "${_free_kb:-unset}"\n'
    )
    return subprocess.run(
        [SYSTEM_BASH, "-c", script],
        capture_output=True, text=True, timeout=60,
        env=env if env is not None else _gate_env(overrides, project_dir))


def test_recovery_enabled_accepts_every_yaml_boolean_spelling() -> None:
    """PyYAML resolves True/yes/on (any case) as booleans, as the trainer
    reads this key; a textual `= true` would skip every recovery guard."""
    for spelling in ("true", "True", "TRUE", "yes", "Yes", "YES",
                     "on", "On", "ON"):
        completed = _run_gate(_RECOVERY_BLOCK, {
            "POLICY_RECOVERY_ENABLED": spelling, "MIN_FREE_DISK_GB": "0"})
        assert completed.returncode == 0, (spelling, completed.stderr)
        assert "__ACCEPTED__ recovery=true" in completed.stdout
    for spelling in ("false", "False", "FALSE", "no", "No", "NO",
                     "off", "Off", "OFF", ""):
        completed = _run_gate(_RECOVERY_BLOCK, {
            "POLICY_RECOVERY_ENABLED": spelling, "MIN_FREE_DISK_GB": "0"})
        assert completed.returncode == 0, (spelling, completed.stderr)
        assert "__ACCEPTED__ recovery=false" in completed.stdout


def test_recovery_enabled_rejects_non_boolean_garbage() -> None:
    completed = _run_gate(_RECOVERY_BLOCK, {
        "POLICY_RECOVERY_ENABLED": "ture", "MIN_FREE_DISK_GB": "0"})
    assert completed.returncode != 0
    assert "not a YAML boolean" in completed.stderr


def test_train_duration_gate_stays_inside_the_trainer_grammar() -> None:
    """Only strings parse_duration() itself raises on may be rejected."""
    for duration in ("2d", "4h", "30m", "10s", "1d12h", "48h",
                     "2 days", "45m30s", "24", "1.5", ".5", "2."):
        completed = _run_gate(_SESSION_BLOCK, {
            "MIN_FREE_DISK_GB": "0", "TRAIN_DURATION": duration})
        assert completed.returncode == 0, (duration, completed.stderr)
    for duration in ("4x", "d", "half a day", "day"):
        completed = _run_gate(_SESSION_BLOCK, {
            "MIN_FREE_DISK_GB": "0", "TRAIN_DURATION": duration})
        assert completed.returncode != 0, duration
        assert "TRAIN_DURATION" in completed.stderr


def test_min_free_disk_gb_must_be_a_non_negative_integer() -> None:
    for value in ("abc", "-5", "", "3.5", "10GB"):
        completed = _run_gate(_SESSION_BLOCK, {"MIN_FREE_DISK_GB": value})
        assert completed.returncode != 0, repr(value)
        assert "MIN_FREE_DISK_GB" in completed.stderr


def test_disk_floor_refuses_launch_below_the_floor(tmp_path: Path) -> None:
    """An absurd floor stands in for a full disk, measured with BSD df."""
    completed = _run_gate(_SESSION_BLOCK, {"MIN_FREE_DISK_GB": "999999999"},
                          project_dir=str(tmp_path))
    assert completed.returncode != 0
    assert "free on" in completed.stderr

    completed = _run_gate(_SESSION_BLOCK, {"MIN_FREE_DISK_GB": "1"},
                          project_dir=str(tmp_path))
    assert completed.returncode == 0, completed.stderr
    free_kb = re.search(r"free_kb=(\d+)", completed.stdout)
    assert free_kb and int(free_kb.group(1)) > 0, completed.stdout


def test_disk_floor_zero_disables_and_missing_df_degrades_to_warning(
    tmp_path: Path,
) -> None:
    completed = _run_gate(_SESSION_BLOCK, {"MIN_FREE_DISK_GB": "0"},
                          project_dir=str(tmp_path))
    assert completed.returncode == 0, completed.stderr
    assert "__ACCEPTED__" in completed.stdout

    # df/awk unreachable (a PATH with neither): warn, don't block the launch.
    env = _gate_env({"MIN_FREE_DISK_GB": "10"}, str(tmp_path))
    env["PATH"] = str(tmp_path)
    completed = _run_gate(_SESSION_BLOCK, {}, env=env)
    assert completed.returncode == 0, completed.stderr
    assert "Could not measure free disk space" in completed.stdout
    assert "free_kb=unset" in completed.stdout


@on_macos
@pytest.mark.parametrize(("setting", "value", "message"), [
    ("TRAIN_DURATION", '"4x"', "TRAIN_DURATION"),
    ("MIN_FREE_DISK_GB", "abc", "MIN_FREE_DISK_GB"),
    ("RESUME_LATEST", '"sometimes"', "RESUME_LATEST"),
    ("RESUME", '"models/missing.pt"', "Resume checkpoint not found"),
])
def test_mac_launcher_stops_at_a_gate_before_any_setup(
    tmp_path: Path, setting: str, value: str, message: str,
) -> None:
    """The whole script, from the top: a bad session setting stops it before
    the console log, conda, the Cython scan and torch -- in a scratch copy."""
    text = MAC_LAUNCHER.read_text(encoding="utf-8")
    assignment = re.compile(rf"^{setting}=\S*", re.MULTILINE)
    assert len(assignment.findall(text)) == 1, setting
    launcher = tmp_path / MAC_LAUNCHER.name
    launcher.write_text(assignment.sub(f"{setting}={value}", text),
                        encoding="utf-8")
    (tmp_path / "config").mkdir()
    shutil.copy(MAC_CONFIG, tmp_path / "config" / MAC_CONFIG.name)
    env = dict(os.environ)
    env.pop("_DAMA_ARM64_REEXEC", None)

    completed = subprocess.run(
        [SYSTEM_BASH, str(launcher)], capture_output=True, text=True,
        timeout=60, env=env, cwd=str(tmp_path))
    assert completed.returncode != 0
    assert message in completed.stderr, completed.stdout + completed.stderr
    assert not (tmp_path / "logs").exists()


# ---------------------------------------------------------------------------
# Platform and Rosetta guards
# ---------------------------------------------------------------------------

_ROSETTA_GUARD_END = (
    'echo "[warn] Still running under Rosetta after re-exec; '
    'continuing as x86_64."\nfi\n')


def _guard_env(shims: dict[str, Path], uname: str,
               translated: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("_DAMA_ARM64_REEXEC", None)
    env.update(FAKE_UNAME=uname, FAKE_TRANSLATED=translated,
               PATH=_path_with(shims["platform"]))
    return env


def _run_guard(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Everything from the top of the script through the Rosetta guard."""
    text = script.read_text(encoding="utf-8")
    block = text[:text.index(_ROSETTA_GUARD_END) + len(_ROSETTA_GUARD_END)]
    return subprocess.run(
        [SYSTEM_BASH, "-c", block + "\necho __PAST_GUARDS__\n",
         str(script), "first arg", "second"],
        capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.parametrize(("script", "counterpart"), MAC_SCRIPTS.items(),
                         ids=_SCRIPT_IDS)
def test_mac_scripts_refuse_other_platforms(
    shims: dict[str, Path], script: Path, counterpart: str,
) -> None:
    completed = _run_guard(script, _guard_env(shims, "Linux", "0"))
    assert completed.returncode != 0
    assert counterpart in completed.stderr
    assert "__PAST_GUARDS__" not in completed.stdout


@pytest.mark.parametrize("script", MAC_SCRIPTS, ids=_SCRIPT_IDS)
def test_rosetta_guard_reexecs_natively_with_the_same_arguments(
    shims: dict[str, Path], script: Path,
) -> None:
    completed = _run_guard(script, _guard_env(shims, "Darwin", "1"))
    assert completed.returncode == 0, completed.stderr
    assert (f"ARCH-SHIM reexec=1 -arm64 /bin/bash {script} first arg second"
            in completed.stdout)
    assert "__PAST_GUARDS__" not in completed.stdout


@pytest.mark.parametrize("script", MAC_SCRIPTS, ids=_SCRIPT_IDS)
def test_rosetta_guard_cannot_loop(shims: dict[str, Path], script: Path) -> None:
    """A re-exec that is somehow still translated continues instead."""
    env = _guard_env(shims, "Darwin", "1")
    env["_DAMA_ARM64_REEXEC"] = "1"
    completed = _run_guard(script, env)
    assert completed.returncode == 0, completed.stderr
    assert "ARCH-SHIM" not in completed.stdout
    assert "Still running under Rosetta" in completed.stdout
    assert "__PAST_GUARDS__" in completed.stdout


@pytest.mark.parametrize("script", MAC_SCRIPTS, ids=_SCRIPT_IDS)
def test_native_terminal_passes_the_guards_untouched(
    shims: dict[str, Path], script: Path,
) -> None:
    completed = _run_guard(script, _guard_env(shims, "Darwin", "0"))
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "__PAST_GUARDS__"


@on_macos
def test_rosetta_reexec_really_runs_the_script_as_arm64(tmp_path: Path) -> None:
    """Through real Rosetta, where the machine has it installed."""
    if subprocess.run(["arch", "-x86_64", "/usr/bin/true"],
                      capture_output=True).returncode != 0:
        pytest.skip("Rosetta is not installed")
    text = STOP_SCRIPT.read_text(encoding="utf-8")
    guards = text[:text.index(_ROSETTA_GUARD_END) + len(_ROSETTA_GUARD_END)]
    probe = tmp_path / "probe.sh"
    probe.write_text(guards + 'echo "ARCH=$(uname -m) ARGS=$*"\n',
                     encoding="utf-8")
    env = dict(os.environ)
    env.pop("_DAMA_ARM64_REEXEC", None)
    completed = subprocess.run(
        ["arch", "-x86_64", "/bin/bash", str(probe), "--kill"],
        capture_output=True, text=True, timeout=60, env=env)
    assert completed.returncode == 0, completed.stderr
    assert "re-running natively as arm64" in completed.stdout
    assert "ARCH=arm64 ARGS=--kill" in completed.stdout


# ---------------------------------------------------------------------------
# Portability helpers
# ---------------------------------------------------------------------------

def _realpath_via(shims: dict[str, Path], target: Path,
                  readlink_ok: bool, python_ok: bool) -> str:
    bin_dirs = [] if readlink_ok else [shims["readlink_without_f"]]
    bin_dirs.append(shims["python3" if python_ok else "broken_python3"])
    function = _function(MAC_LAUNCHER.read_text(encoding="utf-8"), "_realpath")
    completed = subprocess.run(
        [SYSTEM_BASH, "-c",
         "set -euo pipefail\n" + function + '_realpath "$1"\n', "sh", str(target)],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": _path_with(*bin_dirs)})
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_realpath_helper_falls_back_when_readlink_lacks_f(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = real_dir / "model_step_100.pt"
    real.write_bytes(b"x")
    link = tmp_path / "link.pt"
    link.symlink_to(real)
    expected = os.path.realpath(link)

    assert _realpath_via(shims, link, True, True) == expected
    assert _realpath_via(shims, link, False, True) == expected
    # Neither available: the (absolute) path as given, never an empty string.
    assert _realpath_via(shims, link, False, False) == str(link)


def test_sha256_helper_matches_hashlib_with_and_without_sha256sum(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "baseline.pt"
    payload.write_bytes(b"pinned recovery baseline")
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    function = _function(MAC_LAUNCHER.read_text(encoding="utf-8"), "_sha256")
    script = "set -euo pipefail\n" + function + '_sha256 "$1"\n'

    paths = [os.environ.get("PATH", "")]
    shasum = shutil.which("shasum")
    if shasum:
        # Only shasum reachable: the macOS-before-sha256sum case.
        only_shasum = tmp_path / "only_shasum"
        only_shasum.mkdir()
        (only_shasum / "shasum").symlink_to(shasum)
        paths.append(str(only_shasum))
    for path in paths:
        completed = subprocess.run(
            [SYSTEM_BASH, "-c", script, "sh", str(payload)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PATH": path})
        assert completed.returncode == 0, (path, completed.stderr)
        assert completed.stdout.split()[0].lower() == expected, path


# ---------------------------------------------------------------------------
# Accelerator gate: driven by the selected config's device.type
# ---------------------------------------------------------------------------

def _run_accelerator_check(tmp_path: Path, config_text: str):
    body = _extract(MAC_LAUNCHER.read_text(encoding="utf-8"),
                    "<<'PYCHECK' || {\n", "\nPYCHECK\n")
    body = body[len("<<'PYCHECK' || {\n"):]
    check = tmp_path / "check.py"
    check.write_text(body + "\n", encoding="utf-8")
    config = tmp_path / f"config_{uuid.uuid4().hex[:8]}.yaml"
    config.write_text(config_text, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-W", "ignore", str(check), str(config)],
        capture_output=True, text=True, env=env, timeout=300)


@on_macos
def test_accelerator_gate_follows_config_device_type(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("yaml")
    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()

    completed = _run_accelerator_check(tmp_path, 'device:\n  type: "cpu"\n')
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Accelerator: cpu" in completed.stdout

    completed = _run_accelerator_check(tmp_path, 'device:\n  type: "tpu"\n')
    assert completed.returncode != 0
    assert "not auto, cuda, mps or cpu" in completed.stdout

    # No device section means CUDA, exactly as config_from_yaml() reads it.
    for text in ('device:\n  type: "cuda"\n', "model:\n  channels: 128\n"):
        completed = _run_accelerator_check(tmp_path, text)
        if cuda_ok:
            assert completed.returncode == 0, completed.stdout
        else:
            assert completed.returncode != 0, text
            assert "macOS has no CUDA" in completed.stdout

    completed = _run_accelerator_check(tmp_path, 'device:\n  type: "auto"\n')
    assert completed.returncode == 0, completed.stdout + completed.stderr
    resolved = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    assert f"Accelerator: {resolved} (config device.type: auto)" in completed.stdout

    completed = _run_accelerator_check(
        tmp_path, MAC_CONFIG.read_text(encoding="utf-8"))
    if mps_ok:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Accelerator: mps" in completed.stdout
    else:
        assert completed.returncode != 0
        assert "MPS (Metal) is not available" in completed.stdout


# ---------------------------------------------------------------------------
# The Mac training config
# ---------------------------------------------------------------------------

def test_mac_config_targets_mps_with_the_local_architecture() -> None:
    yaml = pytest.importorskip("yaml")
    mac = yaml.safe_load(MAC_CONFIG.read_text(encoding="utf-8"))
    local = yaml.safe_load(
        (PROJECT_ROOT / "config" / "training_config_local_retrain.yaml")
        .read_text(encoding="utf-8"))

    assert mac["device"]["type"] in ("mps", "auto")
    assert mac["device"]["compile"]["enabled"] is False
    # Checkpoints stay interchangeable with the other local configs.
    assert mac["model"] == local["model"]
    # No sensors to read on macOS: an enabled check would never fire.
    assert mac["thermal_protection"]["enabled"] is False
    assert "recovery_experiment" not in mac


def test_mac_config_bounds_the_replay_files_kept_in_memory() -> None:
    """Every persisted self-play file stays parsed in the trainer's RAM until
    it is pruned (~0.5GB each here); the trainer default of 60 outgrows 36GB
    of unified memory. config_from_yaml() reads the key from dataloader."""
    yaml = pytest.importorskip("yaml")
    mac = yaml.safe_load(MAC_CONFIG.read_text(encoding="utf-8"))
    assert 0 < mac["dataloader"]["replay_max_files"] <= 10


def test_mac_config_outputs_never_collide_with_other_configs() -> None:
    """A Mac run must never resume from, or overwrite, another machine's run
    when models/ is copied across."""
    yaml = pytest.importorskip("yaml")
    mac_paths = yaml.safe_load(MAC_CONFIG.read_text(encoding="utf-8"))["paths"]
    defaults = {  # config_from_yaml() fallbacks for configs that omit a key
        "checkpoint_dir": "models/checkpoints", "latest_model": "models/latest.pt",
        "replay_dir": "data/replay", "log_dir": "logs",
        "stats_file": "models/training_stats.json",
    }
    for other in sorted((PROJECT_ROOT / "config").rglob("training_config*.yaml")):
        if other == MAC_CONFIG:
            continue
        other_paths = (yaml.safe_load(other.read_text(encoding="utf-8"))
                       or {}).get("paths") or {}
        for key, default in defaults.items():
            assert mac_paths[key] != other_paths.get(key, default), (other.name, key)

    # Where the game's model list looks (settings_dialog.discover_models).
    assert re.fullmatch(r"models/checkpoints[^/]*", mac_paths["checkpoint_dir"])
    assert re.fullmatch(r"models/latest[^/]*\.pt", mac_paths["latest_model"])


# ---------------------------------------------------------------------------
# eval_checkpoints_mac.sh
# ---------------------------------------------------------------------------

def _eval_env(tmp_path: Path, checkpoint_dir: Path,
              shims: dict[str, Path]) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("CONDA_ENV", "TRAINING_CONFIG", "FAKE_EVAL", "_DAMA_ARM64_REEXEC"):
        env.pop(name, None)
    # Marks the env active so the conda activation is skipped and the test
    # never depends on a conda install; this interpreter stands in for it.
    env["CONDA_DEFAULT_ENV"] = "dama"
    env["PATH"] = _path_with(shims["python3"])
    env.update({
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "RESULTS_FILE": str(tmp_path / "out" / "results.jsonl"),
        "PLOT_OUTPUT": str(tmp_path / "out" / "plot.png"),
        "LOCK_DIR": str(tmp_path / "out" / ".lock"),
        "TEST_STATS_DIR": str(tmp_path / "out" / "stats"),
    })
    return env


def _fake_evaluator(tmp_path: Path, shims: dict[str, Path], body: str,
                    env: dict[str, str]) -> None:
    """Answer eval_checkpoint_once with ``body``; every other python call is
    the real interpreter."""
    fake = tmp_path / "fake_eval.py"
    fake.write_text(textwrap.dedent(body), encoding="utf-8")
    env["FAKE_EVAL"] = str(fake)
    env["PATH"] = _path_with(shims["python3"])


_RECORD = """\
    import json, sys
    args = sys.argv[1:]
    step = int(args[args.index("--step") + 1])
    name = args[args.index("--checkpoint-name") + 1]
    print(json.dumps({
        "checkpoint": name, "step": step, "total_games": 100,
        "ml_wins": 40, "algo_wins": 50, "draws": 10, "ml_win_rate": 0.4,
        "ml_as_p1_win_rate": 0.5, "ml_as_p2_win_rate": 0.3,
        "avg_game_length": 42.0, "algo_difficulty": "easy",
        "timestamp": "t"}))
    """


@on_macos
def test_eval_checkpoints_handles_an_empty_directory(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    completed = subprocess.run(
        [SYSTEM_BASH, str(EVAL_SCRIPT), "--once"],
        capture_output=True, text=True, timeout=120,
        env=_eval_env(tmp_path, empty, shims), cwd=str(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"No checkpoints found in {empty}." in completed.stdout
    assert not (tmp_path / "out" / ".lock").exists()


@on_macos
def test_eval_checkpoints_walks_checkpoints_in_step_order(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    """The mapfile replacement: every checkpoint, in version order, once."""
    pytest.importorskip("matplotlib")
    checkpoints = tmp_path / "ckpts"
    checkpoints.mkdir()
    for step in (10, 2, 100, 1):
        (checkpoints / f"model_step_{step}.pt").write_bytes(b"")
    env = _eval_env(tmp_path, checkpoints, shims)
    _fake_evaluator(tmp_path, shims, _RECORD, env)

    completed = subprocess.run(
        [SYSTEM_BASH, str(EVAL_SCRIPT), "--once"],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    order = re.findall(r"EVAL: (model_step_\d+\.pt)", completed.stdout)
    assert order == [f"model_step_{step}.pt" for step in (1, 2, 10, 100)]
    assert "Evaluated 4 checkpoint(s)." in completed.stdout

    rerun = subprocess.run(
        [SYSTEM_BASH, str(EVAL_SCRIPT), "--once"],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert "No new checkpoints to evaluate." in rerun.stdout
    lines = (tmp_path / "out" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 4


@on_macos
def test_eval_checkpoints_runs_the_env_python_not_an_earlier_python3(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    """With the base env active and Homebrew ahead of it on PATH, `conda
    activate dama` leaves Homebrew's python3 (no torch, no yaml) first on
    PATH; only `python` is the env's."""
    pytest.importorskip("matplotlib")
    checkpoints = tmp_path / "ckpts"
    checkpoints.mkdir()
    (checkpoints / "model_step_1.pt").write_bytes(b"")
    env = _eval_env(tmp_path, checkpoints, shims)
    _fake_evaluator(tmp_path, shims, _RECORD, env)
    env["PATH"] = _path_with(shims["broken_python3"], shims["python3"])

    completed = subprocess.run(
        [SYSTEM_BASH, str(EVAL_SCRIPT), "--once"],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Evaluated 1 checkpoint(s)." in completed.stdout
    assert (tmp_path / "out" / "plot.png").is_file()


@on_macos
def test_eval_checkpoints_refuses_an_interpreter_without_the_packages(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    """Every evaluation would fail, and --once used to report only that no
    new checkpoint needed evaluating."""
    checkpoints = tmp_path / "ckpts"
    checkpoints.mkdir()
    (checkpoints / "model_step_1.pt").write_bytes(b"")
    env = _eval_env(tmp_path, checkpoints, shims)
    env["PATH"] = _path_with(shims["broken_python"])

    completed = subprocess.run(
        [SYSTEM_BASH, str(EVAL_SCRIPT), "--once"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
    assert completed.returncode != 0
    assert "cannot import torch and yaml" in completed.stderr
    assert "EVAL:" not in completed.stdout
    assert not (tmp_path / "out" / ".lock").exists()


@on_macos
def test_eval_checkpoints_once_fails_when_an_evaluation_fails(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    checkpoints = tmp_path / "ckpts"
    checkpoints.mkdir()
    (checkpoints / "model_step_1.pt").write_bytes(b"")
    env = _eval_env(tmp_path, checkpoints, shims)
    _fake_evaluator(tmp_path, shims, "raise SystemExit(1)\n", env)

    completed = subprocess.run(
        [SYSTEM_BASH, str(EVAL_SCRIPT), "--once"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "1 evaluation(s) failed." in completed.stdout
    assert "No new checkpoints to evaluate." not in completed.stdout
    assert not (tmp_path / "out" / ".lock").exists()


def _scratch_eval_project(tmp_path: Path, config_text: str) -> Path:
    """eval_checkpoints_mac.sh in a scratch project, so its PROJECT_DIR --
    and every default path derived from it -- is inside tmp_path."""
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    shutil.copy(EVAL_SCRIPT, project / EVAL_SCRIPT.name)
    (project / "config" / MAC_CONFIG.name).write_text(config_text,
                                                      encoding="utf-8")
    return project


def _default_eval_env(shims: dict[str, Path]) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("CHECKPOINT_DIR", "RESULTS_FILE", "PLOT_OUTPUT", "LOCK_DIR",
                 "TEST_STATS_DIR", "TRAINING_CONFIG", "CONDA_ENV", "FAKE_EVAL",
                 "_DAMA_ARM64_REEXEC"):
        env.pop(name, None)
    env["CONDA_DEFAULT_ENV"] = "dama"
    env["PATH"] = _path_with(shims["python3"])
    return env


@on_macos
def test_eval_checkpoints_defaults_follow_the_mac_config(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    """No override needed: the Mac config's checkpoint_dir is watched, and
    the results, plot and lock carry its _mac suffix."""
    pytest.importorskip("matplotlib")
    project = _scratch_eval_project(
        tmp_path, MAC_CONFIG.read_text(encoding="utf-8"))
    script = project / EVAL_SCRIPT.name
    models = project / "models"
    env = _default_eval_env(shims)

    completed = subprocess.run(
        [SYSTEM_BASH, str(script), "--once"], capture_output=True, text=True,
        timeout=120, env=env, cwd=str(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (f"No checkpoints found in {models / 'checkpoints_mac'}."
            in completed.stdout)

    # The results file is found under its Mac name, and the plot is written
    # under its Mac name.
    fake = tmp_path / "record.py"
    fake.write_text(textwrap.dedent(_RECORD), encoding="utf-8")
    record = subprocess.run(
        [sys.executable, str(fake), "--step", "1000", "--checkpoint-name",
         "model_step_001000.pt"], capture_output=True, text=True, check=True)
    (models / "eval_results_mac.jsonl").write_text(record.stdout)
    completed = subprocess.run(
        [SYSTEM_BASH, str(script), "--replot"], capture_output=True, text=True,
        timeout=300, env=env, cwd=str(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (models / "eval_progress_mac.png").is_file(), completed.stdout
    assert not (models / "eval_results.jsonl").exists()

    # The lock is the Mac one: held by a live process, it refuses a second run.
    lock = models / ".eval_checkpoints_mac.lock"
    lock.mkdir()
    (lock / "pid").write_text(f"{os.getpid()}\n")
    completed = subprocess.run(
        [SYSTEM_BASH, str(script), "--once"], capture_output=True, text=True,
        timeout=120, env=env, cwd=str(tmp_path))
    assert completed.returncode != 0
    assert f"already running (pid {os.getpid()})" in completed.stdout


@pytest.mark.parametrize(("paths_yaml", "directory", "suffix"), [
    ('paths:\n  checkpoint_dir: "models/checkpoints_trial"\n',
     "models/checkpoints_trial", "_trial"),
    # No key: config_from_yaml()'s default, still never the unsuffixed name.
    ("model:\n  channels: 128\n", "models/checkpoints", "_checkpoints"),
], ids=["checkpoints_trial", "trainer-default"])
@on_macos
def test_eval_checkpoints_follows_another_training_config(
    tmp_path: Path, shims: dict[str, Path], paths_yaml: str, directory: str,
    suffix: str,
) -> None:
    project = _scratch_eval_project(
        tmp_path, MAC_CONFIG.read_text(encoding="utf-8"))
    (project / "config" / "other.yaml").write_text(paths_yaml, encoding="utf-8")
    env = _default_eval_env(shims)
    env["TRAINING_CONFIG"] = "config/other.yaml"
    lock = project / "models" / f".eval_checkpoints{suffix}.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(f"{os.getpid()}\n")

    completed = subprocess.run(
        [SYSTEM_BASH, str(project / EVAL_SCRIPT.name), "--once"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
    assert completed.returncode != 0
    assert "already running" in completed.stdout
    assert (project / directory).is_dir()

    lock.joinpath("pid").unlink()
    lock.rmdir()
    completed = subprocess.run(
        [SYSTEM_BASH, str(project / EVAL_SCRIPT.name), "--once"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"No checkpoints found in {project / directory}." in completed.stdout


@on_macos
def test_eval_checkpoints_refuses_a_missing_training_config(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    project = _scratch_eval_project(tmp_path, "")
    env = _default_eval_env(shims)
    env["TRAINING_CONFIG"] = "config/missing.yaml"
    completed = subprocess.run(
        [SYSTEM_BASH, str(project / EVAL_SCRIPT.name), "--once"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
    assert completed.returncode != 0
    assert "Training config not found" in completed.stderr
    assert not (project / "models").exists()


@on_macos
def test_eval_checkpoints_ctrl_c_stops_the_run_and_releases_the_lock(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    """Ctrl+C mid-evaluation ends the run. It used to only release the lock:
    the interrupted evaluation was reported as failed and the next
    checkpoint started, now without a lock."""
    checkpoints = tmp_path / "ckpts"
    checkpoints.mkdir()
    for step in (1, 2):
        (checkpoints / f"model_step_{step}.pt").write_bytes(b"")
    started = tmp_path / "started"
    env = _eval_env(tmp_path, checkpoints, shims)
    _fake_evaluator(tmp_path, shims, f"""\
        import pathlib, time
        pathlib.Path({str(started)!r}).touch()
        time.sleep(120)
        """, env)
    lock = tmp_path / "out" / ".lock"

    proc = subprocess.Popen(
        [SYSTEM_BASH, str(EVAL_SCRIPT), "--once"], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, env=env, cwd=str(tmp_path),
        start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while not started.exists():
            assert time.monotonic() < deadline, "evaluation never started"
            assert proc.poll() is None, proc.communicate()[0]
            time.sleep(0.1)
        assert lock.is_dir()
        # What Ctrl+C does: SIGINT to the whole foreground process group.
        os.killpg(proc.pid, signal.SIGINT)
        output = proc.communicate(timeout=60)[0]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
    assert proc.returncode == 130, output
    assert output.count("EVAL: ") == 1, output
    assert not lock.exists()


# ---------------------------------------------------------------------------
# stop_training_mac.sh: spawn-started workers
# ---------------------------------------------------------------------------

_TRAINER_PATTERNS = r"dama\.ai\.ml\.trainer|micro-trainer|micro_trainer"
_LAUNCHER_PATTERNS = r"|script\.sh|train_server\.sh|bash .*local_train_mac\.sh"

_FAKE_TRAINER = """\
import multiprocessing as mp, signal, sys, time
from concurrent.futures import ProcessPoolExecutor

def work(_):
    time.sleep(600)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    # A trainer that never finishes stopping: SIGTERM does not stop it within
    # the STOP_TIMEOUT the tests give stop_training_mac.sh, so it ends up
    # SIGKILLed.
    signal.signal(signal.SIGTERM, lambda *_: None)
    pool = ProcessPoolExecutor(max_workers=2)
    futures = [pool.submit(work, i) for i in range(2)]
    while len(pool._processes) < 2:
        time.sleep(0.1)
    with open(sys.argv[2], "w") as handle:
        handle.write(" ".join(str(pid) for pid in pool._processes))
    time.sleep(600)
"""


def _safe_stop_copy(tmp_path: Path) -> tuple[Path, str, str]:
    """stop_training_mac.sh with the real trainer and launcher patterns
    swapped for unique markers, so it can never touch an actual training run
    on the machine running the tests. Returns the copy, the trainer marker,
    and the launcher's file name. Its PROJECT_DIR is tmp_path."""
    marker = f"zzfaketrainer{uuid.uuid4().hex[:12]}"
    # Unrelated to the trainer marker, which would match the file name too.
    launcher_name = f"zzfakelauncher{uuid.uuid4().hex[:12]}.sh"
    text = STOP_SCRIPT.read_text(encoding="utf-8")
    assert text.count(_TRAINER_PATTERNS) >= 5, \
        "trainer patterns changed; update this test"
    assert _TRAINER_PATTERNS + _LAUNCHER_PATTERNS in text, \
        "launcher patterns changed; update this test"
    text = text.replace(_LAUNCHER_PATTERNS,
                        "|bash .*" + launcher_name.replace(".", r"\."))
    text = text.replace(_TRAINER_PATTERNS, marker)
    stop_copy = tmp_path / STOP_SCRIPT.name
    stop_copy.write_text(text, encoding="utf-8")
    return stop_copy, marker, launcher_name


def _start_fake_trainer(tmp_path: Path, name: str, argv0: str,
                        env: dict[str, str]) -> tuple[subprocess.Popen, list[int]]:
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(_FAKE_TRAINER, encoding="utf-8")
    pid_file = tmp_path / f"{name}_workers.txt"
    trainer = subprocess.Popen([sys.executable, str(fake), argv0, str(pid_file)],
                               env=env)
    deadline = time.monotonic() + 60
    while not pid_file.exists() or not pid_file.read_text().strip():
        if time.monotonic() > deadline or trainer.poll() is not None:
            trainer.kill()
            pytest.fail("fake trainer never started its workers")
        time.sleep(0.2)
    return trainer, [int(pid) for pid in pid_file.read_text().split()]


def _reap(trainers: list[subprocess.Popen], workers: list[int]) -> None:
    for trainer in trainers:
        if trainer.poll() is None:
            trainer.kill()
            trainer.wait(timeout=30)
    for pid in workers:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


@on_macos
def test_stop_training_reaps_spawned_workers_of_a_killed_trainer(
    tmp_path: Path,
) -> None:
    """Spawned workers match neither the module path nor the process title,
    and they outlive a SIGKILLed parent indefinitely."""
    stop_copy, marker, _ = _safe_stop_copy(tmp_path)
    text = stop_copy.read_text(encoding="utf-8")
    functions = tmp_path / "stop_functions.sh"
    functions.write_text(
        _extract(text, "# Get the directory where this script is located",
                 'case "${1:-}" in'), encoding="utf-8")

    trainer, workers = _start_fake_trainer(tmp_path, "trainer", marker,
                                           dict(os.environ))
    try:
        listing = subprocess.run(
            [SYSTEM_BASH, "-c", f'. "{functions}"; list_spawned_workers'],
            capture_output=True, text=True, timeout=60)
        assert set(workers) <= {int(pid) for pid in listing.stdout.split()}

        completed = subprocess.run(
            [SYSTEM_BASH, str(stop_copy), "--kill"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "STOP_TIMEOUT": "3"})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "force-killing it" in completed.stdout
        trainer.wait(timeout=30)

        deadline = time.monotonic() + 10
        while any(_alive(pid) for pid in workers) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not [pid for pid in workers if _alive(pid)], completed.stdout
        assert "Stopped 2 leftover self-play worker(s)." in completed.stdout
        assert "Training session stopped." in completed.stdout
    finally:
        _reap([trainer], workers)


def _ppid(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else -1


@on_macos
def test_stop_training_reaps_orphans_of_this_checkout_only(
    tmp_path: Path,
) -> None:
    """A trainer killed some other way (kill -9, Activity Monitor, a crash)
    left its workers to launchd before stop_training_mac.sh ever ran. They
    are found by this checkout's PYTHONPATH; other orphans are left alone."""
    stop_copy, _, _ = _safe_stop_copy(tmp_path)
    ours_env = {**os.environ, "PYTHONPATH": f"{tmp_path}/src:"}
    theirs_env = {**os.environ, "PYTHONPATH": f"{tmp_path}/src2:"}
    trainers: list[subprocess.Popen] = []
    workers: list[int] = []
    try:
        ours, our_workers = _start_fake_trainer(
            tmp_path, "ours", "zzorphan-ours", ours_env)
        trainers.append(ours)
        workers += our_workers
        theirs, their_workers = _start_fake_trainer(
            tmp_path, "theirs", "zzorphan-theirs", theirs_env)
        trainers.append(theirs)
        workers += their_workers
        for trainer in trainers:
            trainer.kill()
            trainer.wait(timeout=30)
        deadline = time.monotonic() + 10
        while any(_ppid(pid) != 1 for pid in workers):
            assert time.monotonic() < deadline, "workers were not orphaned"
            time.sleep(0.1)

        completed = subprocess.run(
            [SYSTEM_BASH, str(stop_copy), "--kill"],
            capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        # Two workers; the orphaned resource tracker is not signalled (it
        # ignores SIGTERM and exits once the workers are gone).
        assert "Stopped 2 leftover self-play worker(s)." in completed.stdout

        deadline = time.monotonic() + 10
        while any(_alive(pid) for pid in our_workers) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not [pid for pid in our_workers if _alive(pid)]
        assert all(_alive(pid) for pid in their_workers)
    finally:
        _reap(trainers, workers)


@on_macos
def test_stop_training_all_stops_a_launcher_still_in_its_checks(
    tmp_path: Path, shims: dict[str, Path],
) -> None:
    """A launcher still in its checks would start a trainer right after the
    stop; an editor or `tail -f` on the same file must be left alone."""
    stop_copy, _, launcher_name = _safe_stop_copy(tmp_path)
    launcher = tmp_path / launcher_name
    launcher.write_text("while :; do sleep 0.2; done\n", encoding="utf-8")
    running = subprocess.Popen([SYSTEM_BASH, str(launcher)])
    tail = subprocess.Popen(["tail", "-f", str(launcher)],
                            stdout=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        completed = subprocess.run(
            [SYSTEM_BASH, str(stop_copy)], capture_output=True, text=True,
            timeout=120, env={**os.environ, "PATH": _path_with(shims["no_crontab"])})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "No cron entry found for runner.sh." in completed.stdout
        assert "Training processes stopped." in completed.stdout
        assert running.wait(timeout=10) == -signal.SIGTERM
        assert tail.poll() is None
    finally:
        for proc in (running, tail):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)


_SLOW_STOPPING_TRAINER = """\
import signal, sys, time

def stop(*_):
    # The real trainer stops at the next batch, waits for self-play and saves a
    # final checkpoint first -- longer than stop_training.sh's 3 seconds.
    time.sleep(5)
    open(sys.argv[2], "w").close()
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
open(sys.argv[3], "w").close()
time.sleep(600)
"""


@on_macos
def test_stop_training_lets_the_trainer_save_before_it_exits(
    tmp_path: Path,
) -> None:
    stop_copy, marker, _ = _safe_stop_copy(tmp_path)
    fake = tmp_path / "slow_trainer.py"
    fake.write_text(_SLOW_STOPPING_TRAINER, encoding="utf-8")
    saved = tmp_path / "final_checkpoint"
    ready = tmp_path / "ready"
    trainer = subprocess.Popen(
        [sys.executable, str(fake), marker, str(saved), str(ready)])
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert time.monotonic() < deadline, "fake trainer never started"
            assert trainer.poll() is None
            time.sleep(0.1)

        completed = subprocess.run(
            [SYSTEM_BASH, str(stop_copy), "--kill"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "STOP_TIMEOUT": "60"})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert trainer.wait(timeout=30) == 0
        assert saved.exists()
        assert "force-killing" not in completed.stdout
        assert "Training session stopped." in completed.stdout
    finally:
        if trainer.poll() is None:
            trainer.kill()
            trainer.wait(timeout=30)


def test_stop_training_with_nothing_running(tmp_path: Path) -> None:
    stop_copy, _, _ = _safe_stop_copy(tmp_path)
    completed = subprocess.run(
        [SYSTEM_BASH, str(stop_copy), "--kill"],
        capture_output=True, text=True, timeout=120,
        env={k: v for k, v in os.environ.items() if k != "_DAMA_ARM64_REEXEC"})
    if sys.platform != "darwin":
        assert completed.returncode != 0
        return
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "No active training session found."
