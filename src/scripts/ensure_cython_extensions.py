"""Fail-closed readiness guard for locally built Cython extensions.

The project does not ship platform-specific extension binaries.  Launchers use
this helper with the same interpreter that will run the trainer so a missing,
wrong-ABI, or stale local build cannot silently select Python fallbacks.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time


BuildRunner = Callable[[Sequence[str], Path], None]

# These modules are the accelerator contract. Avoid recursively walking the
# whole source tree on drvfs during every launcher preflight, and make adding a
# new required binary an explicit reviewable change.
_EXTENSION_SOURCES = (
    Path("dama/ai/algorithmic/_fast_search.pyx"),
    Path("dama/ai/ml/_fast_encode.pyx"),
    Path("dama/ai/ml/_fast_score.pyx"),
    Path("dama/ai/ml/_fast_stat.pyx"),
)
_BUILD_DEPENDENCIES = (
    Path("setup_cython.py"),
    Path("scripts/setup_cython.py"),
    Path("scripts/ensure_cython_extensions.py"),
)
_BUILD_MANIFEST = Path("build/cython_extensions.v2.json")
_BUILD_MANIFEST_VERSION = 2
_BUILD_LOCK = Path("build/cython_extensions.lock")
_BUILD_LOCK_PARENT_PID_ENV = "_DAMA_CYTHON_BUILD_LOCK_PARENT_PID"
_BUILD_LOCK_ROOT_ENV = "_DAMA_CYTHON_BUILD_LOCK_ROOT"
_BUILD_DISTRIBUTIONS = ("Cython", "numpy", "setuptools")
_BUILD_ENVIRONMENT_VARIABLES = (
    "ARCHFLAGS",
    "AR",
    "ARFLAGS",
    "CC",
    "CFLAGS",
    "CPP",
    "CPPFLAGS",
    "CXX",
    "CXXFLAGS",
    "LDCXXSHARED",
    "LDFLAGS",
    "LDSHARED",
    "RANLIB",
    # GCC/Clang and MSVC also consume these directly, without setuptools
    # adding them to its command line or changing the compiler's banner.
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "INCLUDE",
    "LIB",
    "CL",
    "_CL_",
    "LINK",
    "_LINK_",
)
_BUILD_SYSCONFIG_VARIABLES = (
    "AR",
    "ARFLAGS",
    "CC",
    "CCSHARED",
    "CFLAGS",
    "CPPFLAGS",
    "CXX",
    "LDCXXSHARED",
    "LDFLAGS",
    "LDSHARED",
    "SHLIB_SUFFIX",
    "SOABI",
)
_ACTIVE_BUILD_LOCK_ROOTS: set[Path] = set()


@dataclass(frozen=True)
class ExtensionStatus:
    """Expected extension targets and the subset requiring a rebuild."""

    source_root: Path
    modules: tuple[str, ...]
    targets: tuple[Path, ...]
    stale_targets: tuple[Path, ...]

    @property
    def needs_rebuild(self) -> bool:
        return bool(self.stale_targets)


def _default_build_runner(command: Sequence[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment.pop(_BUILD_LOCK_PARENT_PID_ENV, None)
    environment.pop(_BUILD_LOCK_ROOT_ENV, None)
    root = cwd.resolve()
    if root in _ACTIVE_BUILD_LOCK_ROOTS:
        # The compatibility entrypoint also locks direct builds. Tell this
        # immediate child that its parent already owns the same lock so a
        # launcher-triggered rebuild does not deadlock by reacquiring it.
        environment[_BUILD_LOCK_PARENT_PID_ENV] = str(os.getpid())
        environment[_BUILD_LOCK_ROOT_ENV] = os.fspath(root)
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def parent_holds_cython_build_lock(source_root: Path) -> bool:
    """Return whether this process is a guarded build child of the lock owner."""
    parent_pid = os.environ.get(_BUILD_LOCK_PARENT_PID_ENV)
    parent_root = os.environ.get(_BUILD_LOCK_ROOT_ENV)
    if parent_pid != str(os.getppid()) or not parent_root:
        return False
    try:
        return Path(parent_root).resolve() == source_root.resolve()
    except OSError:
        return False


@contextmanager
def cython_build_lock(source_root: Path):
    """Serialize launcher readiness checks and any rebuild they trigger."""
    lock_path = source_root.resolve() / _BUILD_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()

        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (
                        errno.EACCES,
                        errno.EAGAIN,
                        errno.EDEADLK,
                    ):
                        raise
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)

        root = source_root.resolve()
        _ACTIVE_BUILD_LOCK_ROOTS.add(root)
        try:
            yield
        finally:
            _ACTIVE_BUILD_LOCK_ROOTS.discard(root)
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _extension_layout(
    source_root: Path,
    extension_suffix: str,
    platform_system: str,
) -> tuple[tuple[Path, ...], tuple[str, ...], tuple[Path, ...]]:
    sources = tuple(
        source_root / relative for relative in _EXTENSION_SOURCES
        if not (
            platform_system in ("Windows", "Darwin")
            and relative.stem == "_fast_stat"
        )
    )
    modules = tuple(
        ".".join(source.relative_to(source_root).with_suffix("").parts)
        for source in sources
    )
    targets = tuple(
        source.with_name(f"{source.stem}{extension_suffix}")
        for source in sources
    )
    return sources, modules, targets


@contextmanager
def _regular_file_reader(path: Path, mode: str, *, encoding: str | None = None):
    """Reject nonregular build artifacts without waiting on a FIFO open."""
    def open_nonblocking(name: str, flags: int) -> int:
        return os.open(name, flags | getattr(os, "O_NONBLOCK", 0))

    with open(path, mode, encoding=encoding, opener=open_nonblocking) as stream:
        # Validate the opened descriptor so a path replacement cannot turn a
        # preceding regular-file check into a blocking or unbounded read.
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError(errno.EINVAL, "Expected a regular build file", str(path))
        yield stream


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file_reader(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in _BUILD_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _compiler_banner(command: str | None) -> dict[str, object] | None:
    """Return a bounded identity for one configured compiler command."""
    if not command:
        return None
    try:
        arguments = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        arguments = []
    if not arguments:
        return {"command": command, "probe": None, "returncode": None}

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    try:
        completed = subprocess.run(
            [*arguments, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError):
        return {"command": command, "probe": None, "returncode": None}

    # Version tools occasionally print configuration essays. Four non-empty
    # lines are enough to distinguish the driver without bloating the manifest.
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "command": command,
        "probe": lines[:4],
        "returncode": completed.returncode,
    }


def _native_cpu_identity(
    *, platform_system: str, platform_machine: str,
) -> dict[str, object]:
    """Fingerprint CPU features that can change ``-march=native`` output."""
    identity: dict[str, object] = {
        "machine": platform_machine,
        "processor": platform.processor(),
    }
    if (
        platform_system != "Linux"
        or platform_machine not in ("x86_64", "AMD64")
    ):
        return identity

    try:
        first_processor = Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).split("\n\n", 1)[0]
    except OSError:
        return identity
    fields = {}
    for line in first_processor.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    for key in ("vendor_id", "cpu family", "model", "stepping", "model name"):
        if key in fields:
            identity[key] = fields[key]
    feature_text = fields.get("flags") or fields.get("Features")
    if feature_text:
        identity["features"] = sorted(feature_text.split())
    return identity


def build_environment_fingerprint(
    *,
    platform_system: str | None = None,
    platform_machine: str | None = None,
) -> dict[str, object]:
    """Return build-tool and machine inputs that live outside the source tree."""
    system = platform.system() if platform_system is None else platform_system
    machine = platform.machine() if platform_machine is None else platform_machine
    sysconfig_values = {
        key: sysconfig.get_config_var(key)
        for key in _BUILD_SYSCONFIG_VARIABLES
    }
    compiler_commands = {
        "CC": os.environ.get("CC") or sysconfig_values.get("CC"),
        "CXX": os.environ.get("CXX") or sysconfig_values.get("CXX"),
    }
    # setuptools accepts independent shared-linker commands. Probe them only
    # when overridden: otherwise the effective driver is already represented
    # by CC/CXX, and repeating their usually long sysconfig flag strings adds
    # launcher work without adding identity.
    for name in ("LDSHARED", "LDCXXSHARED"):
        if os.environ.get(name):
            compiler_commands[name] = os.environ[name]
    return {
        "python_version": sys.version,
        "distributions": _distribution_versions(),
        "sysconfig": sysconfig_values,
        "environment": {
            key: os.environ.get(key)
            for key in _BUILD_ENVIRONMENT_VARIABLES
        },
        "compiler_banners": {
            key: _compiler_banner(command)
            for key, command in compiler_commands.items()
        },
        "cpu": _native_cpu_identity(
            platform_system=system, platform_machine=machine),
    }


def build_input_fingerprints(
    source_root: Path,
    *,
    platform_system: str | None = None,
) -> dict[str, str]:
    """Return the exact source and recipe hashes consumed by a build."""
    root = source_root.resolve()
    system = platform.system() if platform_system is None else platform_system
    sources, _, _ = _extension_layout(root, "", system)
    inputs = (*sources, *(root / path for path in _BUILD_DEPENDENCIES))
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in inputs
    }


def _build_manifest_payload(
    source_root: Path,
    *,
    extension_suffix: str,
    platform_system: str,
) -> dict:
    _, modules, targets = _extension_layout(
        source_root, extension_suffix, platform_system)
    return {
        "version": _BUILD_MANIFEST_VERSION,
        "python_cache_tag": sys.implementation.cache_tag,
        "extension_suffix": extension_suffix,
        "platform_system": platform_system,
        "platform_machine": platform.machine(),
        "modules": list(modules),
        "build_environment": build_environment_fingerprint(
            platform_system=platform_system,
            platform_machine=platform.machine(),
        ),
        "inputs": build_input_fingerprints(
            source_root, platform_system=platform_system),
        "outputs": {
            path.relative_to(source_root).as_posix(): _sha256_file(path)
            for path in targets
        },
    }


def _manifest_matches(
    source_root: Path,
    *,
    extension_suffix: str,
    platform_system: str,
) -> bool:
    manifest_path = source_root / _BUILD_MANIFEST
    try:
        with _regular_file_reader(manifest_path, "r", encoding="utf-8") as stream:
            stored = json.load(stream)
        expected = _build_manifest_payload(
            source_root,
            extension_suffix=extension_suffix,
            platform_system=platform_system,
        )
    except (OSError, UnicodeError, ValueError, TypeError):
        return False
    return stored == expected


def write_build_manifest(
    source_root: Path,
    *,
    extension_suffix: str | None = None,
    platform_system: str | None = None,
    expected_inputs: dict[str, str] | None = None,
    expected_environment: dict[str, object] | None = None,
) -> Path:
    """Atomically record exact build inputs and published extension bytes."""
    root = source_root.resolve()
    suffix = (
        sysconfig.get_config_var("EXT_SUFFIX")
        if extension_suffix is None
        else extension_suffix
    )
    if not suffix:
        raise RuntimeError("The selected Python did not report EXT_SUFFIX.")
    system = platform.system() if platform_system is None else platform_system
    payload = _build_manifest_payload(
        root, extension_suffix=suffix, platform_system=system)
    if expected_inputs is not None and payload["inputs"] != expected_inputs:
        changed = sorted(
            key for key in set(payload["inputs"]) | set(expected_inputs)
            if payload["inputs"].get(key) != expected_inputs.get(key)
        )
        raise RuntimeError(
            "Cython build inputs changed while the build was running; "
            "content manifest was not written: " + ", ".join(changed)
        )
    if (
        expected_environment is not None
        and payload["build_environment"] != expected_environment
    ):
        raise RuntimeError(
            "Cython build environment changed while the build was running; "
            "content manifest was not written."
        )
    manifest_path = root / _BUILD_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        dir=manifest_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        try:
            # Retain raw ownership if stream construction partially succeeds
            # before raising; a wrapper must not release a reusable fd number.
            stream = os.fdopen(
                fd, "w", encoding="utf-8", newline="\n", closefd=False,
            )
            try:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                # Buffered close can fail too; retain the original write or
                # interruption error while still releasing the stream.
                try:
                    stream.close()
                except OSError:
                    pass
                raise
            else:
                stream.close()
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        else:
            # Close once before publication, including on native Windows.
            os.close(fd)
        os.replace(temporary, manifest_path)
        try:
            directory_fd = os.open(
                manifest_path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Native Windows and some mounted filesystems cannot fsync a
            # directory. The same-directory replacement is still atomic.
            pass
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return manifest_path


def extension_status(
    source_root: Path,
    *,
    extension_suffix: str | None = None,
    platform_system: str | None = None,
) -> ExtensionStatus:
    """Return exact-interpreter extension targets that are missing or stale."""
    root = source_root.resolve()
    suffix = (
        sysconfig.get_config_var("EXT_SUFFIX")
        if extension_suffix is None
        else extension_suffix
    )
    if not suffix:
        raise RuntimeError("The selected Python did not report EXT_SUFFIX.")

    system = platform.system() if platform_system is None else platform_system
    sources, modules, targets = _extension_layout(root, suffix, system)
    missing_sources = tuple(source for source in sources if not source.is_file())
    if missing_sources:
        names = ", ".join(str(path) for path in missing_sources)
        raise RuntimeError(f"Required Cython source is missing: {names}")

    dependencies = tuple(root / relative for relative in _BUILD_DEPENDENCIES)
    for dependency in dependencies:
        if not dependency.is_file():
            raise RuntimeError(f"Cython build dependency is missing: {dependency}")

    stale_targets: list[Path] = []
    recipe_mtime = max(path.stat().st_mtime_ns for path in dependencies)
    for source, target in zip(sources, targets, strict=True):
        try:
            target_mtime = target.stat().st_mtime_ns
        except OSError:
            stale_targets.append(target)
            continue
        if max(source.stat().st_mtime_ns, recipe_mtime) > target_mtime:
            stale_targets.append(target)

    # Timestamps are a quick rebuild signal, not source identity. Archives,
    # rsync, and explicit utime calls can preserve an old mtime across changed
    # bytes. The content manifest also catches modified/truncated binaries.
    if not stale_targets and not _manifest_matches(
        root, extension_suffix=suffix, platform_system=system):
        stale_targets.extend(targets)

    return ExtensionStatus(
        source_root=root,
        modules=modules,
        targets=targets,
        stale_targets=tuple(stale_targets),
    )


def ensure_cython_extensions(
    source_root: Path,
    *,
    extension_suffix: str | None = None,
    platform_system: str | None = None,
    build_runner: BuildRunner = _default_build_runner,
    verify_imports: bool = True,
) -> ExtensionStatus:
    """Rebuild stale targets once, then verify every accelerator imports."""
    with cython_build_lock(source_root):
        # Recheck only after acquiring the lock. Another launcher may have
        # completed the required build while this process was waiting.
        status = extension_status(
            source_root,
            extension_suffix=extension_suffix,
            platform_system=platform_system,
        )
        if status.needs_rebuild:
            command = (
                sys.executable,
                str(status.source_root / "setup_cython.py"),
                "build_ext",
                "--inplace",
                "--force",
            )
            build_runner(command, status.source_root)
            status = extension_status(
                source_root,
                extension_suffix=extension_suffix,
                platform_system=platform_system,
            )
            if status.needs_rebuild:
                names = ", ".join(str(path) for path in status.stale_targets)
                raise RuntimeError(
                    "Cython build completed without publishing current targets: "
                    f"{names}"
                )

        if verify_imports:
            importlib.invalidate_caches()
            for module, target in zip(status.modules, status.targets, strict=True):
                imported = importlib.import_module(module)
                location = Path(imported.__file__ or "").resolve()
                expected = target.resolve()
                if location != expected:
                    raise RuntimeError(
                        f"{module} imported from {location}, expected {expected}."
                    )
        return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing setup_cython.py and the dama package.",
    )
    args = parser.parse_args(argv)
    status = ensure_cython_extensions(args.source_root)
    print(f"Cython extensions ready for {sys.executable}:")
    for target in status.targets:
        print(f"  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
