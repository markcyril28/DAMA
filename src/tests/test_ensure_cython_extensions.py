"""Contracts for the cross-platform Cython extension readiness guard."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Lock
from time import sleep
from types import SimpleNamespace

import pytest

import setup_cython as build_entrypoint
import scripts.ensure_cython_extensions as cython_guard
from scripts.ensure_cython_extensions import (
    build_environment_fingerprint,
    build_input_fingerprints,
    ensure_cython_extensions,
    extension_status,
    write_build_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_direct_build_overlap_probe(
    start,
    target_name: str,
    active_name: str,
    overlap_name: str,
    arguments: tuple[str, ...],
) -> None:
    """Run one simulated direct build in a fresh process."""
    import setup_cython as child_entrypoint
    import scripts.ensure_cython_extensions as child_guard

    target = Path(target_name)
    active = Path(active_name)
    overlap = Path(overlap_name)
    status = SimpleNamespace(targets=(target,), needs_rebuild=True)

    child_guard.extension_status = lambda _root: status
    child_guard.build_input_fingerprints = lambda _root: {"input": "stable"}
    child_guard.build_environment_fingerprint = lambda: {"env": "stable"}

    def publish_manifest(*_args, **_kwargs):
        assert arguments == ("build_ext", "--inplace", "--force")
        return target.parent

    child_guard.write_build_manifest = publish_manifest

    def simulated_build(*_args, **_kwargs):
        try:
            active.mkdir()
            owns_marker = True
        except FileExistsError:
            overlap.touch()
            owns_marker = False
        sleep(0.25)
        replacement = target.with_name(f"{target.name}.{os.getpid()}.new")
        replacement.write_bytes(str(os.getpid()).encode("ascii"))
        os.replace(replacement, target)
        if owns_marker:
            active.rmdir()

    child_entrypoint.runpy.run_path = simulated_build
    child_entrypoint.sys.argv = ["setup_cython.py", *arguments]
    start.wait(timeout=5)
    child_entrypoint.main()


def _source_tree(root: Path) -> Path:
    source_root = root / "src"
    for relative in (
        "dama/ai/algorithmic/_fast_search.pyx",
        "dama/ai/ml/_fast_encode.pyx",
        "dama/ai/ml/_fast_score.pyx",
        "dama/ai/ml/_fast_stat.pyx",
        "setup_cython.py",
        "scripts/setup_cython.py",
        "scripts/ensure_cython_extensions.py",
    ):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    return source_root


def test_status_uses_exact_abi_suffix_and_excludes_posix_module_on_windows(
    tmp_path: Path,
) -> None:
    source_root = _source_tree(tmp_path)
    search = source_root / "dama/ai/algorithmic/_fast_search.pyx"
    wrong_abi = search.with_name("_fast_search.cp310-win_amd64.pyd")
    wrong_abi.write_bytes(b"wrong ABI")

    status = extension_status(
        source_root,
        extension_suffix=".cp311-win_amd64.pyd",
        platform_system="Windows",
    )

    assert len(status.modules) == 3
    assert "dama.ai.ml._fast_stat" not in status.modules
    assert len(status.stale_targets) == 3
    assert wrong_abi not in status.targets
    assert search.with_name("_fast_search.cp311-win_amd64.pyd") in (
        status.stale_targets
    )


def test_recipe_change_rebuilds_once_and_rechecks_every_target(
    tmp_path: Path,
) -> None:
    source_root = _source_tree(tmp_path)
    suffix = ".cp311-win_amd64.pyd"
    initial = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    calls = []

    def publish(command, cwd):
        calls.append((tuple(command), cwd))
        newest_dependency = max(
            path.stat().st_mtime_ns
            for path in (
                source_root / "setup_cython.py",
                source_root / "scripts/setup_cython.py",
                *source_root.rglob("*.pyx"),
            )
        )
        for target in initial.targets:
            target.write_bytes(b"extension")
            os.utime(
                target,
                ns=(newest_dependency + 1_000_000, newest_dependency + 1_000_000),
            )
        write_build_manifest(
            source_root,
            extension_suffix=suffix,
            platform_system="Windows",
        )

    status = ensure_cython_extensions(
        source_root,
        extension_suffix=suffix,
        platform_system="Windows",
        build_runner=publish,
        verify_imports=False,
    )

    assert not status.needs_rebuild
    assert len(calls) == 1
    command, cwd = calls[0]
    assert command[0] == sys.executable
    assert command[-3:] == ("build_ext", "--inplace", "--force")
    assert cwd == source_root.resolve()

    second = ensure_cython_extensions(
        source_root,
        extension_suffix=suffix,
        platform_system="Windows",
        build_runner=publish,
        verify_imports=False,
    )
    assert not second.needs_rebuild
    assert len(calls) == 1


def test_concurrent_guards_serialize_one_stale_build(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    suffix = ".cp311-win_amd64.pyd"
    targets = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows"
    ).targets
    start = Barrier(2)
    publication = Lock()
    calls = []

    def publish(command, cwd):
        calls.append((tuple(command), cwd))
        sleep(0.1)
        with publication:
            newest_dependency = max(
                path.stat().st_mtime_ns
                for path in source_root.rglob("*")
                if path.is_file()
            )
            for target in targets:
                target.write_bytes(b"extension")
                os.utime(
                    target,
                    ns=(newest_dependency + 1_000_000,) * 2,
                )
            write_build_manifest(
                source_root,
                extension_suffix=suffix,
                platform_system="Windows",
            )

    def ensure_after_barrier():
        start.wait(timeout=5)
        return ensure_cython_extensions(
            source_root,
            extension_suffix=suffix,
            platform_system="Windows",
            build_runner=publish,
            verify_imports=False,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(ensure_after_barrier),
            pool.submit(ensure_after_barrier),
        )
        results = [future.result(timeout=10) for future in futures]

    assert len(calls) == 1
    assert all(not result.needs_rebuild for result in results)


@pytest.mark.parametrize("other_arguments", [
    ("build_ext", "--inplace", "--force"),
    ("build", "--force"),
    ("sdist",),
    ("--name",),
])
def test_concurrent_direct_build_entrypoints_serialize(
    tmp_path: Path, other_arguments: tuple[str, ...],
) -> None:
    """Every recipe invocation can regenerate C and must take the build lock."""
    target = tmp_path / "accelerator.test.so"
    active = tmp_path / "active-build"
    overlap = tmp_path / "overlap-observed"
    target.write_bytes(b"old extension")
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(
            target=_run_direct_build_overlap_probe,
            args=(start, str(target), str(active), str(overlap), arguments),
        )
        for arguments in (
            ("build_ext", "--inplace", "--force"), other_arguments,
        )
    ]

    for process in processes:
        process.start()
    try:
        start.set()
        for process in processes:
            process.join(timeout=10)

        assert [process.exitcode for process in processes] == [0, 0]
        assert not overlap.exists()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_content_change_with_preserved_mtime_requires_rebuild(
    tmp_path: Path,
) -> None:
    source_root = _source_tree(tmp_path)
    suffix = ".cp311-win_amd64.pyd"
    initial = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    newest_input = max(
        path.stat().st_mtime_ns
        for path in source_root.rglob("*.py")
    )
    for source in source_root.rglob("*.pyx"):
        newest_input = max(newest_input, source.stat().st_mtime_ns)
    for target in initial.targets:
        target.write_bytes(b"extension")
        os.utime(target, ns=(newest_input + 1_000_000,) * 2)
    write_build_manifest(
        source_root,
        extension_suffix=suffix,
        platform_system="Windows",
    )
    assert not extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows"
    ).needs_rebuild

    source = source_root / "dama/ai/algorithmic/_fast_search.pyx"
    source_mtime = source.stat().st_mtime_ns
    source.write_text("# changed with preserved timestamp\n", encoding="utf-8")
    os.utime(source, ns=(source_mtime, source_mtime))

    changed = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    assert changed.needs_rebuild
    assert changed.stale_targets == changed.targets


def test_output_change_with_preserved_mtime_requires_rebuild(
    tmp_path: Path,
) -> None:
    source_root = _source_tree(tmp_path)
    suffix = ".cp311-win_amd64.pyd"
    initial = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    newest_input = max(
        path.stat().st_mtime_ns for path in source_root.rglob("*")
        if path.is_file()
    )
    for target in initial.targets:
        target.write_bytes(b"extension")
        os.utime(target, ns=(newest_input + 1_000_000,) * 2)
    write_build_manifest(
        source_root,
        extension_suffix=suffix,
        platform_system="Windows",
    )

    target = initial.targets[0]
    target_mtime = target.stat().st_mtime_ns
    target.write_bytes(b"corrupted!")
    os.utime(target, ns=(target_mtime, target_mtime))

    changed = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    assert changed.needs_rebuild
    assert changed.stale_targets == changed.targets


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
@pytest.mark.parametrize("failure_site", ("manifest", "target"))
def test_readiness_rejects_fifo_without_waiting_for_a_writer(
    tmp_path: Path, failure_site: str,
) -> None:
    """Damaged build inputs must request repair without blocking launchers."""
    source_root = _source_tree(tmp_path)
    suffix = ".test.so"
    initial = extension_status(
        source_root, extension_suffix=suffix, platform_system="Linux")
    for target in initial.targets:
        target.write_bytes(b"extension")
    manifest = write_build_manifest(
        source_root, extension_suffix=suffix, platform_system="Linux")
    damaged = manifest if failure_site == "manifest" else initial.targets[0]
    damaged.unlink()
    os.mkfifo(damaged)

    # Bound the actual blocking open in a disposable process; a thread would
    # remain stuck even after its test times out.
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.ensure_cython_extensions import extension_status
status = extension_status(
    Path(sys.argv[2]), extension_suffix=".test.so", platform_system="Linux")
assert status.needs_rebuild
assert status.stale_targets == status.targets
"""
    subprocess.run(
        [sys.executable, "-c", script, str(PROJECT_ROOT / "src"), str(source_root)],
        check=True, capture_output=True, text=True, timeout=2,
    )


def test_manifest_rejects_inputs_changed_during_build(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    suffix = ".cp311-win_amd64.pyd"
    initial = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    for target in initial.targets:
        target.write_bytes(b"compiled from old source")
    expected_inputs = build_input_fingerprints(
        source_root, platform_system="Windows")

    changed = source_root / "dama/ai/ml/_fast_score.pyx"
    changed.write_text("# changed during build\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="build inputs changed while the build was running"
    ):
        write_build_manifest(
            source_root,
            extension_suffix=suffix,
            platform_system="Windows",
            expected_inputs=expected_inputs,
        )
    assert not (source_root / "build/cython_extensions.v2.json").exists()


@pytest.mark.parametrize("failure_site", ("fdopen", "replace"))
@pytest.mark.parametrize("cleanup_failure", (False, True))
def test_manifest_failure_does_not_close_a_reused_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    failure_site: str, cleanup_failure: bool,
) -> None:
    """An unwound stream or completed close must not transfer raw ownership twice."""
    manifest = tmp_path / "build/cython_extensions.v2.json"
    manifest.parent.mkdir()
    manifest.write_bytes(b"previous manifest")
    unrelated = tmp_path / "unrelated.bin"
    raw_descriptors = []
    unrelated_descriptors = []
    real_mkstemp = cython_guard.tempfile.mkstemp
    real_fdopen = os.fdopen
    failure = OSError("synthetic manifest publication failure")

    def capture_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        raw_descriptors.append(fd)
        return fd, name

    def open_unrelated():
        unrelated_descriptors.append(os.open(
            unrelated, os.O_WRONLY | os.O_CREAT, 0o600))

    def fail_fdopen(fd, *args, **kwargs):
        # A real partially constructed wrapper may close itself before its
        # caller learns whether it acquired the underlying descriptor.
        stream = real_fdopen(fd, *args, **kwargs)
        stream.close()
        open_unrelated()
        raise failure

    def fail_replace(_source, _target):
        with pytest.raises(OSError):
            os.fstat(raw_descriptors[0])
        open_unrelated()
        assert unrelated_descriptors[0] == raw_descriptors[0]
        raise failure

    def fail_unlink(_path, *args, **kwargs):
        raise OSError("synthetic temporary cleanup failure")

    monkeypatch.setattr(
        cython_guard, "_build_manifest_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(cython_guard.tempfile, "mkstemp", capture_mkstemp)
    if failure_site == "fdopen":
        monkeypatch.setattr(cython_guard.os, "fdopen", fail_fdopen)
    else:
        monkeypatch.setattr(cython_guard.os, "replace", fail_replace)
    if cleanup_failure:
        monkeypatch.setattr(Path, "unlink", fail_unlink)

    try:
        with pytest.raises(OSError) as observed:
            write_build_manifest(tmp_path)
        assert observed.value is failure
        assert len(unrelated_descriptors) == 1
        assert os.write(unrelated_descriptors[0], b"still open") == 10
        if raw_descriptors[0] != unrelated_descriptors[0]:
            with pytest.raises(OSError):
                os.fstat(raw_descriptors[0])
        assert manifest.read_bytes() == b"previous manifest"
        if not cleanup_failure:
            assert not list(manifest.parent.glob(".*.tmp"))
    finally:
        for fd in unrelated_descriptors:
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.parametrize("failure_type", (
    OSError, RuntimeError, KeyboardInterrupt, SystemExit, None,
))
def test_manifest_stream_close_preserves_first_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_type,
) -> None:
    """A buffered close error must not hide serialization or interruption."""
    manifest = tmp_path / "build/cython_extensions.v2.json"
    manifest.parent.mkdir()
    manifest.write_bytes(b"previous manifest")
    original = failure_type("manifest write failed") if failure_type else None
    close_failure = OSError("manifest stream close failed")
    real_fdopen = os.fdopen
    descriptors = []

    class FailingCloseStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            self.close()

        def close(self):
            self.stream.close()
            raise close_failure

    def failing_close_fdopen(fd, *args, **kwargs):
        descriptors.append(fd)
        return FailingCloseStream(real_fdopen(fd, *args, **kwargs))

    def fail_dump(_payload, stream, **_kwargs):
        stream.write("incomplete manifest")
        raise original

    monkeypatch.setattr(
        cython_guard, "_build_manifest_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(cython_guard.os, "fdopen", failing_close_fdopen)
    if original is not None:
        monkeypatch.setattr(cython_guard.json, "dump", fail_dump)

    expected = original if original is not None else close_failure
    with pytest.raises(type(expected)) as observed:
        write_build_manifest(tmp_path)

    assert observed.value is expected
    assert manifest.read_bytes() == b"previous manifest"
    assert not list(manifest.parent.glob(".*.tmp"))
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("environment_variable", (
    "CFLAGS", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
    "COMPILER_PATH", "GCC_EXEC_PREFIX", "INCLUDE", "LIB", "CL", "_CL_",
    "LINK", "_LINK_",
))
def test_build_environment_change_requires_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, environment_variable: str,
) -> None:
    # These compiler inputs can change output without changing its banner.
    monkeypatch.setattr(cython_guard, "_compiler_banner", lambda command: command)
    source_root = _source_tree(tmp_path)
    suffix = ".cp311-win_amd64.pyd"
    initial = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    newest_input = max(
        path.stat().st_mtime_ns for path in source_root.rglob("*")
        if path.is_file()
    )
    for target in initial.targets:
        target.write_bytes(b"extension")
        os.utime(target, ns=(newest_input + 1_000_000,) * 2)
    write_build_manifest(
        source_root,
        extension_suffix=suffix,
        platform_system="Windows",
    )
    assert not extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows"
    ).needs_rebuild

    monkeypatch.setenv(
        environment_variable, "dama-manifest-environment-test")

    changed = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    assert changed.needs_rebuild
    assert changed.stale_targets == changed.targets


def test_build_environment_tracks_setuptools_linker_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = build_environment_fingerprint()

    monkeypatch.setenv("LDSHARED", "dama-synthetic-linker --shared")

    after = build_environment_fingerprint()
    assert after != before
    assert after["environment"]["LDSHARED"] == (
        "dama-synthetic-linker --shared"
    )


def test_build_environment_probes_effective_compiler_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = {
        "CC": "dama-synthetic-cc --driver-mode=gcc",
        "CXX": "dama-synthetic-cxx --driver-mode=g++",
        "LDSHARED": "dama-synthetic-linker --shared",
        "LDCXXSHARED": "dama-synthetic-cxx-linker --shared",
    }
    for name, command in commands.items():
        monkeypatch.setenv(name, command)
    monkeypatch.setattr(
        cython_guard,
        "_compiler_banner",
        lambda command: {"command": command},
    )

    fingerprint = build_environment_fingerprint()

    assert fingerprint["compiler_banners"] == {
        name: {"command": command}
        for name, command in commands.items()
    }


def test_manifest_rejects_environment_changed_during_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source_root = _source_tree(tmp_path)
    suffix = ".cp311-win_amd64.pyd"
    initial = extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows")
    for target in initial.targets:
        target.write_bytes(b"compiled before environment change")
    expected_inputs = build_input_fingerprints(
        source_root, platform_system="Windows")
    expected_environment = build_environment_fingerprint(
        platform_system="Windows")

    monkeypatch.setenv("CFLAGS", "-DDAMA_MANIFEST_ENVIRONMENT_RACE=1")

    with pytest.raises(
        RuntimeError, match="build environment changed while the build was running"
    ):
        write_build_manifest(
            source_root,
            extension_suffix=suffix,
            platform_system="Windows",
            expected_inputs=expected_inputs,
            expected_environment=expected_environment,
        )
    assert not (source_root / "build/cython_extensions.v2.json").exists()


def test_build_failure_is_fail_closed(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)

    def fail_build(command, cwd):
        raise RuntimeError("compiler failed")

    with pytest.raises(RuntimeError, match="compiler failed"):
        ensure_cython_extensions(
            source_root,
            extension_suffix=".cp311-win_amd64.pyd",
            platform_system="Windows",
            build_runner=fail_build,
            verify_imports=False,
        )


def test_successful_build_without_current_targets_is_fail_closed(
    tmp_path: Path,
) -> None:
    """A compiler exit code alone is not evidence that binaries were published."""
    source_root = _source_tree(tmp_path)

    with pytest.raises(
        RuntimeError, match="without publishing current targets"
    ):
        ensure_cython_extensions(
            source_root,
            extension_suffix=".cp311-win_amd64.pyd",
            platform_system="Windows",
            build_runner=lambda _command, _cwd: None,
            verify_imports=False,
        )


def test_build_entrypoint_does_not_attest_a_stale_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    target = tmp_path / "accelerator.test.so"
    target.write_bytes(b"old extension")
    stale = SimpleNamespace(targets=(target,), needs_rebuild=True)
    wrote_manifest = False

    def unexpected_manifest(_source_root):
        nonlocal wrote_manifest
        wrote_manifest = True

    monkeypatch.setattr(cython_guard, "extension_status", lambda _root: stale)
    monkeypatch.setattr(cython_guard, "write_build_manifest", unexpected_manifest)
    monkeypatch.setattr(build_entrypoint.runpy, "run_path", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        build_entrypoint.sys,
        "argv",
        ["setup_cython.py", "build_ext", "--inplace", "--dry-run"],
    )

    with pytest.raises(RuntimeError, match="did not republish stale targets"):
        build_entrypoint.main()
    assert not wrote_manifest


def test_build_entrypoint_pins_inputs_across_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    target = tmp_path / "accelerator.test.so"
    target.write_bytes(b"old extension")
    status = SimpleNamespace(targets=(target,), needs_rebuild=True)
    expected_inputs = {"accelerator.pyx": "before-build"}
    expected_environment = {"compiler": "before-build"}

    monkeypatch.setattr(cython_guard, "extension_status", lambda _root: status)
    monkeypatch.setattr(
        cython_guard,
        "build_input_fingerprints",
        lambda _root: expected_inputs,
    )
    monkeypatch.setattr(
        cython_guard,
        "build_environment_fingerprint",
        lambda: expected_environment,
    )

    def simulated_build(*_args, **_kwargs):
        replacement = target.with_suffix(".new")
        replacement.write_bytes(b"compiled from old source")
        os.replace(replacement, target)

    def reject_changed_inputs(
        _root, *, expected_inputs=None, expected_environment=None,
    ):
        assert expected_inputs == {"accelerator.pyx": "before-build"}
        assert expected_environment == {"compiler": "before-build"}
        raise RuntimeError("Cython build inputs changed while the build was running")

    monkeypatch.setattr(build_entrypoint.runpy, "run_path", simulated_build)
    monkeypatch.setattr(cython_guard, "write_build_manifest", reject_changed_inputs)
    monkeypatch.setattr(
        build_entrypoint.sys,
        "argv",
        ["setup_cython.py", "build_ext", "--inplace", "--force"],
    )

    with pytest.raises(
        RuntimeError, match="build inputs changed while the build was running"
    ):
        build_entrypoint.main()


@pytest.mark.parametrize(
    "extra_option",
    (
        "--debug",
        "--compiler=unix",
        "--include-dirs=/synthetic/include",
        "--cython-directives=boundscheck=True",
    ),
)
def test_build_entrypoint_does_not_attest_noncanonical_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra_option: str,
) -> None:
    target = tmp_path / "accelerator.test.so"
    target.write_bytes(b"old extension")
    wrote_manifest = False

    def simulated_build(*_args, **_kwargs):
        replacement = target.with_suffix(".new")
        replacement.write_bytes(b"noncanonical extension")
        os.replace(replacement, target)

    def unexpected_manifest(*_args, **_kwargs):
        nonlocal wrote_manifest
        wrote_manifest = True

    status = SimpleNamespace(targets=(target,), needs_rebuild=True)
    monkeypatch.setattr(cython_guard, "extension_status", lambda _root: status)
    monkeypatch.setattr(build_entrypoint.runpy, "run_path", simulated_build)
    monkeypatch.setattr(
        cython_guard, "write_build_manifest", unexpected_manifest)
    monkeypatch.setattr(
        build_entrypoint.sys,
        "argv",
        [
            "setup_cython.py",
            "build_ext",
            "--inplace",
            "--force",
            extra_option,
        ],
    )

    build_entrypoint.main()

    assert target.read_bytes() == b"noncanonical extension"
    assert not wrote_manifest


def test_windows_launcher_runs_guard_before_trainer_import() -> None:
    launcher = (PROJECT_ROOT / "local_train.ps1").read_text(encoding="utf-8")

    guard = launcher.index("ensure_cython_extensions.py")
    preflight = launcher.index("$PreflightCode = @'")
    trainer = launcher.index("import dama.ai.ml.trainer")
    assert guard < preflight < trainer
    assert "Cython extension readiness failed" in launcher
    assert "Training was not started" in launcher


def test_environment_setups_use_canonical_fail_closed_build() -> None:
    canonical = "python setup_cython.py build_ext --inplace --force"
    for name in ("setup_conda.sh", "setup_conda_server.sh"):
        setup_script = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert setup_script.count(canonical) == 1
        assert "setup_cython.py build --build-base" not in setup_script
        assert "Cython extension build failed" not in setup_script
