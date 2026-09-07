"""Contracts for the cross-platform Cython extension readiness guard."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
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
    child_guard.write_build_manifest = lambda *_args, **_kwargs: target.parent

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
    child_entrypoint.sys.argv = [
        "setup_cython.py", "build_ext", "--inplace", "--force",
    ]
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


def test_concurrent_direct_build_entrypoints_serialize(tmp_path: Path) -> None:
    """The documented direct build path must not share its build tree."""
    target = tmp_path / "accelerator.test.so"
    active = tmp_path / "active-build"
    overlap = tmp_path / "overlap-observed"
    target.write_bytes(b"old extension")
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(
            target=_run_direct_build_overlap_probe,
            args=(start, str(target), str(active), str(overlap)),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert not overlap.exists()


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


def test_build_environment_change_requires_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
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
    assert not extension_status(
        source_root, extension_suffix=suffix, platform_system="Windows"
    ).needs_rebuild

    monkeypatch.setenv("CFLAGS", "-DDAMA_MANIFEST_ENVIRONMENT_TEST=1")

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
