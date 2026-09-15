"""Safety contracts for publishing locally built Cython extensions."""

import errno
from functools import partial
from mmap import ACCESS_READ, mmap
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

from Cython import Build
import pytest
import setuptools
from setuptools.dist import Distribution


@pytest.fixture
def build_recipe(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    """Load the recipe without compiling or invoking setuptools commands."""
    captured: dict = {}
    monkeypatch.setattr(Build, "cythonize", lambda extensions, **_: extensions)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    recipe = Path(__file__).parents[1] / "scripts" / "setup_cython.py"
    return runpy.run_path(str(recipe)), captured


@pytest.mark.parametrize("force_option", ("--force", "-f"))
def test_force_build_also_forces_cython_regeneration(
    monkeypatch: pytest.MonkeyPatch, force_option: str,
) -> None:
    captured: dict = {}

    def capture_cythonize(extensions, **kwargs):
        captured.update(kwargs)
        return extensions

    monkeypatch.setattr(Build, "cythonize", capture_cythonize)
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys, "argv", ["setup_cython.py", "build_ext", force_option])
    recipe = Path(__file__).parents[1] / "scripts" / "setup_cython.py"

    runpy.run_path(str(recipe))

    assert captured["force"] is True


def test_build_recipe_ignores_external_distutils_configuration(
    build_recipe: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ambient config must not inject hidden options into certified builds."""
    namespace, captured = build_recipe
    user_config = tmp_path / ".pydistutils.cfg"
    user_config.write_text(
        "[build_ext]\ndebug=1\ncompiler=unix\n", encoding="utf-8")
    extra_config = tmp_path / "extra.cfg"
    extra_config.write_text(
        "[build_ext]\nparallel=8\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DIST_EXTRA_CONFIG", str(extra_config))

    distribution_class = captured["distclass"]
    assert distribution_class is namespace["HermeticBuildDistribution"]
    distribution = distribution_class()
    distribution.parse_config_files()

    assert distribution.find_config_files() == []
    assert "build_ext" not in distribution.command_options


def test_build_recipe_ignores_local_pyproject_extension_overrides(
    build_recipe: tuple[dict, dict], tmp_path: Path,
) -> None:
    """TOML must not append an untracked replacement for a certified target."""
    _, captured = build_recipe
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="dama-local-override"\nversion="0.1"\n'
        '[tool.setuptools]\n'
        'ext-modules=[{name="dama.ai.ml._fast_score",'
        'sources=["alternate_score.c"],'
        'define-macros=[["DAMA_AMBIENT_OVERRIDE","1"]]}]\n',
        encoding="utf-8",
    )
    expected_extensions = list(captured["ext_modules"])
    distribution = captured["distclass"]({
        "src_root": str(tmp_path),
        "ext_modules": list(expected_extensions),
    })

    distribution.parse_config_files()

    assert distribution.ext_modules == expected_extensions


@pytest.mark.skipif(
    not Path("/proc/self/maps").is_file(),
    reason="Mapped-target detection requires Linux /proc process maps.",
)
def test_inplace_build_detects_a_mapped_target(
    build_recipe: tuple[dict, dict], tmp_path: Path,
) -> None:
    namespace, _ = build_recipe
    suffix = namespace["sysconfig"].get_config_var("EXT_SUFFIX")
    target = tmp_path / f"package/accelerator{suffix}"
    target.parent.mkdir()
    target.write_bytes(b"old extension")
    extension = SimpleNamespace(name="package.accelerator")

    with target.open("rb") as stream, mmap(
        stream.fileno(), 0, access=ACCESS_READ,
    ):
        owner = namespace["_mapped_inplace_extension_owner"](
            [extension], source_root=tmp_path)
        for inplace_option in ("--inplace", "-i"):
            with pytest.raises(
                SystemExit, match=rf"PID {os.getpid()} is using it"
            ):
                namespace["_refuse_mapped_inplace_build"](
                    [extension], argv=["build_ext", inplace_option],
                    source_root=tmp_path,
                )

    assert owner == (os.getpid(), target)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows does not allow replacement of an open mapped file.",
)
def test_atomic_inplace_copy_hides_partial_binary_and_preserves_reader(
    build_recipe: tuple[dict, dict], tmp_path: Path,
) -> None:
    namespace, captured = build_recipe
    command_class = captured["cmdclass"]["build_ext"]
    assert command_class is namespace["AtomicBuildExt"]
    command = command_class(Distribution())
    command.initialize_options()
    command.inplace = True
    command.dry_run = False
    command.force = True
    command.verbose = 0

    source = tmp_path / "built.so"
    target = tmp_path / "active.so"
    source.write_bytes(b"B" * (8 * 1024 * 1024))
    target.write_bytes(b"A" * (8 * 1024 * 1024))
    original_inode = target.stat().st_ino

    with target.open("rb") as stream, mmap(
        stream.fileno(), 0, access=ACCESS_READ,
    ) as mapped:
        result = command.copy_file(source, target)
        assert mapped[:16] == b"A" * 16

    assert result == (target, True)
    assert target.read_bytes()[:16] == b"B" * 16
    assert target.stat().st_ino != original_inode
    assert not list(tmp_path.glob(".active.so.*.tmp"))


def test_atomic_inplace_copy_publishes_completed_binary(
    build_recipe: tuple[dict, dict], tmp_path: Path,
) -> None:
    """Exercise native file flushing on every platform without live mappings."""
    _, captured = build_recipe
    command = captured["cmdclass"]["build_ext"](Distribution())
    command.initialize_options()
    command.inplace = True
    command.dry_run = False
    command.force = True
    command.verbose = 0
    suffix = ".pyd" if os.name == "nt" else ".so"
    source = tmp_path / f"built{suffix}"
    target = tmp_path / f"active{suffix}"
    source.write_bytes(b"new completed extension")
    target.write_bytes(b"old extension")

    assert command.copy_file(source, target) == (target, True)

    assert target.read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


@pytest.mark.parametrize("cleanup_failure", (False, True))
def test_atomic_inplace_copy_failure_keeps_previous_binary(
    build_recipe: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, cleanup_failure: bool,
) -> None:
    namespace, captured = build_recipe
    command = captured["cmdclass"]["build_ext"](Distribution())
    command.initialize_options()
    command.inplace = True
    command.dry_run = False
    command.force = True
    command.verbose = 0
    source = tmp_path / "built.so"
    target = tmp_path / "active.so"
    source.write_bytes(b"new")
    target.write_bytes(b"old")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("synthetic replace failure")

    def fail_unlink(_path: Path, **_kwargs) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(namespace["os"], "replace", fail_replace)
    if cleanup_failure:
        monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="synthetic replace failure"):
        command.copy_file(source, target)

    assert target.read_bytes() == b"old"
    assert bool(list(tmp_path.glob(".active.so.*.tmp"))) is cleanup_failure


def test_atomic_inplace_copy_cleans_temporary_after_initial_close_failure(
    build_recipe: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed first close must clean its name without retrying the fd."""
    namespace, captured = build_recipe
    command = captured["cmdclass"]["build_ext"](Distribution())
    command.initialize_options()
    command.inplace = True
    command.dry_run = False
    source = tmp_path / "built.so"
    target = tmp_path / "active.so"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    real_close = os.close
    close_failure = OSError("initial temporary close failed")
    closed = []

    def fail_close(fd):
        closed.append(fd)
        real_close(fd)
        raise close_failure

    monkeypatch.setattr(namespace["os"], "close", fail_close)

    with pytest.raises(OSError) as observed:
        command.copy_file(source, target)

    assert observed.value is close_failure
    assert len(closed) == 1
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".active.so.*.tmp"))


def test_atomic_inplace_copy_flushes_windows_binary_through_writable_handle(
    build_recipe: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Windows os.fsync uses FlushFileBuffers, which rejects read-only handles."""
    fcntl = pytest.importorskip("fcntl")
    namespace, captured = build_recipe
    command = captured["cmdclass"]["build_ext"](Distribution())
    command.initialize_options()
    command.inplace = True
    command.dry_run = False
    command.force = True
    command.verbose = 0
    source = tmp_path / "built.pyd"
    target = tmp_path / "active.pyd"
    probe = tmp_path / "probe.bin"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    probe.write_bytes(b"probe")
    real_fsync = os.fsync
    flushed_access_modes = []

    def windows_fsync(fd: int) -> None:
        access_mode = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        flushed_access_modes.append(access_mode)
        if access_mode == os.O_RDONLY:
            raise OSError(errno.EBADF, "FlushFileBuffers needs GENERIC_WRITE")
        real_fsync(fd)

    fsync_file = namespace["_fsync_file"]
    monkeypatch.setattr(namespace["os"], "fsync", windows_fsync)
    # The POSIX read-only flush is exactly what failed on Windows.
    with pytest.raises(OSError, match="GENERIC_WRITE"):
        fsync_file(probe, windows=False)

    monkeypatch.setitem(
        type(command).copy_file.__globals__, "_fsync_file",
        partial(fsync_file, windows=True))
    assert command.copy_file(source, target) == (target, True)

    assert target.read_bytes() == b"new"
    assert os.O_RDWR in flushed_access_modes
    assert not list(tmp_path.glob(".active.pyd.*.tmp"))
