"""Safety contracts for publishing locally built Cython extensions."""

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


def test_atomic_inplace_copy_failure_keeps_previous_binary(
    build_recipe: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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

    monkeypatch.setattr(namespace["os"], "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        command.copy_file(source, target)

    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".active.so.*.tmp"))
