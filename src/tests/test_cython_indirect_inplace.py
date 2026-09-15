"""Indirect in-place builds must refuse extensions already mapped by a reader."""

from contextlib import ExitStack
from mmap import ACCESS_READ, mmap
import os
from pathlib import Path
import runpy
import shutil
import sys
import sysconfig

from Cython import Build
import pytest
import setuptools
from setuptools.command.build_ext import build_ext


@pytest.mark.skipif(
    not Path("/proc/self/maps").is_file(),
    reason="Mapped-target detection requires Linux /proc process maps.",
)
@pytest.mark.parametrize("parent_command", ("develop", "editable_wheel", "build_ext"))
@pytest.mark.parametrize("mapped", (False, True))
def test_finalized_inplace_build_checks_mapped_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    parent_command: str, mapped: bool,
) -> None:
    """Use setuptools' parent-command configuration without installing anything."""
    captured = {}
    monkeypatch.setattr(Build, "cythonize", lambda extensions, **_kwargs: extensions)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    arguments = [parent_command]
    if parent_command == "build_ext":
        arguments.append("--inplace")
    monkeypatch.setattr(sys, "argv", ["setup_cython.py", *arguments])
    recipe = tmp_path / "scripts/setup_cython.py"
    recipe.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "scripts/setup_cython.py", recipe)
    runpy.run_path(str(recipe))

    extension = setuptools.Extension("package.accelerator", sources=["unused.c"])
    distribution = captured["distclass"]({
        "name": "dama-inplace-safety-probe",
        "script_args": arguments,
        "ext_modules": [extension],
        "cmdclass": captured["cmdclass"],
    })
    assert distribution.parse_command_line()
    if parent_command == "develop":
        # These are the actual settings made by install_for_development().
        distribution.get_command_obj("develop").reinitialize_command(
            "build_ext", inplace=True)
    elif parent_command == "editable_wheel":
        distribution.get_command_obj("editable_wheel")._set_editable_mode()
    command = distribution.get_command_obj("build_ext")
    command.ensure_finalized()
    assert command.inplace is True or command.inplace == 1

    compiler_calls = []
    monkeypatch.setattr(build_ext, "run", lambda self: compiler_calls.append(self))
    target = tmp_path / f"package/accelerator{sysconfig.get_config_var('EXT_SUFFIX')}"
    target.parent.mkdir()
    target.write_bytes(b"live extension")
    # Mapping begins after the recipe's early CLI guard. This also covers an
    # importer arriving before the effective command starts compiling.
    with ExitStack() as stack:
        if mapped:
            stream = stack.enter_context(target.open("rb"))
            stack.enter_context(mmap(stream.fileno(), 0, access=ACCESS_READ))
            with pytest.raises(SystemExit, match=rf"PID {os.getpid()} is using it"):
                distribution.run_command("build_ext")
            assert compiler_calls == []
        else:
            distribution.run_command("build_ext")
            assert compiler_calls == [command]

    assert target.read_bytes() == b"live extension"
