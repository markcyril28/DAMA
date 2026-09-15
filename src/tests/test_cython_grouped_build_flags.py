"""Grouped setuptools flags retain the Cython safety and regeneration checks."""

import os
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace

from Cython import Build
import pytest
import setuptools
from setuptools.command.build_ext import build_ext


@pytest.fixture
def recipe_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    def capture_cythonize(extensions, **options):
        captured.update(options)
        return extensions

    monkeypatch.setattr(Build, "cythonize", capture_cythonize)
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    # Load the unchanged recipe in an empty source tree so real mapped project
    # extensions cannot interfere with this command-line-only probe.
    recipe = tmp_path / "scripts/setup_cython.py"
    recipe.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "scripts/setup_cython.py", recipe)

    def load(arguments):
        monkeypatch.setattr(sys, "argv", ["setup_cython.py", *arguments])
        namespace = runpy.run_path(str(recipe))
        return namespace, captured

    return load


@pytest.mark.parametrize("flags", ("-if", "-fi", "-gif"))
def test_grouped_inplace_flags_refuse_a_mapped_target(recipe_loader, flags):
    namespace, _ = recipe_loader([])
    target = Path("mapped_extension.so")
    namespace["_refuse_mapped_inplace_build"].__globals__[
        "_mapped_inplace_extension_owner"
    ] = lambda *_args, **_kwargs: (os.getpid(), target)

    # Verify this is a real setuptools spelling, not a synthetic parser case.
    distribution = setuptools.Distribution({
        "script_args": ["build_ext", flags],
        "cmdclass": {"build_ext": build_ext},
    })
    assert distribution.parse_command_line()
    command = distribution.get_command_obj("build_ext")
    assert command.inplace and command.force

    with pytest.raises(SystemExit, match="Refusing to rebuild mapped"):
        namespace["_refuse_mapped_inplace_build"](
            [SimpleNamespace(name="mapped_extension")], ["build_ext", flags])


@pytest.mark.parametrize("flags", ("-if", "-fi", "-gif"))
def test_grouped_force_flags_regenerate_cython(
    recipe_loader, flags: str,
):
    _, captured = recipe_loader(["build_ext", flags])
    assert captured["force"] is True


@pytest.mark.parametrize("arguments", (
    ["build_ext", "-I/path/if"],
    ["build_ext", "-L/path/fi"],
    ["build_ext", "-j4"],
    ["build_ext", "-I", "-if"],
    ["build_ext", "--include-dirs=-if"],
    ["build_ext", "--include-dirs", "-if"],
    ["build_ext", "build", "-f"],
))
def test_option_values_and_other_commands_are_not_build_ext_flags(
    recipe_loader, arguments,
):
    namespace, _ = recipe_loader([])
    flag_requested = namespace["_build_ext_flag_requested"]
    assert not flag_requested(arguments, "inplace")
    assert not flag_requested(arguments, "force")


@pytest.mark.parametrize("arguments", (
    ["build_ext", "--force", "build_ext", "--inplace"],
    ["build_ext", "build_ext", "-if"],
    ["build_ext", "-I", "build_ext", "build_ext", "-fi"],
    ["build_ext", "-I", "build_ext", "-f"],
    ["build_ext", "-I", "-if"],
    ["build_ext", "build", "-f"],
))
def test_repeated_commands_and_command_like_values_match_setuptools(
    recipe_loader, arguments,
):
    namespace, _ = recipe_loader([])
    distribution = setuptools.Distribution({
        "script_args": arguments,
        "cmdclass": {"build_ext": build_ext},
    })
    assert distribution.parse_command_line()
    command = distribution.get_command_obj("build_ext")
    flag_requested = namespace["_build_ext_flag_requested"]
    for flag in ("force", "inplace"):
        assert flag_requested(arguments, flag) == bool(getattr(command, flag))


@pytest.mark.parametrize("arguments", (
    ["build", "-gf"],
    ["build", "-fg"],
    ["build", "-fv"],
    ["build", "-fv", "build_ext", "--inplace"],
    ["build", "-qf", "build_ext", "--inplace"],
    ["build_ext", "--inplace", "build", "-fq"],
    ["build", "--force", "build_ext", "--inplace"],
    ["build_ext", "--force"],
    ["build_ext", "-I", "-f"],
    ["build_ext", "--include-dirs", "--force"],
    ["build", "--executable", "-f"],
    ["build", "--executable=-f"],
))
def test_cython_regeneration_matches_inherited_build_force(
    recipe_loader, arguments,
):
    """Cython and the C compiler must consume the same effective force flag."""
    distribution = setuptools.Distribution({
        "script_args": arguments,
        "cmdclass": {"build_ext": build_ext},
        "ext_modules": [],
    })
    assert distribution.parse_command_line()
    command = distribution.get_command_obj("build_ext")
    # build_ext inherits unset options from the build command at finalization.
    command.ensure_finalized()

    _, captured = recipe_loader(arguments)

    assert captured["force"] == bool(command.force)
