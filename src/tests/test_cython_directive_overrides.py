"""Explicit Cython directives apply before source generation and compilation."""

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

from Cython import Build
import pytest
import setuptools


@pytest.fixture
def directive_recipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    def capture_cythonize(extensions, **options):
        captured["cythonize"] = options
        return extensions

    monkeypatch.setattr(Build, "cythonize", capture_cythonize)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    recipe = tmp_path / "scripts/setup_cython.py"
    recipe.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "scripts/setup_cython.py", recipe)

    def load(arguments):
        monkeypatch.setattr(sys, "argv", ["setup_cython.py", *arguments])
        runpy.run_path(str(recipe))
        distribution = captured["distclass"]({
            "script_args": arguments,
            "ext_modules": captured["ext_modules"],
            "cmdclass": captured["cmdclass"],
        })
        distribution.parse_command_line()
        command = distribution.get_command_obj("build_ext")
        command.ensure_finalized()
        return captured, command

    return load


@pytest.mark.parametrize("arguments, expected", (
    (["build_ext", "--cython-directives=boundscheck=True"], {"boundscheck": True}),
    (["build_ext", "--cython-directives", "boundscheck=True,wraparound=True"],
     {"boundscheck": True, "wraparound": True}),
    (["build_ext", "--cython-directives=boundscheck=False"], {"boundscheck": False}),
    (["build_ext", "--cython-directives=boundscheck=False", "build_ext",
      "--cython-directives=boundscheck=True"], {"boundscheck": True}),
))
def test_directive_override_reaches_generation_and_compiler(
    directive_recipe, arguments, expected,
):
    captured, command = directive_recipe(arguments)

    # A directive change must regenerate existing C even without --force.
    assert captured["cythonize"]["force"] is True
    for name, value in expected.items():
        assert captured["cythonize"]["compiler_directives"][name] == value
    # Cython's build_ext expects a mapping; its CLI leaves a raw string here.
    assert command.cython_directives == expected


def test_directive_looking_include_value_keeps_canonical_defaults(directive_recipe):
    captured, command = directive_recipe([
        "build_ext", "--include-dirs", "--cython-directives=boundscheck=True",
    ])

    assert captured["cythonize"]["force"] is False
    assert captured["cythonize"]["compiler_directives"]["boundscheck"] is False
    assert command.cython_directives == {}


def test_invalid_directive_fails_before_cython_generation(
    directive_recipe, monkeypatch: pytest.MonkeyPatch,
):
    generated = []
    monkeypatch.setattr(Build, "cythonize", lambda *args, **kwargs: generated.append(args))

    with pytest.raises(ValueError, match="boundscheck directive must be set"):
        directive_recipe(["build_ext", "--cython-directives=boundscheck=invalid"])

    assert generated == []


def test_real_build_honors_boundscheck_override(tmp_path: Path):
    """Compile tiny isolated targets, then exercise their checked indexing."""
    source_root = Path(__file__).parents[1]
    for relative in (
        "setup_cython.py", "scripts/setup_cython.py", "scripts/ensure_cython_extensions.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, target)
    extensions = []
    for relative in (
        "dama/ai/algorithmic/_fast_search.pyx", "dama/ai/ml/_fast_encode.pyx",
        "dama/ai/ml/_fast_score.pyx", "dama/ai/ml/_fast_stat.pyx",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "def lookup(double[:] items, Py_ssize_t index):\n"
            "    return items[index]\n", encoding="utf-8",
        )
        extensions.append(setuptools.Extension(
            ".".join(Path(relative).with_suffix("").parts), [str(target)]))
    # Leave newer unchecked generated C behind, as a prior canonical build
    # would. The override has no --force flag to repair that cache for it.
    Build.cythonize(
        extensions, force=True, quiet=True,
        compiler_directives={"boundscheck": False, "language_level": "3"},
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    command = [
        sys.executable, "setup_cython.py", "build_ext", "--inplace",
        "--cython-directives=boundscheck=True",
    ]
    completed = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (tmp_path / "build/cython_extensions.v2.json").exists()

    probe = subprocess.run(
        [sys.executable, "-c",
         "from array import array\n"
         "from dama.ai.ml._fast_score import lookup\n"
         "assert lookup(array('d', [17.0]), 0) == 17.0\n"
         "try:\n"
         "    lookup(array('d', [17.0]), 1)\n"
         "except IndexError:\n"
         "    print('checked indexing')\n"
         "else:\n"
         "    raise AssertionError('boundscheck override did not take effect')\n"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stdout.strip() == "checked indexing"
