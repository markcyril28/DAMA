"""Compatibility entrypoint for ``python setup_cython.py``."""

from pathlib import Path
import runpy


def main() -> None:
    script = Path(__file__).resolve().parent / "scripts" / "setup_cython.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
