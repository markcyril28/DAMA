"""Compatibility entrypoint for ``python setup_cython.py``."""

from contextlib import nullcontext
from pathlib import Path
import runpy
import sys


def _is_canonical_manifest_build(arguments: list[str]) -> bool:
    """Return whether an invocation matches the launcher's build contract."""
    return (
        len(arguments) == 3
        and arguments.count("build_ext") == 1
        and sum(option in arguments for option in ("--inplace", "-i")) == 1
        and sum(option in arguments for option in ("--force", "-f")) == 1
    )


def _target_identities(
    targets: tuple[Path, ...],
) -> tuple[tuple[int, int, int, int, int] | None, ...]:
    identities = []
    for target in targets:
        try:
            stat = target.stat()
        except OSError:
            identities.append(None)
            continue
        identities.append((
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ))
    return tuple(identities)


def _run_setup(source_root: Path, script: Path, arguments: list[str]) -> None:
    inplace = "build_ext" in arguments and any(
        argument in ("--inplace", "-i") for argument in arguments
    )
    manifest_build = _is_canonical_manifest_build(arguments)
    if inplace:
        from scripts.ensure_cython_extensions import (
            build_environment_fingerprint,
            build_input_fingerprints,
            extension_status,
            write_build_manifest,
        )

        before_status = extension_status(source_root)
        before_identities = _target_identities(before_status.targets)
        if manifest_build:
            before_inputs = build_input_fingerprints(source_root)
            before_environment = build_environment_fingerprint()

    runpy.run_path(str(script), run_name="__main__")
    if inplace:
        after_identities = _target_identities(before_status.targets)
        republished = tuple(
            before != after
            for before, after in zip(
                before_identities, after_identities, strict=True)
        )
        if any(republished) and not all(republished):
            raise RuntimeError(
                "In-place Cython build republished only some expected targets; "
                "content manifest was not written. Re-run with --force."
            )
        if not any(republished):
            after_status = extension_status(source_root)
            if after_status.needs_rebuild:
                raise RuntimeError(
                    "In-place Cython build did not republish stale targets; "
                    "content manifest was not written. Re-run with --force."
                )
            return
        if not manifest_build:
            print(
                "Cython build used noncanonical options; the production "
                "content manifest was not written. The readiness guard will "
                "rebuild any changed targets before training."
            )
            return
        manifest = write_build_manifest(
            source_root,
            expected_inputs=before_inputs,
            expected_environment=before_environment,
        )
        print(f"wrote Cython build manifest: {manifest}")


def main() -> None:
    source_root = Path(__file__).resolve().parent
    script = source_root / "scripts" / "setup_cython.py"
    arguments = sys.argv[1:]
    if "build_ext" not in arguments:
        _run_setup(source_root, script, arguments)
        return

    from scripts.ensure_cython_extensions import (
        cython_build_lock,
        parent_holds_cython_build_lock,
    )

    lock = (
        nullcontext()
        if parent_holds_cython_build_lock(source_root)
        else cython_build_lock(source_root)
    )
    with lock:
        _run_setup(source_root, script, arguments)


if __name__ == "__main__":
    main()
