"""Build Cython extensions for performance-critical encoding functions."""

import os
import platform
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile

from Cython.Build import cythonize
import numpy as np
from setuptools import Distribution, Extension, setup
from setuptools.command.build_ext import build_ext


class HermeticBuildDistribution(Distribution):
    """Keep extension output independent of ambient distutils config files."""

    def find_config_files(self) -> list[str]:
        # setuptools otherwise reads system and user distutils configuration,
        # a local setup.cfg, and DIST_EXTRA_CONFIG. Those files can inject
        # output-changing build_ext options behind an otherwise canonical
        # command line, outside the content manifest's source contract.
        return []


def _mapped_inplace_extension_owner(
    extensions: list[Extension], source_root: Path | None = None,
) -> tuple[int, Path] | None:
    """Return one process mapping an in-place build target, if observable."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None

    root = (source_root or Path(__file__).resolve().parents[1]).resolve()
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    if not suffix:
        return None
    targets = {}
    for extension in extensions:
        module_path = root.joinpath(*extension.name.split("."))
        target = Path(f"{module_path}{suffix}")
        try:
            target_stat = target.stat()
        except OSError:
            continue
        targets[(
            os.major(target_stat.st_dev),
            os.minor(target_stat.st_dev),
            target_stat.st_ino,
        )] = target
    if not targets:
        return None

    for maps_path in proc_root.glob("[0-9]*/maps"):
        try:
            lines = maps_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            fields = line.split(maxsplit=5)
            if len(fields) < 5:
                continue
            try:
                device_major, device_minor = (
                    int(part, 16) for part in fields[3].split(":", 1)
                )
                inode = int(fields[4])
            except (ValueError, TypeError):
                continue
            target = targets.get((device_major, device_minor, inode))
            if target is not None:
                return int(maps_path.parent.name), target
    return None


def _refuse_mapped_inplace_build(
    extensions: list[Extension], argv: list[str] | None = None,
    source_root: Path | None = None,
) -> None:
    """Fail before compiling when a requested in-place target is live."""
    arguments = sys.argv[1:] if argv is None else argv
    inplace_requested = any(
        argument in ("--inplace", "-i") for argument in arguments)
    if "build_ext" not in arguments or not inplace_requested:
        return
    owner = _mapped_inplace_extension_owner(extensions, source_root=source_root)
    if owner is None:
        return
    pid, target = owner
    raise SystemExit(
        f"Refusing to rebuild mapped Cython extension {target}; "
        f"PID {pid} is using it."
    )


class AtomicBuildExt(build_ext):
    """Publish in-place extension binaries with one atomic replacement."""

    def copy_file(
        self,
        infile,
        outfile,
        preserve_mode=True,
        preserve_times=True,
        link=None,
        level=1,
    ):
        # setuptools normally unlinks the destination and streams the new
        # binary into its final name. A process importing during that window
        # can observe a missing or partial shared library. The temporary file
        # lives beside the target, so os.replace() is atomic on every supported
        # filesystem. Non-in-place and unusual link builds retain setuptools.
        if not self.inplace or self.dry_run or link is not None:
            return super().copy_file(
                infile, outfile, preserve_mode, preserve_times, link, level)

        source = Path(os.fsdecode(infile))
        target = Path(os.fsdecode(outfile))
        self.announce(f"atomically copying {source} -> {target}", level=level)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            source_stat = source.stat()
            if preserve_times:
                os.utime(
                    temporary,
                    ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
                )
            if preserve_mode:
                os.chmod(temporary, source_stat.st_mode)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            try:
                directory_fd = os.open(
                    target.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Windows and some mounted filesystems cannot fsync a
                # directory. The same-directory replacement is still atomic.
                pass
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return outfile, True


# Compiler optimization flags — these are the single highest-impact change
# for CPU-bound self-play throughput. The alpha-beta search spends 99% of
# game time in C; -O3 enables auto-vectorization, loop unrolling, and
# inlining that default -O2 misses. -march=native targets the exact CPU
# (AVX2/SSE4.2 on modern x86). -ffast-math allows reordering of FP
# operations.
# MSVC rejects the GCC/Clang spellings above (it ignores them with warning
# D9002 and silently builds an unoptimized extension), so the Windows toolchain
# needs its own equivalents: /O2 for full optimization, /fp:fast for the
# -ffast-math relaxations, /arch:AVX2 as the portable stand-in for
# -march=native, which MSVC does not have.
if platform.system() == "Windows":
    _compile_args = ["/O2", "/fp:fast"]
    _link_args = []
    if platform.machine() in ("x86_64", "AMD64"):
        _compile_args.append("/arch:AVX2")
else:
    _compile_args = ["-O3", "-ffast-math"]
    _link_args = []
    if platform.machine() in ("x86_64", "AMD64"):
        _compile_args.append("-march=native")

# Only the .pyx sources are handed to cythonize(). Everything that depends on
# the machine doing the build -- the compiler flags chosen above, and numpy's
# absolute include path -- is attached to the Extension objects afterwards.
#
# Cython copies whatever it is given into a metadata block at the top of the
# generated .c, and those .c files ARE tracked (only the .so/.pyd are ignored).
# Passing the build settings in up front therefore made the generated source
# machine-specific: the committed _fast_encode.c carried a contributor's
# "C:\Users\...\site-packages\numpy" paths, and _fast_search.c flipped
# between the MSVC and GCC flag spellings every time the platform that last
# built it changed -- so simply running local_train.sh, whose staleness guard
# rebuilds in place, dirtied a tracked file. Attaching them after cythonize
# keeps the compile line identical while taking all of that back out of the
# generated source.
#
# This does not make _fast_encode.c fully machine-independent: `cimport numpy`
# makes Cython emit source-reference comments naming numpy's __init__.pxd by
# path, and suppressing those means turning off code comments in the generated
# C altogether, which is a worse trade. That one file still differs per
# machine; the other two no longer do.
extensions = [
    Extension("dama.ai.ml._fast_encode", sources=["dama/ai/ml/_fast_encode.pyx"]),
    Extension("dama.ai.ml._fast_score", sources=["dama/ai/ml/_fast_score.pyx"]),
    Extension(
        "dama.ai.algorithmic._fast_search",
        sources=["dama/ai/algorithmic/_fast_search.pyx"],
    ),
]

# The corpus metadata helper uses POSIX stat(2) plus C11 threads. Native
# Windows keeps the exact ThreadPoolExecutor fallback in corpus.py.
if platform.system() != "Windows":
    extensions.append(
        Extension("dama.ai.ml._fast_stat", sources=["dama/ai/ml/_fast_stat.pyx"])
    )

_refuse_mapped_inplace_build(extensions)

# ``build_ext --force`` recompiles generated C but does not tell Cython to
# regenerate that C.  Keep both layers coupled: the readiness guard uses a
# forced build after a content mismatch, including preserved-mtime ``.pyx``
# changes that Cython's timestamp check cannot see on its own.
_force_cython = any(
    argument in ("--force", "-f") for argument in sys.argv[1:]
)

ext_modules = cythonize(
    extensions,
    force=_force_cython,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "language_level": "3",
        "profile": False,
        "linetrace": False,
    },
)

for _ext in ext_modules:
    _ext.extra_compile_args = list(_compile_args)
    _ext.extra_link_args = list(_link_args)

# The recursive search keeps several board, move, hash, and heuristic values
# live across calls. Let GCC rename result registers to remove false output and
# anti-dependencies without forcing the loop unrolling that enlarged and slowed
# this extension in earlier measurements.
if platform.system() != "Windows":
    for _ext in ext_modules:
        if _ext.name.endswith("_fast_search"):
            _ext.extra_compile_args.append("-frename-registers")

# The search's capture emission and periodic deadline checks retain external
# libc calls in the hot path. On ELF hosts, address them through the GOT
# directly instead of paying an extra PLT trampoline on every call.
if platform.system() == "Linux":
    for _ext in ext_modules:
        if _ext.name.endswith("_fast_search"):
            _ext.extra_compile_args.append("-fno-plt")

# Keep the branch-heavy move-generation and search loops on full 32-byte fetch
# boundaries on x86 Linux. This is isolated to the search extension so the
# tensor encoders retain their established code layout.
if platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"):
    for _ext in ext_modules:
        if _ext.name.endswith("_fast_search"):
            _ext.extra_compile_args.append("-falign-loops=32")

# _fast_encode is the only one that uses the numpy C API.
for _ext in ext_modules:
    if _ext.name.endswith("_fast_encode"):
        _ext.include_dirs = [np.get_include()]
        _ext.define_macros = [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": AtomicBuildExt},
    distclass=HermeticBuildDistribution,
)
