"""Build Cython extensions for performance-critical encoding functions."""

from Cython.Build import cythonize
from setuptools import Extension, setup
import platform
import numpy as np


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

ext_modules = cythonize(
    extensions,
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

# _fast_encode is the only one that uses the numpy C API.
for _ext in ext_modules:
    if _ext.name.endswith("_fast_encode"):
        _ext.include_dirs = [np.get_include()]
        _ext.define_macros = [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

setup(ext_modules=ext_modules)

