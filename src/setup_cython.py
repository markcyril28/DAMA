"""Build Cython extensions for performance-critical encoding functions."""

import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Compiler optimization flags — these are the single highest-impact change
# for CPU-bound self-play throughput. The alpha-beta search spends 99% of
# game time in C; -O3 enables auto-vectorization, loop unrolling, and
# inlining that the default -O2 misses. -march=native targets the exact
# CPU (AVX2/SSE4.2 on modern x86). -ffast-math allows reordering of FP
# operations (safe here: no NaN/Inf edge cases in evaluation or encoding).
_compile_args = ["-O3", "-ffast-math"]
_link_args = []
if platform.machine() in ("x86_64", "AMD64"):
    _compile_args.append("-march=native")

extensions = [
    Extension(
        "dama.ai.ml._fast_encode",
        sources=["dama/ai/ml/_fast_encode.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=_compile_args,
        extra_link_args=_link_args,
    ),
    Extension(
        "dama.ai.ml._fast_score",
        sources=["dama/ai/ml/_fast_score.pyx"],
        extra_compile_args=_compile_args,
        extra_link_args=_link_args,
    ),
    Extension(
        "dama.ai.algorithmic._fast_search",
        sources=["dama/ai/algorithmic/_fast_search.pyx"],
        extra_compile_args=_compile_args,
        extra_link_args=_link_args,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
            "profile": False,
            "linetrace": False,
        },
    ),
)
