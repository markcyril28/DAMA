"""Build Cython extensions for performance-critical encoding functions."""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "dama.ai.ml._fast_encode",
        sources=["dama/ai/ml/_fast_encode.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "dama.ai.ml._fast_score",
        sources=["dama/ai/ml/_fast_score.pyx"],
    ),
    Extension(
        "dama.ai.algorithmic._fast_search",
        sources=["dama/ai/algorithmic/_fast_search.pyx"],
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
        },
    ),
)
