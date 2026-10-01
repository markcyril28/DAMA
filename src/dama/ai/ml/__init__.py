"""Machine Learning based AI opponent."""

import sys

if sys.platform == 'darwin':
    # macOS loads the *_mac.py variants of the modules whose CUDA/Linux code
    # the Apple Silicon port cannot share (see _platform_mac).
    from . import _platform_mac
    _platform_mac.install()
