"""macOS module variants.

The CUDA/Linux code in ``dataset``, ``model``, ``model_vs_algo``,
``selfplay``, ``stats_collector`` and ``trainer`` is left as it is for the
CUDA/ROCm hosts. Where the Apple Silicon (MPS) port conflicts with it, the
macOS version lives beside it as ``<module>_mac.py``.

On macOS, importing ``dama.ai.ml.<module>`` loads ``<module>_mac.py`` under
that same module name, so every ``from .dataset import ...``, pickled class
reference and spawned worker resolves to the macOS variant without any other
file knowing about it. Other platforms never import this module.

Set ``DAMA_DISABLE_MAC_VARIANTS=1`` to load the shared modules on a Mac.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PREFIX = __package__ + '.'

# Modules that have a ``<name>_mac.py`` variant.
MAC_VARIANTS = (
    'dataset',
    'model',
    'model_vs_algo',
    'selfplay',
    'stats_collector',
    'trainer',
)


class _MacVariantFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        name = fullname[len(_PREFIX):]
        if name not in MAC_VARIANTS:
            return None
        return importlib.util.spec_from_file_location(
            fullname, _HERE / f'{name}_mac.py')


def install() -> None:
    """Route the variant modules to their macOS files (macOS only)."""
    if sys.platform != 'darwin' or os.environ.get('DAMA_DISABLE_MAC_VARIANTS'):
        return
    if not any(isinstance(f, _MacVariantFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MacVariantFinder())
