"""Backend-neutral helpers for the three devices the trainer supports.

The training code was written against CUDA (NVIDIA locally, ROCm/HIP on the
server, which PyTorch also exposes as ``cuda``). Apple Silicon Macs have no
CUDA; their GPU is reached through Metal Performance Shaders (``mps``). These
helpers give each CUDA-only call site one place to ask "what does this device
support" instead of repeating ``torch.cuda.*`` guards that raise on a Mac.

Every function accepts a ``torch.device``, a device string, or ``None``
(meaning CPU) and degrades to a harmless no-op / zero on backends that lack
the feature, so callers never need their own try/except.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional, Union

import torch

DeviceLike = Union[torch.device, str, None]

# 'auto' resolves to the best backend present; the rest name one explicitly.
DEVICE_CHOICES = ('auto', 'cuda', 'mps', 'cpu')


def _type(device: DeviceLike) -> str:
    if device is None:
        return 'cpu'
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


def mps_available() -> bool:
    """True when this PyTorch build can run on an Apple GPU right now."""
    backend = getattr(torch.backends, 'mps', None)
    try:
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def is_available(device: DeviceLike) -> bool:
    kind = _type(device)
    if kind == 'cuda':
        return torch.cuda.is_available()
    if kind == 'mps':
        return mps_available()
    return kind == 'cpu'


def default_device_type() -> str:
    """Best available backend: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    if mps_available():
        return 'mps'
    return 'cpu'


def resolve_device_type(requested: Optional[str]) -> str:
    """Map a configured device name ('auto' included) to a concrete type."""
    if requested is None or str(requested).lower() == 'auto':
        return default_device_type()
    return torch.device(str(requested).lower()).type


def is_gpu(device: DeviceLike) -> bool:
    return _type(device) in ('cuda', 'mps')


def synchronize(device: DeviceLike) -> None:
    """Block until queued kernels on ``device`` finish (no-op on CPU)."""
    kind = _type(device)
    if kind == 'cuda':
        torch.cuda.current_stream().synchronize()
    elif kind == 'mps':
        torch.mps.synchronize()


def empty_cache(device: DeviceLike) -> None:
    kind = _type(device)
    if kind == 'cuda':
        torch.cuda.empty_cache()
    elif kind == 'mps':
        torch.mps.empty_cache()


def memory_allocated(device: DeviceLike) -> int:
    """Bytes currently held by live tensors on ``device`` (0 for CPU)."""
    kind = _type(device)
    try:
        if kind == 'cuda':
            return int(torch.cuda.memory_allocated(device))
        if kind == 'mps':
            return int(torch.mps.current_allocated_memory())
    except Exception:
        pass
    return 0


def memory_reserved(device: DeviceLike) -> int:
    """Bytes the allocator holds from the driver, cached blocks included."""
    kind = _type(device)
    try:
        if kind == 'cuda':
            return int(torch.cuda.memory_reserved(device))
        if kind == 'mps':
            return int(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    return 0


def _system_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def total_memory(device: DeviceLike) -> int:
    """Memory the device may use, in bytes.

    On Apple Silicon the GPU shares system RAM, so this is Metal's
    recommended working-set limit (about 75% of RAM) rather than a VRAM size.
    """
    kind = _type(device)
    try:
        if kind == 'cuda':
            return int(torch.cuda.get_device_properties(device).total_memory)
        if kind == 'mps':
            recommended = getattr(torch.mps, 'recommended_max_memory', None)
            if recommended is not None:
                return int(recommended())
            return int(_system_ram_bytes() * 0.75)
    except Exception:
        pass
    return _system_ram_bytes()


def free_memory(device: DeviceLike) -> int:
    return max(0, total_memory(device) - memory_allocated(device))


def _mac_chip_name() -> str:
    try:
        out = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return 'Apple Silicon'


def device_name(device: DeviceLike) -> str:
    kind = _type(device)
    try:
        if kind == 'cuda':
            return torch.cuda.get_device_name(device)
        if kind == 'mps':
            return f"{_mac_chip_name()} GPU (MPS)"
    except Exception:
        pass
    return platform.processor() or platform.machine() or 'CPU'


def manual_seed_all(seed: int) -> None:
    """Seed every accelerator RNG present (torch.manual_seed covers CPU)."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_available():
        torch.mps.manual_seed(seed)


def bf16_supported(device: DeviceLike) -> bool:
    kind = _type(device)
    if kind == 'cuda':
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False
    if kind == 'mps':
        # bfloat16 on MPS needs macOS 14+ and a recent PyTorch; probe it.
        try:
            x = torch.ones(2, 2, device='mps', dtype=torch.bfloat16)
            (x @ x).sum().item()
            return True
        except Exception:
            return False
    return True


def amp_supported(device: DeviceLike) -> bool:
    """Whether torch.autocast works for this device type in this build."""
    kind = _type(device)
    if kind == 'cuda':
        return True
    if kind == 'mps':
        try:
            with torch.autocast(device_type='mps', dtype=torch.float16):
                pass
            return True
        except Exception:
            return False
    return False


def compile_supported(device: DeviceLike) -> bool:
    """torch.compile is only exercised (and its Triton/cudagraph settings
    only apply) on CUDA; the MPS Inductor backend is still experimental."""
    return _type(device) == 'cuda'
