"""Shared value types for the Laya bridge client and move policy."""

from dataclasses import dataclass
from typing import Tuple

# The worker's truncation check follows laya.common.build_sequence, a private
# function; any other version must be re-verified before it is allowed.
SUPPORTED_LAYA_VERSIONS: Tuple[str, ...] = ("0.3.6",)


@dataclass(frozen=True)
class BridgeSpec:
    """Everything that identifies one Laya worker process."""
    python: str  # absolute path of the interpreter that has laya installed
    model: str = "convaiinnovations/laya"  # hub repo id or local checkpoint directory
    subfolder: str = ""  # "" = English root, or "multilingual" / "typed-decisions"
    device: str = "auto"  # auto, cuda, cpu
    hf_home: str = ""  # "" = inherit HF_HOME from the environment
    offline: bool = True  # HF_HUB_OFFLINE=1: never download weights implicitly
    startup_timeout_sec: float = 180.0
    request_timeout_sec: float = 60.0
    idle_shutdown_sec: float = 900.0  # 0 disables the idle shutdown

    def process_key(self) -> tuple:
        """Fields that require a new worker process when they change."""
        return (self.python, self.model, self.subfolder, self.device,
                self.hf_home, self.offline)


@dataclass(frozen=True)
class BridgeChoice:
    """One answer from the worker's ``choose`` operation."""
    probabilities: Tuple[float, ...]  # aligned to the options that were sent
    choice_index: int  # argmax, as an index into the options that were sent
    confidence: float
    device: str  # device the forward pass ran on: cuda or cpu
    elapsed_ms: float  # worker-side time for the predict call
    head_max_len: int  # head budget used for this call (raised only when needed)
