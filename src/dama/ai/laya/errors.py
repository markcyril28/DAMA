"""Errors raised by the Laya player."""


class LayaError(RuntimeError):
    """Base class for Laya player failures."""


class LayaUnavailableError(LayaError):
    """The Laya bridge cannot be started or used.

    Covers a missing interpreter or package, a failed checkpoint load, a crashed
    or timed-out worker, an unsupported laya version, and a device mismatch.
    """


class LayaBudgetError(LayaError):
    """The move options or the state do not fit Laya's input without truncation."""
