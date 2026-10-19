"""Timing instrumentation.

A GPU-synchronised timestamp and a wall-clock context manager, for splitting a
training step or an inference call into its phases. Orthogonal to the logic it
measures, which is why the same helpers serve both the model internals
(``modeling_*``) and the deploy scripts.
"""

import time
from contextlib import contextmanager
from typing import Dict, Optional

import torch

from tau0_vla.utils.device import synchronize


def cuda_sync() -> None:
    """Synchronize the accelerator stream; a no-op without one.

    Kept under its historical name because ``tau0_vla`` callers and the deploy
    scripts import it; it dispatches to XPU, CUDA or CPU.
    """
    synchronize()


def sync_ts(enabled: bool) -> float:
    """Return the current time in seconds, syncing the CUDA stream first when ``enabled``.

    For timing **purely asynchronous GPU sub-stages**: kernels are dispatched
    asynchronously, so without a sync ``perf_counter`` measures how long the CPU
    took to dispatch rather than how long the GPU took to run. Adjacent stages can
    share one return value as their common boundary, halving the number of syncs.

    Args:
        enabled: Whether timing is on. When off, returns ``0.0`` immediately and
            costs no synchronization.

    Returns:
        The value of ``time.perf_counter()``, or ``0.0`` when ``enabled=False``.
    """
    if enabled:
        cuda_sync()
        return time.perf_counter()
    return 0.0
