"""Device-agnostic helpers for the XPU / CUDA / CPU backends.

Every place that used to ask ``torch.cuda`` directly goes through this module, so
the training, deploy and proposal paths share one code path across accelerators
instead of forking per device.

Two rules this module encodes, both learned from experience:

* **A missing CUDA build must never be a crash.** ``torch.cuda.synchronize()``
  raises ``AssertionError: Torch not compiled with CUDA enabled`` on a wheel
  built without CUDA, which is what an XPU or CPU-only install has. Callers use
  :func:`synchronize` / :func:`empty_cache`, which are no-ops when the requested
  backend is unavailable.
* **Autocast state is per device type.** ``torch.get_autocast_gpu_dtype()``
  always reports the *CUDA* dtype (float16 by default) even when an XPU autocast
  context is active, where the dtype is bfloat16. Attention kernels that read it
  then cast activations to the wrong dtype, so use :func:`autocast_dtype`.

Detection order is XPU, then CUDA, then CPU: a host can expose an Intel GPU next
to an NVIDIA one, and an explicit ``device=`` argument always wins over
detection.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

__all__ = [
    "autocast_dtype",
    "available_device_types",
    "default_device",
    "default_device_type",
    "device_count",
    "device_type",
    "empty_cache",
    "get_device_rng_state",
    "is_autocast_enabled",
    "is_available",
    "resolve_compute_dtype",
    "resolve_device",
    "set_device_rng_state",
    "synchronize",
]

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in *this* build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def available_device_types() -> list[str]:
    """Accelerator types present on this host, in detection order, ending with ``cpu``."""
    return [device_type for device_type in _ACCELERATORS if is_available(device_type)] + ["cpu"]


def default_device_type() -> str:
    """Best available backend: ``xpu`` > ``cuda`` > ``cpu``."""
    return available_device_types()[0]


def default_device(index: Optional[int] = None) -> torch.device:
    """Device for the best available backend, optionally with an explicit index."""
    device_type = default_device_type()
    if index is None or device_type == "cpu":
        return torch.device(device_type)
    return torch.device(device_type, int(index))


def resolve_device(device: DeviceLike) -> torch.device:
    """Resolve ``None`` / str / :class:`torch.device` to a concrete device.

    ``None`` auto-detects with :func:`default_device`. Explicit values are
    honoured verbatim — an unsupported device still fails loudly at the first
    kernel instead of being silently replaced by a different backend.
    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        return default_device()
    return torch.device(device)


def device_type(device: Any) -> str:
    """Backend type (``"xpu"`` / ``"cuda"`` / ``"cpu"``) of a device, string or tensor."""
    if isinstance(device, torch.Tensor):
        return device.device.type
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def _resolved_type(device: DeviceLike) -> str:
    """Backend type for ``device``, falling back to detection when it is ``None``."""
    return device_type(device) if device is not None else default_device_type()


def device_count(backend: Optional[str] = None) -> int:
    """Number of visible devices of a backend; 0 when the backend is unavailable."""
    resolved = backend or default_device_type()
    if resolved == "xpu":
        return torch.xpu.device_count() if is_available("xpu") else 0
    if resolved == "cuda":
        return torch.cuda.device_count() if is_available("cuda") else 0
    return 0


def synchronize(device: DeviceLike = None) -> None:
    """Synchronize the accelerator stream for ``device``; no-op on CPU.

    Safe on builds without CUDA: ``torch.cuda.synchronize()`` itself raises there,
    so availability is checked first.
    """
    resolved = _resolved_type(device)
    if resolved == "xpu" and is_available("xpu"):
        torch.xpu.synchronize()
    elif resolved == "cuda" and is_available("cuda"):
        torch.cuda.synchronize()


def empty_cache(device: DeviceLike = None) -> None:
    """Release cached allocator blocks for ``device``; no-op on CPU."""
    resolved = _resolved_type(device)
    if resolved == "xpu" and is_available("xpu"):
        torch.xpu.empty_cache()
    elif resolved == "cuda" and is_available("cuda"):
        torch.cuda.empty_cache()


def is_autocast_enabled(device: DeviceLike = None) -> bool:
    """Whether autocast is active for ``device``'s backend."""
    resolved = _resolved_type(device)
    if resolved not in _ACCELERATORS:
        return torch.is_autocast_enabled("cpu")
    try:
        return torch.is_autocast_enabled(resolved)
    except TypeError:  # torch < 2.4 has no per-device argument
        return torch.is_autocast_enabled()


def autocast_dtype(device: DeviceLike = None) -> torch.dtype:
    """Dtype autocast uses for ``device``'s backend (bfloat16 on XPU by default).

    Never hard-code ``torch.get_autocast_gpu_dtype()``: under an XPU autocast
    context it still reports the CUDA dtype (float16), not the XPU one.
    """
    resolved = _resolved_type(device)
    if resolved not in _ACCELERATORS:
        return torch.float32
    try:
        return torch.get_autocast_dtype(resolved)
    except (AttributeError, TypeError):  # torch < 2.4
        return torch.get_autocast_gpu_dtype()


def get_device_rng_state(device: DeviceLike = None) -> Optional[list[torch.Tensor]]:
    """RNG state of the accelerator, or ``None`` when it is unavailable.

    Used to restore determinism around deploy warmup, which samples noise from the
    device RNG.
    """
    resolved = _resolved_type(device)
    if resolved == "xpu" and is_available("xpu"):
        return torch.xpu.get_rng_state_all()
    if resolved == "cuda" and is_available("cuda"):
        return torch.cuda.get_rng_state_all()
    return None


def set_device_rng_state(state: Optional[list[torch.Tensor]], device: DeviceLike = None) -> None:
    """Restore the state returned by :func:`get_device_rng_state`; no-op when ``None``."""
    if state is None:
        return
    resolved = _resolved_type(device)
    if resolved == "xpu" and is_available("xpu"):
        torch.xpu.set_rng_state_all(state)
    elif resolved == "cuda" and is_available("cuda"):
        torch.cuda.set_rng_state_all(state)


def resolve_compute_dtype(module: Any, tensor: torch.Tensor) -> torch.dtype:
    """Dtype to compute in when ``tensor`` arrives as float32 (a LayerNorm output).

    Attention kernels receive fp32 projections because the norm runs in fp32, and
    have to cast back before the kernel. Precedence: the active autocast dtype for
    ``tensor``'s device, the module's quantization dtype, the first Linear weight
    dtype, then the input dtype. Reading ``torch.get_autocast_gpu_dtype()`` here is
    wrong on XPU, where autocast runs bfloat16 while that getter reports float16.
    """
    if is_autocast_enabled(tensor.device):
        return autocast_dtype(tensor.device)
    config = getattr(module, "config", None)
    pre_quantization_dtype = getattr(config, "_pre_quantization_dtype", None)
    if pre_quantization_dtype is not None:
        return pre_quantization_dtype
    if hasattr(module, "modules"):
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                return layer.weight.dtype
    return tensor.dtype
