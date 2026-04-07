"""Centralized cross-platform device selection for CorridorKey."""

import logging
import os

logger = logging.getLogger(__name__)

DEVICE_ENV_VAR = "CORRIDORKEY_DEVICE"
VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def is_rocm_system() -> bool:
    """Detect if the system has AMD ROCm available (without importing torch).

    Checks: /opt/rocm (Linux), HIP_PATH env var (Windows), HIP_VISIBLE_DEVICES
    (any platform), CORRIDORKEY_ROCM=1 (explicit opt-in for cases where
    auto-detection fails, e.g. pip-installed ROCm on Windows).
    """
    return (
        os.path.exists("/opt/rocm")
        or os.environ.get("HIP_PATH") is not None
        or os.environ.get("HIP_VISIBLE_DEVICES") is not None
        or os.environ.get("CORRIDORKEY_ROCM") == "1"
    )


def setup_rocm_env() -> None:
    """Set ROCm environment variables and apply optional patches.

    Must be called before importing torch so that env vars are visible to
    PyTorch's initialization. This module intentionally avoids importing
    torch at module level to make that possible. Safe to call on non-ROCm
    systems (no-op).
    """
    if not is_rocm_system():
        return
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    # Level 4 = suppress info/debug but keep warnings and errors visible
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")
    # Enable GTT (system RAM as GPU overflow) on Linux for 16GB cards.
    # pytorch-rocm-gtt must be installed separately: pip install pytorch-rocm-gtt
    try:
        import pytorch_rocm_gtt

        pytorch_rocm_gtt.patch()
    except ImportError:
        pass  # not installed — expected on most systems
    except Exception:
        logger.warning("pytorch-rocm-gtt is installed but patch() failed", exc_info=True)


def detect_best_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Auto-selected device: %s", device)
    return device


def resolve_device(requested: str | None = None) -> str:
    """Resolve device from explicit request > env var > auto-detect.

    Args:
        requested: Device string from CLI arg. None or "auto" triggers
                   env var lookup then auto-detection.

    Returns:
        Validated device string ("cuda", "mps", or "cpu").

    Raises:
        RuntimeError: If the requested backend is unavailable.
    """
    import torch

    # CLI arg takes priority, then env var, then auto
    device = requested
    if device is None or device == "auto":
        device = os.environ.get(DEVICE_ENV_VAR, "auto")

    if device == "auto":
        return detect_best_device()

    device = device.lower()
    if device not in VALID_DEVICES:
        raise RuntimeError(f"Unknown device '{device}'. Valid options: {', '.join(VALID_DEVICES)}")

    # Validate the explicit request
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build."
            )
    elif device == "mps":
        if not hasattr(torch.backends, "mps"):
            raise RuntimeError(
                "MPS requested but this PyTorch build has no MPS support. Install PyTorch >= 1.12 with MPS backend."
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but not available on this machine. Requires Apple Silicon (M1+) with macOS 12.3+."
            )

    return device


def clear_device_cache(device) -> None:
    """Clear GPU memory cache if applicable (no-op for CPU)."""
    import torch

    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()
