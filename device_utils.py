"""Centralized cross-platform device selection for CorridorKey."""

import logging
import os

import torch

logger = logging.getLogger(__name__)

DEVICE_ENV_VAR = "CORRIDORKEY_DEVICE"
VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def detect_best_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
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


def clear_device_cache(device: torch.device | str) -> None:
    """Clear GPU memory cache if applicable (no-op for CPU)."""
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


class RuntimeThreadPool:
    """Context manager to handle parallel I/O with constrained compute threads.

    On high-core-count systems (e.g. Threadripper), nested thread pools in
    OpenCV, PyTorch, and OpenEXR can cause massive context-switching overhead
    when running inside a top-level ThreadPoolExecutor.

    This context manager:
    1. Determines optimal worker count based on CPU and workload type.
    2. Temporarily constrains library internal threads to 1 per worker.
    3. Provides a managed ThreadPoolExecutor for I/O-heavy tasks.
    """

    def __init__(self, is_video: bool):
        self.is_video = is_video
        self.max_workers = 1 if is_video else min(32, (os.cpu_count() or 1) + 4)
        self.executor = None
        self._prev_cv2_threads = None
        self._prev_torch_threads = None
        self._prev_exr_threads = os.environ.get("OPENEXR_NUM_THREADS")

    def __enter__(self):
        # VideoCapture is not thread-safe; use a single worker
        if self.is_video:
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=1)
            return self.executor

        # Constrain library threads to prevent thrashing
        import cv2
        import torch

        self._prev_cv2_threads = cv2.getNumThreads()
        self._prev_torch_threads = torch.get_num_threads()

        cv2.setNumThreads(0)
        torch.set_num_threads(1)
        os.environ["OPENEXR_NUM_THREADS"] = "1"

        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self.executor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)

        if not self.is_video:
            import cv2
            import torch

            if self._prev_cv2_threads is not None:
                cv2.setNumThreads(self._prev_cv2_threads)
            if self._prev_torch_threads is not None:
                torch.set_num_threads(self._prev_torch_threads)

            if self._prev_exr_threads is not None:
                os.environ["OPENEXR_NUM_THREADS"] = self._prev_exr_threads
            else:
                os.environ.pop("OPENEXR_NUM_THREADS", None)
