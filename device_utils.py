"""Centralized cross-platform device selection for CorridorKey."""

import json
import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEVICE_ENV_VAR = "CORRIDORKEY_DEVICE"
VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def is_rocm_system() -> bool:
    """Detect if the system has AMD ROCm available (before or after torch import).

    Checks: /opt/rocm (Linux), HIP_PATH (Windows, default C:\\hip),
    HIP_VISIBLE_DEVICES (any platform), CORRIDORKEY_ROCM=1 (explicit opt-in).
    """
    return (
        os.path.exists("/opt/rocm")
        or os.environ.get("HIP_PATH") is not None
        or os.environ.get("HIP_VISIBLE_DEVICES") is not None
        or os.environ.get("CORRIDORKEY_ROCM") == "1"
    )


def setup_rocm_env() -> None:
    """Set ROCm environment variables and apply optional patches.

    Must be called before importing torch. Safe to call on non-ROCm systems (no-op).
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
    except Exception:
        pass  # not installed, or patch failed — non-fatal


import torch  # noqa: E402 — deferred so setup_rocm_env() can run first


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


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float


def _enumerate_nvidia() -> list[GPUInfo] | None:
    """Enumerate NVIDIA GPUs via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        gpus: list[GPUInfo] = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vram_total_gb=float(parts[2]) / 1024,
                        vram_free_gb=float(parts[3]) / 1024,
                    )
                )
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _enumerate_amd() -> list[GPUInfo] | None:
    """Enumerate AMD GPUs via amd-smi (ROCm). Returns None if unavailable.

    Tries amd-smi first (modern), then rocm-smi (legacy).
    """
    # Try amd-smi (ROCm 6.0+)
    try:
        result = subprocess.run(
            ["amd-smi", "static", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            gpus: list[GPUInfo] = []
            for i, gpu in enumerate(data):
                try:
                    name = gpu.get("asic", {}).get("market_name", f"AMD GPU {i}")
                    vram_info = gpu.get("vram", {})
                    total_mb = vram_info.get("size", {}).get("value", 0)
                    total_gb = float(total_mb) / 1024 if total_mb else 0
                    gpus.append(GPUInfo(index=i, name=name, vram_total_gb=total_gb, vram_free_gb=total_gb))
                except (KeyError, TypeError, ValueError):
                    logger.debug("Failed to parse amd-smi entry %d, skipping", i)
            if gpus:
                # Try to get live VRAM usage from monitor
                try:
                    mon = subprocess.run(
                        ["amd-smi", "monitor", "--vram", "--json"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if mon.returncode == 0:
                        mon_data = json.loads(mon.stdout)
                        for entry in mon_data:
                            idx = entry.get("gpu", 0)
                            used_pct = entry.get("vram_use", 0)
                            if idx < len(gpus) and gpus[idx].vram_total_gb > 0:
                                used_gb = gpus[idx].vram_total_gb * float(used_pct) / 100
                                gpus[idx].vram_free_gb = gpus[idx].vram_total_gb - used_gb
                except Exception:
                    pass
                return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # Fallback: rocm-smi (legacy, deprecated but still ships)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n")[1:]:  # skip header
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx = int(parts[0]) if parts[0].isdigit() else len(gpus)
                    total_b = int(parts[1]) if parts[1].isdigit() else 0
                    used_b = int(parts[2]) if parts[2].isdigit() else 0
                    gpus.append(
                        GPUInfo(
                            index=idx,
                            name=f"AMD GPU {idx}",
                            vram_total_gb=total_b / (1024**3),
                            vram_free_gb=(total_b - used_b) / (1024**3),
                        )
                    )
            if gpus:
                return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def enumerate_gpus() -> list[GPUInfo]:
    """List all available GPUs with VRAM info.

    Tries nvidia-smi (NVIDIA), then amd-smi/rocm-smi (AMD ROCm),
    then falls back to torch.cuda API.
    Returns an empty list on non-GPU systems.
    """
    # NVIDIA
    gpus = _enumerate_nvidia()
    if gpus is not None:
        return gpus

    # AMD ROCm
    gpus = _enumerate_amd()
    if gpus is not None:
        return gpus

    # Fallback to torch (works for both NVIDIA and ROCm via HIP)
    if torch.cuda.is_available():
        fallback: list[GPUInfo] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024**3)
            fallback.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    vram_total_gb=total,
                    vram_free_gb=total,  # can't query free without setting device
                )
            )
        return fallback

    return []


def clear_device_cache(device: torch.device | str) -> None:
    """Clear GPU memory cache if applicable (no-op for CPU)."""
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()
