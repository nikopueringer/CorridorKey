"""System info endpoints — device, VRAM, model unloading, weight downloads."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading

from fastapi import APIRouter, HTTPException

from ..deps import get_service
from ..schemas import DeviceResponse, VRAMResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["system"])

# Base dir for weight paths
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Weight download state
_download_status: dict[str, dict] = {}
_download_lock = threading.Lock()


def _weights_info() -> dict:
    """Check which weights are installed."""
    ck_dir = os.path.join(_BASE_DIR, "CorridorKeyModule", "checkpoints")
    ck_files = [f for f in os.listdir(ck_dir) if f.endswith(".pth")] if os.path.isdir(ck_dir) else []

    gvm_dir = os.path.join(_BASE_DIR, "gvm_core", "weights")
    # GVM weights have vae/, unet/, scheduler/ subdirs — check for unet/config.json
    gvm_config = os.path.isfile(os.path.join(gvm_dir, "unet", "config.json")) if os.path.isdir(gvm_dir) else False

    vm_dir = os.path.join(_BASE_DIR, "VideoMaMaInferenceModule", "checkpoints", "VideoMaMa")
    vm_exists = os.path.isdir(vm_dir) and len(os.listdir(vm_dir)) > 0

    return {
        "corridorkey": {
            "installed": len(ck_files) > 0,
            "path": ck_dir,
            "detail": ck_files[0] if ck_files else None,
            "size_hint": "~300 MB",
        },
        "gvm": {
            "installed": gvm_config,
            "path": gvm_dir,
            "detail": "vae + unet + scheduler" if gvm_config else None,
            "size_hint": "~10 GB",
        },
        "videomama": {
            "installed": vm_exists,
            "path": vm_dir,
            "detail": f"{len(os.listdir(vm_dir))} files" if vm_exists else None,
            "size_hint": "~5 GB",
        },
    }


@router.get("/device", response_model=DeviceResponse)
def get_device():
    service = get_service()
    return DeviceResponse(device=service._device)


def _nvidia_smi_vram() -> dict | None:
    """Query nvidia-smi for system-wide VRAM usage (all processes, not just PyTorch)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,name", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]
        total_mb, used_mb, free_mb, name = [x.strip() for x in line.split(",")]
        total = float(total_mb) / 1024
        used = float(used_mb) / 1024
        free = float(free_mb) / 1024
        return {
            "total": total,
            "reserved": used,
            "allocated": used,
            "free": free,
            "name": name,
        }
    except Exception:
        return None


@router.get("/vram", response_model=VRAMResponse)
def get_vram():
    # Prefer nvidia-smi for system-wide VRAM (includes other processes like Unreal)
    smi = _nvidia_smi_vram()
    if smi:
        return VRAMResponse(
            total=smi["total"],
            reserved=smi["reserved"],
            allocated=smi["allocated"],
            free=smi["free"],
            name=smi["name"],
            available=True,
        )
    # Fallback to PyTorch (only sees its own allocations)
    service = get_service()
    info = service.get_vram_info()
    if not info:
        return VRAMResponse(available=False)
    return VRAMResponse(
        total=info.get("total", 0),
        reserved=info.get("reserved", 0),
        allocated=info.get("allocated", 0),
        free=info.get("free", 0),
        name=info.get("name", ""),
        available=True,
    )


@router.get("/vram-limit")
def get_vram_limit_setting():
    from ..worker import get_vram_limit

    return {"vram_limit_gb": get_vram_limit()}


@router.post("/vram-limit")
def set_vram_limit_setting(vram_limit_gb: float):
    from ..worker import set_vram_limit

    set_vram_limit(vram_limit_gb)
    return {"status": "ok", "vram_limit_gb": vram_limit_gb}


@router.post("/unload")
def unload_engines():
    service = get_service()
    service.unload_engines()
    return {"status": "unloaded"}


@router.get("/weights")
def get_weights():
    """Check installed weights status."""
    info = _weights_info()
    # Merge in download status
    with _download_lock:
        for key, status in _download_status.items():
            if key in info:
                info[key]["download"] = status
    return info


def _run_download(name: str, cmd: list[str], target_dir: str) -> None:
    """Run a weight download in a background thread."""
    try:
        with _download_lock:
            _download_status[name] = {"status": "downloading", "error": None}

        logger.info(f"Starting weight download: {name} → {target_dir}")
        os.makedirs(target_dir, exist_ok=True)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_BASE_DIR,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            error = result.stderr.strip()[-500:] if result.stderr else "Unknown error"
            logger.error(f"Weight download failed for {name}: {error}")
            with _download_lock:
                _download_status[name] = {"status": "failed", "error": error}
        else:
            logger.info(f"Weight download complete: {name}")
            with _download_lock:
                _download_status[name] = {"status": "complete", "error": None}

    except subprocess.TimeoutExpired:
        with _download_lock:
            _download_status[name] = {"status": "failed", "error": "Download timed out (1 hour)"}
    except Exception as e:
        with _download_lock:
            _download_status[name] = {"status": "failed", "error": str(e)}


def _hf_bin() -> str:
    """Find the huggingface-hub CLI binary."""
    import shutil as _shutil

    # Check the venv first (covers Docker where PATH may not include .venv/bin)
    venv_hf = os.path.join(_BASE_DIR, ".venv", "bin", "huggingface-cli")
    if os.path.isfile(venv_hf):
        return venv_hf
    # Also check for 'hf' alias
    venv_hf2 = os.path.join(_BASE_DIR, ".venv", "bin", "hf")
    if os.path.isfile(venv_hf2):
        return venv_hf2
    # Try PATH
    found = _shutil.which("huggingface-cli") or _shutil.which("hf")
    if found:
        return found
    # Last resort: run via python -m
    return "huggingface-cli"


@router.post("/weights/download/{name}")
def download_weights(name: str):
    """Start downloading weights for a model. Runs in the background."""
    with _download_lock:
        if name in _download_status and _download_status[name].get("status") == "downloading":
            return {"status": "already_downloading"}

    hf = _hf_bin()

    # Build the download command — use python -m as fallback if hf CLI not found
    python_bin = os.path.join(_BASE_DIR, ".venv", "bin", "python")
    if not os.path.isfile(python_bin):
        python_bin = sys.executable

    def _dl_cmd(repo: str, local_dir: str, extra_args: list[str] | None = None) -> list[str]:
        """Build huggingface download command with python -m fallback."""
        try:
            # Test if hf CLI works
            subprocess.run([hf, "--version"], capture_output=True, timeout=5)
            cmd = [hf, "download", repo, "--local-dir", local_dir]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            cmd = [python_bin, "-m", "huggingface_hub", "download", repo, "--local-dir", local_dir]
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    if name == "corridorkey":
        target = os.path.join(_BASE_DIR, "CorridorKeyModule", "checkpoints")
        cmd = _dl_cmd("nikopueringer/CorridorKey_v1.0", target, ["CorridorKey_v1.0.pth"])
        thread = threading.Thread(target=_run_download, args=(name, cmd, target), daemon=True)
        thread.start()
        return {"status": "started", "size_hint": "~300 MB"}

    elif name == "gvm":
        target = os.path.join(_BASE_DIR, "gvm_core", "weights")
        cmd = _dl_cmd("geyongtao/gvm", target)
        thread = threading.Thread(target=_run_download, args=(name, cmd, target), daemon=True)
        thread.start()
        return {"status": "started", "size_hint": "~10 GB"}

    elif name == "videomama":
        target = os.path.join(_BASE_DIR, "VideoMaMaInferenceModule", "checkpoints", "VideoMaMa")
        cmd = _dl_cmd("SammyLim/VideoMaMa", target)
        thread = threading.Thread(target=_run_download, args=(name, cmd, target), daemon=True)
        thread.start()
        return {"status": "started", "size_hint": "~5 GB"}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown weight set: {name}. Valid: corridorkey, gvm, videomama")
