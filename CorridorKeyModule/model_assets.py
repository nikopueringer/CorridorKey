from __future__ import annotations

import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DEFAULT_CORRIDORKEY_CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"
DEFAULT_GVM_WEIGHTS_DIR = PROJECT_ROOT / "gvm_core" / "weights"
DEFAULT_VIDEOMAMA_CHECKPOINTS_DIR = PROJECT_ROOT / "VideoMaMaInferenceModule" / "checkpoints"

TORCH_EXT = ".pth"
MLX_EXT = ".safetensors"

CORRIDORKEY_REPO_ID = "nikopueringer/CorridorKey_v1.0"
CORRIDORKEY_TORCH_FILENAME = "CorridorKey_v1.0.pth"
CORRIDORKEY_MLX_FILENAME = "corridorkey_mlx.safetensors"
CORRIDORKEY_MLX_REPO = "nikopueringer/corridorkey-mlx"
CORRIDORKEY_MLX_TAG = "v1.0.0"

GVM_REPO_ID = "geyongtao/gvm"
VIDEOMAMA_REPO_ID = "SammyLim/VideoMaMa"
VIDEOMAMA_BASE_REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
VIDEOMAMA_BASE_LICENSE_URL = "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt"

DOWNLOAD_ATTEMPTS = 3
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def mlx_runtime_available() -> bool:
    """Return True when MLX can actually be used on this machine."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return importlib.util.find_spec("corridorkey_mlx") is not None


def ensure_corridorkey_assets(
    *,
    ensure_torch: bool = True,
    ensure_mlx: bool = False,
    download_mlx_if_available: bool = False,
    checkpoint_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Ensure the main CorridorKey checkpoints needed for first-run inference exist.

    Behavior:
      - Torch weights are downloaded when requested and missing.
      - MLX weights are downloaded when explicitly requested and missing.
      - If the checkpoint folder is otherwise empty, MLX is auto-downloaded as well
        when `download_mlx_if_available=True` and the local runtime supports MLX.
    """
    checkpoint_dir = _ensure_dir(checkpoint_dir or DEFAULT_CORRIDORKEY_CHECKPOINT_DIR)
    lock_name = f"corridorkey:{checkpoint_dir.resolve()}"

    with _get_lock(lock_name):
        torch_files = _find_checkpoint_files(checkpoint_dir, TORCH_EXT)
        mlx_files = _find_checkpoint_files(checkpoint_dir, MLX_EXT)
        was_empty = not torch_files and not mlx_files

        if ensure_torch and not torch_files:
            _download_corridorkey_torch(checkpoint_dir)
            torch_files = _find_checkpoint_files(checkpoint_dir, TORCH_EXT)

        should_download_mlx = False
        if ensure_mlx and not mlx_files:
            should_download_mlx = True
        elif download_mlx_if_available and was_empty and not mlx_files and mlx_runtime_available():
            should_download_mlx = True

        if should_download_mlx:
            _download_corridorkey_mlx(checkpoint_dir)

    return checkpoint_dir


def ensure_gvm_weights(weights_dir: str | os.PathLike[str] | None = None) -> Path:
    """Ensure the GVM alpha-generation weights are available locally."""
    weights_dir = _ensure_dir(weights_dir or DEFAULT_GVM_WEIGHTS_DIR)
    lock_name = f"gvm:{weights_dir.resolve()}"

    with _get_lock(lock_name):
        if _gvm_weights_ready(weights_dir):
            return weights_dir

        logger.info("Downloading GVM weights to %s", weights_dir)
        _download_with_retries(
            label="GVM weights",
            action=lambda: snapshot_download(
                repo_id=GVM_REPO_ID,
                local_dir=str(weights_dir),
                local_dir_use_symlinks=False,
                allow_patterns=["vae/*", "scheduler/*", "unet/*"],
            ),
        )

        if not _gvm_weights_ready(weights_dir):
            raise RuntimeError(f"GVM download completed but required files are still missing in {weights_dir}")

    return weights_dir


def ensure_videomama_weights(
    checkpoints_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Ensure both VideoMaMa checkpoints are present locally."""
    checkpoints_dir = _ensure_dir(checkpoints_dir or DEFAULT_VIDEOMAMA_CHECKPOINTS_DIR)
    base_dir = _ensure_dir(checkpoints_dir / "stable-video-diffusion-img2vid-xt")
    unet_dir = _ensure_dir(checkpoints_dir / "VideoMaMa")
    lock_name = f"videomama:{checkpoints_dir.resolve()}"

    with _get_lock(lock_name):
        if not _videomama_base_ready(base_dir):
            logger.info("Downloading VideoMaMa base weights to %s", base_dir)
            _download_with_retries(
                label="VideoMaMa base weights",
                action=lambda: snapshot_download(
                    repo_id=VIDEOMAMA_BASE_REPO_ID,
                    local_dir=str(base_dir),
                    local_dir_use_symlinks=False,
                    allow_patterns=["feature_extractor/*", "image_encoder/*", "vae/*"],
                ),
                gated_repo_url=VIDEOMAMA_BASE_LICENSE_URL,
            )

        if not _videomama_unet_ready(unet_dir):
            logger.info("Downloading VideoMaMa UNet weights to %s", unet_dir)
            _download_with_retries(
                label="VideoMaMa UNet weights",
                action=lambda: snapshot_download(
                    repo_id=VIDEOMAMA_REPO_ID,
                    local_dir=str(unet_dir),
                    local_dir_use_symlinks=False,
                    allow_patterns=["unet/*"],
                ),
            )

        if not _videomama_base_ready(base_dir):
            raise RuntimeError(f"VideoMaMa base download completed but files are still missing in {base_dir}")
        if not _videomama_unet_ready(unet_dir):
            raise RuntimeError(f"VideoMaMa UNet download completed but files are still missing in {unet_dir}")

    return base_dir, unet_dir


def _download_corridorkey_torch(checkpoint_dir: Path) -> Path:
    logger.info("Downloading CorridorKey Torch checkpoint to %s", checkpoint_dir)

    cached_path = _download_with_retries(
        label="CorridorKey Torch checkpoint",
        action=lambda: hf_hub_download(
            repo_id=CORRIDORKEY_REPO_ID,
            filename=CORRIDORKEY_TORCH_FILENAME,
        ),
    )

    destination = checkpoint_dir / CORRIDORKEY_TORCH_FILENAME
    _copy_atomic(Path(cached_path), destination)
    return destination


def _download_corridorkey_mlx(checkpoint_dir: Path) -> Path:
    if not mlx_runtime_available():
        raise RuntimeError("MLX weights were requested, but MLX is not available on this machine.")

    logger.info("Downloading CorridorKey MLX checkpoint to %s", checkpoint_dir)
    release_tag = os.environ.get("CORRIDORKEY_MLX_WEIGHTS_TAG", CORRIDORKEY_MLX_TAG)
    repo_override = os.environ.get("CORRIDORKEY_MLX_WEIGHTS_REPO", CORRIDORKEY_MLX_REPO)

    def action() -> Path:
        command = [
            sys.executable,
            "-m",
            "corridorkey_mlx",
            "weights",
            "download",
            "--tag",
            release_tag,
            "--asset",
            CORRIDORKEY_MLX_FILENAME,
            "--print-path",
        ]
        env = os.environ.copy()
        env["CORRIDORKEY_MLX_WEIGHTS_REPO"] = repo_override

        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        except subprocess.CalledProcessError:
            logger.info(
                "corridorkey_mlx CLI download failed for %s@%s; falling back to direct download.",
                repo_override,
                release_tag,
            )
            return _download_corridorkey_mlx_direct(
                checkpoint_dir, repo_override=repo_override, release_tag=release_tag
            )

        cached_path = _extract_path_from_output(completed)
        if cached_path is None or not cached_path.exists():
            logger.info(
                "corridorkey_mlx CLI did not return a usable path for %s@%s; falling back to direct download.",
                repo_override,
                release_tag,
            )
            return _download_corridorkey_mlx_direct(
                checkpoint_dir, repo_override=repo_override, release_tag=release_tag
            )
        return cached_path

    cached_path = _download_with_retries(label="CorridorKey MLX checkpoint", action=action)

    destination = checkpoint_dir / CORRIDORKEY_MLX_FILENAME
    if cached_path.resolve() != destination.resolve():
        _copy_atomic(cached_path, destination)
    return destination


def _download_corridorkey_mlx_direct(checkpoint_dir: Path, *, repo_override: str, release_tag: str) -> Path:
    destination = checkpoint_dir / CORRIDORKEY_MLX_FILENAME
    if destination.exists():
        return destination

    download_url = f"https://github.com/{repo_override}/releases/download/{release_tag}/{CORRIDORKEY_MLX_FILENAME}"
    tmp_path = destination.with_name(f".{destination.name}.download-{os.getpid()}-{threading.get_ident()}")

    try:
        urllib.request.urlretrieve(download_url, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return destination


def _download_with_retries(
    *,
    label: str,
    action: Callable[[], Any],
    gated_repo_url: str | None = None,
) -> Any:
    last_exc: Exception | None = None

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return action()
        except GatedRepoError as exc:
            last_exc = exc
            break
        except RepositoryNotFoundError as exc:
            last_exc = exc
            break
        except subprocess.CalledProcessError as exc:
            last_exc = exc
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError, RuntimeError) as exc:
            last_exc = exc

        if attempt < DOWNLOAD_ATTEMPTS:
            logger.warning("%s download failed on attempt %d/%d. Retrying...", label, attempt, DOWNLOAD_ATTEMPTS)
            time.sleep(min(2**attempt, 8))

    assert last_exc is not None
    raise _wrap_download_error(label=label, exc=last_exc, gated_repo_url=gated_repo_url) from last_exc


def _wrap_download_error(*, label: str, exc: Exception, gated_repo_url: str | None = None) -> RuntimeError:
    if isinstance(exc, GatedRepoError):
        if gated_repo_url:
            return RuntimeError(
                f"{label} could not be downloaded because the repository is gated. "
                f"Accept the license at {gated_repo_url}, then retry."
            )
        return RuntimeError(f"{label} could not be downloaded because the repository is gated.")

    if isinstance(exc, RepositoryNotFoundError):
        return RuntimeError(f"{label} could not be downloaded because the source repository was not found.")

    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        return RuntimeError(f"{label} download failed: {details}")

    if isinstance(exc, HfHubHTTPError):
        return RuntimeError(f"{label} download failed with an HTTP error: {exc}")

    if isinstance(exc, LocalEntryNotFoundError):
        return RuntimeError(f"{label} download failed because the files could not be fetched from Hugging Face.")

    return RuntimeError(f"{label} download failed: {exc}")


def _find_checkpoint_files(directory: Path, ext: str) -> list[Path]:
    return sorted(path for path in directory.glob(f"*{ext}") if path.is_file())


def _ensure_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    tmp_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}")

    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _extract_path_from_output(completed: subprocess.CompletedProcess[str]) -> Path | None:
    for stream in (completed.stdout, completed.stderr):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        for line in reversed(lines):
            candidate = Path(line).expanduser()
            if candidate.suffix == MLX_EXT:
                return candidate
    return None


def _has_weight_file(directory: Path) -> bool:
    if not directory.exists():
        return False
    return any(
        path.is_file() and path.suffix in {".bin", ".pt", ".pth", ".safetensors"} for path in directory.rglob("*")
    )


def _gvm_weights_ready(weights_dir: Path) -> bool:
    return (
        _has_weight_file(weights_dir / "vae")
        and (weights_dir / "scheduler" / "scheduler_config.json").is_file()
        and _has_weight_file(weights_dir / "unet")
    )


def _videomama_base_ready(base_dir: Path) -> bool:
    return (
        (base_dir / "feature_extractor" / "preprocessor_config.json").is_file()
        and _has_weight_file(base_dir / "image_encoder")
        and _has_weight_file(base_dir / "vae")
    )


def _videomama_unet_ready(unet_dir: Path) -> bool:
    return _has_weight_file(unet_dir / "unet")


def _get_lock(name: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(name)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[name] = lock
        return lock
