"""Inference model download and presence checks.

The CLI's ``init`` command calls these functions. Download logic lives here
(Application Layer) so the CLI stays thin and other consumers can reuse it.

The inference model is a single checkpoint file. Its URL and expected filename
are defined as module-level constants so they can be updated in one place.

Alpha generator models are owned by their respective packages (e.g.
``corridorkey-gbm``) and are not managed here.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

from corridorkey_core.engine_factory import MLX_EXT, TORCH_EXT

from corridorkey.config import CorridorKeyConfig

logger = logging.getLogger(__name__)

# Inference model download URL - update when a new checkpoint is released.
MODEL_DOWNLOAD_URL = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"

# Expected filename after download.
MODEL_FILENAME = "CorridorKey_v1.0.pth"

# SHA-256 checksum of the expected file. Empty string disables verification.
MODEL_CHECKSUM_SHA256 = "a03827f58e8c79b2ca26031bf67c77db5390dc1718c1ffc5b7aed8b57315788f"


def is_model_present(config: CorridorKeyConfig) -> bool:
    """Return True if at least one valid checkpoint exists in checkpoint_dir.

    Checks for both Torch (.pth) and MLX (.safetensors) checkpoints so the
    function works regardless of which backend the user has installed.

    Args:
        config: Loaded CorridorKeyConfig with a resolved checkpoint_dir.

    Returns:
        True if one or more checkpoint files are found.
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return False
    return any(any(checkpoint_dir.glob(f"*{ext}")) for ext in (TORCH_EXT, MLX_EXT))


def download_model(
    config: CorridorKeyConfig,
    on_progress: Callable[[int, int], None] | None = None,
    url: str | None = None,
    filename: str | None = None,
    checksum: str = MODEL_CHECKSUM_SHA256,
) -> Path:
    """Download the inference model checkpoint into checkpoint_dir.

    Uses an atomic write (download to a .tmp file, then os.replace) so a
    partial download never leaves a corrupt checkpoint in place.

    The URL and filename are resolved in this order:
        1. Explicit ``url``/``filename`` arguments (highest priority)
        2. ``config.model_download_url`` / ``config.model_filename``
        3. Built-in ``MODEL_DOWNLOAD_URL`` / ``MODEL_FILENAME`` constants

    Args:
        config: Loaded CorridorKeyConfig. checkpoint_dir is created if missing.
        on_progress: Called with (bytes_downloaded, total_bytes). total_bytes
            is 0 when the server does not report Content-Length.
        url: Override download URL. Defaults to config or built-in constant.
        filename: Override target filename. Defaults to config or built-in constant.
        checksum: Expected SHA-256 hex digest. Pass an empty string to skip
            verification.

    Returns:
        Absolute path to the downloaded checkpoint file.

    Raises:
        RuntimeError: If the download fails or the checksum does not match.
    """
    resolved_url = url or config.model_download_url or MODEL_DOWNLOAD_URL
    resolved_filename = filename or config.model_filename or MODEL_FILENAME
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dest = checkpoint_dir / resolved_filename
    tmp = checkpoint_dir / (resolved_filename + ".tmp")

    logger.info("Downloading inference model from %s", resolved_url)
    logger.info("Destination: %s", dest)

    try:
        with urllib.request.urlopen(resolved_url) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 64  # 64 KB

            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total)

    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Model download failed: {e}") from e

    if checksum:
        logger.info("Verifying checksum...")
        actual = _sha256(tmp)
        if actual != checksum.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {resolved_filename}.\n"
                f"  Expected: {checksum}\n"
                f"  Got:      {actual}\n"
                "The downloaded file has been removed. Try again."
            )
        logger.info("Checksum OK")

    os.replace(tmp, dest)
    logger.info("Model saved: %s", dest)
    return dest


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()
