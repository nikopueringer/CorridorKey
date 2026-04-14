"""Backend factory — selects Torch or MLX engine and normalizes output contracts."""

from __future__ import annotations

import errno
import glob
import logging
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
TORCH_EXT = ".pth"  # DEPRECATED: remove after .pth sunset
SAFETENSORS_EXT = ".safetensors"
# Torch backend accepts either extension; safetensors is preferred when both are present.
TORCH_EXTS = (SAFETENSORS_EXT, TORCH_EXT)
MLX_EXT = ".safetensors"
DEFAULT_IMG_SIZE = 2048

BACKEND_ENV_VAR = "CORRIDORKEY_BACKEND"
VALID_BACKENDS = ("auto", "torch", "mlx")

# Update HF_REPO_ID and HF_CHECKPOINT_FILENAME_* if a new model version is released.
HF_REPO_ID = "nikopueringer/CorridorKey_v1.0"
HF_CHECKPOINT_FILENAME_SAFETENSORS = "CorridorKey.safetensors"
HF_CHECKPOINT_FILENAME = "CorridorKey.pth"  # DEPRECATED: remove after .pth sunset


def resolve_backend(requested: str | None = None) -> str:
    """Resolve backend: CLI flag > env var > auto-detect.

    Auto mode: Apple Silicon + corridorkey_mlx importable + .safetensors found → mlx.
    Otherwise → torch.

    Raises RuntimeError if explicit backend is unavailable.
    """
    if requested is None or requested.lower() == "auto":
        backend = os.environ.get(BACKEND_ENV_VAR, "auto").lower()
    else:
        backend = requested.lower()

    if backend == "auto":
        return _auto_detect_backend()

    if backend not in VALID_BACKENDS:
        raise RuntimeError(f"Unknown backend '{backend}'. Valid: {', '.join(VALID_BACKENDS)}")

    if backend == "mlx":
        _validate_mlx_available()

    return backend


CHECKPOINT_DIR = os.path.join("CorridorKeyModule", "checkpoints")
MLX_MODEL_URL = "https://github.com/nikopueringer/corridorkey-mlx/releases/download/v1.0.0/corridorkey_mlx.safetensors"
MLX_MODEL_FILENAME = "corridorkey_mlx.safetensors"


def _auto_detect_backend() -> str:
    """Try MLX on Apple Silicon, fall back to Torch."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        logger.info("Not Apple Silicon — using torch backend")
        return "torch"

    try:
        import corridorkey_mlx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        logger.info("corridorkey_mlx not installed — using torch backend")
        return "torch"

        # Auto-download logic for the .safetensors file
    model_path = os.path.join(CHECKPOINT_DIR, MLX_MODEL_FILENAME)
    cache_path = model_path + ".tmp"

    if not os.path.exists(model_path):
        logger.info(f"MLX checkpoint not found. Downloading to {model_path}...")
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)

            # Create CorridorKeyModule/checkpoints/ if it doesn't exist
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            # Download the file
            urllib.request.urlretrieve(MLX_MODEL_URL, cache_path)
            os.rename(cache_path, model_path)
            logger.info("Download complete.")

        except Exception as e:
            logger.error(f"Failed to download MLX checkpoint: {e}")
            logger.info("Falling back to torch backend due to download failure.")

            # Clean up corrupted/partial file if the download failed midway
            if os.path.exists(model_path):
                os.remove(model_path)

            return "torch"

    logger.info("Apple Silicon + MLX available — using mlx backend")
    return "mlx"


def _validate_mlx_available() -> None:
    """Raise RuntimeError with actionable message if MLX can't be used."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX backend requires Apple Silicon (M1+ Mac)")

    try:
        import corridorkey_mlx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as err:
        raise RuntimeError(
            "MLX backend requested but corridorkey_mlx is not installed. "
            "Install with: uv pip install corridorkey-mlx@git+https://github.com/cmoyates/corridorkey-mlx.git"
        ) from err


def _copy_to_checkpoint_dir(cached_path: str, dest: Path) -> Path:
    """Copy a HuggingFace-cached file into CHECKPOINT_DIR, mapping ENOSPC to a friendly error."""
    try:
        shutil.copy2(cached_path, dest)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise OSError(
                errno.ENOSPC,
                "Not enough disk space to save checkpoint (~300 MB required). "
                f"Free up space in {CHECKPOINT_DIR} and try again.",
            ) from exc
        raise
    logger.info("Checkpoint saved to %s", dest)
    return dest


def _ensure_torch_checkpoint_pth_fallback() -> Path:
    """DEPRECATED: remove after .pth sunset.

    Download the legacy .pth checkpoint from HuggingFace. Used only when the
    official .safetensors file is not yet published to the HF repo.
    """
    dest = Path(CHECKPOINT_DIR) / HF_CHECKPOINT_FILENAME
    hf_url = f"https://huggingface.co/{HF_REPO_ID}"

    from huggingface_hub import hf_hub_download

    logger.info("Downloading legacy .pth CorridorKey checkpoint from %s ...", hf_url)

    try:
        cached_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_CHECKPOINT_FILENAME,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download CorridorKey checkpoint from {hf_url}. "
            "Check your network connection and try again. "
            f"Original error: {exc}"
        ) from exc

    return _copy_to_checkpoint_dir(cached_path, dest)


def _ensure_torch_checkpoint() -> Path:
    """Download the Torch checkpoint from HuggingFace if not present.

    Prefers the safer .safetensors format. If the HF repo does not yet host a
    .safetensors file (transitional), falls back to the legacy .pth download.

    Returns the path to the downloaded checkpoint file.

    Raises:
        RuntimeError: Network or download failure.
        OSError: Disk space or filesystem error.
    """
    dest = Path(CHECKPOINT_DIR) / HF_CHECKPOINT_FILENAME_SAFETENSORS
    hf_url = f"https://huggingface.co/{HF_REPO_ID}"

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    logger.info("Downloading CorridorKey checkpoint (.safetensors) from %s ...", hf_url)

    try:
        cached_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_CHECKPOINT_FILENAME_SAFETENSORS,
        )
    except EntryNotFoundError:
        # DEPRECATED: remove after .pth sunset.
        # The HF repo doesn't have the .safetensors yet — fall back to .pth so
        # this code can ship before the safetensors upload lands.
        logger.info(
            "No %s found on the HF repo yet — falling back to legacy .pth.",
            HF_CHECKPOINT_FILENAME_SAFETENSORS,
        )
        return _ensure_torch_checkpoint_pth_fallback()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download CorridorKey checkpoint from {hf_url}. "
            "Check your network connection and try again. "
            f"Original error: {exc}"
        ) from exc

    return _copy_to_checkpoint_dir(cached_path, dest)


def _find_single(ext: str) -> list[str]:
    return glob.glob(os.path.join(CHECKPOINT_DIR, f"*{ext}"))


def _discover_checkpoint(ext: str) -> Path:
    """Find exactly one checkpoint for the requested backend.

    For Torch (``ext == TORCH_EXT``): accepts both ``.safetensors`` and ``.pth``,
    preferring ``.safetensors`` when both are present. Auto-downloads when
    nothing is found locally.

    For MLX (``ext == MLX_EXT``): strictly ``.safetensors`` as before.

    Raises FileNotFoundError (0 found) or ValueError (>1 of the chosen format).
    """
    if ext == TORCH_EXT:
        safetensors_matches = _find_single(SAFETENSORS_EXT)
        pth_matches = _find_single(TORCH_EXT)

        if safetensors_matches and pth_matches:
            logger.info(
                "Both .safetensors and .pth checkpoints present in %s — preferring .safetensors.",
                CHECKPOINT_DIR,
            )

        # Prefer safetensors
        matches = safetensors_matches or pth_matches
        chosen_ext = SAFETENSORS_EXT if safetensors_matches else TORCH_EXT

        if not matches:
            return _ensure_torch_checkpoint()

        if len(matches) > 1:
            names = [os.path.basename(f) for f in matches]
            raise ValueError(f"Multiple {chosen_ext} checkpoints in {CHECKPOINT_DIR}: {names}. Keep exactly one.")

        return Path(matches[0])

    # MLX path — strict .safetensors match.
    matches = _find_single(ext)

    if len(matches) == 0:
        other_ext = TORCH_EXT
        other_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{other_ext}"))
        hint = ""
        if other_files:
            hint = f" (Found {other_ext} files — did you mean --backend=torch?)"
        raise FileNotFoundError(f"No {ext} checkpoint found in {CHECKPOINT_DIR}.{hint}")

    if len(matches) > 1:
        names = [os.path.basename(f) for f in matches]
        raise ValueError(f"Multiple {ext} checkpoints in {CHECKPOINT_DIR}: {names}. Keep exactly one.")

    return Path(matches[0])


def _wrap_mlx_output(raw: dict, despill_strength: float, auto_despeckle: bool, despeckle_size: int) -> dict:
    """Normalize MLX uint8 output to match Torch float32 contract.

    Torch contract:
      alpha:     [H,W,1] float32 0-1
      fg:        [H,W,3] float32 0-1 sRGB
      comp:      [H,W,3] float32 0-1 sRGB
      processed: [H,W,4] float32 linear premul RGBA
    """
    from CorridorKeyModule.core import color_utils as cu

    # alpha: uint8 [H,W] → float32 [H,W,1]
    alpha_raw = raw["alpha"]
    alpha = alpha_raw.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    # fg: uint8 [H,W,3] → float32 [H,W,3] (sRGB)
    fg = raw["fg"].astype(np.float32) / 255.0

    # Apply despeckle (MLX stubs this)
    if auto_despeckle:
        processed_alpha = cu.clean_matte_opencv(alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
    else:
        processed_alpha = alpha

    # Apply despill (MLX stubs this)
    fg_despilled = cu.despill_opencv(fg, green_limit_mode="average", strength=despill_strength)

    # Composite over checkerboard for comp output
    h, w = fg.shape[:2]
    bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
    bg_lin = cu.srgb_to_linear(bg_srgb)
    fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
    comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
    comp_srgb = cu.linear_to_srgb(comp_lin)

    # Build processed: [H,W,4] linear premul RGBA
    fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
    processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

    return {
        "alpha": alpha,  # raw prediction (before despeckle), matches Torch
        "fg": fg,  # raw sRGB prediction, matches Torch
        "comp": comp_srgb,  # sRGB composite on checker
        "processed": processed_rgba,  # linear premul RGBA
    }


class _MLXEngineAdapter:
    """Wraps CorridorKeyMLXEngine to match Torch output contract."""

    def __init__(self, raw_engine):
        self._engine = raw_engine
        logger.info("MLX adapter active: despill and despeckle are handled by the adapter layer, not native MLX")

    def process_frame(
        self,
        image,
        mask_linear,
        refiner_scale=1.0,
        input_is_linear=False,
        fg_is_straight=True,
        despill_strength=1.0,
        auto_despeckle=True,
        despeckle_size=400,
        **_kwargs,
    ):
        """Delegate to MLX engine, then normalize output to Torch contract."""
        # MLX engine expects uint8 input — convert if float
        if image.dtype != np.uint8:
            image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            image_u8 = image

        if mask_linear.dtype != np.uint8:
            mask_u8 = (np.clip(mask_linear, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            mask_u8 = mask_linear

        # Squeeze mask to 2D for MLX (it validates [H,W] or [H,W,1])
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[:, :, 0]

        raw = self._engine.process_frame(
            image_u8,
            mask_u8,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            fg_is_straight=fg_is_straight,
            despill_strength=0.0,  # disable MLX stubs — adapter applies these
            auto_despeckle=False,
            despeckle_size=despeckle_size,
        )

        return _wrap_mlx_output(raw, despill_strength, auto_despeckle, despeckle_size)


DEFAULT_MLX_TILE_SIZE = 512
DEFAULT_MLX_TILE_OVERLAP = 64


def create_engine(
    backend: str | None = None,
    device: str | None = None,
    img_size: int = DEFAULT_IMG_SIZE,
    tile_size: int | None = DEFAULT_MLX_TILE_SIZE,
    overlap: int = DEFAULT_MLX_TILE_OVERLAP,
):
    """Factory: returns an engine with process_frame() matching the Torch contract.

    Args:
        tile_size: MLX only — tile size for tiled inference (default 512).
            Set to None to disable tiling and use full-frame inference.
        overlap: MLX only — overlap pixels between tiles (default 64).
    """
    backend = resolve_backend(backend)

    if backend == "mlx":
        ckpt = _discover_checkpoint(MLX_EXT)
        from corridorkey_mlx import CorridorKeyMLXEngine  # type: ignore[import-not-found]

        raw_engine = CorridorKeyMLXEngine(str(ckpt), img_size=img_size, tile_size=tile_size, overlap=overlap)
        mode = f"tiled (tile={tile_size}, overlap={overlap})" if tile_size else "full-frame"
        logger.info("MLX engine loaded: %s [%s]", ckpt.name, mode)
        return _MLXEngineAdapter(raw_engine)
    else:
        ckpt = _discover_checkpoint(TORCH_EXT)
        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        logger.info("Torch engine loaded: %s (device=%s)", ckpt.name, device)
        return CorridorKeyEngine(
            checkpoint_path=str(ckpt), device=device or "cpu", img_size=img_size, model_precision=torch.float16
        )
