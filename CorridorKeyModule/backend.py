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

from CorridorKeyModule.core.color_utils import SCREEN_COLOR_CHOICES

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
HF_CHECKPOINT_FILENAME_SAFETENSORS = "CorridorKey_v1.0.safetensors"
HF_CHECKPOINT_FILENAME = "CorridorKey_v1.0.pth"  # DEPRECATED: remove after .pth sunset

# Dedicated blue-screen weights (CorridorKeyBlue). Same architecture as the
# green checkpoint — only the trained weights differ.
HF_REPO_ID_BLUE = "nikopueringer/CorridorKeyBlue_1.0"
HF_CHECKPOINT_FILENAME_BLUE_SAFETENSORS = "CorridorKeyBlue_1.0.safetensors"
HF_CHECKPOINT_FILENAME_BLUE = "CorridorKeyBlue_1.0.pth"  # DEPRECATED: remove after .pth sunset

# Re-exported alias for callers/tests that import VALID_SCREEN_COLORS from this module.
VALID_SCREEN_COLORS = SCREEN_COLOR_CHOICES
BLUE_FILENAME_TOKEN = "blue"  # case-insensitive substring marking a blue checkpoint


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


def _hf_repo_for_color(screen_color: str) -> str:
    return HF_REPO_ID_BLUE if screen_color == "blue" else HF_REPO_ID


def _hf_safetensors_for_color(screen_color: str) -> str:
    return HF_CHECKPOINT_FILENAME_BLUE_SAFETENSORS if screen_color == "blue" else HF_CHECKPOINT_FILENAME_SAFETENSORS


def _hf_pth_for_color(screen_color: str) -> str:
    return HF_CHECKPOINT_FILENAME_BLUE if screen_color == "blue" else HF_CHECKPOINT_FILENAME


def _ensure_torch_checkpoint_pth_fallback(screen_color: str = "green") -> Path:
    """DEPRECATED: remove after .pth sunset.

    Download the legacy .pth checkpoint from HuggingFace. Used only when the
    official .safetensors file is not yet published to the HF repo.
    """
    repo_id = _hf_repo_for_color(screen_color)
    pth_filename = _hf_pth_for_color(screen_color)
    dest = Path(CHECKPOINT_DIR) / pth_filename
    hf_url = f"https://huggingface.co/{repo_id}"

    from huggingface_hub import hf_hub_download

    logger.info("Downloading legacy .pth CorridorKey (%s) checkpoint from %s ...", screen_color, hf_url)

    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=pth_filename,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download CorridorKey checkpoint from {hf_url}. "
            "Check your network connection and try again. "
            f"Original error: {exc}"
        ) from exc

    return _copy_to_checkpoint_dir(cached_path, dest)


def _ensure_torch_checkpoint(screen_color: str = "green") -> Path:
    """Download the Torch checkpoint from HuggingFace if not present.

    Prefers the safer .safetensors format. If the HF repo does not yet host a
    .safetensors file (transitional), falls back to the legacy .pth download.

    Returns the path to the downloaded checkpoint file.

    Raises:
        RuntimeError: Network or download failure.
        OSError: Disk space or filesystem error.
    """
    repo_id = _hf_repo_for_color(screen_color)
    safetensors_filename = _hf_safetensors_for_color(screen_color)
    dest = Path(CHECKPOINT_DIR) / safetensors_filename
    hf_url = f"https://huggingface.co/{repo_id}"

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    logger.info("Downloading CorridorKey (%s) checkpoint (.safetensors) from %s ...", screen_color, hf_url)

    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=safetensors_filename,
        )
    except EntryNotFoundError:
        # DEPRECATED: remove after .pth sunset.
        logger.info(
            "No %s found on the HF repo yet — falling back to legacy .pth.",
            safetensors_filename,
        )
        return _ensure_torch_checkpoint_pth_fallback(screen_color)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download CorridorKey checkpoint from {hf_url}. "
            "Check your network connection and try again. "
            f"Original error: {exc}"
        ) from exc

    return _copy_to_checkpoint_dir(cached_path, dest)


def _find_single(ext: str) -> list[str]:
    return glob.glob(os.path.join(CHECKPOINT_DIR, f"*{ext}"))


def _filter_by_color(paths: list[str], screen_color: str) -> list[str]:
    """Keep only checkpoint files whose basename matches the requested screen color.

    Convention: filenames containing the substring 'blue' (case-insensitive)
    are blue weights; everything else is treated as green.
    """
    is_blue_target = screen_color == "blue"
    out: list[str] = []
    for p in paths:
        is_blue_file = BLUE_FILENAME_TOKEN in os.path.basename(p).lower()
        if is_blue_file == is_blue_target:
            out.append(p)
    return out


def _discover_checkpoint(ext: str, screen_color: str = "green") -> Path:
    """Find exactly one checkpoint for the requested backend and screen color.

    For Torch (``ext == TORCH_EXT``): accepts both ``.safetensors`` and ``.pth``,
    preferring ``.safetensors`` when both are present. Auto-downloads when
    nothing is found locally for the requested color.

    For MLX (``ext == MLX_EXT``): strictly ``.safetensors``. Blue is not yet
    supported on MLX — raises RuntimeError.

    Raises FileNotFoundError (0 found, no auto-download) or ValueError (>1 match).
    """
    if screen_color not in VALID_SCREEN_COLORS:
        raise ValueError(f"Unknown screen_color '{screen_color}'. Valid: {', '.join(VALID_SCREEN_COLORS)}")

    if ext == TORCH_EXT:
        safetensors_matches = _filter_by_color(_find_single(SAFETENSORS_EXT), screen_color)
        pth_matches = _filter_by_color(_find_single(TORCH_EXT), screen_color)

        if safetensors_matches and pth_matches:
            logger.info(
                "Both .safetensors and .pth %s checkpoints present in %s — preferring .safetensors.",
                screen_color,
                CHECKPOINT_DIR,
            )

        matches = safetensors_matches or pth_matches
        chosen_ext = SAFETENSORS_EXT if safetensors_matches else TORCH_EXT

        if not matches:
            return _ensure_torch_checkpoint(screen_color)

        if len(matches) > 1:
            names = [os.path.basename(f) for f in matches]
            raise ValueError(
                f"Multiple {chosen_ext} {screen_color} checkpoints in {CHECKPOINT_DIR}: {names}. Keep exactly one."
            )

        return Path(matches[0])

    # MLX path — strict .safetensors match.
    if screen_color == "blue":
        raise RuntimeError(
            "Blue-screen support is not yet available on the MLX backend. "
            "Use --backend torch with --screen-color blue, or wait for the MLX blue release."
        )

    matches = _filter_by_color(_find_single(ext), screen_color)

    if len(matches) == 0:
        other_ext = TORCH_EXT
        other_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{other_ext}"))
        hint = ""
        if other_files:
            hint = f" (Found {other_ext} files — did you mean --backend=torch?)"
        raise FileNotFoundError(f"No {ext} {screen_color} checkpoint found in {CHECKPOINT_DIR}.{hint}")

    if len(matches) > 1:
        names = [os.path.basename(f) for f in matches]
        raise ValueError(f"Multiple {ext} {screen_color} checkpoints in {CHECKPOINT_DIR}: {names}. Keep exactly one.")

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
    fg_despilled = cu.despill_opencv(fg, limit_mode="average", strength=despill_strength)

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
        screen_channel: int = 1,
        **_kwargs,
    ):
        """Delegate to MLX engine, then normalize output to Torch contract.

        ``screen_channel`` is accepted for API parity with the Torch engine but
        only ``1`` (green) is supported here — the MLX backend has no blue
        checkpoint yet, so the despill in ``_wrap_mlx_output`` is hard-wired to
        the green channel. Calling with ``screen_channel != 1`` is a programmer
        error (the public ``create_engine`` rejects MLX + blue earlier); we
        raise instead of silently returning a green-keyed result.
        """
        if screen_channel != 1:
            raise NotImplementedError(
                f"_MLXEngineAdapter does not support screen_channel={screen_channel}. "
                "MLX has no blue-screen checkpoint yet; use the Torch backend with "
                "--screen-color blue, or wait for the MLX blue release."
            )
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
    screen_color: str = "green",
):
    """Factory: returns an engine with process_frame() matching the Torch contract.

    Args:
        tile_size: MLX only — tile size for tiled inference (default 512).
            Set to None to disable tiling and use full-frame inference.
        overlap: MLX only — overlap pixels between tiles (default 64).
        screen_color: 'green' (default) or 'blue'. Selects which checkpoint to
            load. Blue is currently Torch-only.
    """
    if screen_color not in VALID_SCREEN_COLORS:
        raise ValueError(f"Unknown screen_color '{screen_color}'. Valid: {', '.join(VALID_SCREEN_COLORS)}")
    backend = resolve_backend(backend)

    if backend == "mlx":
        ckpt = _discover_checkpoint(MLX_EXT, screen_color=screen_color)
        from corridorkey_mlx import CorridorKeyMLXEngine  # type: ignore[import-not-found]

        raw_engine = CorridorKeyMLXEngine(str(ckpt), img_size=img_size, tile_size=tile_size, overlap=overlap)
        mode = f"tiled (tile={tile_size}, overlap={overlap})" if tile_size else "full-frame"
        logger.info("MLX engine loaded: %s [%s, screen=%s]", ckpt.name, mode, screen_color)
        return _MLXEngineAdapter(raw_engine)
    else:
        ckpt = _discover_checkpoint(TORCH_EXT, screen_color=screen_color)
        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        logger.info("Torch engine loaded: %s (device=%s, screen=%s)", ckpt.name, device, screen_color)
        return CorridorKeyEngine(
            checkpoint_path=str(ckpt), device=device or "cpu", img_size=img_size, model_precision=torch.float16
        )
