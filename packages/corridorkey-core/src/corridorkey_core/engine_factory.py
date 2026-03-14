"""Engine factory — selects Torch or MLX backend and normalizes output contracts.

This is the Core Layer's public entry point for constructing an inference engine.
Callers receive an object with a process_frame() method that always returns the
Torch contract regardless of which backend is running underneath.

Torch output contract:
    alpha:     [H, W, 1] float32 0-1  (raw prediction, before despeckle)
    fg:        [H, W, 3] float32 0-1  sRGB straight
    comp:      [H, W, 3] float32 0-1  sRGB composite over checkerboard
    processed: [H, W, 4] float32      linear premultiplied RGBA

Backend resolution order:
    explicit argument > CORRIDORKEY_BACKEND env var > auto-detect

Auto-detect:
    Apple Silicon + corridorkey_mlx importable + .safetensors present → mlx
    Otherwise → torch
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TORCH_EXT = ".pth"
MLX_EXT = ".safetensors"
DEFAULT_IMG_SIZE = 2048
DEFAULT_MLX_TILE_SIZE = 512
DEFAULT_MLX_TILE_OVERLAP = 64

BACKEND_ENV_VAR = "CORRIDORKEY_BACKEND"
VALID_BACKENDS = ("auto", "torch", "mlx")


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def resolve_backend(requested: str | None = None) -> str:
    """Resolve which backend to use.

    Priority: explicit argument > CORRIDORKEY_BACKEND env var > auto-detect.

    Args:
        requested: "torch", "mlx", "auto", or None (treated as "auto").

    Returns:
        "torch" or "mlx".

    Raises:
        RuntimeError: If an explicit backend is requested but unavailable.
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


def _mlx_available() -> bool:
    """Return True if corridorkey_mlx is importable."""
    return importlib.util.find_spec("corridorkey_mlx") is not None


def _auto_detect_backend() -> str:
    """Try MLX on Apple Silicon, fall back to Torch."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        logger.info("Not Apple Silicon — using torch backend")
        return "torch"

    if not _mlx_available():
        logger.info("corridorkey_mlx not installed — using torch backend")
        return "torch"

    logger.info("Apple Silicon + MLX available — using mlx backend")
    return "mlx"


def _validate_mlx_available() -> None:
    """Raise RuntimeError with an actionable message if MLX cannot be used."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX backend requires Apple Silicon (M1+ Mac)")

    if not _mlx_available():
        raise RuntimeError(
            "MLX backend requested but corridorkey_mlx is not installed. "
            "Install with: uv pip install corridorkey-mlx@git+https://github.com/cmoyates/corridorkey-mlx.git"
        )


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def discover_checkpoint(checkpoint_dir: str, ext: str) -> Path:
    """Find exactly one checkpoint file with the given extension in checkpoint_dir.

    Args:
        checkpoint_dir: Directory to search.
        ext: File extension including the dot, e.g. ".pth" or ".safetensors".

    Returns:
        Path to the single matching checkpoint file.

    Raises:
        FileNotFoundError: If no matching file is found.
        ValueError: If more than one matching file is found.
    """
    matches = glob.glob(os.path.join(checkpoint_dir, f"*{ext}"))

    if len(matches) == 0:
        other_ext = MLX_EXT if ext == TORCH_EXT else TORCH_EXT
        other_files = glob.glob(os.path.join(checkpoint_dir, f"*{other_ext}"))
        hint = ""
        if other_files:
            other_backend = "mlx" if other_ext == MLX_EXT else "torch"
            hint = f" (Found {other_ext} files — did you mean --backend={other_backend}?)"
        raise FileNotFoundError(f"No {ext} checkpoint found in {checkpoint_dir}.{hint}")

    if len(matches) > 1:
        names = [os.path.basename(f) for f in matches]
        raise ValueError(f"Multiple {ext} checkpoints in {checkpoint_dir}: {names}. Keep exactly one.")

    return Path(matches[0])


# ---------------------------------------------------------------------------
# MLX output normalization
# ---------------------------------------------------------------------------


def _wrap_mlx_output(raw: dict, despill_strength: float, auto_despeckle: bool, despeckle_size: int) -> dict:
    """Normalize MLX uint8 output to match the Torch float32 contract.

    MLX engines return uint8 arrays and stub out despill/despeckle.
    This adapter applies both post-processing steps and builds the full
    four-key output dict that the Application Layer expects.
    """
    from corridorkey_core.compositing import (
        clean_matte,
        composite_straight,
        create_checkerboard,
        despill,
        linear_to_srgb,
        premultiply,
        srgb_to_linear,
    )

    # alpha: uint8 [H, W] → float32 [H, W, 1]
    alpha_raw = raw["alpha"]
    alpha = alpha_raw.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    # fg: uint8 [H, W, 3] → float32 [H, W, 3] sRGB
    fg = raw["fg"].astype(np.float32) / 255.0

    # Despeckle (MLX stubs this — adapter applies it)
    processed_alpha = (
        clean_matte(alpha, area_threshold=despeckle_size, dilation=25, blur_size=5) if auto_despeckle else alpha
    )

    # Despill (MLX stubs this — adapter applies it)
    fg_despilled = despill(fg, green_limit_mode="average", strength=despill_strength)

    # Composite over checkerboard for the comp output
    h, w = fg.shape[:2]
    bg_srgb = create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
    bg_lin = srgb_to_linear(bg_srgb)
    fg_despilled_lin = srgb_to_linear(fg_despilled)
    comp_lin = composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
    comp_srgb = linear_to_srgb(comp_lin)

    # Build processed: [H, W, 4] linear premultiplied RGBA
    fg_premul_lin = premultiply(fg_despilled_lin, processed_alpha)
    processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

    return {
        "alpha": alpha,  # raw prediction (before despeckle), matches Torch
        "fg": fg,  # raw sRGB prediction, matches Torch
        "comp": comp_srgb,  # sRGB composite on checkerboard
        "processed": processed_rgba,  # linear premultiplied RGBA
    }


# ---------------------------------------------------------------------------
# MLX adapter
# ---------------------------------------------------------------------------


class _MLXEngineAdapter:
    """Wraps CorridorKeyMLXEngine to expose the same process_frame() contract as CorridorKeyEngine.

    MLX engines expect uint8 inputs and return uint8 outputs with despill/despeckle
    stubbed out. This adapter handles the conversion in both directions so the
    Application Layer never needs to know which backend is active.
    """

    def __init__(self, raw_engine) -> None:
        self._engine = raw_engine
        logger.info("MLX adapter active: despill and despeckle are handled by the adapter, not native MLX")

    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict:
        """Delegate to the MLX engine then normalize output to the Torch contract."""
        # MLX engine expects uint8 — convert if float
        image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        mask_u8 = (
            (np.clip(mask_linear, 0.0, 1.0) * 255).astype(np.uint8) if mask_linear.dtype != np.uint8 else mask_linear
        )

        # MLX validates [H, W] or [H, W, 1] — squeeze to 2D
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[:, :, 0]

        raw = self._engine.process_frame(
            image_u8,
            mask_u8,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            fg_is_straight=fg_is_straight,
            despill_strength=0.0,  # disabled — adapter applies despill
            auto_despeckle=False,  # disabled — adapter applies despeckle
            despeckle_size=despeckle_size,
        )

        return _wrap_mlx_output(raw, despill_strength, auto_despeckle, despeckle_size)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_engine(
    checkpoint_dir: str,
    backend: str | None = None,
    device: str | None = None,
    img_size: int = DEFAULT_IMG_SIZE,
    tile_size: int | None = DEFAULT_MLX_TILE_SIZE,
    overlap: int = DEFAULT_MLX_TILE_OVERLAP,
):
    """Create and return an inference engine for the resolved backend.

    The returned object always exposes process_frame() with the Torch output
    contract, regardless of whether Torch or MLX is running underneath.

    Args:
        checkpoint_dir: Directory containing the checkpoint file(s).
        backend: "torch", "mlx", "auto", or None. Resolved via resolve_backend().
        device: Torch device string (e.g. "cuda", "cpu"). Torch only.
        img_size: Square resolution the model runs at internally.
        tile_size: MLX only — tile size for tiled inference. None = full-frame.
        overlap: MLX only — overlap pixels between tiles.

    Returns:
        An engine object with a process_frame() method matching the Torch contract.
    """
    backend = resolve_backend(backend)

    if backend == "mlx":
        ckpt = discover_checkpoint(checkpoint_dir, MLX_EXT)
        from corridorkey_mlx import CorridorKeyMLXEngine  # type: ignore[import-not-found]

        raw_engine = CorridorKeyMLXEngine(str(ckpt), img_size=img_size, tile_size=tile_size, overlap=overlap)
        mode = f"tiled (tile={tile_size}, overlap={overlap})" if tile_size else "full-frame"
        logger.info("MLX engine loaded: %s [%s]", ckpt.name, mode)
        return _MLXEngineAdapter(raw_engine)

    # Torch
    ckpt = discover_checkpoint(checkpoint_dir, TORCH_EXT)
    from corridorkey_core.inference_engine import CorridorKeyEngine

    logger.info("Torch engine loaded: %s (device=%s)", ckpt.name, device or "cpu")
    return CorridorKeyEngine(checkpoint_path=str(ckpt), device=device or "cpu", img_size=img_size)
