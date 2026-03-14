"""Backend factory — selects Torch or MLX engine and normalizes output contracts."""

from __future__ import annotations

import functools
import glob
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
TORCH_EXT = ".pth"
MLX_EXT = ".safetensors"
DEFAULT_IMG_SIZE = 2048

BACKEND_ENV_VAR = "CORRIDORKEY_BACKEND"
VALID_BACKENDS = ("auto", "torch", "mlx")

# Adaptive despeckle bypass: if the alpha matte is already mostly binary
# (near 0 or 1), CCL is wasted work — the matte has no speckle noise.
# We check the fraction of pixels in the "transition zone" (partially
# transparent) and skip despeckle if it's below this threshold.
DESPECKLE_BYPASS_THRESHOLD = 0.01
DESPECKLE_TRANSITION_LOW = 0.05
DESPECKLE_TRANSITION_HIGH = 0.95


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

    safetensor_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{MLX_EXT}"))
    if not safetensor_files:
        logger.info("No %s checkpoint found — using torch backend", MLX_EXT)
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


def _discover_checkpoint(ext: str) -> Path:
    """Find exactly one checkpoint with the given extension.

    Raises FileNotFoundError (0 found) or ValueError (>1 found).
    Includes cross-reference hints when wrong extension files exist.
    """
    matches = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{ext}"))

    if len(matches) == 0:
        other_ext = MLX_EXT if ext == TORCH_EXT else TORCH_EXT
        other_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{other_ext}"))
        hint = ""
        if other_files:
            other_backend = "mlx" if other_ext == MLX_EXT else "torch"
            hint = f" (Found {other_ext} files — did you mean --backend={other_backend}?)"
        raise FileNotFoundError(f"No {ext} checkpoint found in {CHECKPOINT_DIR}.{hint}")

    if len(matches) > 1:
        names = [os.path.basename(f) for f in matches]
        raise ValueError(f"Multiple {ext} checkpoints in {CHECKPOINT_DIR}: {names}. Keep exactly one.")

    return Path(matches[0])


@functools.lru_cache(maxsize=4)
def _get_checkerboard(
    w: int,
    h: int,
    checker_size: int = 128,
    color1: float = 0.15,
    color2: float = 0.55,
) -> np.ndarray:
    """Return a cached checkerboard pattern, creating it only on first call per resolution."""
    from CorridorKeyModule.core import color_utils as cu

    return cu.create_checkerboard(w, h, checker_size=checker_size, color1=color1, color2=color2)


def _wrap_mlx_output(
    raw: dict,
    despill_strength: float,
    auto_despeckle: bool,
    despeckle_size: int,
    enabled_outputs: frozenset[str] = frozenset({"fg", "matte", "comp", "processed"}),
) -> dict:
    """Normalize MLX uint8 output to match Torch float32 contract.

    Torch contract:
      alpha:     [H,W,1] float32 0-1
      fg:        [H,W,3] float32 0-1 sRGB
      comp:      [H,W,3] float32 0-1 sRGB  (skipped if not in enabled_outputs)
      processed: [H,W,4] float32 linear premul RGBA  (skipped if not in enabled_outputs)
    """
    from CorridorKeyModule.core import color_utils as cu

    need_comp = "comp" in enabled_outputs
    need_processed = "processed" in enabled_outputs
    need_postprocess = need_comp or need_processed

    # --- Normalize MLX uint8 outputs to float32 [0,1] ---
    # MLX engine outputs uint8 for memory efficiency on Apple Silicon,
    # but downstream code (despill, compositing, EXR writes) expects float32.
    alpha_raw = raw["alpha"]
    if alpha_raw.dtype == np.uint8:
        alpha_float = alpha_raw.astype(np.float32) / np.float32(255.0)
    else:
        alpha_float = np.asarray(alpha_raw, dtype=np.float32)
    if alpha_float.ndim == 2:
        alpha_float = alpha_float[:, :, np.newaxis]

    fg_raw = raw["fg"]
    if fg_raw.dtype == np.uint8:
        fg_float = fg_raw.astype(np.float32) / np.float32(255.0)
    else:
        fg_float = np.asarray(fg_raw, dtype=np.float32)

    result: dict = {
        "alpha": alpha_float,
        "fg": fg_float,
    }

    if not need_postprocess:
        return result

    # Adaptive despeckle bypass: count pixels in the "transition zone"
    # (partially transparent). If nearly all pixels are fully opaque or
    # fully transparent, the matte is clean and CCL is wasted work.
    if auto_despeckle:
        alpha_2d = alpha_float[:, :, 0] if alpha_float.ndim == 3 else alpha_float
        transition_pixel_count = np.count_nonzero(
            (alpha_2d > DESPECKLE_TRANSITION_LOW) & (alpha_2d < DESPECKLE_TRANSITION_HIGH)
        )
        transition_ratio = transition_pixel_count / alpha_2d.size
        if transition_ratio < DESPECKLE_BYPASS_THRESHOLD:
            processed_alpha = alpha_float
        else:
            processed_alpha = cu.clean_matte(alpha_float, area_threshold=despeckle_size, dilation=25, blur_size=5)
    else:
        processed_alpha = alpha_float

    # Despill is disabled inside MLX engine (it lacks the numpy-based impl),
    # so the adapter applies it here on the normalized float32 output
    fg_despilled = cu.despill(fg_float, green_limit_mode="average", strength=despill_strength)

    # Both comp and processed outputs need linear-space foreground
    fg_despilled_linear = cu.srgb_to_linear(fg_despilled)

    if need_comp:
        frame_height, frame_width = fg_float.shape[:2]
        checkerboard_srgb = _get_checkerboard(frame_width, frame_height)
        checkerboard_linear = cu.srgb_to_linear(checkerboard_srgb)
        composite_linear = cu.composite_straight(fg_despilled_linear, checkerboard_linear, processed_alpha)
        result["comp"] = cu.linear_to_srgb(composite_linear)

    if need_processed:
        # EXR output requires linear premultiplied RGBA
        fg_premultiplied_linear = cu.premultiply(fg_despilled_linear, processed_alpha)
        result["processed"] = np.concatenate([fg_premultiplied_linear, processed_alpha], axis=-1)

    return result


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
        enabled_outputs=frozenset({"fg", "matte", "comp", "processed"}),
    ):
        """Delegate to MLX engine, then normalize output to Torch contract."""

        # MLX engine expects uint8 input — the Torch pipeline uses float32,
        # so we convert at the adapter boundary to avoid changing callers
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            image_uint8 = image

        if mask_linear.dtype != np.uint8:
            mask_uint8 = (np.clip(mask_linear, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            mask_uint8 = mask_linear

        # MLX engine validates [H,W] shape for masks
        if mask_uint8.ndim == 3:
            mask_uint8 = mask_uint8[:, :, 0]

        inference_start = time.perf_counter()
        raw = self._engine.process_frame(
            image_uint8,
            mask_uint8,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            fg_is_straight=fg_is_straight,
            despill_strength=0.0,  # disable MLX stubs — adapter applies these in _wrap_mlx_output
            auto_despeckle=False,  # same — handled by adapter for consistency with Torch
            despeckle_size=despeckle_size,
        )
        inference_elapsed = time.perf_counter() - inference_start

        postprocess_start = time.perf_counter()
        result = _wrap_mlx_output(raw, despill_strength, auto_despeckle, despeckle_size, enabled_outputs)
        postprocess_elapsed = time.perf_counter() - postprocess_start

        logger.debug(
            "MLX frame: inference=%.0fms  postprocess=%.0fms",
            inference_elapsed * 1000,
            postprocess_elapsed * 1000,
        )

        # _timing is a side-channel for per-phase profiling — consumed by the
        # writer thread in clip_manager.py via res.pop("_timing").
        # Torch engine does not set this; only the MLX adapter does.
        result["_timing"] = {"mlx_inference": inference_elapsed, "postprocess": postprocess_elapsed}
        return result


DEFAULT_MLX_TILE_SIZE = 768
DEFAULT_MLX_TILE_OVERLAP = 128


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
        return CorridorKeyEngine(checkpoint_path=str(ckpt), device=device or "cpu", img_size=img_size)
