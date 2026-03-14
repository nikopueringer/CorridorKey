"""Backend factory — selects Torch or MLX engine and normalizes output contracts."""

from __future__ import annotations

import glob
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
TORCH_EXT = ".pth"
MLX_EXT = ".safetensors"
DEFAULT_IMG_SIZE = 2048

BACKEND_ENV_VAR = "CORRIDORKEY_BACKEND"
VALID_BACKENDS = ("auto", "torch", "mlx")


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


_checkerboard_cache: dict[tuple[int, int, int, float, float], np.ndarray] = {}


def _get_checkerboard(
    w: int,
    h: int,
    checker_size: int = 128,
    color1: float = 0.15,
    color2: float = 0.55,
) -> np.ndarray:
    """Return a cached checkerboard pattern, creating it only on first call per resolution."""
    from CorridorKeyModule.core import color_utils as cu

    key = (w, h, checker_size, color1, color2)
    if key not in _checkerboard_cache:
        _checkerboard_cache[key] = cu.create_checkerboard(w, h, checker_size=checker_size, color1=color1, color2=color2)
    return _checkerboard_cache[key]


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

    # alpha: uint8 [H,W] → float32 [H,W,1]
    alpha_raw = raw["alpha"]
    # Zero-copy when possible: avoid .astype if already float32
    if alpha_raw.dtype == np.uint8:
        alpha = alpha_raw.astype(np.float32) / np.float32(255.0)
    else:
        alpha = np.asarray(alpha_raw, dtype=np.float32)
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    # fg: uint8 [H,W,3] → float32 [H,W,3] (sRGB)
    fg_raw = raw["fg"]
    if fg_raw.dtype == np.uint8:
        fg = fg_raw.astype(np.float32) / np.float32(255.0)
    else:
        fg = np.asarray(fg_raw, dtype=np.float32)

    result: dict = {
        "alpha": alpha,
        "fg": fg,
    }

    if not need_postprocess:
        return result

    # Apply despeckle — skip if alpha is already clean (low noise)
    if auto_despeckle:
        # Adaptive bypass: if alpha is mostly binary (near 0 or 1), skip expensive CCL
        alpha_2d = alpha[:, :, 0] if alpha.ndim == 3 else alpha
        mid_range = np.count_nonzero((alpha_2d > 0.05) & (alpha_2d < 0.95))
        mid_ratio = mid_range / alpha_2d.size
        if mid_ratio < 0.01:
            # Less than 1% of pixels in transition zone — matte is clean
            processed_alpha = alpha
        else:
            processed_alpha = cu.clean_matte(alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
    else:
        processed_alpha = alpha

    # Apply despill (MLX stubs this)
    fg_despilled = cu.despill(fg, green_limit_mode="average", strength=despill_strength)

    # Colorspace conversion needed for both comp and processed
    fg_despilled_lin = cu.srgb_to_linear(fg_despilled)

    if need_comp:
        h, w = fg.shape[:2]
        bg_srgb = _get_checkerboard(w, h)
        bg_lin = cu.srgb_to_linear(bg_srgb)
        comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        result["comp"] = cu.linear_to_srgb(comp_lin)

    if need_processed:
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
        result["processed"] = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

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
        import time

        # Fast path: skip inference if mask is all-background or all-foreground
        mask_2d = mask_linear[:, :, 0] if mask_linear.ndim == 3 else mask_linear
        mask_min, mask_max = float(mask_2d.min()), float(mask_2d.max())

        if mask_max < 0.01:
            # Pure background — alpha=0, fg=black
            h, w = image.shape[:2]
            result = {
                "alpha": np.zeros((h, w, 1), dtype=np.float32),
                "fg": np.zeros((h, w, 3), dtype=np.float32),
                "_timing": {"mlx_inference": 0.0, "postprocess": 0.0},
            }
            logger.debug("Frame skipped: mask is all-background")
            return result

        if mask_min > 0.99:
            # Pure foreground — alpha=1, fg=input
            h, w = image.shape[:2]
            fg = image if image.dtype == np.float32 else image.astype(np.float32) / 255.0
            result = {
                "alpha": np.ones((h, w, 1), dtype=np.float32),
                "fg": fg,
                "_timing": {"mlx_inference": 0.0, "postprocess": 0.0},
            }
            logger.debug("Frame skipped: mask is all-foreground")
            return result

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

        t_mlx_start = time.perf_counter()
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
        t_mlx = time.perf_counter() - t_mlx_start

        t_post_start = time.perf_counter()
        result = _wrap_mlx_output(raw, despill_strength, auto_despeckle, despeckle_size, enabled_outputs)
        t_post = time.perf_counter() - t_post_start

        logger.debug(
            "MLX frame: inference=%.0fms  postprocess=%.0fms",
            t_mlx * 1000,
            t_post * 1000,
        )

        result["_timing"] = {"mlx_inference": t_mlx, "postprocess": t_post}
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
