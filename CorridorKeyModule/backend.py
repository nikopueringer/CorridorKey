"""Backend factory — selects Torch or MLX engine and normalizes output contracts."""

from __future__ import annotations

import glob
import importlib
import logging
import os
import platform
import shutil
import subprocess
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
MLX_INSTALL_SPEC = "corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLX_CONVERTER_REPO_URL = "https://github.com/nikopueringer/corridorkey-mlx.git"
MLX_CONVERTER_REPO_DIR = PROJECT_ROOT / "corridorkey-mlx"
DEFAULT_MLX_CHECKPOINT_NAME = f"corridorkey_mlx{MLX_EXT}"
_MLX_INSTALL_ATTEMPTED = False


def resolve_backend(requested: str | None = None) -> str:
    """Resolve backend: CLI flag > env var > auto-detect.

    Auto mode prefers MLX on Apple Silicon and auto-installs the MLX runtime
    when needed.
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
    """Prefer MLX on Apple Silicon, otherwise fall back to Torch."""
    if not _is_apple_silicon():
        logger.info("Not Apple Silicon — using torch backend")
        return "torch"

    if not _mlx_runtime_available(auto_install=True):
        logger.info("Apple Silicon detected but corridorkey_mlx could not be installed — using torch backend")
        return "torch"

    logger.info("Apple Silicon detected — preferring mlx backend")
    return "mlx"


def _validate_mlx_available() -> None:
    """Raise RuntimeError with actionable message if MLX can't be used."""
    if not _is_apple_silicon():
        raise RuntimeError("MLX backend requires Apple Silicon (M1+ Mac)")

    if not _mlx_runtime_available(auto_install=True):
        raise RuntimeError(
            "MLX backend requested but corridorkey_mlx is unavailable and automatic installation failed. "
            f"Tried: {_install_command_summary()}"
        )


def _is_apple_silicon() -> bool:
    """Return True when running on an Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _mlx_runtime_available(*, auto_install: bool = False) -> bool:
    """Check whether the MLX runtime package can be imported."""
    if _can_import_mlx_runtime():
        return True

    if not auto_install:
        return False

    return _install_mlx_runtime()


def _can_import_mlx_runtime() -> bool:
    """Return True when corridorkey_mlx can be imported."""
    try:
        import corridorkey_mlx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False

    return True


def _install_mlx_runtime() -> bool:
    """Attempt a one-time runtime install of corridorkey_mlx."""
    global _MLX_INSTALL_ATTEMPTED

    if _MLX_INSTALL_ATTEMPTED:
        return _can_import_mlx_runtime()

    _MLX_INSTALL_ATTEMPTED = True

    if sys.version_info < (3, 11):
        logger.warning("Automatic MLX install requires Python 3.11+")
        return False

    logger.info("corridorkey_mlx not installed — attempting automatic install")

    for cmd in _install_commands():
        try:
            logger.info("Attempting MLX install via: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as err:
            logger.warning("Automatic corridorkey_mlx install failed: %s", err)
            continue

        if result.returncode == 0:
            importlib.invalidate_caches()
            if _can_import_mlx_runtime():
                logger.info("corridorkey_mlx installed successfully")
                return True

            logger.warning("corridorkey_mlx install completed but the package is still unavailable")
            return False

        details = (result.stderr or result.stdout or "").strip()
        if details:
            logger.warning("Automatic corridorkey_mlx install failed: %s", details.splitlines()[-1])
        else:
            logger.warning("Automatic corridorkey_mlx install failed with exit code %s", result.returncode)

    return False


def _install_commands() -> list[list[str]]:
    """Return installer commands in preferred order for the active interpreter."""
    commands: list[list[str]] = []

    uv_path = shutil.which("uv")
    if uv_path:
        commands.append([uv_path, "pip", "install", "--python", sys.executable, MLX_INSTALL_SPEC])

    commands.append([sys.executable, "-m", "pip", "install", MLX_INSTALL_SPEC])
    return commands


def _install_command_summary() -> str:
    """Human-readable summary of attempted installer commands."""
    return " or ".join(" ".join(cmd) for cmd in _install_commands())


def _ensure_mlx_checkpoint() -> Path:
    """Return an MLX checkpoint, converting from the Torch checkpoint when needed."""
    matches = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{MLX_EXT}"))

    if len(matches) == 1:
        return Path(matches[0])

    if len(matches) > 1:
        names = [os.path.basename(f) for f in matches]
        raise ValueError(f"Multiple {MLX_EXT} checkpoints in {CHECKPOINT_DIR}: {names}. Keep exactly one.")

    torch_checkpoint = _discover_checkpoint(TORCH_EXT)
    output_path = Path(CHECKPOINT_DIR) / DEFAULT_MLX_CHECKPOINT_NAME

    logger.info(
        "No %s checkpoint found — attempting automatic MLX conversion from %s",
        MLX_EXT,
        torch_checkpoint.name,
    )
    _convert_torch_checkpoint_to_mlx(torch_checkpoint, output_path)

    if not output_path.is_file():
        raise FileNotFoundError(
            f"Automatic MLX conversion completed but no {MLX_EXT} checkpoint was created at {output_path}."
        )

    return output_path


def _convert_torch_checkpoint_to_mlx(torch_checkpoint: Path, output_path: Path) -> None:
    """Clone the converter repo if needed, then generate MLX weights."""
    git_path = shutil.which("git")
    uv_path = shutil.which("uv")

    if not git_path:
        raise RuntimeError("Automatic MLX weight conversion requires `git` to be installed.")
    if not uv_path:
        raise RuntimeError("Automatic MLX weight conversion requires `uv` to be installed.")

    repo_dir = MLX_CONVERTER_REPO_DIR
    if not repo_dir.exists():
        logger.info("Cloning MLX converter repo into %s", repo_dir)
        _run_checked_command([git_path, "clone", MLX_CONVERTER_REPO_URL, str(repo_dir)], cwd=PROJECT_ROOT)

    convert_script = repo_dir / "scripts" / "convert_weights.py"
    if not convert_script.is_file():
        raise RuntimeError(f"MLX converter script not found at {convert_script}")

    logger.info("Syncing MLX converter dependencies in %s (including reference group)", repo_dir)
    _run_checked_command([uv_path, "sync", "--group", "reference"], cwd=repo_dir)

    logger.info("Converting Torch checkpoint %s -> %s", torch_checkpoint.name, output_path.name)
    _run_checked_command(
        [
            uv_path,
            "run",
            "--group",
            "reference",
            "python",
            "scripts/convert_weights.py",
            "--checkpoint",
            str(torch_checkpoint),
            "--output",
            str(output_path),
        ],
        cwd=repo_dir,
    )


def _run_checked_command(cmd: list[str], *, cwd: Path) -> None:
    """Run a subprocess and raise a concise error if it fails."""
    try:
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    except Exception as err:
        raise RuntimeError(f"Command `{' '.join(cmd)}` failed to start: {err}") from err

    if result.returncode == 0:
        return

    details = (result.stderr or result.stdout or "").strip()
    if details:
        raise RuntimeError(f"Command `{' '.join(cmd)}` failed: {details.splitlines()[-1]}")

    raise RuntimeError(f"Command `{' '.join(cmd)}` failed with exit code {result.returncode}")


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
        processed_alpha = cu.clean_matte(alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
    else:
        processed_alpha = alpha

    # Apply despill (MLX stubs this)
    fg_despilled = cu.despill(fg, green_limit_mode="average", strength=despill_strength)

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
        ckpt = _ensure_mlx_checkpoint()
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
