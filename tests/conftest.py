"""Shared pytest configuration and fixtures for CorridorKey tests."""

import platform
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _has_gpu():
    """Check if any GPU backend (CUDA or MPS) is available."""
    if torch.cuda.is_available():
        return True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True

    return False


def _has_mlx():
    """Check if MLX is available (Apple Silicon + corridorkey_mlx installed)."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import corridorkey_mlx  # noqa: F401

        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU/MLX tests when hardware is unavailable."""
    if not _has_gpu():
        skip_gpu = pytest.mark.skip(reason="No GPU available (neither CUDA nor MPS)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    if not _has_mlx():
        skip_mlx = pytest.mark.skip(reason="MLX not available (requires Apple Silicon + corridorkey_mlx)")
        for item in items:
            if "mlx" in item.keywords:
                item.add_marker(skip_mlx)


# ---------------------------------------------------------------------------
# Basic frame/mask fixtures (used by color_utils and inference_engine tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_frame_rgb():
    """Small 64x64 RGB frame as float32 in [0, 1] (sRGB)."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64, 3), dtype=np.float32)


@pytest.fixture
def sample_mask():
    """Matching 64x64 single-channel alpha mask as float32 in [0, 1]."""
    rng = np.random.default_rng(42)
    mask = rng.random((64, 64), dtype=np.float32)
    # Make it more mask-like: threshold to create distinct FG/BG regions
    return (mask > 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Clip directory structure fixtures (used by clip_manager tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_clip_dir(tmp_path):
    """Creates a temporary directory with the expected clip structure.

    Layout::

        tmp_path/
          shot_a/
            Input/
              frame_0000.png
              frame_0001.png
            AlphaHint/
              frame_0000.png
              frame_0001.png
            VideoMamaMaskHint/
          shot_b/
            Input/
              frame_0000.png
            AlphaHint/         (empty — needs generation)
            VideoMamaMaskHint/

    PNG files are tiny 4x4 images so tests run fast.
    """
    import cv2

    tiny_img = np.zeros((4, 4, 3), dtype=np.uint8)
    tiny_img[1:3, 1:3] = 255  # small white square
    tiny_mask = np.zeros((4, 4), dtype=np.uint8)
    tiny_mask[1:3, 1:3] = 255

    # shot_a — fully ready (Input + AlphaHint populated)
    shot_a = tmp_path / "shot_a"
    for subdir in ["Input", "AlphaHint", "VideoMamaMaskHint"]:
        (shot_a / subdir).mkdir(parents=True)

    for i in range(2):
        cv2.imwrite(str(shot_a / "Input" / f"frame_{i:04d}.png"), tiny_img)
        cv2.imwrite(str(shot_a / "AlphaHint" / f"frame_{i:04d}.png"), tiny_mask)

    # shot_b — Input only, empty AlphaHint (needs generation)
    shot_b = tmp_path / "shot_b"
    for subdir in ["Input", "AlphaHint", "VideoMamaMaskHint"]:
        (shot_b / subdir).mkdir(parents=True)

    cv2.imwrite(str(shot_b / "Input" / "frame_0000.png"), tiny_img)

    return tmp_path


# ---------------------------------------------------------------------------
# Mock inference engine fixture (used by inference_engine tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_greenformer():
    """A mock GreenFormer model that returns deterministic tensors.

    Returns alpha=0.8 and fg=0.6 everywhere, sized to match the input.
    No GPU or model weights needed.
    """

    def fake_forward(x):
        b, c, h, w = x.shape
        return {
            "alpha": torch.full((b, 1, h, w), 0.8, device=x.device),
            "fg": torch.full((b, 3, h, w), 0.6, device=x.device),
        }

    model = MagicMock()
    model.side_effect = fake_forward
    model.refiner = None
    model.use_refiner = False
    return model
