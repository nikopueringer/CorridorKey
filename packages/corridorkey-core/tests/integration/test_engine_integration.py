"""Integration tests for CorridorKeyEngine.

These tests require a real checkpoint file and a CUDA GPU. They are skipped
by default and must be opted into with --run-gpu.

Set the checkpoint path via the CK_CHECKPOINT_PATH environment variable:

    CK_CHECKPOINT_PATH=/path/to/model.pth mise run test-gpu
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_CHECKPOINT_ENV = "CK_CHECKPOINT_PATH"


def _checkpoint_path() -> Path | None:
    raw = os.environ.get(_CHECKPOINT_ENV)
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


@pytest.fixture(scope="module")
def engine():
    """Load CorridorKeyEngine once for the entire module."""
    from corridorkey_core.inference_engine import CorridorKeyEngine

    ckpt = _checkpoint_path()
    if ckpt is None:
        pytest.skip(f"Set {_CHECKPOINT_ENV} to a valid .pth file to run integration tests")

    return CorridorKeyEngine(checkpoint_path=ckpt, device="cuda", img_size=2048)


@pytest.fixture
def sample_frame():
    """1080p synthetic frame and mask."""
    image = np.random.rand(1080, 1920, 3).astype(np.float32)
    mask = np.random.rand(1080, 1920).astype(np.float32)
    return image, mask


@pytest.mark.gpu
class TestProcessFrameContract:
    """Verify the output dict keys, shapes, dtypes, and value ranges."""

    def test_output_keys(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert set(result.keys()) == {"alpha", "fg", "comp", "processed"}

    def test_alpha_shape(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert result["alpha"].shape == (1080, 1920, 1)

    def test_fg_shape(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert result["fg"].shape == (1080, 1920, 3)

    def test_comp_shape(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert result["comp"].shape == (1080, 1920, 3)

    def test_processed_shape(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert result["processed"].shape == (1080, 1920, 4)

    def test_alpha_range(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert result["alpha"].min() >= 0.0
        assert result["alpha"].max() <= 1.0

    def test_fg_range(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert result["fg"].min() >= 0.0
        assert result["fg"].max() <= 1.0

    def test_comp_range(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        assert result["comp"].min() >= 0.0
        assert result["comp"].max() <= 1.0

    def test_all_outputs_float32(self, engine, sample_frame):
        image, mask = sample_frame
        result = engine.process_frame(image, mask)
        for key, arr in result.items():
            assert arr.dtype == np.float32, f"{key} dtype is {arr.dtype}, expected float32"


@pytest.mark.gpu
class TestProcessFrameInputHandling:
    """Verify the engine handles different input formats correctly."""

    def test_uint8_image_accepted(self, engine):
        image = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        mask = np.random.rand(256, 256).astype(np.float32)
        result = engine.process_frame(image, mask)
        assert result["alpha"].shape == (256, 256, 1)

    def test_3d_mask_accepted(self, engine):
        image = np.random.rand(256, 256, 3).astype(np.float32)
        mask = np.random.rand(256, 256, 1).astype(np.float32)
        result = engine.process_frame(image, mask)
        assert result["alpha"].shape == (256, 256, 1)

    def test_linear_input_flag(self, engine):
        image = np.random.rand(256, 256, 3).astype(np.float32)
        mask = np.random.rand(256, 256).astype(np.float32)
        result = engine.process_frame(image, mask, input_is_linear=True)
        assert result["alpha"].shape == (256, 256, 1)

    def test_despeckle_disabled(self, engine):
        image = np.random.rand(256, 256, 3).astype(np.float32)
        mask = np.random.rand(256, 256).astype(np.float32)
        result = engine.process_frame(image, mask, auto_despeckle=False)
        assert result["alpha"].shape == (256, 256, 1)

    def test_despill_disabled(self, engine):
        image = np.random.rand(256, 256, 3).astype(np.float32)
        mask = np.random.rand(256, 256).astype(np.float32)
        result = engine.process_frame(image, mask, despill_strength=0.0)
        assert result["alpha"].shape == (256, 256, 1)
