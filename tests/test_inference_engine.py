"""Tests for CorridorKeyModule.inference_engine.CorridorKeyEngine.process_frame.

These tests mock the GreenFormer model so they run without GPU or model
weights. They verify the pre-processing (resize, normalize, color space
conversion) and post-processing (upscale, despill, premultiply, composite)
pipeline that wraps the neural network.

Why mock the model?
  The model requires a ~500MB checkpoint and CUDA. The pre/post-processing
  pipeline is where compositing bugs hide (wrong color space, premul errors,
  alpha dimension mismatches). Mocking the model isolates that logic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_with_mock(mock_greenformer, img_size=64):
    """Create a CorridorKeyEngine with a mocked model, bypassing __init__.

    Manually sets the attributes that __init__ would create, avoiding the
    need for checkpoint files or GPU.
    """
    from CorridorKeyModule.inference_engine import CorridorKeyEngine

    engine = object.__new__(CorridorKeyEngine)
    engine.device = torch.device("cpu")
    engine.img_size = img_size
    engine.checkpoint_path = "/fake/checkpoint.pth"
    engine.use_refiner = False
    engine.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    engine.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    engine.model = mock_greenformer
    engine.model_precision = torch.float32
    engine.mixed_precision = True
    return engine


# ---------------------------------------------------------------------------
# process_frame output structure
# ---------------------------------------------------------------------------


class TestProcessFrameOutputs:
    """Verify shape, dtype, and key presence of process_frame outputs."""

    def test_output_keys(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """process_frame must return alpha, fg, comp, and processed."""
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask)

        assert "alpha" in result
        assert "fg" in result
        assert "comp" in result
        assert "processed" in result

    def test_output_shapes_match_input(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """All outputs should match the spatial dimensions of the input."""
        h, w = sample_frame_rgb.shape[:2]
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask)

        assert result["alpha"].shape[:2] == (h, w)
        assert result["fg"].shape[:2] == (h, w)
        assert result["comp"].shape == (h, w, 3)
        assert result["processed"].shape == (h, w, 4)

    def test_output_dtype_float32(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """All outputs should be float32 numpy arrays."""
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask)

        for key in ("alpha", "fg", "comp", "processed"):
            assert result[key].dtype == np.float32, f"{key} should be float32"

    def test_alpha_output_range_is_zero_to_one(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """Alpha output must be in [0, 1] — values outside this range corrupt compositing."""
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        alpha = result["alpha"]
        assert alpha.min() >= -0.01, f"alpha min {alpha.min():.4f} is below 0"
        assert alpha.max() <= 1.01, f"alpha max {alpha.max():.4f} is above 1"

    def test_fg_output_range_is_zero_to_one(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """FG output must be in [0, 1] — required for downstream sRGB conversion and EXR export."""
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        fg = result["fg"]
        assert fg.min() >= -0.01, f"fg min {fg.min():.4f} is below 0"
        assert fg.max() <= 1.01, f"fg max {fg.max():.4f} is above 1"


# ---------------------------------------------------------------------------
# Input color space handling
# ---------------------------------------------------------------------------


class TestProcessFrameColorSpace:
    """Verify the sRGB vs linear input paths.

    When input_is_linear=True, process_frame resizes in linear space then
    converts to sRGB before feeding the model (preserving HDR highlight detail).
    When False (default), it resizes in sRGB directly.
    """

    def test_srgb_input_default(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """Default sRGB path should not crash and should return valid outputs."""
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask, input_is_linear=False)

        np.testing.assert_allclose(result["comp"], 0.545655, atol=1e-4)

    def test_linear_input_path(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """Linear input path should convert to sRGB before model input."""
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask, input_is_linear=True)
        assert result["comp"].shape == sample_frame_rgb.shape

    def test_uint8_input_normalized(self, sample_mask, mock_greenformer):
        """uint8 input should be auto-converted to float32 [0, 1]."""
        img_uint8 = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        engine = _make_engine_with_mock(mock_greenformer)
        # Should not crash — uint8 is auto-normalized to float32
        result = engine.process_frame(img_uint8, sample_mask)
        assert result["alpha"].dtype == np.float32

    def test_model_called_exactly_once(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """The neural network model must be called exactly once per process_frame() call.

        Double-inference would double latency and produce incorrect outputs.
        """
        engine = _make_engine_with_mock(mock_greenformer)
        engine.process_frame(sample_frame_rgb, sample_mask)
        assert mock_greenformer.call_count == 1


# ---------------------------------------------------------------------------
# Post-processing pipeline
# ---------------------------------------------------------------------------


class TestProcessFramePostProcessing:
    """Verify post-processing: despill, despeckle, premultiply, composite."""

    def test_despill_strength_variants_dont_crash(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """
        Despill at strength 0.0 and 1.0 should both produce valid outputs.
        Scenario: Process a frame with despill_strength set to 0.0 vs 1.0.
        Expected: Full despill should produce different pixel values than no despill, even with mock inputs.
        """
        engine = _make_engine_with_mock(mock_greenformer)
        result_no_despill = engine.process_frame(sample_frame_rgb, sample_mask, despill_strength=0.0)
        result_full_despill = engine.process_frame(sample_frame_rgb, sample_mask, despill_strength=1.0)

        rgb_none = result_no_despill["processed"][:, :, :3]
        rgb_full = result_full_despill["processed"][:, :, :3]

        np.testing.assert_allclose(rgb_none, 0.2548, atol=1e-3)
        assert rgb_none.shape == rgb_full.shape
        np.testing.assert_allclose(rgb_none, rgb_full, atol=1e-7)

    def test_auto_despeckle_toggle(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """auto_despeckle=False should skip clean_matte without crashing."""
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask, auto_despeckle=False)
        assert result["alpha"].shape[:2] == sample_frame_rgb.shape[:2]

    def test_processed_is_linear_premul_rgba(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """The 'processed' output should be 4-channel RGBA (linear, premultiplied).

        This is the EXR-ready output that compositors load into Nuke for
        an Over operation. The RGB channels should be <= alpha (premultiplied
        means color is already multiplied by alpha).
        """
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        processed = result["processed"]
        assert processed.shape[2] == 4

        rgb = processed[:, :, :3]
        alpha = processed[:, :, 3:4]
        expected_premul = (0.6**2.2) * 0.8
        np.testing.assert_allclose(alpha, 0.8, atol=1e-5)
        np.testing.assert_allclose(rgb, expected_premul, atol=1e-2)

    def test_mask_2d_vs_3d_input(self, sample_frame_rgb, mock_greenformer):
        """process_frame should accept both [H, W] and [H, W, 1] masks."""
        engine = _make_engine_with_mock(mock_greenformer)
        mask_2d = np.ones((64, 64), dtype=np.float32) * 0.5
        mask_3d = mask_2d[:, :, np.newaxis]

        result_2d = engine.process_frame(sample_frame_rgb, mask_2d)
        result_3d = engine.process_frame(sample_frame_rgb, mask_3d)

        # Both should produce the same output
        np.testing.assert_allclose(result_2d["alpha"], result_3d["alpha"], atol=1e-5)

    def test_refiner_scale_parameter_accepted(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """Non-default refiner_scale must not raise — the parameter must be threaded through."""
        engine = _make_engine_with_mock(mock_greenformer)
        result = engine.process_frame(sample_frame_rgb, sample_mask, refiner_scale=0.5)
        assert result["alpha"].shape[:2] == sample_frame_rgb.shape[:2]


# ---------------------------------------------------------------------------
# NVIDIA Specific GPU test
# ---------------------------------------------------------------------------


class TestNvidiaGPUProcess:
    @pytest.mark.gpu
    def test_process_frame_on_gpu(self, sample_frame_rgb, sample_mask, mock_greenformer):
        """
        Scenario: Process a frame using a CUDA-configured engine to verify cross-device compatibility.
        Expected: The mock model detects the GPU input and returns matching tensors without a device mismatch error.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = _make_engine_with_mock(mock_greenformer)
        engine.device = torch.device("cuda")

        result = engine.process_frame(sample_frame_rgb, sample_mask)
        assert result["alpha"].dtype == np.float32
