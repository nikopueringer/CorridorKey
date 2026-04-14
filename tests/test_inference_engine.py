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

from CorridorKeyModule.core import color_utils as cu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_with_mock(mock_greenformer, img_size=64, device="cpu"):
    """Create a CorridorKeyEngine with a mocked model, bypassing __init__.

    Manually sets the attributes that __init__ would create, avoiding the
    need for checkpoint files or GPU.
    """
    from CorridorKeyModule.inference_engine import CorridorKeyEngine

    engine = object.__new__(CorridorKeyEngine)
    engine.device = torch.device(device)
    engine.img_size = img_size
    engine.checkpoint_path = "/fake/checkpoint.pth"
    engine.use_refiner = False
    engine.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=torch.device(device)).reshape(3, 1, 1)
    engine.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=torch.device(device)).reshape(3, 1, 1)
    engine.model = mock_greenformer
    engine.model_precision = torch.float32
    engine.mixed_precision = True
    return engine


# ---------------------------------------------------------------------------
# process_frame output structure
# ---------------------------------------------------------------------------


class TestProcessFrameOutputs:
    """Verify shape, dtype, and key presence of process_frame outputs."""

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_output_keys(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """process_frame must return alpha, fg, comp, and processed."""
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")[0]
        else:
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")

        assert "alpha" in result
        assert "fg" in result
        assert "comp" in result
        assert "processed" in result

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_output_shapes_match_input(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """All outputs should match the spatial dimensions of the input."""
        h, w = sample_frame_rgb.shape[:2]
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")[0]
        else:
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")

        assert result["alpha"].shape[:2] == (h, w)
        assert result["fg"].shape[:2] == (h, w)
        assert result["comp"].shape == (h, w, 3)
        assert result["processed"].shape == (h, w, 4)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_output_dtype_float32(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """All outputs should be float32 numpy arrays."""
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")[0]
        else:
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")

        for key in ("alpha", "fg", "comp", "processed"):
            assert result[key].dtype == np.float32, f"{key} should be float32"

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_alpha_output_range_is_zero_to_one(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """Alpha output must be in [0, 1] — values outside this range corrupt compositing."""
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")[0]
        else:
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")
        alpha = result["alpha"]
        assert alpha.min() >= -0.01, f"alpha min {alpha.min():.4f} is below 0"
        assert alpha.max() <= 1.01, f"alpha max {alpha.max():.4f} is above 1"

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_fg_output_range_is_zero_to_one(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """FG output must be in [0, 1] — required for downstream sRGB conversion and EXR export."""
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")[0]
        else:
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")
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

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_srgb_input_default(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """Default sRGB path should not crash and should return valid outputs."""
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(
                sample_frame_rgb, sample_mask, input_is_linear=False, post_process_on_gpu=backend == "torch"
            )[0]
        else:
            result = engine.process_frame(
                sample_frame_rgb, sample_mask, input_is_linear=False, post_process_on_gpu=backend == "torch"
            )

        np.testing.assert_allclose(result["comp"], 0.545655, atol=1e-4)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_linear_input_path(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """Linear input path should convert to sRGB before model input."""
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(
                sample_frame_rgb, sample_mask, input_is_linear=True, post_process_on_gpu=backend == "torch"
            )[0]
        else:
            result = engine.process_frame(
                sample_frame_rgb, sample_mask, input_is_linear=True, post_process_on_gpu=backend == "torch"
            )
        assert result["comp"].shape == sample_frame_rgb.shape[1:] if batched else sample_frame_rgb.shape

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_uint8_input_normalized(self, sample_mask, mock_greenformer, backend, batched):
        """uint8 input should be auto-converted to float32 [0, 1]."""
        img_uint8 = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            img_uint8 = np.stack([img_uint8] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(img_uint8, sample_mask, post_process_on_gpu=backend == "torch")[0]
        else:
            result = engine.process_frame(img_uint8, sample_mask, post_process_on_gpu=backend == "torch")
        assert result["alpha"].dtype == np.float32

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_model_called_exactly_once(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """The neural network model must be called exactly once per process_frame() call.

        Double-inference would double latency and produce incorrect outputs.
        """
        engine = _make_engine_with_mock(mock_greenformer)
        engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")
        assert mock_greenformer.call_count == 1


# ---------------------------------------------------------------------------
# Post-processing pipeline
# ---------------------------------------------------------------------------


class TestProcessFramePostProcessing:
    """Verify post-processing: despill, despeckle, premultiply, composite."""

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_despill_strength_reduces_green_in_spill_pixels(self, sample_frame_rgb, sample_mask, backend, batched):
        """despill_strength=1.0 must reduce green in spill pixels; strength=0.0 must leave it unchanged.

        The default mock_greenformer returns uniform gray (R=G=B=0.6) which has no
        green spill by definition: limit=(R+B)/2=0.6=G so spill_amount=0 always.
        This test uses a green-heavy fg mock (R=0.2, G=0.8, B=0.2) to force
        spill_amount > 0 and verify the despill path actually runs and reduces green.
        """
        from unittest.mock import MagicMock

        def green_heavy_forward(x):
            b, c, h, w = x.shape
            fg = torch.zeros(b, 3, h, w, dtype=torch.float32)
            fg[:, 0, :, :] = 0.2  # R
            fg[:, 1, :, :] = 0.8  # G — heavy green spill: G >> (R+B)/2
            fg[:, 2, :, :] = 0.2  # B
            return {
                "alpha": torch.full((b, 1, h, w), 0.8, dtype=torch.float32),
                "fg": fg,
            }

        green_mock = MagicMock()
        green_mock.side_effect = green_heavy_forward
        green_mock.refiner = None
        green_mock.use_refiner = False

        engine = _make_engine_with_mock(green_mock)

        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result_no_despill = engine.process_frame(
                sample_frame_rgb, sample_mask, despill_strength=0.0, post_process_on_gpu=backend == "torch"
            )[0]
            result_full_despill = engine.process_frame(
                sample_frame_rgb, sample_mask, despill_strength=1.0, post_process_on_gpu=backend == "torch"
            )[0]
        else:
            result_no_despill = engine.process_frame(
                sample_frame_rgb, sample_mask, despill_strength=0.0, post_process_on_gpu=backend == "torch"
            )
            result_full_despill = engine.process_frame(
                sample_frame_rgb, sample_mask, despill_strength=1.0, post_process_on_gpu=backend == "torch"
            )

        rgb_none = result_no_despill["processed"][:, :, :3]
        rgb_full = result_full_despill["processed"][:, :, :3]

        # Both outputs must be valid shapes and in-range
        assert rgb_none.shape == rgb_full.shape
        assert rgb_none.min() >= 0.0
        assert rgb_full.min() >= 0.0

        # Green channel must be reduced by despill (spill_amount > 0 is guaranteed by construction)
        assert rgb_full[:, :, 1].mean() < rgb_none[:, :, 1].mean(), (
            "despill_strength=1.0 should reduce the green channel relative to strength=0.0 when G > (R+B)/2"
        )

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_auto_despeckle_toggle(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """auto_despeckle=False should skip clean_matte without crashing."""
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(
                sample_frame_rgb, sample_mask, auto_despeckle=False, post_process_on_gpu=backend == "torch"
            )[0]
            sample_frame_rgb = sample_frame_rgb[0]  # for the shape assertion below
        else:
            result = engine.process_frame(
                sample_frame_rgb, sample_mask, auto_despeckle=False, post_process_on_gpu=backend == "torch"
            )
        assert result["alpha"].shape[:2] == sample_frame_rgb.shape[:2]

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_processed_is_linear_premul_rgba(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """The 'processed' output should be 4-channel RGBA (linear, premultiplied).

        This is the EXR-ready output that compositors load into Nuke for
        an Over operation. The RGB channels should be <= alpha (premultiplied
        means color is already multiplied by alpha).
        """
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")[0]
        else:
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")
        processed = result["processed"]
        assert processed.shape[2] == 4

        rgb = processed[:, :, :3]
        alpha = processed[:, :, 3:4]
        # Use srgb_to_linear rather than the gamma 2.2 approximation (x**2.2).
        # LLM_HANDOVER.md Bug History: "Do not apply a pure mathematical Gamma 2.2
        # curve; use the piecewise real sRGB transfer functions defined in color_utils.py."
        # The difference between the two at FG=0.6 is ~0.005, which the previous
        # atol=1e-2 was too loose to catch — a gamma 2.2 regression would have passed.
        expected_premul = cu.srgb_to_linear(np.float32(0.6)) * 0.8
        np.testing.assert_allclose(alpha, 0.8, atol=1e-5)
        np.testing.assert_allclose(rgb, expected_premul, atol=1e-4)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_mask_2d_vs_3d_input(self, sample_frame_rgb, mock_greenformer, backend, batched):
        """process_frame should accept both [H, W] and [H, W, 1] masks."""
        engine = _make_engine_with_mock(mock_greenformer)
        mask_2d = np.ones((64, 64), dtype=np.float32) * 0.5
        mask_3d = mask_2d[:, :, np.newaxis]

        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            mask_2d = np.stack([mask_2d] * 2, axis=0)
            mask_3d = np.stack([mask_3d] * 2, axis=0)
            result_2d = engine.process_frame(sample_frame_rgb, mask_2d, post_process_on_gpu=backend == "torch")[0]
            result_3d = engine.process_frame(sample_frame_rgb, mask_3d, post_process_on_gpu=backend == "torch")[0]
        else:
            result_2d = engine.process_frame(sample_frame_rgb, mask_2d, post_process_on_gpu=backend == "torch")
            result_3d = engine.process_frame(sample_frame_rgb, mask_3d, post_process_on_gpu=backend == "torch")

        # Both should produce the same output
        np.testing.assert_allclose(result_2d["alpha"], result_3d["alpha"], atol=1e-5)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_refiner_scale_parameter_accepted(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """Non-default refiner_scale must not raise — the parameter must be threaded through."""
        engine = _make_engine_with_mock(mock_greenformer)
        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(
                sample_frame_rgb, sample_mask, refiner_scale=0.5, post_process_on_gpu=backend == "torch"
            )[0]
            sample_frame_rgb = sample_frame_rgb[0]  # for the shape assertion below
        else:
            result = engine.process_frame(
                sample_frame_rgb, sample_mask, refiner_scale=0.5, post_process_on_gpu=backend == "torch"
            )
        assert result["alpha"].shape[:2] == sample_frame_rgb.shape[:2]


# ---------------------------------------------------------------------------
# NVIDIA Specific GPU test
# ---------------------------------------------------------------------------


class TestNvidiaGPUProcess:
    @pytest.mark.gpu
    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("batched", [True, False])
    def test_process_frame_on_gpu(self, sample_frame_rgb, sample_mask, mock_greenformer, backend, batched):
        """
        Scenario: Process a frame using a CUDA-configured engine.
        Expected: Input tensors are moved to CUDA before the model is called,
        confirmed by asserting the device of the tensor the mock received.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        captured_device: list[torch.device] = []
        original_side_effect = mock_greenformer.side_effect

        def spy_forward(x):
            captured_device.append(x.device)
            return original_side_effect(x)

        mock_greenformer.side_effect = spy_forward

        engine = _make_engine_with_mock(mock_greenformer, device="cuda")

        if batched:
            sample_frame_rgb = np.stack([sample_frame_rgb] * 2, axis=0)
            sample_mask = np.stack([sample_mask] * 2, axis=0)
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")
            result = result[0]
        else:
            result = engine.process_frame(sample_frame_rgb, sample_mask, post_process_on_gpu=backend == "torch")
        assert result["alpha"].dtype == np.float32
        assert len(captured_device) == 1, "Model should be called exactly once"
        assert captured_device[0].type == "cuda", f"Expected model input on cuda, got {captured_device[0]}"


# ---------------------------------------------------------------------------
# Checkpoint format dispatch (.safetensors vs .pth)
# ---------------------------------------------------------------------------


class TestLoadModelFormatDispatch:
    """Verify _load_model routes to the right loader based on file extension.

    The engine must support both .safetensors (preferred) and .pth (legacy).
    These tests exercise the branch in _load_model without needing the real
    GreenFormer or full model weights — we patch GreenFormer and feed it a
    tiny random state dict in each format.
    """

    def _make_random_state_dict(self) -> dict[str, torch.Tensor]:
        """A handful of leaf tensors that load_state_dict(strict=False) ignores cleanly."""
        return {
            "dummy.weight": torch.zeros(4, 4),
            "dummy.bias": torch.zeros(4),
        }

    def _patch_greenformer(self, monkeypatch):
        """Replace GreenFormer with a stub exposing the state-dict API _load_model needs."""
        from unittest.mock import MagicMock

        import CorridorKeyModule.inference_engine as engine_mod

        stub = MagicMock()
        stub.state_dict.return_value = self._make_random_state_dict()
        stub.to.return_value = stub
        stub.eval.return_value = stub
        stub.load_state_dict.return_value = ([], [])

        factory = MagicMock(return_value=stub)
        monkeypatch.setattr(engine_mod, "GreenFormer", factory)
        return stub

    def test_safetensors_checkpoint_loads_via_safetensors_library(self, tmp_path, monkeypatch):
        """A .safetensors path must be dispatched to safetensors.torch.load_file."""
        from unittest import mock

        from safetensors.torch import save_file

        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        ckpt = tmp_path / "model.safetensors"
        save_file(self._make_random_state_dict(), str(ckpt))

        self._patch_greenformer(monkeypatch)
        # torch.load must NOT be called when the checkpoint is safetensors.
        with mock.patch("torch.load", side_effect=AssertionError("torch.load called for .safetensors")):
            engine = object.__new__(CorridorKeyEngine)
            engine.device = torch.device("cpu")
            engine.img_size = 64
            engine.checkpoint_path = str(ckpt)
            engine.use_refiner = False
            engine.model_precision = torch.float32
            engine._is_rocm = False

            model = engine._load_model()
            assert model is not None

    def test_pth_checkpoint_loads_via_torch_load(self, tmp_path, monkeypatch):
        """A .pth path must be dispatched to torch.load (legacy path)."""
        from unittest import mock

        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        ckpt = tmp_path / "model.pth"
        torch.save(self._make_random_state_dict(), str(ckpt))

        self._patch_greenformer(monkeypatch)
        with mock.patch(
            "safetensors.torch.load_file",
            side_effect=AssertionError("safetensors.load_file called for .pth"),
        ):
            engine = object.__new__(CorridorKeyEngine)
            engine.device = torch.device("cpu")
            engine.img_size = 64
            engine.checkpoint_path = str(ckpt)
            engine.use_refiner = False
            engine.model_precision = torch.float32
            engine._is_rocm = False

            model = engine._load_model()
            assert model is not None
