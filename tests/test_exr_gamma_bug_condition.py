"""Bug condition exploration tests for EXR gamma correction.

These tests encode the EXPECTED (correct) behavior for EXR+sRGB gamma handling.
On UNFIXED code they are expected to FAIL, confirming the bug exists.

**Validates: Requirements 1.1, 1.2, 1.3**

Defect 1 (1.1): run_inference EXR path ignores input_is_linear=False — no gamma correction applied.
Defect 2 (1.2): process_frame receives raw linear data with input_is_linear=False semantics.
Defect 3 (1.3): read_image_frame uses naive pow(1/2.2) instead of piecewise sRGB transfer function.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from backend.frame_io import read_image_frame
from CorridorKeyModule.core.color_utils import linear_to_srgb

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Float32 pixel values in [0, 1] — the valid linear range for EXR data
linear_pixel_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Small float32 RGB arrays representing EXR frame data
# Using small sizes (4x4 to 16x16) to keep tests fast
linear_pixel_arrays = arrays(
    dtype=np.float32,
    shape=st.tuples(
        st.integers(min_value=4, max_value=16),
        st.integers(min_value=4, max_value=16),
        st.just(3),
    ),
    elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)

# Strategy that specifically targets the sRGB piecewise threshold region
# Values near 0.0031308 where naive pow(1/2.2) diverges from piecewise sRGB
threshold_pixel_values = st.one_of(
    # Values below the threshold (linear segment of sRGB)
    st.floats(min_value=0.0, max_value=0.0031308, allow_nan=False, allow_infinity=False),
    # Values right around the threshold
    st.floats(min_value=0.002, max_value=0.005, allow_nan=False, allow_infinity=False),
    # Values above the threshold (power segment of sRGB)
    st.floats(min_value=0.0031308, max_value=1.0, allow_nan=False, allow_infinity=False),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_exr(path: str, rgb_data: np.ndarray) -> None:
    """Write a float32 RGB array as a BGR EXR file via OpenCV."""
    bgr = cv2.cvtColor(rgb_data.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


# ---------------------------------------------------------------------------
# Defect 3: naive pow(1/2.2) vs piecewise sRGB in read_image_frame
# ---------------------------------------------------------------------------


class TestDefect3NaivePowVsPiecewiseSRGB:
    """**Validates: Requirements 1.3**

    read_image_frame(exr_path, gamma_correct_exr=True) should produce output
    matching linear_to_srgb() from color_utils.py, NOT np.power(data, 1/2.2).
    """

    @given(data=linear_pixel_arrays)
    @settings(max_examples=50, deadline=None)
    def test_gamma_corrected_exr_matches_piecewise_srgb(self, data: np.ndarray) -> None:
        """When gamma_correct_exr=True, the result must match the piecewise
        sRGB transfer function, not the naive pow(1/2.2) approximation.

        **Validates: Requirements 1.3**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            exr_path = os.path.join(tmpdir, "test.exr")
            _write_exr(exr_path, data)

            result = read_image_frame(exr_path, gamma_correct_exr=True)
            assert result is not None, "read_image_frame returned None"

            # The expected output: piecewise sRGB transfer function
            raw = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
            if raw.ndim == 3 and raw.shape[2] == 4:
                raw = raw[:, :, :3]
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            linear_clamped = np.maximum(raw_rgb, 0.0).astype(np.float32)
            expected = linear_to_srgb(linear_clamped)

            np.testing.assert_allclose(
                result,
                expected,
                atol=1e-6,
                err_msg=(
                    "read_image_frame with gamma_correct_exr=True does not match "
                    "linear_to_srgb(). It likely uses naive pow(1/2.2) instead of "
                    "the piecewise sRGB transfer function."
                ),
            )

    @given(
        pixel_val=threshold_pixel_values,
    )
    @settings(max_examples=100, deadline=None)
    def test_threshold_region_divergence(self, pixel_val: float) -> None:
        """Specifically test values near the 0.0031308 threshold where
        naive pow(1/2.2) and piecewise sRGB diverge most.

        **Validates: Requirements 1.3**
        """
        # Create a small 2x2 image with the test pixel value
        data = np.full((2, 2, 3), pixel_val, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            exr_path = os.path.join(tmpdir, "threshold_test.exr")
            _write_exr(exr_path, data)

            result = read_image_frame(exr_path, gamma_correct_exr=True)
            assert result is not None

            expected = linear_to_srgb(data)

            np.testing.assert_allclose(
                result,
                expected,
                atol=1e-6,
                err_msg=(
                    f"At pixel value {pixel_val} (near sRGB threshold 0.0031308), "
                    f"read_image_frame produces {result.flat[0]:.8f} but piecewise "
                    f"sRGB expects {expected.flat[0]:.8f}. "
                    f"Naive pow(1/2.2) would give {np.power(pixel_val, 1.0 / 2.2):.8f}."
                ),
            )


# ---------------------------------------------------------------------------
# Defect 1 & 2: run_inference EXR path ignores gamma + wrong semantics
# ---------------------------------------------------------------------------


class TestDefect1And2RunInferenceEXRPath:
    """**Validates: Requirements 1.1, 1.2**

    When input is an EXR image sequence with input_is_linear=False,
    the frame passed to process_frame() must have sRGB gamma applied
    and process_frame must receive input_is_linear=False with properly
    gamma-corrected data.
    """

    @given(data=linear_pixel_arrays)
    @settings(max_examples=30, deadline=None)
    def test_exr_srgb_frame_is_gamma_corrected(self, data: np.ndarray) -> None:
        """Given an EXR image sequence with input_is_linear=False, the frame
        passed to process_frame() must NOT be raw linear data — it must have
        sRGB gamma applied.

        **Validates: Requirements 1.1, 1.2**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up a minimal clip structure with one EXR frame
            input_dir = os.path.join(tmpdir, "Input")
            alpha_dir = os.path.join(tmpdir, "AlphaHint")
            os.makedirs(input_dir)
            os.makedirs(alpha_dir)

            h, w = data.shape[:2]

            # Write EXR input frame
            exr_path = os.path.join(input_dir, "frame_00000.exr")
            _write_exr(exr_path, data)

            # Write a matching alpha mask (simple grayscale PNG)
            mask = np.full((h, w), 128, dtype=np.uint8)
            cv2.imwrite(os.path.join(alpha_dir, "frame_00000.png"), mask)

            # What we expect: the linear data should be gamma-corrected
            # via piecewise sRGB before reaching process_frame
            expected_srgb = linear_to_srgb(np.maximum(data, 0.0).astype(np.float32))

            # We'll capture what process_frame actually receives by patching it
            captured_args = {}

            def mock_process_frame(image, mask_linear, *, input_is_linear=False, **kwargs):
                captured_args["image"] = image.copy()
                captured_args["input_is_linear"] = input_is_linear
                # Return minimal valid result
                h_img, w_img = image.shape[:2]
                return {
                    "alpha": np.zeros((h_img, w_img, 1), dtype=np.float32),
                    "fg": np.zeros((h_img, w_img, 3), dtype=np.float32),
                    "comp": np.zeros((h_img, w_img, 3), dtype=np.float32),
                    "processed": np.zeros((h_img, w_img, 4), dtype=np.float32),
                }

            # Build a mock clip that looks like an EXR image sequence
            mock_clip = MagicMock()
            mock_clip.name = "test_clip"
            mock_clip.root_path = tmpdir
            mock_clip.input_asset.type = "sequence"
            mock_clip.input_asset.path = input_dir
            mock_clip.input_asset.frame_count = 1
            mock_clip.alpha_asset.type = "sequence"
            mock_clip.alpha_asset.path = alpha_dir
            mock_clip.alpha_asset.frame_count = 1

            # Create mock settings with input_is_linear=False (sRGB selected)
            mock_settings = MagicMock()
            mock_settings.input_is_linear = False
            mock_settings.despill_strength = 1.0
            mock_settings.auto_despeckle = False
            mock_settings.despeckle_size = 400
            mock_settings.refiner_scale = 1.0

            # Mock the engine
            mock_engine = MagicMock()
            mock_engine.process_frame = mock_process_frame

            # Patch create_engine where it's imported from inside run_inference
            with patch("CorridorKeyModule.backend.create_engine", return_value=mock_engine):
                from clip_manager import run_inference

                run_inference(
                    [mock_clip],
                    device="cpu",
                    max_frames=1,
                    settings=mock_settings,
                )

            assert "image" in captured_args, "process_frame was never called — clip setup may be wrong"

            actual_image = captured_args["image"]
            actual_is_linear = captured_args["input_is_linear"]

            # Defect 1: The frame should be gamma-corrected (sRGB), not raw linear
            # On buggy code, img_srgb contains raw linear data
            np.testing.assert_allclose(
                actual_image,
                expected_srgb,
                atol=1e-5,
                err_msg=(
                    "Frame passed to process_frame() is raw linear data, not "
                    "sRGB gamma-corrected. The EXR branch in run_inference() "
                    "does not apply gamma correction when input_is_linear=False."
                ),
            )

            # Defect 2: input_is_linear should be False (sRGB semantics)
            assert actual_is_linear is False, (
                f"process_frame received input_is_linear={actual_is_linear}, "
                f"expected False. The EXR data has wrong semantics."
            )
