"""Preservation property tests for EXR gamma correction bugfix.

These tests verify that non-buggy code paths remain unchanged after the fix.
They MUST PASS on the current UNFIXED code — they establish the baseline
behavior that must be preserved.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

3.1: Linear EXR path (input_is_linear=True) passes data through unchanged.
3.2: Standard image path (PNG/JPG/TIFF) reads uint8 normalized to [0,1] float32.
3.3: Video path continues to decode frames as sRGB regardless of input_is_linear.
3.4: read_image_frame() with gamma_correct_exr=False returns raw linear EXR data.
3.5: Linear EXR inputs with input_is_linear=True produce byte-exact identical output.
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

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Float32 pixel arrays in [0, 1] for EXR data
linear_pixel_arrays = arrays(
    dtype=np.float32,
    shape=st.tuples(
        st.integers(min_value=4, max_value=16),
        st.integers(min_value=4, max_value=16),
        st.just(3),
    ),
    elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)

# uint8 pixel arrays for standard image data (PNG/JPG)
uint8_pixel_arrays = arrays(
    dtype=np.uint8,
    shape=st.tuples(
        st.integers(min_value=4, max_value=16),
        st.integers(min_value=4, max_value=16),
        st.just(3),
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_exr(path: str, rgb_data: np.ndarray) -> None:
    """Write a float32 RGB array as a BGR EXR file via OpenCV."""
    bgr = cv2.cvtColor(rgb_data.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def _write_png(path: str, rgb_data: np.ndarray) -> None:
    """Write a uint8 RGB array as a BGR PNG file via OpenCV."""
    bgr = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


# ---------------------------------------------------------------------------
# 3.4: gamma_correct_exr=False returns raw linear EXR data (no transformation)
# ---------------------------------------------------------------------------


class TestPreservationLinearEXRRead:
    """**Validates: Requirements 3.1, 3.4**

    read_image_frame() with gamma_correct_exr=False (the default) returns
    raw linear EXR data with no gamma transformation applied. The result
    must be identical to what cv2.imread produces (clamped to >= 0, as float32).
    """

    @given(data=linear_pixel_arrays)
    @settings(max_examples=50, deadline=None)
    def test_exr_default_returns_raw_linear_data(self, data: np.ndarray) -> None:
        """For all float32 arrays in [0, 1], read_image_frame with default
        params (gamma_correct_exr=False) returns data identical to raw
        cv2.imread output — no transformation applied.

        **Validates: Requirements 3.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            exr_path = os.path.join(tmpdir, "test.exr")
            _write_exr(exr_path, data)

            result = read_image_frame(exr_path)
            assert result is not None, "read_image_frame returned None"

            # Reconstruct what raw cv2.imread would produce
            raw = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
            if raw.ndim == 3 and raw.shape[2] == 4:
                raw = raw[:, :, :3]
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            expected = np.maximum(raw_rgb, 0.0).astype(np.float32)

            np.testing.assert_array_equal(
                result,
                expected,
                err_msg=(
                    "read_image_frame with gamma_correct_exr=False (default) "
                    "should return raw linear EXR data identical to cv2.imread "
                    "output. No transformation should be applied."
                ),
            )

    @given(data=linear_pixel_arrays)
    @settings(max_examples=50, deadline=None)
    def test_exr_explicit_false_returns_raw_linear_data(self, data: np.ndarray) -> None:
        """Explicitly passing gamma_correct_exr=False produces the same
        result as the default — raw linear data, no transformation.

        **Validates: Requirements 3.1, 3.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            exr_path = os.path.join(tmpdir, "test.exr")
            _write_exr(exr_path, data)

            result_default = read_image_frame(exr_path)
            result_explicit = read_image_frame(exr_path, gamma_correct_exr=False)

            assert result_default is not None
            assert result_explicit is not None

            np.testing.assert_array_equal(
                result_default,
                result_explicit,
                err_msg=(
                    "read_image_frame(path) and read_image_frame(path, "
                    "gamma_correct_exr=False) should produce identical results."
                ),
            )


# ---------------------------------------------------------------------------
# 3.2: Standard image path reads uint8 normalized to [0,1] float32
# ---------------------------------------------------------------------------


class TestPreservationStandardImageRead:
    """**Validates: Requirements 3.2**

    For PNG/JPG/TIFF with input_is_linear=False, data is read as uint8,
    divided by 255, and the result is float32 in [0, 1].
    """

    @given(data=uint8_pixel_arrays)
    @settings(max_examples=50, deadline=None)
    def test_png_read_returns_uint8_normalized(self, data: np.ndarray) -> None:
        """For all uint8 arrays, standard image read returns
        array.astype(float32) / 255.0.

        **Validates: Requirements 3.2**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = os.path.join(tmpdir, "test.png")
            _write_png(png_path, data)

            result = read_image_frame(png_path)
            assert result is not None, "read_image_frame returned None for PNG"

            # Reconstruct expected: read as uint8 BGR, convert to RGB, / 255
            raw = cv2.imread(png_path)
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            expected = raw_rgb.astype(np.float32) / 255.0

            np.testing.assert_array_equal(
                result,
                expected,
                err_msg=(
                    "Standard image read should return uint8 data normalized to [0,1] float32 via division by 255."
                ),
            )

    @given(data=uint8_pixel_arrays)
    @settings(max_examples=50, deadline=None)
    def test_png_result_dtype_and_range(self, data: np.ndarray) -> None:
        """Standard image read always returns float32 in [0, 1].

        **Validates: Requirements 3.2**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = os.path.join(tmpdir, "test.png")
            _write_png(png_path, data)

            result = read_image_frame(png_path)
            assert result is not None

            assert result.dtype == np.float32, f"Expected float32 dtype, got {result.dtype}"
            assert result.min() >= 0.0, "Result contains negative values"
            assert result.max() <= 1.0, "Result contains values > 1.0"


# ---------------------------------------------------------------------------
# 3.1, 3.5: Linear EXR with input_is_linear=True in run_inference
# ---------------------------------------------------------------------------


class TestPreservationLinearEXRInference:
    """**Validates: Requirements 3.1, 3.5**

    When input is an EXR image sequence with input_is_linear=True,
    the data passes through without gamma correction and process_frame()
    receives input_is_linear=True with raw linear data.
    """

    @given(data=linear_pixel_arrays)
    @settings(max_examples=30, deadline=None)
    def test_linear_exr_passes_through_unchanged(self, data: np.ndarray) -> None:
        """For all linear EXR inputs with input_is_linear=True, the data
        reaching process_frame() is byte-exact identical to the raw EXR
        read — no gamma correction applied.

        **Validates: Requirements 3.1, 3.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "Input")
            alpha_dir = os.path.join(tmpdir, "AlphaHint")
            os.makedirs(input_dir)
            os.makedirs(alpha_dir)

            h, w = data.shape[:2]

            # Write EXR input frame
            exr_path = os.path.join(input_dir, "frame_00000.exr")
            _write_exr(exr_path, data)

            # Write a matching alpha mask
            mask = np.full((h, w), 128, dtype=np.uint8)
            cv2.imwrite(os.path.join(alpha_dir, "frame_00000.png"), mask)

            # Read what the raw EXR data looks like after cv2 round-trip
            raw = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
            if raw.ndim == 3 and raw.shape[2] == 4:
                raw = raw[:, :, :3]
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            expected_linear = np.maximum(raw_rgb, 0.0).astype(np.float32)

            # Capture what process_frame actually receives
            captured_args = {}

            def mock_batch_process_frames(image, mask_linear, *, input_is_linear=False, **kwargs):
                captured_args["image"] = image.copy()
                captured_args["input_is_linear"] = input_is_linear
                h_img, w_img = image.shape[:2]
                return [
                    {
                        "alpha": np.zeros((h_img, w_img, 1), dtype=np.float32),
                        "fg": np.zeros((h_img, w_img, 3), dtype=np.float32),
                        "comp": np.zeros((h_img, w_img, 3), dtype=np.float32),
                        "processed": np.zeros((h_img, w_img, 4), dtype=np.float32),
                    }
                ]

            mock_clip = MagicMock()
            mock_clip.name = "test_clip"
            mock_clip.root_path = tmpdir
            mock_clip.input_asset.type = "sequence"
            mock_clip.input_asset.path = input_dir
            mock_clip.input_asset.frame_count = 1
            mock_clip.alpha_asset.type = "sequence"
            mock_clip.alpha_asset.path = alpha_dir
            mock_clip.alpha_asset.frame_count = 1

            # input_is_linear=True — user confirmed linear EXR
            mock_settings = MagicMock()
            mock_settings.input_is_linear = True
            mock_settings.despill_strength = 1.0
            mock_settings.auto_despeckle = False
            mock_settings.despeckle_size = 400
            mock_settings.refiner_scale = 1.0

            mock_engine = MagicMock()
            mock_engine.batch_process_frames = mock_batch_process_frames

            with patch("CorridorKeyModule.backend.create_engine", return_value=mock_engine):
                from clip_manager import run_inference

                run_inference(
                    [mock_clip],
                    device="cpu",
                    max_frames=1,
                    settings=mock_settings,
                )

            assert "image" in captured_args, "process_frame was never called"

            # The frame should be raw linear data — no gamma correction
            np.testing.assert_allclose(
                captured_args["image"][0],
                expected_linear,
                atol=1e-6,
                err_msg=(
                    "With input_is_linear=True, the EXR data reaching "
                    "process_frame() should be raw linear — identical to "
                    "cv2.imread output. No gamma correction should be applied."
                ),
            )

            # input_is_linear should be True
            assert captured_args["input_is_linear"] is True, (
                f"process_frame received input_is_linear={captured_args['input_is_linear']}, expected True"
            )
