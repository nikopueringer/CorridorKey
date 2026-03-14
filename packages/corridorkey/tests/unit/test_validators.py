"""Unit tests for validators.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from corridorkey.errors import FrameMismatchError, FrameReadError, MaskChannelError, WriteFailureError
from corridorkey.validators import (
    ensure_output_dirs,
    normalize_mask_channels,
    normalize_mask_dtype,
    validate_frame_counts,
    validate_frame_read,
    validate_write,
)


class TestValidateFrameCounts:
    def test_equal_counts_returns_count(self):
        assert validate_frame_counts("clip", 100, 100) == 100

    def test_mismatch_non_strict_returns_minimum(self):
        assert validate_frame_counts("clip", 100, 90) == 90

    def test_mismatch_strict_raises(self):
        with pytest.raises(FrameMismatchError, match="clip"):
            validate_frame_counts("clip", 100, 90, strict=True)

    def test_zero_counts(self):
        assert validate_frame_counts("clip", 0, 0) == 0


class TestNormalizeMaskChannels:
    def test_2d_passthrough(self):
        mask = np.ones((4, 4), dtype=np.float32)
        result = normalize_mask_channels(mask)
        assert result.shape == (4, 4)

    def test_3d_extracts_first_channel(self):
        mask = np.zeros((4, 4, 3), dtype=np.float32)
        mask[:, :, 0] = 0.5
        mask[:, :, 1] = 1.0
        result = normalize_mask_channels(mask)
        assert result.shape == (4, 4)
        assert np.allclose(result, 0.5)

    def test_zero_channels_raises(self):
        mask = np.zeros((4, 4, 0), dtype=np.float32)
        with pytest.raises(MaskChannelError):
            normalize_mask_channels(mask, "clip", 0)

    def test_4d_raises(self):
        mask = np.zeros((4, 4, 3, 1), dtype=np.float32)
        with pytest.raises(MaskChannelError):
            normalize_mask_channels(mask, "clip", 0)

    def test_output_dtype_is_float32(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        result = normalize_mask_channels(mask)
        assert result.dtype == np.float32


class TestNormalizeMaskDtype:
    def test_uint8_normalized(self):
        mask = np.array([[255, 0]], dtype=np.uint8)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        assert np.isclose(result[0, 0], 1.0)
        assert np.isclose(result[0, 1], 0.0)

    def test_uint16_normalized(self):
        mask = np.array([[65535]], dtype=np.uint16)
        result = normalize_mask_dtype(mask)
        assert np.isclose(result[0, 0], 1.0, atol=1e-5)

    def test_float32_passthrough(self):
        mask = np.array([[0.5]], dtype=np.float32)
        result = normalize_mask_dtype(mask)
        assert result is mask  # same object, no copy

    def test_float64_cast(self):
        mask = np.array([[0.75]], dtype=np.float64)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        assert np.isclose(result[0, 0], 0.75)


class TestValidateFrameRead:
    def test_valid_frame_returned(self):
        frame = np.zeros((4, 4, 3), dtype=np.float32)
        result = validate_frame_read(frame, "clip", 0, "/path")
        assert result is frame

    def test_none_raises(self):
        with pytest.raises(FrameReadError, match="clip"):
            validate_frame_read(None, "clip", 5, "/path/frame.exr")


class TestValidateWrite:
    def test_success_no_raise(self):
        validate_write(True, "clip", 0, "/path")

    def test_failure_raises(self):
        with pytest.raises(WriteFailureError, match="clip"):
            validate_write(False, "clip", 3, "/path/frame.exr")


class TestEnsureOutputDirs:
    def test_creates_all_subdirs(self, tmp_path: Path):
        dirs = ensure_output_dirs(str(tmp_path / "clip"))
        for key in ("root", "fg", "matte", "comp", "processed"):
            assert key in dirs
            assert Path(dirs[key]).is_dir()

    def test_idempotent(self, tmp_path: Path):
        clip_root = str(tmp_path / "clip")
        ensure_output_dirs(clip_root)
        ensure_output_dirs(clip_root)  # should not raise

    def test_root_is_output_subdir(self, tmp_path: Path):
        dirs = ensure_output_dirs(str(tmp_path / "clip"))
        assert dirs["root"].endswith("Output")
