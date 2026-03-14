"""Unit tests for errors.py."""

from __future__ import annotations

from unittest.mock import patch

from corridorkey.errors import (
    ClipScanError,
    CorridorKeyError,
    ExtractionError,
    FFmpegNotFoundError,
    FrameMismatchError,
    FrameReadError,
    InvalidStateTransitionError,
    JobCancelledError,
    MaskChannelError,
    VRAMInsufficientError,
    WriteFailureError,
)


class TestErrorHierarchy:
    def test_all_inherit_from_base(self):
        errors = [
            ClipScanError("x"),
            FrameMismatchError("clip", 10, 5),
            FrameReadError("clip", 0, "/path"),
            WriteFailureError("clip", 0, "/path"),
            MaskChannelError("clip", 0, 2),
            VRAMInsufficientError(10.0, 4.0),
            InvalidStateTransitionError("clip", "RAW", "COMPLETE"),
            JobCancelledError("clip"),
            ExtractionError("clip", "detail"),
        ]
        for err in errors:
            assert isinstance(err, CorridorKeyError)
            assert isinstance(err, Exception)


class TestFrameMismatchError:
    def test_attributes(self):
        err = FrameMismatchError("shot1", 100, 90)
        assert err.clip_name == "shot1"
        assert err.input_count == 100
        assert err.alpha_count == 90

    def test_message_contains_counts(self):
        err = FrameMismatchError("shot1", 100, 90)
        assert "100" in str(err)
        assert "90" in str(err)
        assert "shot1" in str(err)


class TestFrameReadError:
    def test_attributes(self):
        err = FrameReadError("shot1", 42, "/frames/0042.exr")
        assert err.clip_name == "shot1"
        assert err.frame_index == 42
        assert err.path == "/frames/0042.exr"

    def test_message(self):
        err = FrameReadError("shot1", 42, "/frames/0042.exr")
        assert "42" in str(err)
        assert "shot1" in str(err)


class TestJobCancelledError:
    def test_without_frame_index(self):
        err = JobCancelledError("shot1")
        assert err.frame_index is None
        assert "shot1" in str(err)
        assert "frame" not in str(err)

    def test_with_frame_index(self):
        err = JobCancelledError("shot1", 15)
        assert err.frame_index == 15
        assert "15" in str(err)


class TestInvalidStateTransitionError:
    def test_attributes(self):
        err = InvalidStateTransitionError("shot1", "RAW", "COMPLETE")
        assert err.clip_name == "shot1"
        assert err.current_state == "RAW"
        assert err.target_state == "COMPLETE"

    def test_message(self):
        err = InvalidStateTransitionError("shot1", "RAW", "COMPLETE")
        assert "RAW" in str(err)
        assert "COMPLETE" in str(err)


class TestVRAMInsufficientError:
    def test_attributes(self):
        err = VRAMInsufficientError(22.7, 8.0)
        assert err.required_gb == 22.7
        assert err.available_gb == 8.0

    def test_message_formatted(self):
        err = VRAMInsufficientError(22.7, 8.0)
        assert "22.7" in str(err)
        assert "8.0" in str(err)


class TestFFmpegNotFoundError:
    def test_darwin_hint(self):
        with patch("corridorkey.errors.sys") as mock_sys:
            mock_sys.platform = "darwin"
            err = FFmpegNotFoundError("ffmpeg")
        assert "brew" in str(err)

    def test_linux_hint(self):
        with patch("corridorkey.errors.sys") as mock_sys:
            mock_sys.platform = "linux"
            err = FFmpegNotFoundError("ffprobe")
        assert "apt" in str(err)
        assert "ffprobe" in str(err)

    def test_windows_hint(self):
        with patch("corridorkey.errors.sys") as mock_sys:
            mock_sys.platform = "win32"
            err = FFmpegNotFoundError("ffmpeg")
        assert "choco" in str(err)

    def test_default_binary_is_ffmpeg(self):
        err = FFmpegNotFoundError()
        assert "ffmpeg" in str(err)
