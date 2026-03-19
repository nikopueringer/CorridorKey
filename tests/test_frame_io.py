"""Tests for backend.frame_io — frame reading utilities.

Focuses on input validation and edge cases that don't require real video
files or model weights.
"""

from __future__ import annotations

from backend.frame_io import read_video_frame_at, read_video_mask_at


class TestReadVideoFrameAtNegativeIndex:
    """read_video_frame_at must return None for negative frame indices."""

    def test_negative_one_returns_none(self, tmp_path):
        """frame_index=-1 must return None without raising."""
        # A valid-looking path is enough — the guard fires before VideoCapture
        result = read_video_frame_at(str(tmp_path / "fake.mp4"), frame_index=-1)
        assert result is None

    def test_large_negative_returns_none(self, tmp_path):
        """Large negative values must also return None."""
        result = read_video_frame_at(str(tmp_path / "fake.mp4"), frame_index=-999)
        assert result is None

    def test_zero_does_not_trigger_guard(self, tmp_path):
        """frame_index=0 is valid and must not be caught by the negative guard.

        The file doesn't exist so cap.read() fails, returning None via the
        existing 'ret' check — not the new guard. We just confirm no TypeError
        or unexpected exception is raised.
        """
        # Should return None (file not found path), not raise
        result = read_video_frame_at(str(tmp_path / "fake.mp4"), frame_index=0)
        assert result is None


class TestReadVideoMaskAtNegativeIndex:
    """read_video_mask_at must return None for negative frame indices."""

    def test_negative_one_returns_none(self, tmp_path):
        """frame_index=-1 must return None without raising."""
        result = read_video_mask_at(str(tmp_path / "fake.mp4"), frame_index=-1)
        assert result is None

    def test_large_negative_returns_none(self, tmp_path):
        """Large negative values must also return None."""
        result = read_video_mask_at(str(tmp_path / "fake.mp4"), frame_index=-999)
        assert result is None

    def test_zero_does_not_trigger_guard(self, tmp_path):
        """frame_index=0 is valid and must not be caught by the negative guard."""
        result = read_video_mask_at(str(tmp_path / "fake.mp4"), frame_index=0)
        assert result is None
