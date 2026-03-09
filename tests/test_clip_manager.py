"""Tests for clip_manager.py utility functions and ClipEntry discovery.

These tests verify the non-interactive parts of clip_manager: file type
detection, Windows→Linux path mapping, and the ClipEntry asset discovery
that scans directory trees to find Input/AlphaHint pairs.

No GPU, model weights, or interactive input required.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from clip_manager import (
    ClipAsset,
    ClipEntry,
    is_image_file,
    is_video_file,
    map_path,
    organize_target,
)

# ---------------------------------------------------------------------------
# is_image_file / is_video_file
# ---------------------------------------------------------------------------


class TestFileTypeDetection:
    """Verify extension-based file type helpers.

    These are used everywhere in clip_manager to decide how to read inputs.
    A missed extension means a valid frame silently disappears from the batch.
    """

    @pytest.mark.parametrize(
        "filename",
        [
            "frame.png",
            "SHOT_001.EXR",
            "plate.jpg",
            "ref.JPEG",
            "scan.tif",
            "deep.tiff",
            "comp.bmp",
        ],
    )
    def test_image_extensions_recognized(self, filename):
        assert is_image_file(filename)

    @pytest.mark.parametrize(
        "filename",
        [
            "frame.mp4",
            "CLIP.MOV",
            "take.avi",
            "rushes.mkv",
        ],
    )
    def test_video_extensions_recognized(self, filename):
        assert is_video_file(filename)

    @pytest.mark.parametrize(
        "filename",
        [
            "readme.txt",
            "notes.pdf",
            "project.nk",
            "scene.blend",
            ".DS_Store",
        ],
    )
    def test_non_media_rejected(self, filename):
        assert not is_image_file(filename)
        assert not is_video_file(filename)

    def test_image_is_not_video(self):
        """Image and video extensions must not overlap."""
        assert not is_video_file("frame.png")
        assert not is_video_file("plate.exr")

    def test_video_is_not_image(self):
        assert not is_image_file("clip.mp4")
        assert not is_image_file("rushes.mov")


# ---------------------------------------------------------------------------
# map_path
# ---------------------------------------------------------------------------


class TestMapPath:
    r"""Windows→Linux path mapping.

    The tool is designed for studios running a Linux render farm with
    Windows workstations.  V:\ maps to /mnt/ssd-storage.
    """

    def test_basic_mapping(self):
        result = map_path(r"V:\Projects\Shot1")
        assert result == "/mnt/ssd-storage/Projects/Shot1"

    def test_case_insensitive_drive_letter(self):
        result = map_path(r"v:\projects\shot1")
        assert result == "/mnt/ssd-storage/projects/shot1"

    def test_trailing_whitespace_stripped(self):
        result = map_path(r"  V:\Projects\Shot1  ")
        assert result == "/mnt/ssd-storage/Projects/Shot1"

    def test_backslashes_converted(self):
        result = map_path(r"V:\Deep\Nested\Path\Here")
        assert "\\" not in result

    def test_non_v_drive_passthrough(self):
        """Paths not on V: are normalized and returned unchanged."""
        linux_path = "/mnt/other/data"
        assert map_path(linux_path) == linux_path

    def test_linux_path_is_normalized(self):
        assert map_path("/mnt/other/../other/data") == "/mnt/other/data"

    def test_home_path_is_expanded(self):
        home = os.path.expanduser("~")
        assert map_path("~/clips") == os.path.join(home, "clips")

    def test_drive_root_only(self):
        result = map_path("V:\\")
        assert result == "/mnt/ssd-storage/"


# ---------------------------------------------------------------------------
# ClipAsset
# ---------------------------------------------------------------------------


class TestClipAsset:
    """ClipAsset wraps a directory of images or a video file and counts frames."""

    def test_sequence_frame_count(self, tmp_path):
        """Image sequence: frame count = number of image files in directory."""
        seq_dir = tmp_path / "Input"
        seq_dir.mkdir()
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(5):
            cv2.imwrite(str(seq_dir / f"frame_{i:04d}.png"), tiny)

        asset = ClipAsset(str(seq_dir), "sequence")
        assert asset.frame_count == 5

    def test_sequence_ignores_non_image_files(self, tmp_path):
        """Non-image files (thumbs.db, .nk, etc.) should not be counted."""
        seq_dir = tmp_path / "Input"
        seq_dir.mkdir()
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.imwrite(str(seq_dir / "frame_0000.png"), tiny)
        (seq_dir / "thumbs.db").write_text("junk")
        (seq_dir / "notes.txt").write_text("notes")

        asset = ClipAsset(str(seq_dir), "sequence")
        assert asset.frame_count == 1

    def test_empty_sequence(self, tmp_path):
        """Empty directory → 0 frames."""
        seq_dir = tmp_path / "Input"
        seq_dir.mkdir()
        asset = ClipAsset(str(seq_dir), "sequence")
        assert asset.frame_count == 0


# ---------------------------------------------------------------------------
# ClipEntry.find_assets
# ---------------------------------------------------------------------------


class TestClipEntryFindAssets:
    """ClipEntry.find_assets() discovers Input and AlphaHint from a shot directory.

    This is the core discovery logic that decides what's ready for inference
    vs. what still needs alpha generation.
    """

    def test_finds_image_sequence_input(self, tmp_clip_dir):
        """shot_a has Input/ with 2 PNGs → input_asset is a sequence."""
        entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
        entry.find_assets()
        assert entry.input_asset is not None
        assert entry.input_asset.type == "sequence"
        assert entry.input_asset.frame_count == 2

    def test_finds_alpha_hint(self, tmp_clip_dir):
        """shot_a has AlphaHint/ with 2 PNGs → alpha_asset is populated."""
        entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
        entry.find_assets()
        assert entry.alpha_asset is not None
        assert entry.alpha_asset.type == "sequence"
        assert entry.alpha_asset.frame_count == 2

    def test_empty_alpha_hint_is_none(self, tmp_clip_dir):
        """shot_b has empty AlphaHint/ → alpha_asset is None (needs generation)."""
        entry = ClipEntry("shot_b", str(tmp_clip_dir / "shot_b"))
        entry.find_assets()
        assert entry.input_asset is not None
        assert entry.alpha_asset is None

    def test_missing_input_raises(self, tmp_path):
        """A shot with no Input directory or video raises ValueError."""
        empty_shot = tmp_path / "empty_shot"
        empty_shot.mkdir()
        entry = ClipEntry("empty_shot", str(empty_shot))
        with pytest.raises(ValueError, match="No 'Input' directory or video file found"):
            entry.find_assets()

    def test_empty_input_dir_raises(self, tmp_path):
        """An empty Input/ directory raises ValueError."""
        shot = tmp_path / "bad_shot"
        (shot / "Input").mkdir(parents=True)
        entry = ClipEntry("bad_shot", str(shot))
        with pytest.raises(ValueError, match="'Input' directory is empty"):
            entry.find_assets()

    def test_validate_pair_frame_count_mismatch(self, tmp_path):
        """Mismatched Input/AlphaHint frame counts raise ValueError."""
        shot = tmp_path / "mismatch"
        (shot / "Input").mkdir(parents=True)
        (shot / "AlphaHint").mkdir(parents=True)

        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        tiny_mask = np.zeros((4, 4), dtype=np.uint8)

        # 3 input frames, 2 alpha frames
        for i in range(3):
            cv2.imwrite(str(shot / "Input" / f"frame_{i:04d}.png"), tiny)
        for i in range(2):
            cv2.imwrite(str(shot / "AlphaHint" / f"frame_{i:04d}.png"), tiny_mask)

        entry = ClipEntry("mismatch", str(shot))
        entry.find_assets()
        with pytest.raises(ValueError, match="Frame count mismatch"):
            entry.validate_pair()

    def test_validate_pair_matching_counts_ok(self, tmp_clip_dir):
        """Matching frame counts pass validation without error."""
        entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
        entry.find_assets()
        entry.validate_pair()  # should not raise


# ---------------------------------------------------------------------------
# organize_target
# ---------------------------------------------------------------------------


class TestOrganizeTarget:
    """organize_target() sets up the hint directory structure for a shot.

    It creates AlphaHint/ and VideoMamaMaskHint/ directories if missing.
    """

    def test_creates_hint_directories(self, tmp_path):
        """Missing hint directories should be created."""
        shot = tmp_path / "shot_x"
        (shot / "Input").mkdir(parents=True)
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.imwrite(str(shot / "Input" / "frame_0000.png"), tiny)

        organize_target(str(shot))

        assert (shot / "AlphaHint").is_dir()
        assert (shot / "VideoMamaMaskHint").is_dir()

    def test_existing_hint_dirs_preserved(self, tmp_clip_dir):
        """Existing hint directories and their contents are not disturbed."""
        shot_a = tmp_clip_dir / "shot_a"
        alpha_files_before = sorted(os.listdir(shot_a / "AlphaHint"))

        organize_target(str(shot_a))

        alpha_files_after = sorted(os.listdir(shot_a / "AlphaHint"))
        assert alpha_files_before == alpha_files_after

    def test_moves_loose_images_to_input(self, tmp_path):
        """Loose image files in a shot dir get moved into Input/."""
        shot = tmp_path / "messy_shot"
        shot.mkdir()
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.imwrite(str(shot / "frame_0000.png"), tiny)
        cv2.imwrite(str(shot / "frame_0001.png"), tiny)

        organize_target(str(shot))

        assert (shot / "Input").is_dir()
        input_files = os.listdir(shot / "Input")
        assert len(input_files) == 2
        # Original loose files should be gone
        assert not (shot / "frame_0000.png").exists()
