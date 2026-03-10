"""Tests for clip_manager.py utility functions and ClipEntry discovery.

These tests verify the non-interactive parts of clip_manager: file type
detection, Windows→Linux path mapping, and the ClipEntry asset discovery
that scans directory trees to find Input/AlphaHint pairs.

No GPU, model weights, or interactive input required.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from clip_manager import (
    ClipAsset,
    ClipEntry,
    generate_alphas,
    is_image_file,
    is_video_file,
    map_path,
    organize_clips,
    organize_target,
    scan_clips,
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
        """Paths not on V: are returned as-is (may already be Linux paths)."""
        linux_path = "/mnt/other/data"
        assert map_path(linux_path) == linux_path

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


# ---------------------------------------------------------------------------
# generate_alphas
# ---------------------------------------------------------------------------


class TestGenerateAlphas:
    """generate_alphas() runs GVM on clips that are missing an AlphaHint."""

    def test_no_op_when_all_clips_have_alpha(self, tmp_clip_dir):
        """All clips already have alpha → GVM is never invoked."""
        entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
        entry.find_assets()
        assert entry.alpha_asset is not None

        with patch("clip_manager.get_gvm_processor") as mock_gvm:
            generate_alphas([entry])

        mock_gvm.assert_not_called()

    def test_calls_process_sequence_for_missing_alpha(self, tmp_clip_dir):
        """Clips with alpha_asset=None trigger a GVM process_sequence call."""
        entry = ClipEntry("shot_b", str(tmp_clip_dir / "shot_b"))
        entry.find_assets()
        assert entry.alpha_asset is None

        mock_proc = MagicMock()
        mock_proc.process_sequence.return_value = None

        with patch("clip_manager.get_gvm_processor", return_value=mock_proc):
            generate_alphas([entry])

        mock_proc.process_sequence.assert_called_once()

    def test_import_error_is_logged_not_raised(self, tmp_clip_dir, caplog):
        """ImportError from get_gvm_processor must not propagate — log only."""
        entry = ClipEntry("shot_b", str(tmp_clip_dir / "shot_b"))
        entry.find_assets()

        with patch("clip_manager.get_gvm_processor", side_effect=ImportError("GVM not installed")):
            with caplog.at_level(logging.ERROR, logger="clip_manager"):
                generate_alphas([entry])  # must not raise

        assert "GVM" in caplog.text

    def test_renames_output_to_match_input_stems(self, tmp_clip_dir):
        """Generated PNGs are renamed to {input_stem}_alphaHint_{i:04d}.png."""
        entry = ClipEntry("shot_b", str(tmp_clip_dir / "shot_b"))
        entry.find_assets()
        # shot_b has one input frame: frame_0000.png → expected stem: frame_0000

        def fake_process_sequence(*args, **kwargs):
            # Simulate GVM writing a mask PNG with an arbitrary name
            out_dir = kwargs["direct_output_dir"]
            open(os.path.join(out_dir, "gvm_0000.png"), "wb").close()

        mock_proc = MagicMock()
        mock_proc.process_sequence.side_effect = fake_process_sequence

        with patch("clip_manager.get_gvm_processor", return_value=mock_proc):
            generate_alphas([entry])

        alpha_dir = str(tmp_clip_dir / "shot_b" / "AlphaHint")
        files = sorted(os.listdir(alpha_dir))
        assert files == ["frame_0000_alphaHint_0000.png"]


# ---------------------------------------------------------------------------
# organize_clips
# ---------------------------------------------------------------------------


class TestOrganizeClips:
    """organize_clips() structures a batch directory of shots."""

    def test_missing_directory_returns_cleanly(self, tmp_path, caplog):
        """Non-existent clips_dir logs a warning and does not raise."""
        missing = str(tmp_path / "does_not_exist")
        with caplog.at_level(logging.WARNING, logger="clip_manager"):
            organize_clips(missing)
        assert "Clips directory" in caplog.text

    def test_loose_video_moved_to_named_subdir(self, tmp_path):
        """A loose .mp4 in the clips dir is moved to {name}/Input.mp4."""
        (tmp_path / "my_shot.mp4").write_bytes(b"\x00" * 16)

        organize_clips(str(tmp_path))

        assert (tmp_path / "my_shot").is_dir()
        assert (tmp_path / "my_shot" / "Input.mp4").is_file()
        assert not (tmp_path / "my_shot.mp4").exists()

    def test_hint_dirs_created_for_organized_video(self, tmp_path):
        """AlphaHint and VideoMamaMaskHint dirs are created alongside a moved video."""
        (tmp_path / "clip_a.mp4").write_bytes(b"\x00")

        organize_clips(str(tmp_path))

        assert (tmp_path / "clip_a" / "AlphaHint").is_dir()
        assert (tmp_path / "clip_a" / "VideoMamaMaskHint").is_dir()

    def test_ignored_clips_and_output_dirs_skipped(self, tmp_path):
        """IgnoredClips and Output subdirs are not passed to organize_target."""
        (tmp_path / "IgnoredClips").mkdir()
        (tmp_path / "Output").mkdir()

        with patch("clip_manager.organize_target") as mock_ot:
            organize_clips(str(tmp_path))
            called_paths = [call.args[0] for call in mock_ot.call_args_list]

        assert not any("IgnoredClips" in p for p in called_paths)
        assert not any("Output" in p for p in called_paths)

    def test_calls_organize_target_on_shot_subdirs(self, tmp_clip_dir):
        """Existing shot subdirs are each passed through organize_target."""
        with patch("clip_manager.organize_target") as mock_ot:
            organize_clips(str(tmp_clip_dir))
            called_paths = [call.args[0] for call in mock_ot.call_args_list]

        shot_paths = {str(tmp_clip_dir / "shot_a"), str(tmp_clip_dir / "shot_b")}
        assert shot_paths.issubset(set(called_paths))


# ---------------------------------------------------------------------------
# scan_clips
# ---------------------------------------------------------------------------


class TestScanClips:
    """scan_clips() discovers clip entries from CLIPS_DIR (monkeypatched)."""

    def test_creates_clips_dir_and_returns_empty_if_missing(self, tmp_path, monkeypatch):
        """A missing CLIPS_DIR is created automatically and [] is returned."""
        import clip_manager

        missing = str(tmp_path / "ClipsForInference")
        monkeypatch.setattr(clip_manager, "CLIPS_DIR", missing)

        result = scan_clips()

        assert result == []
        assert os.path.isdir(missing)

    def test_returns_clips_with_valid_input(self, tmp_clip_dir, monkeypatch):
        """Clips whose Input directories exist are included in the result."""
        import clip_manager

        monkeypatch.setattr(clip_manager, "CLIPS_DIR", str(tmp_clip_dir))
        result = scan_clips()
        names = {c.name for c in result}

        assert "shot_a" in names
        assert "shot_b" in names  # valid input even without alpha

    def test_excludes_frame_count_mismatch(self, tmp_clip_dir, monkeypatch):
        """A clip with mismatched Input/AlphaHint frame counts is excluded."""
        import clip_manager

        mismatch = tmp_clip_dir / "mismatch_shot"
        (mismatch / "Input").mkdir(parents=True)
        (mismatch / "AlphaHint").mkdir()
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        tiny_mask = np.zeros((4, 4), dtype=np.uint8)
        for i in range(3):
            cv2.imwrite(str(mismatch / "Input" / f"frame_{i:04d}.png"), tiny)
        cv2.imwrite(str(mismatch / "AlphaHint" / "frame_0000.png"), tiny_mask)

        monkeypatch.setattr(clip_manager, "CLIPS_DIR", str(tmp_clip_dir))
        result = scan_clips()
        names = {c.name for c in result}

        assert "mismatch_shot" not in names
        assert "shot_a" in names  # valid shot still found

    def test_skips_hidden_and_underscore_dirs(self, tmp_clip_dir, monkeypatch):
        """Directories starting with '.' or '_' are never returned."""
        import clip_manager

        (tmp_clip_dir / ".hidden").mkdir()
        (tmp_clip_dir / "_temp").mkdir()
        monkeypatch.setattr(clip_manager, "CLIPS_DIR", str(tmp_clip_dir))

        result = scan_clips()
        names = {c.name for c in result}

        assert ".hidden" not in names
        assert "_temp" not in names
