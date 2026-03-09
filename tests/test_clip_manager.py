"""Tests for clip_manager.py utility functions and ClipEntry discovery.

These tests verify the non-interactive parts of clip_manager: file type
detection, Windows→Linux path mapping, and the ClipEntry asset discovery
that scans directory trees to find Input/AlphaHint pairs.

No GPU, model weights, or interactive input required.
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from clip_manager import (
    ClipAsset,
    ClipEntry,
    is_image_file,
    is_video_file,
    map_path,
    generate_alphas,
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
# generate_alphas
# ---------------------------------------------------------------------------

class TestGenerateAlphas:
    """
    Tests for the generate_alphas orchestrator.
    Focuses on GVM integration, directory cleanup, and filename remapping.
    """

    @patch("clip_manager.get_gvm_processor")
    def test_all_clips_valid_skips_generation(self, mock_get_processor, caplog):
        """
        Scenario: All provided clips already have an alpha_asset populated.
        Expected: Function logs that no generation is needed and returns without calling GVM.
        """
        caplog.set_level("INFO")
        clip = ClipEntry("shot_a", "/tmp/shot_a")
        clip.alpha_asset = MagicMock()

        generate_alphas([clip])
        
        assert "All clips have valid Alpha assets" in caplog.text
        mock_get_processor.assert_not_called()

    @patch("clip_manager.get_gvm_processor")
    def test_gvm_missing_exits_gracefully(self, mock_get_processor, caplog):
        """
        Scenario: GVM requirements are not installed (ImportError).
        Expected: Function logs the error and returns early without crashing.
        """
        mock_get_processor.side_effect = ImportError("No module named 'gvm'")

        clip = ClipEntry("shot_a", "/tmp/shot_a")
        clip.alpha_asset = None 

        generate_alphas([clip])
        
        assert "GVM Import Error" in caplog.text
        assert "Skipping GVM generation" in caplog.text

    @patch("clip_manager.shutil.rmtree")
    @patch("clip_manager.os.path.exists", return_value=True)
    @patch("clip_manager.get_gvm_processor")
    def test_existing_alpha_dir_is_cleaned(self, _, __, mock_rmtree):
        """
        Scenario: An old AlphaHint folder already exists.
        Expected: The existing folder is deleted (rmtree) before new generation starts.
        """
        clip = ClipEntry("shot_a", "/tmp/shot_a")
        clip.alpha_asset = None
        clip.input_asset = MagicMock()

        with patch("clip_manager.os.makedirs"):
            generate_alphas([clip])
        
        mock_rmtree.assert_called_once_with(os.path.join("/tmp/shot_a", "AlphaHint"))

    def test_naming_remap_sequence(self, tmp_path):
        """
        Scenario: Input is a sequence (frame_A.png). GVM outputs generic output_0.png.
        Expected: Output is renamed to frame_A_alphaHint_0000.png.
        """
        shot_dir = tmp_path / "shot_01"
        input_dir = shot_dir / "Input"
        alpha_dir = shot_dir / "AlphaHint"
        input_dir.mkdir(parents=True)
        alpha_dir.mkdir()

        (input_dir / "frame_A.png").write_text("data")
        (alpha_dir / "output_0.png").write_text("mask_data")

        clip = ClipEntry("shot_01", str(shot_dir))
        clip.input_asset = ClipAsset(path=str(input_dir), asset_type="sequence")
        clip.alpha_asset = None

        with (patch("clip_manager.get_gvm_processor"),
            patch("clip_manager.resolve_device"),
            patch("clip_manager.os.rename") as mock_rename,
            patch("clip_manager.shutil.rmtree"),
            patch("clip_manager.is_image_file", return_value=True)):
            
            generate_alphas([clip])

            expected_old = os.path.join(str(alpha_dir), "output_0.png")
            expected_new = os.path.join(str(alpha_dir), "frame_A_alphaHint_0000.png")
            
            mock_rename.assert_called_once_with(expected_old, expected_new)
    
    def test_naming_remap_video(self, tmp_path):
        """
        Scenario: Input is a single video file ('my_clip.mp4').
        Expected: Generated alpha frames use 'my_clip' as the stem for renaming.
        """
        shot_dir = tmp_path / "shot_01"
        alpha_dir = shot_dir / "AlphaHint"
        shot_dir.mkdir()
        alpha_dir.mkdir()

        video_path = str(shot_dir / "my_clip.mp4")
        clip = ClipEntry("shot_01", str(shot_dir))
        clip.input_asset = ClipAsset(path=video_path, asset_type="video")
        clip.alpha_asset = None

        with (patch("clip_manager.get_gvm_processor"),
            patch("clip_manager.resolve_device"),
            patch("clip_manager.os.rename") as mock_rename,
            patch("clip_manager.os.listdir", return_value=["gvm_output.png"])):
            generate_alphas([clip])

            expected_new = os.path.join(str(alpha_dir), "my_clip_alphaHint_0000.png")
            
            args, _ = mock_rename.call_args
            assert args[1] == expected_new

    @patch("clip_manager.get_gvm_processor")
    def test_empty_output_logs_error(self, _mock_get_processor, caplog):
        """
        Scenario: The processor completes but the AlphaHint directory is empty.
        Expected: Logs an error indicating no PNGs were found and continues.
        """
        clip = ClipEntry("shot_a", "/tmp/shot_a")
        clip.alpha_asset = None
        clip.input_asset = MagicMock()

        with (patch("clip_manager.os.path.exists", return_value=True),
            patch("clip_manager.shutil.rmtree"),
            patch("clip_manager.os.makedirs"),
            patch("clip_manager.os.listdir", return_value=[])):
            generate_alphas([clip])
            
            assert "no PNGs found" in caplog.text

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
# organize_clips
# ---------------------------------------------------------------------------

class TestOrganizeClips:
    """
    Legacy wrapper tests for organizing the main Clips directory.

    This handles moving loose video files into structured folders and then
    triggering organize_target on all subdirectories.
    """

    def test_organize_loose_video_file(self, tmp_path):
        """
        Tests that a loose .mp4 file is moved into its own folder.

        Scenario: A directory contains a loose video file like 'shot_001.mp4'.
        Expected: A new folder 'shot_001' is created, containing 'Input.mp4' and an empty 'AlphaHint' directory.
        """
        clips_dir = tmp_path / "ClipsForInference"
        clips_dir.mkdir()

        video_file = clips_dir / "shot_001.mp4"
        video_file.write_text("test_video_data")

        with patch("clip_manager.organize_target") as mock_target:
            organize_clips(str(clips_dir))

        target_folder = clips_dir / "shot_001"
        assert target_folder.is_dir(), f"Folder {target_folder} was not created!"
        assert (target_folder / "Input.mp4").exists()
        assert (target_folder / "AlphaHint").exists()

        mock_target.assert_called_with(str(target_folder))

    def test_skips_video_if_folder_exists(self, tmp_path, caplog):
        """
        Tests that a video is skipped if a folder with its name already exists.

        Scenario: Both 'shot_001.mp4' and a folder named 'shot_001' exist.
        Expected: The original file is left alone, and a conflict warning is logged.
        """
        clips_dir = tmp_path / "ClipsForInference"
        clips_dir.mkdir()

        video_path = clips_dir / "shot_001.mp4"
        video_path.write_text("data")

        conflict_folder = clips_dir / "shot_001"
        conflict_folder.mkdir()



        organize_clips(str(clips_dir))
        assert video_path.exists(), "The video was moved even though a folder existed!"
        assert "already exists" in caplog.text

    def test_ignores_protected_folders(self, tmp_path):
        """
        Tests that 'Output' and 'IgnoredClips' folders are not processed.

        Scenario: Directory contains a valid shot folder plus 'Output' and 'IgnoredClips'.
        Expected: 'organize_target' is called exactly once (only for the valid shot).
        """
        clips_dir = tmp_path / "ClipsForInference"
        clips_dir.mkdir()

        (clips_dir / "shot_001").mkdir()
        (clips_dir / "Output").mkdir()
        (clips_dir / "IgnoredClips").mkdir()

        with patch("clip_manager.organize_target") as mock_target:


            organize_clips(str(clips_dir))

        mock_target.assert_any_call(str(clips_dir / "shot_001"))

        assert mock_target.call_count == 1, f"Expected 1 call, but got {mock_target.call_count}"

    def test_handles_nonexistent_directory(self, caplog):
        """
        Tests that the function exits gracefully if the directory is missing.

        Scenario: The provided path does not exist on the filesystem.
        Expected: Function logs a 'directory not found' warning and returns early.
        """
        fake_path = "/tmp/ghost_directory_12345"

        organize_clips(fake_path)

        assert "directory not found" in caplog.text
        assert fake_path in caplog.text

    def test_batch_organization_mix(self, tmp_path):
        """
        Tests that the function handles a mix of loose videos and folders at once.

        Scenario: Directory contains one loose video and one already existing folder.
        Expected: The video is migrated, and 'organize_target' is called for both.
        """
        clips_dir = tmp_path / "ClipsForInference"
        clips_dir.mkdir()

        video_a = clips_dir / "shot_A.mp4"
        video_a.write_text("video_data")

        folder_b = clips_dir / "shot_B"
        folder_b.mkdir()

        with patch("clip_manager.organize_target") as mock_target:


            organize_clips(str(clips_dir))

        assert (clips_dir / "shot_A" / "Input.mp4").exists()

        mock_target.assert_any_call(str(clips_dir / "shot_A"))
        mock_target.assert_any_call(str(clips_dir / "shot_B"))
        assert mock_target.call_count == 2

# ---------------------------------------------------------------------------
# scan_clips
# ---------------------------------------------------------------------------

class TestScanClips:
    """
    Tests for the scan_clips file orchestrator.
    Ensures directory health, automatic organization, and validation reporting.
    """

    @patch("clip_manager.os.makedirs")
    @patch("clip_manager.os.path.exists", return_value=False)
    def test_empty_start_creates_dir(self, _mock_exists, mock_makedirs):
        """
        Scenario: The Clips directory is missing entirely.
        Expected: Function creates the directory and returns an empty list.
        """
        results = scan_clips()

        assert results == []
        mock_makedirs.assert_called_once()

    def test_noise_filter_skips_hidden_folders(self, tmp_path):
        """
        Scenario: Folder contains .git, _internal, and IgnoredClips.
        Expected: These are ignored and not processed as potential clips.
        """
        clips_dir = tmp_path / "Clips"
        clips_dir.mkdir()
        (clips_dir / ".git").mkdir()
        (clips_dir / "_cache").mkdir()
        (clips_dir / "IgnoredClips").mkdir()
        (clips_dir / "valid_shot").mkdir()

        with (patch("clip_manager.CLIPS_DIR", str(clips_dir)),
             patch("clip_manager.organize_clips"),
             patch("clip_manager.ClipEntry") as mock_entry):

            scan_clips()

            assert mock_entry.call_count == 1
            mock_entry.assert_called_with("valid_shot", str(clips_dir / "valid_shot"))

    def test_clean_run_returns_valid_entries(self, tmp_path):
        """
        Scenario: Multiple valid folders exist.
        Expected: Returns a list of ClipEntry objects that have been validated.
        """
        clips_dir = tmp_path / "Clips"
        clips_dir.mkdir()
        (clips_dir / "shot_001").mkdir()

        with (patch("clip_manager.CLIPS_DIR", str(clips_dir)),
             patch("clip_manager.organize_clips"),
             patch("clip_manager.ClipEntry") as mock_entry):

            instance = mock_entry.return_value

            results = scan_clips()

            assert len(results) == 1
            instance.find_assets.assert_called_once()
            instance.validate_pair.assert_called_once()

    def test_invalid_clip_reporting(self, tmp_path, capsys):
        """
        Scenario: A folder exists but ClipEntry raises a ValueError (e.g., missing video).
        Expected: The clip is added to invalid_clips and printed to the console.
        """
        clips_dir = tmp_path / "Clips"
        clips_dir.mkdir()
        (clips_dir / "broken_shot").mkdir()

        with (patch("clip_manager.CLIPS_DIR", str(clips_dir)),
             patch("clip_manager.organize_clips"),
             patch("clip_manager.ClipEntry") as mock_entry):

            mock_entry.return_value.find_assets.side_effect = ValueError("Missing Input.mp4")
            results = scan_clips()
            captured = capsys.readouterr()

            assert results == []
            assert "INVALID OR SKIPPED CLIPS" in captured.out
            assert "broken_shot: Missing Input.mp4" in captured.out

    def test_crash_recovery_continues_loop(self, tmp_path):
        """
        Scenario: One folder causes an unexpected Exception.
        Expected: The function catches the error for that folder and continues to the next.
        """
        clips_dir = tmp_path / "Clips"
        clips_dir.mkdir()
        (clips_dir / "crash_shot").mkdir()
        (clips_dir / "good_shot").mkdir()

        with (patch("clip_manager.CLIPS_DIR", str(clips_dir)),
             patch("clip_manager.organize_clips"),
             patch("clip_manager.ClipEntry") as mock_entry):

            mock_entry.side_effect = [Exception("Disk Error"), mock_entry.return_value]
            results = scan_clips()

            assert len(results) == 1
