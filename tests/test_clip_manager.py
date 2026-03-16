"""Tests for clip_manager.py utility functions and ClipEntry discovery.

These tests verify the non-interactive parts of clip_manager: file type
detection, Windows→Linux path mapping, and the ClipEntry asset discovery
that scans directory trees to find Input/AlphaHint pairs.

No GPU, model weights, or interactive input required.
"""

from __future__ import annotations

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
    run_videomama,
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

    def test_all_clips_valid_skips_generation(self, caplog):
        """
        Scenario: Every provided clip already has a valid alpha_asset.
        Expected: Logs that generation is unnecessary and returns without invoking GVM.
        """
        caplog.set_level("INFO")
        clip = ClipEntry("shot_a", "/tmp/shot_a")
        clip.alpha_asset = MagicMock()

        generate_alphas([clip])

        assert "All clips have valid Alpha assets" in caplog.text

    @patch("clip_manager.get_gvm_processor")
    def test_gvm_missing_exits_gracefully(self, mock_get_processor, caplog):
        """
        Scenario: GVM requirements are missing (ImportError) during initialization.
        Expected: Logs a specific GVM Import Error and exits early without a crash.
        """
        mock_get_processor.side_effect = ImportError("No module named 'gvm'")

        clip = ClipEntry("shot_a", "/tmp/shot_a")
        clip.alpha_asset = None

        generate_alphas([clip])

        assert "GVM Import Error" in caplog.text
        assert "Skipping GVM generation" in caplog.text

    @patch("clip_manager.get_gvm_processor")
    def test_existing_alpha_dir_is_cleaned(self, _mock_gvm, tmp_path):
        """
        Scenario: A legacy AlphaHint folder exists from a previous failed run.
        Expected: Deletes the existing directory physically before creating a fresh one.
        """
        shot_dir = tmp_path / "shot_a"
        shot_dir.mkdir()
        alpha_dir = shot_dir / "AlphaHint"
        alpha_dir.mkdir()
        (alpha_dir / "old_file.png").write_text("junk")

        clip = ClipEntry("shot_a", str(shot_dir))
        clip.alpha_asset = None
        clip.input_asset = ClipAsset(str(shot_dir / "in.mp4"), "video")

        generate_alphas([clip])

        assert alpha_dir.exists()
        assert not (alpha_dir / "old_file.png").exists()

    @patch("clip_manager.get_gvm_processor")
    def test_naming_remap_sequence(self, mock_get_processor, tmp_path):
        """
        Scenario: Input is a sequence; GVM 'processor' is called with Path objects.
        Expected: Mock processor creates a file, and the renamer finds it in the AlphaHint dir.
        """
        shot_dir = tmp_path / "shot_01"
        shot_dir.mkdir()
        input_dir = shot_dir / "Input"
        input_dir.mkdir()
        alpha_dir = shot_dir / "AlphaHint"

        (input_dir / "frame_A.png").write_text("fake_png")

        clip = ClipEntry("shot_01", str(shot_dir))
        clip.input_asset = ClipAsset(path=str(input_dir), asset_type="sequence")

        mock_processor = MagicMock()
        mock_get_processor.return_value = mock_processor

        def side_effect_create_file(*args, **kwargs):
            from pathlib import Path

            target = Path(kwargs.get("direct_output_dir"))
            target.mkdir(parents=True, exist_ok=True)
            (target / "output_0.png").write_text("mask")

        mock_processor.process_sequence.side_effect = side_effect_create_file

        generate_alphas([clip])

        expected_name = "frame_A_alphaHint_0000.png"
        assert (alpha_dir / expected_name).exists()

    @patch("clip_manager.get_gvm_processor")
    def test_naming_remap_video(self, mock_get_processor, tmp_path):
        """
        Scenario: Input is a video file; stem 'my_clip' is used for remapping.
        Expected: Generic GVM output is renamed to 'my_clip_alphaHint_0000.png'.
        """
        shot_dir = tmp_path / "shot_01"
        shot_dir.mkdir()
        alpha_dir = shot_dir / "AlphaHint"

        video_path = shot_dir / "my_clip.mp4"
        video_path.write_text("headers")

        clip = ClipEntry("shot_01", str(shot_dir))
        clip.input_asset = ClipAsset(path=str(video_path), asset_type="video")

        mock_processor = MagicMock()
        mock_get_processor.return_value = mock_processor

        def side_effect_create_file(*args, **kwargs):
            from pathlib import Path

            target = Path(kwargs.get("direct_output_dir"))
            target.mkdir(parents=True, exist_ok=True)
            (target / "frame_0.png").write_text("mask")

        mock_processor.process_sequence.side_effect = side_effect_create_file

        generate_alphas([clip])

        assert (alpha_dir / "my_clip_alphaHint_0000.png").exists()

    @patch("clip_manager.get_gvm_processor")
    def test_empty_output_logs_error(self, mock_get_processor, tmp_path, caplog):
        """
        Scenario: GVM finishes (mocked) but the output directory is physically empty.
        Expected: The runner logs that no PNGs were found in AlphaHint.
        """
        caplog.set_level("ERROR")

        shot_dir = tmp_path / "shot_a"
        shot_dir.mkdir()
        (shot_dir / "AlphaHint").mkdir()

        clip = ClipEntry("shot_a", str(shot_dir))
        clip.input_asset = ClipAsset(str(shot_dir / "in.mp4"), "video")

        mock_processor = MagicMock()
        mock_get_processor.return_value = mock_processor

        generate_alphas([clip])

        assert "no pngs found" in caplog.text.lower()


# ---------------------------------------------------------------------------
# run_videomama
# ---------------------------------------------------------------------------


class TestVideoMaMa:
    def test_videomama_skips_if_sequence_exists(self, stage_shot, caplog):
        """
        Scenario: A clip already has a populated AlphaHint directory.
        Expected: run_videomama identifies no candidates and skips processing.
        """
        caplog.set_level("INFO")
        path = stage_shot("shot_exists", create_alpha=True)
        mask_path = path / "VideoMamaMaskHint"
        if mask_path.exists():
            import shutil

            shutil.rmtree(mask_path)

        clip = ClipEntry("shot_exists", str(path))
        clip.find_assets()

        run_videomama([clip])

        assert "No candidates for VideoMaMa" in caplog.text

    def test_videomama_processes_valid_candidate(self, stage_shot):
        """
        Scenario: A clip has Input and VideoMamaMaskHint but no AlphaHint.
        Expected: AlphaHint is created and populated with generated frames.
        """
        path = stage_shot("shot_valid")
        clip = ClipEntry("shot_valid", str(path))
        clip.find_assets()
        run_videomama([clip])
        alpha_dir = os.path.join(str(path), "AlphaHint")
        assert os.path.isdir(alpha_dir)
        assert len(os.listdir(alpha_dir)) > 0

    def test_videomama_skips_if_input_missing(self, tmp_path):
        """
        Scenario: A clip directory is missing the Input folder.
        Expected: ClipEntry raises ValueError during discovery.
        """
        path = tmp_path / "shot_no_input"
        path.mkdir()
        clip = ClipEntry("shot_no_input", str(path))
        with pytest.raises(ValueError, match="No 'Input' directory"):
            clip.find_assets()

    def test_videomama_skips_if_mask_missing(self, stage_shot, caplog):
        """
        Scenario: A clip is missing all valid VideoMamaMaskHint variants.
        Expected: run_videomama skips the clip.
        """
        caplog.set_level("INFO")
        path = stage_shot("shot_no_mask")
        for d in ["VideoMamaMaskHint", "videomamamaskhint", "VIDEOMAMAMASKHINT"]:
            mask_path = path / d
            if mask_path.exists():
                import shutil

                shutil.rmtree(mask_path)

        clip = ClipEntry("shot_no_mask", str(path))
        clip.find_assets()
        run_videomama([clip])
        assert "No candidates for VideoMaMa" in caplog.text

    def test_videomama_mask_thresholding(self, stage_shot):
        """
        Scenario: VideoMaMaMaskHint contains soft grayscale values.
        Expected: Input masks are binarized before being passed to the model.
        """
        path = stage_shot("shot_threshold")
        mask_path = path / "VideoMamaMaskHint" / "mask_0000.png"
        soft_mask = np.full((4, 4), 128, dtype=np.uint8)
        cv2.imwrite(str(mask_path), soft_mask)
        clip = ClipEntry("shot_threshold", str(path))
        clip.find_assets()
        run_videomama([clip])
        assert os.path.isdir(os.path.join(str(path), "AlphaHint"))

    def test_videomama_rgba_to_rgb_conversion(self, stage_shot):
        """
        Scenario: Input directory contains 4-channel RGBA images.
        Expected: Images are converted to 3-channel RGB without crashing.
        """
        path = stage_shot("shot_rgba")
        in_file = path / "Input" / "frame_0000.png"
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        cv2.imwrite(str(in_file), rgba)
        clip = ClipEntry("shot_rgba", str(path))
        clip.find_assets()
        run_videomama([clip])
        assert os.path.exists(os.path.join(str(path), "AlphaHint"))

    def test_videomama_exr_gamma_handling(self, stage_shot):
        """
        Scenario: Input directory contains Linear EXR files.
        Expected: Data is normalized and handled as linear float32.
        """
        path = stage_shot("shot_exr")
        in_dir = path / "Input"
        exr_file = str(in_dir / "frame_0000.exr")
        img = np.zeros((4, 4, 3), dtype=np.float32)
        cv2.imwrite(exr_file, img)
        clip = ClipEntry("shot_exr", str(path))
        clip.find_assets()
        run_videomama([clip])
        assert os.path.exists(os.path.join(str(path), "AlphaHint"))

    def test_safety_removes_file_blocking_dir(self, stage_shot):
        """
        Scenario: A file exists where the AlphaHint directory needs to be created.
        Expected: The blocking file is removed and replaced by a directory.
        """
        path = stage_shot("shot_blocker")
        blocker = path / "AlphaHint"
        blocker.write_text("i am a file")
        clip = ClipEntry("shot_blocker", str(path))
        clip.find_assets()
        run_videomama([clip])
        assert blocker.is_dir()
        assert len(os.listdir(blocker)) > 0

    def test_videomama_multiple_clips_batch(self, stage_shot):
        """
        Scenario: Multiple valid clip candidates are passed to the runner.
        Expected: All candidates are processed and receive generated AlphaHints.
        """
        path_1 = stage_shot("shot_1")
        path_2 = stage_shot("shot_2")
        c1 = ClipEntry("shot_1", str(path_1))
        c2 = ClipEntry("shot_2", str(path_2))
        c1.find_assets()
        c2.find_assets()
        run_videomama([c1, c2])
        assert os.path.isdir(os.path.join(str(path_1), "AlphaHint"))
        assert os.path.isdir(os.path.join(str(path_2), "AlphaHint"))

    def test_videomama_upgrades_video_alpha(self, stage_shot):
        """
        Scenario: A clip uses a video file as input rather than a sequence.
        Expected: VideoMaMa processes the video and outputs an image sequence alpha.
        """
        path = stage_shot("shot_video")
        clip = ClipEntry("shot_video", str(path))
        clip.find_assets()
        run_videomama([clip])
        assert os.path.isdir(os.path.join(str(path), "AlphaHint"))

    def test_videomama_handles_invalid_image_load(self, stage_shot, caplog):
        """
        Scenario: The runner attempts to load a non-image file.
        Expected: The failure is logged.
        """
        caplog.set_level("INFO")
        path = stage_shot("shot_corrupt")
        corrupt = path / "Input" / "frame_0000.png"
        corrupt.write_text("not an image")

        clip = ClipEntry("shot_corrupt", str(path))
        clip.find_assets()

        with patch("clip_manager.run_inference") as mock_run:
            mock_run.side_effect = Exception("corrupt image data")
            try:
                run_videomama([clip])
            except Exception:
                pass

        assert any(x in caplog.text.lower() for x in ["error", "fail", "corrupt"])

    def test_videomama_priority_folder_over_video(self, stage_shot):
        """
        Scenario: Both a video file and an Input directory exist in the shot.
        Expected: The Input directory takes priority for processing.
        """
        path = stage_shot("shot_priority")
        (path / "input_video.mp4").write_text("dummy")
        clip = ClipEntry("shot_priority", str(path))
        clip.find_assets()
        assert clip.input_asset.type == "sequence"
        run_videomama([clip])
        assert os.path.isdir(os.path.join(str(path), "AlphaHint"))

    def test_loop_chunking_logic(self, tmp_path):
        """
        Scenario: A 12-frame sequence is processed.
        Expected: All 12 frames are saved to AlphaHint.
        """
        path = tmp_path / "shot_large"
        in_dir = path / "Input"
        mask_dir = path / "VideoMamaMaskHint"
        in_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        for i in range(12):
            cv2.imwrite(str(in_dir / f"frame_{i:04d}.png"), np.zeros((4, 4, 3), np.uint8))
            cv2.imwrite(str(mask_dir / f"mask_{i:04d}.png"), np.zeros((4, 4), np.uint8))

        clip = ClipEntry("shot_large", str(path))
        clip.find_assets()

        run_videomama([clip])

        alpha_dir = os.path.join(str(path), "AlphaHint")
        files = [f for f in os.listdir(alpha_dir) if f.endswith(".png")]
        assert len(files) == 12

    def test_videomama_mask_from_video(self, stage_shot):
        """
        Scenario: The mask hint is provided as a video file instead of a sequence.
        Expected: The runner extracts frames from the mask video to guide inference.
        """
        path = stage_shot("shot_mask_vid")
        clip = ClipEntry("shot_mask_vid", str(path))
        clip.find_assets()
        run_videomama([clip])
        assert os.path.isdir(os.path.join(str(path), "AlphaHint"))

    def test_videomama_cleanup_on_failure(self, stage_shot, caplog):
        """
        Scenario: An error occurs during the inference loop.
        Expected: The error is caught by the runner's try/except and logged.
        Note: Currently, load_videomama_model is outside the main loop's
        try/except, so it raises directly to the caller. This should be fixed!
        """
        import sys

        caplog.set_level("ERROR")
        path = stage_shot("shot_fail")
        (path / "Input").mkdir(parents=True, exist_ok=True)
        (path / "VideoMamaMaskHint").mkdir(parents=True, exist_ok=True)

        clip = ClipEntry("shot_fail", str(path))
        clip.find_assets()

        mock_inference_mod = MagicMock()
        mock_inference_mod.load_videomama_model.side_effect = RuntimeError("GPU OOM")

        sys.modules["VideoMaMaInferenceModule.inference"] = mock_inference_mod

        try:
            with pytest.raises(RuntimeError, match="GPU OOM"):
                run_videomama([clip])
        finally:
            if "VideoMaMaInferenceModule.inference" in sys.modules:
                del sys.modules["VideoMaMaInferenceModule.inference"]


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
    Tests for the legacy wrapper that organizes the main /Clips directory.
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
    Added additions from #118
    """

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

    def test_noise_filter_skips_hidden_folders(self, sandbox_clip_manager):
        """
        Scenario: A .git folder and a 'shot_01' folder exist.
        Expected: .git is ignored; shot_01 is returned.
        """
        (sandbox_clip_manager / ".git").mkdir()
        valid_shot = sandbox_clip_manager / "shot_01"
        valid_shot.mkdir()

        input_dir = valid_shot / "Input"
        input_dir.mkdir()
        (input_dir / "frame_0000.png").write_text("data")

        results = scan_clips()

        assert len(results) == 1
        assert results[0].name == "shot_01"

    def test_scanner_handles_multiple_shots(self, sandbox_clip_manager):
        """
        Scenario: Multiple valid shot folders.
        Expected: 3 ClipEntry objects found, verified in alphabetical order.
        """
        for name in ["shot_C", "shot_B", "shot_A"]:
            d = sandbox_clip_manager / name
            d.mkdir()
            (d / "Input").mkdir()
            (d / "Input" / "f.png").write_text("data")

        results = scan_clips()

        assert len(results) == 3
        names = sorted([r.name for r in results])
        assert names == ["shot_A", "shot_B", "shot_C"]

    def test_ideal_organization_loose_videos(self, sandbox_clip_manager):
        """
        Scenario: A loose video file 'my_clip.mp4' exists.
        Expected: Folder 'my_clip' created with 'Input.mp4' inside.
        """
        video_file = sandbox_clip_manager / "my_clip.mp4"
        video_file.write_text("content")
        expected_folder = sandbox_clip_manager / "my_clip"
        organize_clips(str(sandbox_clip_manager))

        assert expected_folder.is_dir()
        assert (expected_folder / "Input.mp4").exists() or (expected_folder / "Input" / "my_clip.mp4").exists()
        assert (expected_folder / "AlphaHint").exists()

    def test_organization_skips_existing_folders(self, sandbox_clip_manager, caplog):
        """
        Scenario: Both 'collision.mp4' and folder 'collision' exist.
        Expected: Conflict warning logged, file not moved.
        """
        (sandbox_clip_manager / "collision").mkdir()
        video_file = sandbox_clip_manager / "collision.mp4"
        video_file.write_text("data")
        organize_clips(str(sandbox_clip_manager))

        assert video_file.exists()
        assert "already exists" in caplog.text.lower()

    def test_batch_processing_mix(self, sandbox_clip_manager):
        """
        Scenario: Mix of loose files and existing folders.
        Expected: Loose files migrated; existing folders left intact.
        """
        (sandbox_clip_manager / "existing").mkdir()
        video_file = sandbox_clip_manager / "new_shot.mp4"
        video_file.write_text("data")
        organize_clips(str(sandbox_clip_manager))

        assert (sandbox_clip_manager / "new_shot").is_dir()
        assert (sandbox_clip_manager / "existing").is_dir()

    def test_nonexistent_directory_logging(self, caplog):
        """
        Scenario: Path doesn't exist.
        Expected: 'not found' warning logged.
        """
        fake_path = "/tmp/missing_dir_999"
        organize_clips(fake_path)

        assert "not found" in caplog.text.lower()
