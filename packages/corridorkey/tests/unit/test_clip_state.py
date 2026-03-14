"""Unit tests for clip_state.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.clip_state import ClipAsset, ClipEntry, ClipState, scan_clips_dir, scan_project_clips
from corridorkey.errors import ClipScanError, InvalidStateTransitionError


def _make_sequence(path: Path, count: int = 5) -> None:
    """Write count PNG stubs into path."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (path / f"frame_{i:06d}.png").touch()


def _make_clip_dir(root: Path, with_alpha: bool = False, with_mask: bool = False) -> Path:
    """Create a minimal clip directory structure."""
    frames = root / "Frames"
    _make_sequence(frames)
    if with_alpha:
        _make_sequence(root / "AlphaHint")
    if with_mask:
        _make_sequence(root / "VideoMamaMaskHint")
    return root


class TestClipState:
    def test_all_members_present(self):
        states = {s.value for s in ClipState}
        assert states == {"EXTRACTING", "RAW", "MASKED", "READY", "COMPLETE", "ERROR"}


class TestClipEntry:
    def test_transition_raw_to_ready(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.RAW)
        clip.transition_to(ClipState.READY)
        assert clip.state == ClipState.READY

    def test_transition_raw_to_complete_raises(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.RAW)
        with pytest.raises(InvalidStateTransitionError):
            clip.transition_to(ClipState.COMPLETE)

    def test_set_error_sets_message(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.RAW)
        clip.set_error("something broke")
        assert clip.state == ClipState.ERROR
        assert clip.error_message == "something broke"

    def test_transition_clears_error_message(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.ERROR)
        clip.error_message = "old error"
        clip.transition_to(ClipState.RAW)
        assert clip.error_message is None

    def test_output_dir_property(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path))
        assert clip.output_dir == str(tmp_path / "Output")

    def test_has_outputs_false_when_empty(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path))
        assert not clip.has_outputs

    def test_has_outputs_true_when_fg_exists(self, tmp_path: Path):
        fg_dir = tmp_path / "Output" / "FG"
        fg_dir.mkdir(parents=True)
        (fg_dir / "frame_000000.exr").touch()
        clip = ClipEntry(name="shot1", root_path=str(tmp_path))
        assert clip.has_outputs

    def test_processing_lock(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path))
        assert not clip.is_processing
        clip.set_processing(True)
        assert clip.is_processing
        clip.set_processing(False)
        assert not clip.is_processing

    def test_error_to_raw_transition_allowed(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.ERROR)
        clip.transition_to(ClipState.RAW)
        assert clip.state == ClipState.RAW

    def test_complete_to_ready_allowed(self, tmp_path: Path):
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.COMPLETE)
        clip.transition_to(ClipState.READY)
        assert clip.state == ClipState.READY


class TestClipAsset:
    def test_sequence_frame_count(self, tmp_path: Path):
        seq = tmp_path / "Frames"
        _make_sequence(seq, count=10)
        asset = ClipAsset(str(seq), "sequence")
        assert asset.frame_count == 10

    def test_sequence_get_frame_files_naturally_sorted(self, tmp_path: Path):
        seq = tmp_path / "Frames"
        seq.mkdir()
        for name in ["frame_10.png", "frame_2.png", "frame_1.png"]:
            (seq / name).touch()
        asset = ClipAsset(str(seq), "sequence")
        files = asset.get_frame_files()
        assert files == ["frame_1.png", "frame_2.png", "frame_10.png"]

    def test_video_asset_get_frame_files_empty(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.touch()
        asset = ClipAsset(str(video), "video")
        assert asset.get_frame_files() == []

    def test_missing_sequence_dir_frame_count_zero(self, tmp_path: Path):
        asset = ClipAsset(str(tmp_path / "nonexistent"), "sequence")
        assert asset.frame_count == 0


class TestFindAssets:
    def test_finds_frames_sequence(self, tmp_path: Path):
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir)
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.input_asset is not None
        assert clip.input_asset.asset_type == "sequence"

    def test_state_is_raw_without_alpha(self, tmp_path: Path):
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir)
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.state == ClipState.RAW

    def test_state_is_ready_with_alpha(self, tmp_path: Path):
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir, with_alpha=True)
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.state == ClipState.READY

    def test_state_is_masked_with_mask_no_alpha(self, tmp_path: Path):
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir, with_mask=True)
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.state == ClipState.MASKED

    def test_no_input_raises(self, tmp_path: Path):
        clip_dir = tmp_path / "empty"
        clip_dir.mkdir()
        clip = ClipEntry(name="empty", root_path=str(clip_dir))
        with pytest.raises(ClipScanError):
            clip.find_assets()

    def test_loads_in_out_range_from_clip_json(self, tmp_path: Path):
        from corridorkey.project import write_clip_json

        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir)
        write_clip_json(str(clip_dir), {"in_out_range": {"in_point": 5, "out_point": 20}})
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.in_out_range is not None
        assert clip.in_out_range.in_point == 5
        assert clip.in_out_range.out_point == 20

    def test_state_complete_when_all_outputs_present(self, tmp_path: Path):
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir, with_alpha=True)

        # Write manifest and matching output frames
        import json

        out_dir = clip_dir / "Output"
        for subdir in ("FG", "Matte"):
            d = out_dir / subdir
            d.mkdir(parents=True)
            for i in range(5):
                (d / f"frame_{i:06d}.exr").touch()
        manifest = {"version": 1, "enabled_outputs": ["fg", "matte"], "formats": {}, "params": {}}
        (out_dir / ".corridorkey_manifest.json").write_text(json.dumps(manifest))

        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.state == ClipState.COMPLETE


class TestScanClipsDir:
    def test_scans_subdirectories(self, tmp_path: Path):
        for name in ("shot1", "shot2"):
            _make_clip_dir(tmp_path / name)
        clips = scan_clips_dir(str(tmp_path))
        assert len(clips) == 2

    def test_skips_hidden_dirs(self, tmp_path: Path):
        _make_clip_dir(tmp_path / "shot1")
        (tmp_path / ".hidden").mkdir()
        clips = scan_clips_dir(str(tmp_path))
        assert len(clips) == 1

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        clips = scan_clips_dir(str(tmp_path / "nonexistent"))
        assert clips == []

    def test_v2_project_scanned_correctly(self, tmp_path: Path):
        clips_dir = tmp_path / "clips"
        _make_clip_dir(clips_dir / "shot1")
        clips = scan_clips_dir(str(tmp_path))
        assert len(clips) == 1
        assert clips[0].name == "shot1"


class TestScanProjectClips:
    def test_v2_project(self, tmp_path: Path):
        clips_dir = tmp_path / "clips"
        _make_clip_dir(clips_dir / "shot1")
        _make_clip_dir(clips_dir / "shot2")
        clips = scan_project_clips(str(tmp_path))
        assert len(clips) == 2

    def test_v1_project_single_clip(self, tmp_path: Path):
        _make_clip_dir(tmp_path)
        clips = scan_project_clips(str(tmp_path))
        assert len(clips) == 1
