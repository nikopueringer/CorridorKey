"""Regression tests for backend.clip_state scanning behavior."""

from __future__ import annotations

import os

from backend.clip_state import scan_clips_dir
from backend.project import write_clip_json


def _write_frame_sequence(sequence_dir, count: int = 1) -> None:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (sequence_dir / f"frame_{i:04d}.png").touch()


def test_scan_clips_dir_keeps_v2_clips_with_same_display_name(tmp_path):
    """Different clips must not collapse just because their display names match."""
    shared_name = "Hero Plate"

    for project_name, clip_name in (("project_a", "shot_a"), ("project_b", "shot_b")):
        clip_root = tmp_path / project_name / "clips" / clip_name
        _write_frame_sequence(clip_root / "Frames")
        write_clip_json(str(clip_root), {"display_name": shared_name})

    entries = scan_clips_dir(str(tmp_path), allow_standalone_videos=False)

    assert len(entries) == 2
    assert [entry.name for entry in entries] == [shared_name, shared_name]
    assert {os.path.basename(entry.root_path) for entry in entries} == {"shot_a", "shot_b"}


def test_scan_clips_dir_still_prefers_folder_clip_over_loose_video(tmp_path):
    """A valid clip folder should still win over a loose video with the same stem."""
    clip_root = tmp_path / "hero"
    _write_frame_sequence(clip_root / "Input")
    (tmp_path / "hero.mp4").touch()

    entries = scan_clips_dir(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].root_path == str(clip_root)
    assert entries[0].name == "hero"
