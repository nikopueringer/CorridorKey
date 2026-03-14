"""Unit tests for project.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.models import InOutRange
from corridorkey.project import (
    add_clips_to_project,
    create_project,
    get_clip_dirs,
    get_display_name,
    is_image_file,
    is_v2_project,
    is_video_file,
    load_in_out_range,
    read_clip_json,
    read_project_json,
    sanitize_stem,
    save_in_out_range,
    set_display_name,
    write_clip_json,
    write_project_json,
)


class TestSanitizeStem:
    def test_strips_extension(self):
        assert sanitize_stem("my_clip.mp4") == "my_clip"

    def test_replaces_spaces(self):
        assert sanitize_stem("my clip.mp4") == "my_clip"

    def test_collapses_underscores(self):
        assert sanitize_stem("my__clip.mp4") == "my_clip"

    def test_truncates_to_max_len(self):
        long_name = "a" * 100 + ".mp4"
        result = sanitize_stem(long_name, max_len=60)
        assert len(result) <= 60

    def test_no_extension(self):
        assert sanitize_stem("my_clip") == "my_clip"

    def test_special_chars_replaced(self):
        result = sanitize_stem("my-clip (1).mp4")
        assert " " not in result
        assert "(" not in result


class TestIsVideoFile:
    def test_mp4(self):
        assert is_video_file("clip.mp4")

    def test_mov(self):
        assert is_video_file("clip.MOV")

    def test_not_video(self):
        assert not is_video_file("frame.png")
        assert not is_video_file("data.json")


class TestIsImageFile:
    def test_png(self):
        assert is_image_file("frame.png")

    def test_exr(self):
        assert is_image_file("frame.EXR")

    def test_not_image(self):
        assert not is_image_file("clip.mp4")
        assert not is_image_file("data.json")


class TestWriteReadProjectJson:
    def test_roundtrip(self, tmp_path: Path):
        data = {"version": 2, "display_name": "Test Project"}
        write_project_json(str(tmp_path), data)
        result = read_project_json(str(tmp_path))
        assert result == data

    def test_missing_returns_none(self, tmp_path: Path):
        assert read_project_json(str(tmp_path)) is None

    def test_corrupt_returns_none(self, tmp_path: Path):
        (tmp_path / "project.json").write_text("not json")
        assert read_project_json(str(tmp_path)) is None


class TestWriteReadClipJson:
    def test_roundtrip(self, tmp_path: Path):
        data = {"source": {"filename": "clip.mp4"}}
        write_clip_json(str(tmp_path), data)
        result = read_clip_json(str(tmp_path))
        assert result == data

    def test_missing_returns_none(self, tmp_path: Path):
        assert read_clip_json(str(tmp_path)) is None

    def test_corrupt_returns_none(self, tmp_path: Path):
        (tmp_path / "clip.json").write_text("{bad json")
        assert read_clip_json(str(tmp_path)) is None


class TestGetSetDisplayName:
    def test_falls_back_to_folder_name(self, tmp_path: Path):
        clip_dir = tmp_path / "MyClip"
        clip_dir.mkdir()
        assert get_display_name(str(clip_dir)) == "MyClip"

    def test_reads_from_clip_json(self, tmp_path: Path):
        write_clip_json(str(tmp_path), {"display_name": "Pretty Name"})
        assert get_display_name(str(tmp_path)) == "Pretty Name"

    def test_reads_from_project_json_fallback(self, tmp_path: Path):
        write_project_json(str(tmp_path), {"display_name": "Project Name"})
        assert get_display_name(str(tmp_path)) == "Project Name"

    def test_set_writes_to_clip_json(self, tmp_path: Path):
        write_clip_json(str(tmp_path), {})
        set_display_name(str(tmp_path), "New Name")
        assert get_display_name(str(tmp_path)) == "New Name"

    def test_set_writes_to_project_json_when_no_clip_json(self, tmp_path: Path):
        set_display_name(str(tmp_path), "Project Name")
        data = read_project_json(str(tmp_path))
        assert data is not None
        assert data["display_name"] == "Project Name"


class TestSaveLoadInOutRange:
    def test_roundtrip_via_clip_json(self, tmp_path: Path):
        write_clip_json(str(tmp_path), {})
        r = InOutRange(in_point=10, out_point=50)
        save_in_out_range(str(tmp_path), r)
        loaded = load_in_out_range(str(tmp_path))
        assert loaded is not None
        assert loaded.in_point == 10
        assert loaded.out_point == 50

    def test_clear_range(self, tmp_path: Path):
        write_clip_json(str(tmp_path), {})
        save_in_out_range(str(tmp_path), InOutRange(0, 10))
        save_in_out_range(str(tmp_path), None)
        assert load_in_out_range(str(tmp_path)) is None

    def test_missing_returns_none(self, tmp_path: Path):
        assert load_in_out_range(str(tmp_path)) is None


class TestIsV2Project:
    def test_v2_has_clips_subdir(self, tmp_path: Path):
        (tmp_path / "clips").mkdir()
        assert is_v2_project(str(tmp_path))

    def test_v1_no_clips_subdir(self, tmp_path: Path):
        assert not is_v2_project(str(tmp_path))


class TestGetClipDirs:
    def test_v2_returns_clip_subdirs(self, tmp_path: Path):
        clips = tmp_path / "clips"
        (clips / "shot1").mkdir(parents=True)
        (clips / "shot2").mkdir(parents=True)
        result = get_clip_dirs(str(tmp_path))
        names = [Path(p).name for p in result]
        assert sorted(names) == ["shot1", "shot2"]

    def test_v1_returns_project_dir(self, tmp_path: Path):
        result = get_clip_dirs(str(tmp_path))
        assert result == [str(tmp_path)]

    def test_hidden_dirs_excluded(self, tmp_path: Path):
        clips = tmp_path / "clips"
        (clips / "shot1").mkdir(parents=True)
        (clips / ".hidden").mkdir()
        result = get_clip_dirs(str(tmp_path))
        names = [Path(p).name for p in result]
        assert ".hidden" not in names


class TestCreateProject:
    def test_creates_project_folder(self, tmp_path: Path):
        video = tmp_path / "source.mp4"
        video.touch()
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=False)
        assert Path(project_dir).is_dir()

    def test_creates_clips_subdir(self, tmp_path: Path):
        video = tmp_path / "source.mp4"
        video.touch()
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=False)
        assert (Path(project_dir) / "clips").is_dir()

    def test_creates_project_json(self, tmp_path: Path):
        video = tmp_path / "source.mp4"
        video.touch()
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=False)
        data = read_project_json(project_dir)
        assert data is not None
        assert data["version"] == 2

    def test_custom_display_name(self, tmp_path: Path):
        video = tmp_path / "source.mp4"
        video.touch()
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=False, display_name="My Project")
        data = read_project_json(project_dir)
        assert data is not None
        assert data["display_name"] == "My Project"

    def test_empty_paths_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="At least one"):
            create_project([], str(tmp_path / "Projects"))

    def test_copy_source_copies_file(self, tmp_path: Path):
        video = tmp_path / "source.mp4"
        video.write_bytes(b"fake video data")
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=True)
        clips_dir = Path(project_dir) / "clips"
        clip_dirs = [d for d in clips_dir.iterdir() if d.is_dir()]
        assert len(clip_dirs) == 1
        source_dir = clip_dirs[0] / "Source"
        assert (source_dir / "source.mp4").exists()


class TestAddClipsToProject:
    def test_adds_clip_to_existing_project(self, tmp_path: Path):
        video1 = tmp_path / "clip1.mp4"
        video1.touch()
        video2 = tmp_path / "clip2.mp4"
        video2.touch()
        projects_dir = tmp_path / "Projects"

        project_dir = create_project(str(video1), str(projects_dir), copy_source=False)
        new_paths = add_clips_to_project(project_dir, [str(video2)], copy_source=False)

        assert len(new_paths) == 1
        assert Path(new_paths[0]).is_dir()

    def test_project_json_updated(self, tmp_path: Path):
        video1 = tmp_path / "clip1.mp4"
        video1.touch()
        video2 = tmp_path / "clip2.mp4"
        video2.touch()
        projects_dir = tmp_path / "Projects"

        project_dir = create_project(str(video1), str(projects_dir), copy_source=False)
        add_clips_to_project(project_dir, [str(video2)], copy_source=False)

        data = read_project_json(project_dir)
        assert data is not None
        assert len(data["clips"]) == 2
