"""Project folder management - creation, scanning, and metadata.

A project is a timestamped container holding one or more clips:

    Projects/
        260301_093000_Woman_Jumps/
            project.json                    (v2 project-level metadata)
            clips/
                Woman_Jumps/                (ClipEntry.root_path points here)
                    Source/
                        Woman_Jumps_For_Joy.mp4
                    Frames/
                    AlphaHint/
                    Output/FG/ Matte/ Comp/ Processed/
                    clip.json               (per-clip metadata)
                Man_Walks/
                    Source/...

Legacy v1 format (no clips/ subdirectory) is still supported for backward
compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime

from corridorkey.models import InOutRange

logger = logging.getLogger(__name__)

_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"})
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp", ".dpx"})

# Qt-style file dialog filter string for video files.
VIDEO_FILE_FILTER = "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf *.webm *.m4v);;All Files (*)"


def _dedupe_path(parent_dir: str, stem: str) -> tuple[str, str]:
    """Return a unique child path under parent_dir and its final stem.

    If ``{parent_dir}/{stem}`` already exists, appends numeric suffixes
    (``_2``, ``_3``, ...) until a free path is found.

    Args:
        parent_dir: Directory in which to create the child.
        stem: Desired folder name.

    Returns:
        Tuple of (absolute_path, final_stem).
    """
    path = os.path.join(parent_dir, stem)
    if not os.path.exists(path):
        return path, stem

    index = 2
    while True:
        candidate_stem = f"{stem}_{index}"
        candidate_path = os.path.join(parent_dir, candidate_stem)
        if not os.path.exists(candidate_path):
            return candidate_path, candidate_stem
        index += 1


def sanitize_stem(filename: str, max_len: int = 60) -> str:
    """Clean a filename stem for use in folder names.

    Strips the extension, replaces non-alphanumeric characters with
    underscores, collapses consecutive underscores, and truncates.

    Args:
        filename: Original filename (with or without extension).
        max_len: Maximum length of the returned stem.

    Returns:
        Sanitized stem string safe for use as a folder name.
    """
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[^\w\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem[:max_len]


def create_project(
    source_video_paths: str | list[str],
    projects_dir: str,
    *,
    copy_source: bool = True,
    display_name: str | None = None,
) -> str:
    """Create a new v2 project folder for one or more source videos.

    Each video gets its own clip subfolder inside ``clips/``. When
    copy_source is True the video is copied into ``Source/``; otherwise
    only a reference path is stored in clip.json.

    Creates: ``{projects_dir}/YYMMDD_HHMMSS_{stem}/clips/{clip_stem}/Source/...``

    Args:
        source_video_paths: Single video path or list of paths.
        projects_dir: Absolute path to the directory where projects are stored.
            This must always be provided - there is no default.
        copy_source: Copy video files into clip folders when True.
        display_name: Optional project name. Derived from the first video
            filename when None.

    Returns:
        Absolute path to the new project folder.

    Raises:
        ValueError: If source_video_paths is empty.
    """
    if isinstance(source_video_paths, str):
        source_video_paths = [source_video_paths]
    if not source_video_paths:
        raise ValueError("At least one source video path is required")

    root = projects_dir
    os.makedirs(root, exist_ok=True)

    if display_name and display_name.strip():
        clean = display_name.strip()
        name_stem = re.sub(r"[^\w\-]", "_", clean)
        name_stem = re.sub(r"_+", "_", name_stem).strip("_")[:60]
        project_display_name = clean
    else:
        first_filename = os.path.basename(source_video_paths[0])
        name_stem = sanitize_stem(first_filename)
        project_display_name = name_stem.replace("_", " ")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{name_stem}"

    project_dir, _ = _dedupe_path(root, folder_name)

    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_names: list[str] = []
    for video_path in source_video_paths:
        clip_name = _create_clip_folder(clips_dir, video_path, copy_source=copy_source)
        clip_names.append(clip_name)

    write_project_json(
        project_dir,
        {
            "version": 2,
            "created": datetime.now().isoformat(),
            "display_name": project_display_name,
            "clips": clip_names,
        },
    )

    return project_dir


def add_clips_to_project(
    project_dir: str,
    source_video_paths: list[str],
    *,
    copy_source: bool = True,
) -> list[str]:
    """Add new clips to an existing project.

    Args:
        project_dir: Absolute path to the project folder.
        source_video_paths: List of video file paths to add.
        copy_source: Copy videos into clip folders when True.

    Returns:
        List of absolute paths to the new clip subfolders.
    """
    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    new_paths: list[str] = []
    for video_path in source_video_paths:
        clip_name = _create_clip_folder(clips_dir, video_path, copy_source=copy_source)
        new_paths.append(os.path.join(clips_dir, clip_name))

    data = read_project_json(project_dir) or {}
    existing = data.get("clips", [])
    for p in new_paths:
        existing.append(os.path.basename(p))
    data["clips"] = existing
    write_project_json(project_dir, data)

    return new_paths


def _create_clip_folder(
    clips_dir: str,
    video_path: str,
    *,
    copy_source: bool = True,
) -> str:
    """Create a single clip subfolder inside clips_dir.

    Args:
        clips_dir: Parent directory for clip subfolders.
        video_path: Absolute path to the source video.
        copy_source: Copy the video into Source/ when True.

    Returns:
        The clip folder name (not the full path).
    """
    filename = os.path.basename(video_path)
    clip_name = sanitize_stem(filename)

    clip_dir, clip_name = _dedupe_path(clips_dir, clip_name)

    source_dir = os.path.join(clip_dir, "Source")
    os.makedirs(source_dir, exist_ok=True)

    if copy_source:
        target = os.path.join(source_dir, filename)
        if not os.path.isfile(target):
            shutil.copy2(video_path, target)
            logger.info("Copied source video: %s -> %s", video_path, target)
    else:
        logger.info("Referencing source video in place: %s", video_path)

    write_clip_json(
        clip_dir,
        {
            "source": {
                "original_path": os.path.abspath(video_path),
                "filename": filename,
                "copied": copy_source,
            },
        },
    )

    return clip_name


def get_clip_dirs(project_dir: str) -> list[str]:
    """Return absolute paths to all clip subdirectories in a project.

    For v2 projects (with a clips/ subdirectory) scans that directory.
    For v1 projects returns [project_dir] as a single-clip fallback.

    Args:
        project_dir: Absolute path to the project folder.

    Returns:
        Sorted list of absolute clip directory paths.
    """
    clips_dir = os.path.join(project_dir, "clips")
    if os.path.isdir(clips_dir):
        return sorted(
            os.path.join(clips_dir, d)
            for d in os.listdir(clips_dir)
            if os.path.isdir(os.path.join(clips_dir, d)) and not d.startswith(".") and not d.startswith("_")
        )
    return [project_dir]


def is_v2_project(project_dir: str) -> bool:
    """Check whether a project uses the v2 nested clips structure.

    Args:
        project_dir: Absolute path to the project folder.

    Returns:
        True if a clips/ subdirectory exists.
    """
    return os.path.isdir(os.path.join(project_dir, "clips"))


def write_project_json(project_root: str, data: dict) -> None:
    """Atomically write project.json.

    Args:
        project_root: Absolute path to the project folder.
        data: Dict to serialise as JSON.
    """
    path = os.path.join(project_root, "project.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def read_project_json(project_root: str) -> dict | None:
    """Read project.json, returning None if missing or corrupt.

    Args:
        project_root: Absolute path to the project folder.

    Returns:
        Parsed dict, or None on failure.
    """
    path = os.path.join(project_root, "project.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read project.json at %s: %s", path, e)
        return None


def write_clip_json(clip_root: str, data: dict) -> None:
    """Atomically write clip.json (per-clip metadata).

    Args:
        clip_root: Absolute path to the clip folder.
        data: Dict to serialise as JSON.
    """
    path = os.path.join(clip_root, "clip.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def read_clip_json(clip_root: str) -> dict | None:
    """Read clip.json, returning None if missing or corrupt.

    Args:
        clip_root: Absolute path to the clip folder.

    Returns:
        Parsed dict, or None on failure.
    """
    path = os.path.join(clip_root, "clip.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read clip.json at %s: %s", path, e)
        return None


def _read_clip_or_project_json(root: str) -> dict | None:
    """Read clip.json first, falling back to project.json for v1 compatibility.

    Args:
        root: Absolute path to a clip or project folder.

    Returns:
        Parsed dict from whichever file was found, or None.
    """
    data = read_clip_json(root)
    if data is not None:
        return data
    return read_project_json(root)


def get_display_name(root: str) -> str:
    """Get the user-visible name for a clip or project.

    Checks clip.json first, then project.json, falling back to the
    folder name.

    Args:
        root: Absolute path to a clip or project folder.

    Returns:
        Display name string.
    """
    data = _read_clip_or_project_json(root)
    if data and data.get("display_name"):
        return data["display_name"]
    return os.path.basename(root)


def set_display_name(root: str, name: str) -> None:
    """Update the display_name in clip.json or project.json.

    Writes to clip.json when it exists, otherwise to project.json.

    Args:
        root: Absolute path to a clip or project folder.
        name: New display name.
    """
    if os.path.isfile(os.path.join(root, "clip.json")):
        data = read_clip_json(root) or {}
        data["display_name"] = name
        write_clip_json(root, data)
    else:
        data = read_project_json(root) or {}
        data["display_name"] = name
        write_project_json(root, data)


def save_in_out_range(clip_root: str, in_out: InOutRange | None) -> None:
    """Persist an in/out range to clip.json or project.json.

    Pass None to clear the stored range.

    Args:
        clip_root: Absolute path to the clip folder.
        in_out: Range to store, or None to remove.
    """
    if os.path.isfile(os.path.join(clip_root, "clip.json")):
        data = read_clip_json(clip_root) or {}
        if in_out is not None:
            data["in_out_range"] = in_out.to_dict()
        else:
            data.pop("in_out_range", None)
        write_clip_json(clip_root, data)
    else:
        data = read_project_json(clip_root) or {}
        if in_out is not None:
            data["in_out_range"] = in_out.to_dict()
        else:
            data.pop("in_out_range", None)
        write_project_json(clip_root, data)


def load_in_out_range(clip_root: str) -> InOutRange | None:
    """Load an in/out range from clip.json or project.json.

    Args:
        clip_root: Absolute path to the clip folder.

    Returns:
        InOutRange if stored, otherwise None.
    """
    data = _read_clip_or_project_json(clip_root)
    if data and "in_out_range" in data:
        try:
            return InOutRange.from_dict(data["in_out_range"])
        except (KeyError, TypeError):
            return None
    return None


def is_video_file(filename: str) -> bool:
    """Check whether a filename has a recognised video extension.

    Args:
        filename: Filename or path to check.

    Returns:
        True if the extension is in the known video set.
    """
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS


def is_image_file(filename: str) -> bool:
    """Check whether a filename has a recognised image extension.

    Args:
        filename: Filename or path to check.

    Returns:
        True if the extension is in the known image set.
    """
    return os.path.splitext(filename)[1].lower() in _IMAGE_EXTS
