"""Project folder management — creation, scanning, and metadata.

A project is a timestamped container holding one or more clips:
    Projects/
        260301_093000_Woman_Jumps/
            project.json                    (v2 — project-level metadata)
            clips/
                Woman_Jumps/                (ClipEntry.root_path → here)
                    Source/
                        Woman_Jumps_For_Joy.mp4
                    Frames/
                    AlphaHint/
                    Output/FG/ Matte/ Comp/ Processed/
                    clip.json               (per-clip metadata)
                Man_Walks/
                    Source/...

Legacy v1 format (no clips/ dir) is still supported for backward compat.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"})
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp", ".dpx"})
VIDEO_FILE_FILTER = "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf *.webm *.m4v);;All Files (*)"

_app_dir: str | None = None


def _dedupe_path(parent_dir: str, stem: str) -> tuple[str, str]:
    """Return a unique child path under *parent_dir* and its final stem.

    If ``{parent_dir}/{stem}`` already exists, appends numeric suffixes
    (``_2``, ``_3``, ...) until a free path is found.

    Unlike fixed-range probes, this never silently falls back to an existing
    path after enough collisions.
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


def set_app_dir(path: str) -> None:
    """Set the application directory. Called once at startup by main.py."""
    global _app_dir
    _app_dir = path


def projects_root() -> str:
    """Return the Projects root directory, creating it if needed.

    In dev mode: {repo_root}/Projects/
    In frozen mode: {exe_dir}/Projects/
    """
    if _app_dir:
        root = os.path.join(_app_dir, "Projects")
    elif getattr(sys, "frozen", False):
        root = os.path.join(os.path.dirname(sys.executable), "Projects")
    else:
        # Fallback: two levels up from this file (backend/ -> repo root)
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Projects")
    os.makedirs(root, exist_ok=True)
    return root


def sanitize_stem(filename: str, max_len: int = 60) -> str:
    """Clean a filename stem for use in folder names.

    Strips extension, replaces non-alphanumeric chars with underscores,
    collapses runs, and truncates.
    """
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[^\w\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem[:max_len]


def create_project(
    source_video_paths: str | list[str],
    *,
    copy_source: bool = True,
    display_name: str | None = None,
) -> str:
    """Create a new project folder for one or more source videos.

    Creates a v2 project with a ``clips/`` subdirectory.  Each video
    gets its own clip subfolder inside ``clips/``.

    When *copy_source* is True (default), video files are copied into
    each clip's ``Source/`` directory.  When False, the clip stores a
    reference to the original file path.

    Creates: Projects/YYMMDD_HHMMSS_{stem}/clips/{clip_stem}/Source/...

    Args:
        source_video_paths: Single video path (str) or list of paths.
        copy_source: Whether to copy video files into clip folders.
        display_name: Optional project name. If provided, used for both
            the folder name stem and display_name in project.json.
            If None, derived from the first video filename.

    Returns:
        Absolute path to the new project folder.
    """
    # Accept single path for backward compat
    if isinstance(source_video_paths, str):
        source_video_paths = [source_video_paths]
    if not source_video_paths:
        raise ValueError("At least one source video path is required")

    root = projects_root()

    if display_name and display_name.strip():
        clean = display_name.strip()
        # Sanitize for folder name (no splitext — it's not a filename)
        name_stem = re.sub(r"[^\w\-]", "_", clean)
        name_stem = re.sub(r"_+", "_", name_stem).strip("_")[:60]
        project_display_name = clean
    else:
        first_filename = os.path.basename(source_video_paths[0])
        name_stem = sanitize_stem(first_filename)
        project_display_name = name_stem.replace("_", " ")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{name_stem}"

    # Deduplicate if folder already exists (e.g. rapid imports)
    project_dir, _ = _dedupe_path(root, folder_name)

    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_names: list[str] = []
    for video_path in source_video_paths:
        clip_name = _create_clip_folder(
            clips_dir,
            video_path,
            copy_source=copy_source,
        )
        clip_names.append(clip_name)

    # Write project.json (v2 — project-level metadata only)
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
        copy_source: Whether to copy videos into clip folders.

    Returns:
        List of new clip subfolder paths (absolute).
    """
    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    new_paths: list[str] = []
    for video_path in source_video_paths:
        clip_name = _create_clip_folder(
            clips_dir,
            video_path,
            copy_source=copy_source,
        )
        new_paths.append(os.path.join(clips_dir, clip_name))

    # Update project.json clips list
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

    Returns the clip folder name (not full path).
    """
    filename = os.path.basename(video_path)
    clip_name = sanitize_stem(filename)

    # Deduplicate clip folder names within same project
    clip_dir, clip_name = _dedupe_path(clips_dir, clip_name)

    source_dir = os.path.join(clip_dir, "Source")
    os.makedirs(source_dir, exist_ok=True)

    if copy_source:
        target = os.path.join(source_dir, filename)
        if not os.path.isfile(target):
            shutil.copy2(video_path, target)
            logger.info(f"Copied source video: {video_path} -> {target}")
    else:
        logger.info(f"Referencing source video in place: {video_path}")

    # Write clip.json (per-clip metadata)
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

    For v2 projects (with clips/ dir), scans clips/ subdirectories.
    For v1 projects (no clips/ dir), returns [project_dir] as a single clip.
    """
    clips_dir = os.path.join(project_dir, "clips")
    if os.path.isdir(clips_dir):
        return sorted(
            os.path.join(clips_dir, d)
            for d in os.listdir(clips_dir)
            if os.path.isdir(os.path.join(clips_dir, d)) and not d.startswith(".") and not d.startswith("_")
        )
    # v1 fallback: project dir itself is the clip
    return [project_dir]


def is_v2_project(project_dir: str) -> bool:
    """Check if a project uses the v2 nested clips structure."""
    return os.path.isdir(os.path.join(project_dir, "clips"))


def write_project_json(project_root: str, data: dict) -> None:
    """Atomic write of project.json."""
    path = os.path.join(project_root, "project.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def read_project_json(project_root: str) -> dict | None:
    """Read project.json, returning None if missing or corrupt."""
    path = os.path.join(project_root, "project.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read project.json at {path}: {e}")
        return None


def write_clip_json(clip_root: str, data: dict) -> None:
    """Atomic write of clip.json (per-clip metadata)."""
    path = os.path.join(clip_root, "clip.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def read_clip_json(clip_root: str) -> dict | None:
    """Read clip.json, returning None if missing or corrupt."""
    path = os.path.join(clip_root, "clip.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read clip.json at {path}: {e}")
        return None


def _read_clip_or_project_json(root: str) -> dict | None:
    """Read clip.json first, falling back to project.json for v1 compat."""
    data = read_clip_json(root)
    if data is not None:
        return data
    return read_project_json(root)


def get_display_name(root: str) -> str:
    """Get the user-visible name for a clip or project.

    Checks clip.json first, then project.json, falling back to folder name.
    """
    data = _read_clip_or_project_json(root)
    if data and data.get("display_name"):
        return data["display_name"]
    return os.path.basename(root)


def set_display_name(root: str, name: str) -> None:
    """Update display_name. Writes to clip.json if it exists, else project.json."""
    if os.path.isfile(os.path.join(root, "clip.json")):
        data = read_clip_json(root) or {}
        data["display_name"] = name
        write_clip_json(root, data)
    else:
        data = read_project_json(root) or {}
        data["display_name"] = name
        write_project_json(root, data)


def save_in_out_range(clip_root: str, in_out) -> None:
    """Persist in/out range to clip.json (v2) or project.json (v1).

    Pass None to clear.
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


def load_in_out_range(clip_root: str):
    """Load in/out range from clip.json or project.json, or None if not set."""
    data = _read_clip_or_project_json(clip_root)
    if data and "in_out_range" in data:
        try:
            from .clip_state import InOutRange

            return InOutRange.from_dict(data["in_out_range"])
        except (KeyError, TypeError):
            return None
    return None


def is_video_file(filename: str) -> bool:
    """Check if a filename has a video extension."""
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS


def is_image_file(filename: str) -> bool:
    """Check if a filename has an image extension."""
    return os.path.splitext(filename)[1].lower() in _IMAGE_EXTS
