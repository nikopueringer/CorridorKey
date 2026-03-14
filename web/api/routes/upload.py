"""Upload endpoints — video files, image sequences (zip), and alpha hints."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile

from fastapi import APIRouter, HTTPException, UploadFile

from backend.job_queue import GPUJob, JobType
from backend.project import (
    create_project,
    is_image_file,
    is_video_file,
    sanitize_stem,
)

from ..deps import get_queue, get_service
from ..routes.clips import _clip_to_schema, _clips_dir

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/video")
async def upload_video(file: UploadFile, name: str | None = None, auto_extract: bool = True):
    """Upload a video file to create a new project/clip.

    The video is saved into a new project via create_project().
    If auto_extract is True (default), a VIDEO_EXTRACT job is queued
    to extract frames in the background.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not is_video_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Not a supported video format: {file.filename}. "
            "Supported: .mp4, .mov, .avi, .mkv, .mxf, .webm, .m4v",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, file.filename)
        try:
            with open(tmp_path, "wb") as f:
                while chunk := await file.read(8 * 1024 * 1024):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e

        try:
            project_dir = create_project(
                tmp_path,
                copy_source=True,
                display_name=name,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create project: {e}") from e

    # Scan the new clips
    service = get_service()
    clips = service.scan_clips(_clips_dir)
    new_clips = [c for c in clips if c.root_path.startswith(project_dir)]

    # Auto-submit extraction jobs for any clip with a video source
    extract_jobs = []
    if auto_extract:
        queue = get_queue()
        for clip in new_clips:
            has_video = clip.input_asset and clip.input_asset.asset_type == "video"
            no_frames = not os.path.isdir(os.path.join(clip.root_path, "Frames"))
            logger.info(
                f"Upload auto-extract check: clip={clip.name} state={clip.state.value} "
                f"has_video={has_video} no_frames={no_frames}"
            )
            if has_video or clip.state.value == "EXTRACTING":
                job = GPUJob(job_type=JobType.VIDEO_EXTRACT, clip_name=clip.name)
                if queue.submit(job):
                    extract_jobs.append(job.id)
                    logger.info(f"Auto-queued extraction job {job.id} for '{clip.name}'")

    return {
        "status": "ok",
        "project_dir": project_dir,
        "clips": [_clip_to_schema(c) for c in new_clips],
        "extract_jobs": extract_jobs,
    }


@router.post("/frames")
async def upload_frames(file: UploadFile, name: str | None = None):
    """Upload a zip of image frames to create a new clip.

    The zip should contain image files (PNG, EXR, JPG, etc.) at the
    top level or in a single subdirectory. They'll be placed into
    a new project's Frames/ directory.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Expected a .zip file containing image frames")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, file.filename)
        try:
            with open(zip_path, "wb") as f:
                while chunk := await file.read(8 * 1024 * 1024):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e

        # Extract zip
        extract_dir = os.path.join(tmpdir, "extracted")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file") from None

        # Find image files — may be at root or in a single subdirectory
        image_files = [f for f in os.listdir(extract_dir) if is_image_file(f)]
        if not image_files:
            subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
            if len(subdirs) == 1:
                subdir_path = os.path.join(extract_dir, subdirs[0])
                image_files = [f for f in os.listdir(subdir_path) if is_image_file(f)]
                if image_files:
                    extract_dir = subdir_path

        if not image_files:
            raise HTTPException(status_code=400, detail="No image files found in zip")

        # Create project structure manually (create_project expects video)
        from datetime import datetime

        from backend.project import _dedupe_path, projects_root, write_clip_json, write_project_json

        clip_name = sanitize_stem(name or file.filename)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{clip_name}"

        root = projects_root()
        project_dir, _ = _dedupe_path(root, folder_name)
        clips_dir = os.path.join(project_dir, "clips")
        clip_dir, clip_name = _dedupe_path(clips_dir, clip_name)
        frames_dir = os.path.join(clip_dir, "Frames")
        os.makedirs(frames_dir, exist_ok=True)

        for fname in sorted(image_files):
            src = os.path.join(extract_dir, fname)
            dst = os.path.join(frames_dir, fname)
            shutil.copy2(src, dst)

        write_clip_json(clip_dir, {"source": {"type": "uploaded_frames", "original_filename": file.filename}})
        write_project_json(
            project_dir,
            {
                "version": 2,
                "created": datetime.now().isoformat(),
                "display_name": clip_name.replace("_", " "),
                "clips": [clip_name],
            },
        )

    service = get_service()
    clips = service.scan_clips(_clips_dir)
    new_clips = [c for c in clips if c.root_path.startswith(project_dir)]

    return {
        "status": "ok",
        "project_dir": project_dir,
        "clips": [_clip_to_schema(c) for c in new_clips],
        "frame_count": len(image_files),
    }


@router.post("/alpha/{clip_name}")
async def upload_alpha_hint(clip_name: str, file: UploadFile):
    """Upload alpha hint frames (zip) for an existing clip.

    Extracts images into the clip's AlphaHint/ directory.
    Transitions clip from RAW -> READY.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Expected a .zip file containing alpha hint frames")

    service = get_service()
    clips = service.scan_clips(_clips_dir)
    clip = next((c for c in clips if c.name == clip_name), None)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_name}' not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, file.filename)
        try:
            with open(zip_path, "wb") as f:
                while chunk := await file.read(8 * 1024 * 1024):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e

        extract_dir = os.path.join(tmpdir, "extracted")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file") from None

        image_files = [f for f in os.listdir(extract_dir) if is_image_file(f)]
        if not image_files:
            subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
            if len(subdirs) == 1:
                extract_dir = os.path.join(extract_dir, subdirs[0])
                image_files = [f for f in os.listdir(extract_dir) if is_image_file(f)]

        if not image_files:
            raise HTTPException(status_code=400, detail="No image files found in zip")

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        for fname in sorted(image_files):
            src = os.path.join(extract_dir, fname)
            dst = os.path.join(alpha_dir, fname)
            shutil.copy2(src, dst)

    clips = service.scan_clips(_clips_dir)
    updated = next((c for c in clips if c.name == clip_name), None)

    return {
        "status": "ok",
        "clip": _clip_to_schema(updated) if updated else None,
        "alpha_frames": len(image_files),
    }


@router.post("/mask/{clip_name}")
async def upload_videomama_mask(clip_name: str, file: UploadFile):
    """Upload VideoMaMa mask hint frames (zip) for an existing clip.

    Extracts images into the clip's VideoMamaMaskHint/ directory.
    Transitions clip to MASKED state.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Expected a .zip file containing mask frames")

    service = get_service()
    clips = service.scan_clips(_clips_dir)
    clip = next((c for c in clips if c.name == clip_name), None)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_name}' not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, file.filename)
        try:
            with open(zip_path, "wb") as f:
                while chunk := await file.read(8 * 1024 * 1024):
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e

        extract_dir = os.path.join(tmpdir, "extracted")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file") from None

        image_files = [f for f in os.listdir(extract_dir) if is_image_file(f)]
        if not image_files:
            subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
            if len(subdirs) == 1:
                extract_dir = os.path.join(extract_dir, subdirs[0])
                image_files = [f for f in os.listdir(extract_dir) if is_image_file(f)]

        if not image_files:
            raise HTTPException(status_code=400, detail="No image files found in zip")

        mask_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
        os.makedirs(mask_dir, exist_ok=True)

        for fname in sorted(image_files):
            src = os.path.join(extract_dir, fname)
            dst = os.path.join(mask_dir, fname)
            shutil.copy2(src, dst)

    clips = service.scan_clips(_clips_dir)
    updated = next((c for c in clips if c.name == clip_name), None)

    return {
        "status": "ok",
        "clip": _clip_to_schema(updated) if updated else None,
        "mask_frames": len(image_files),
    }
