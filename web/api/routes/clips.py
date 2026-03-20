"""Clip scanning, detail, and deletion endpoints."""

from __future__ import annotations

import logging
import os
import shutil

from fastapi import APIRouter, HTTPException

from ..deps import get_service
from ..schemas import ClipAssetSchema, ClipListResponse, ClipSchema

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clips", tags=["clips"])

# Resolved at startup via app.state.clips_dir
_clips_dir: str = ""


def set_clips_dir(path: str) -> None:
    global _clips_dir
    _clips_dir = path


def _clip_to_schema(clip) -> ClipSchema:
    def _asset(a) -> ClipAssetSchema | None:
        if a is None:
            return None
        return ClipAssetSchema(path=a.path, asset_type=a.asset_type, frame_count=a.frame_count)

    frame_count = 0
    if clip.input_asset:
        frame_count = clip.input_asset.frame_count

    return ClipSchema(
        name=clip.name,
        root_path=clip.root_path,
        state=clip.state.value,
        input_asset=_asset(clip.input_asset),
        alpha_asset=_asset(clip.alpha_asset),
        mask_asset=_asset(clip.mask_asset),
        frame_count=frame_count,
        completed_frames=clip.completed_frame_count(),
        has_outputs=clip.has_outputs,
        warnings=clip.warnings,
        error_message=clip.error_message,
    )


@router.get("", response_model=ClipListResponse)
def list_clips():
    service = get_service()
    try:
        clips = service.scan_clips(_clips_dir)
    except Exception as e:
        logger.error(f"Clip scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return ClipListResponse(
        clips=[_clip_to_schema(c) for c in clips],
        clips_dir=_clips_dir,
    )


@router.get("/{name}", response_model=ClipSchema)
def get_clip(name: str):
    service = get_service()
    try:
        clips = service.scan_clips(_clips_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    for clip in clips:
        if clip.name == name:
            return _clip_to_schema(clip)
    raise HTTPException(status_code=404, detail=f"Clip '{name}' not found")


@router.delete("/{name}")
def delete_clip(name: str):
    """Delete a clip and its entire project directory.

    Removes the clip's root_path. If this was the only clip in a v2 project,
    removes the entire project folder too.
    """
    service = get_service()
    try:
        clips = service.scan_clips(_clips_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    clip = next((c for c in clips if c.name == name), None)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Clip '{name}' not found")

    clip_root = clip.root_path
    if not os.path.isdir(clip_root):
        raise HTTPException(status_code=404, detail="Clip directory not found on disk")

    # Safety: ensure the path is inside the clips dir
    abs_clips = os.path.abspath(_clips_dir)
    abs_clip = os.path.abspath(clip_root)
    if not abs_clip.startswith(abs_clips + os.sep):
        raise HTTPException(status_code=403, detail="Clip path is outside the projects directory")

    try:
        shutil.rmtree(clip_root)
        logger.info(f"Deleted clip directory: {clip_root}")

        # If the parent project's clips/ dir is now empty, remove the project too
        clips_parent = os.path.dirname(clip_root)  # .../clips/
        if os.path.basename(clips_parent) == "clips" and os.path.isdir(clips_parent):
            remaining = [d for d in os.listdir(clips_parent) if not d.startswith(".")]
            if not remaining:
                project_dir = os.path.dirname(clips_parent)
                shutil.rmtree(project_dir)
                logger.info(f"Deleted empty project directory: {project_dir}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}") from e

    return {"status": "deleted", "name": name}


@router.post("/{name}/move")
def move_clip(name: str, target_project: str):
    """Move a clip to a different project.

    The clip directory is physically moved into the target project's clips/ dir.
    Updates project.json in both source and target projects.
    """
    from backend.project import is_v2_project, read_project_json, write_project_json

    service = get_service()
    try:
        clips = service.scan_clips(_clips_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    clip = next((c for c in clips if c.name == name), None)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Clip '{name}' not found")

    # Find target project
    target_dir = os.path.join(_clips_dir, target_project)
    if not os.path.isdir(target_dir) or not is_v2_project(target_dir):
        raise HTTPException(status_code=404, detail=f"Target project '{target_project}' not found")

    target_clips_dir = os.path.join(target_dir, "clips")
    dest = os.path.join(target_clips_dir, os.path.basename(clip.root_path))

    if os.path.exists(dest):
        raise HTTPException(status_code=409, detail=f"A clip named '{name}' already exists in the target project")

    # Safety checks
    abs_root = os.path.abspath(_clips_dir)
    abs_src = os.path.abspath(clip.root_path)
    abs_dst = os.path.abspath(dest)
    if not abs_src.startswith(abs_root + os.sep) or not abs_dst.startswith(abs_root + os.sep):
        raise HTTPException(status_code=403, detail="Path outside projects directory")

    try:
        # Move the clip directory
        shutil.move(clip.root_path, dest)
        logger.info(f"Moved clip '{name}': {clip.root_path} → {dest}")

        # Update target project.json
        target_data = read_project_json(target_dir) or {}
        target_clips = target_data.get("clips", [])
        clip_basename = os.path.basename(dest)
        if clip_basename not in target_clips:
            target_clips.append(clip_basename)
            target_data["clips"] = target_clips
            write_project_json(target_dir, target_data)

        # Clean up source project — use saved path since clip.root_path was moved
        source_clips_parent = os.path.dirname(abs_src)  # use pre-move absolute path
        if os.path.basename(source_clips_parent) == "clips" and os.path.isdir(source_clips_parent):
            remaining = [d for d in os.listdir(source_clips_parent) if not d.startswith(".")]
            source_project = os.path.dirname(source_clips_parent)
            source_data = read_project_json(source_project) or {}
            source_clip_list = source_data.get("clips", [])
            source_clip_list = [c for c in source_clip_list if c != clip_basename]
            source_data["clips"] = source_clip_list
            if remaining:
                write_project_json(source_project, source_data)
            else:
                shutil.rmtree(source_project)
                logger.info(f"Deleted empty source project: {source_project}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to move clip: {e}") from e

    return {"status": "moved", "name": name, "target_project": target_project}
