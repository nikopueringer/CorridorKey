"""Project listing, creation, and management endpoints."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.clip_state import scan_project_clips
from backend.project import (
    _dedupe_path,
    is_v2_project,
    projects_root,
    read_project_json,
    set_display_name,
    write_project_json,
)

from .clips import _clip_to_schema, _clips_dir

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectSchema(BaseModel):
    name: str
    display_name: str
    path: str
    clip_count: int
    created: str | None = None
    clips: list[dict] = []


class CreateProjectRequest(BaseModel):
    name: str


class RenameProjectRequest(BaseModel):
    display_name: str


def _scan_projects() -> list[ProjectSchema]:
    """Scan the projects directory and return project info."""
    root = _clips_dir or projects_root()
    if not os.path.isdir(root):
        return []

    projects = []
    for item in sorted(os.listdir(root)):
        item_path = os.path.join(root, item)
        if not os.path.isdir(item_path) or item.startswith(".") or item.startswith("_"):
            continue

        # Only include v2 projects (have clips/ subdir)
        if not is_v2_project(item_path):
            continue

        data = read_project_json(item_path) or {}
        display = data.get("display_name", item)
        created = data.get("created")

        # Scan clips inside the project
        try:
            clips = scan_project_clips(item_path)
        except Exception:
            clips = []

        projects.append(
            ProjectSchema(
                name=item,
                display_name=display,
                path=item_path,
                clip_count=len(clips),
                created=created,
                clips=[_clip_to_schema(c).__dict__ for c in clips],
            )
        )

    return projects


@router.get("", response_model=list[ProjectSchema])
def list_projects():
    return _scan_projects()


@router.post("", response_model=ProjectSchema)
def create_project(req: CreateProjectRequest):
    """Create a new empty project."""
    root = _clips_dir or projects_root()
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    import re

    name_stem = re.sub(r"[^\w\-]", "_", req.name.strip())
    name_stem = re.sub(r"_+", "_", name_stem).strip("_")[:60]
    folder_name = f"{timestamp}_{name_stem}"

    project_dir, _ = _dedupe_path(root, folder_name)
    clips_dir = os.path.join(project_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    write_project_json(
        project_dir,
        {
            "version": 2,
            "created": datetime.now().isoformat(),
            "display_name": req.name.strip(),
            "clips": [],
        },
    )

    return ProjectSchema(
        name=os.path.basename(project_dir),
        display_name=req.name.strip(),
        path=project_dir,
        clip_count=0,
        created=datetime.now().isoformat(),
    )


@router.patch("/{name}")
def rename_project(name: str, req: RenameProjectRequest):
    """Rename a project's display name."""
    root = _clips_dir or projects_root()
    project_dir = os.path.join(root, name)
    if not os.path.isdir(project_dir):
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")

    set_display_name(project_dir, req.display_name.strip())
    return {"status": "ok", "display_name": req.display_name.strip()}


@router.delete("/{name}")
def delete_project(name: str):
    """Delete a project and all its clips."""
    root = _clips_dir or projects_root()
    project_dir = os.path.join(root, name)

    if not os.path.isdir(project_dir):
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")

    # Safety check
    abs_root = os.path.abspath(root)
    abs_proj = os.path.abspath(project_dir)
    if not abs_proj.startswith(abs_root + os.sep):
        raise HTTPException(status_code=403, detail="Project path outside projects directory")

    try:
        shutil.rmtree(project_dir)
        logger.info(f"Deleted project: {project_dir}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}") from e

    return {"status": "deleted", "name": name}
