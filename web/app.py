"""FastAPI web application for CorridorKey.

Wraps CorridorKeyService with REST endpoints and WebSocket progress.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.clip_state import ClipState
from backend.job_queue import GPUJob, JobType
from backend.service import CorridorKeyService, InferenceParams, OutputConfig

from .thumbnail import get_frame_preview, get_or_create_thumbnail
from .worker import GPUWorker

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent


# --- WebSocket Connection Manager ---


class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, data: dict) -> None:
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# --- Request Models ---


class ScanRequest(BaseModel):
    path: str | None = None


class ClipNameRequest(BaseModel):
    name: str


class AddFileRequest(BaseModel):
    file_path: str


class ImportAlphaRequest(BaseModel):
    clip_name: str
    alpha_path: str  # path to folder of PNG/EXR frames OR a video file


class InferenceRequest(BaseModel):
    input_is_linear: bool = False
    despill_strength: float = 0.5
    auto_despeckle: bool = True
    despeckle_size: int = 400
    refiner_scale: float = 1.0
    # Output config
    fg_enabled: bool = True
    fg_format: str = "exr"
    matte_enabled: bool = True
    matte_format: str = "exr"
    comp_enabled: bool = True
    comp_format: str = "png"
    processed_enabled: bool = True
    processed_format: str = "exr"


class VideoMaMaRequest(BaseModel):
    chunk_size: int = 50


# --- App Factory ---


def create_app(clips_dir: str = "ClipsForInference") -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        service = CorridorKeyService()
        device = service.detect_device()
        logger.info(f"Device: {device}")

        clips = service.scan_clips(clips_dir)
        clips_map = {c.name: c for c in clips}

        loop = asyncio.get_running_loop()
        worker = GPUWorker(service, manager.broadcast, loop)
        worker.set_clips(clips_map)

        # Rescan callback — called by worker after extraction completes
        async def rescan_clips():
            new_clips = service.scan_clips(app.state.clips_dir)
            new_map = {c.name: c for c in new_clips}
            app.state.clips = new_map
            worker.set_clips(new_map)
            await manager.broadcast({"type": "clips_updated"})

        worker._rescan_callback = rescan_clips
        worker.start()

        app.state.service = service
        app.state.worker = worker
        app.state.clips = clips_map
        app.state.clips_dir = clips_dir

        # Auto-queue extraction for any clips stuck in EXTRACTING state
        _auto_queue_extractions(app)

        yield

        # Shutdown
        worker.stop()
        service.unload_engines()

    app = FastAPI(title="CorridorKey", lifespan=lifespan)

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    # --- Routes ---

    @app.get("/")
    async def index():
        return FileResponse(str(WEB_DIR / "templates" / "index.html"))

    @app.get("/api/device")
    async def get_device():
        service: CorridorKeyService = app.state.service
        vram = service.get_vram_info()
        return {
            "device": service._device,
            "vram": vram,
        }

    @app.get("/api/clips")
    async def get_clips():
        clips_map: dict = app.state.clips
        return [_serialize_clip(c) for c in clips_map.values()]

    @app.post("/api/clips/scan")
    async def scan_clips(req: ScanRequest):
        service: CorridorKeyService = app.state.service
        worker: GPUWorker = app.state.worker
        scan_path = req.path or app.state.clips_dir

        if not os.path.isdir(scan_path):
            raise HTTPException(400, f"Directory not found: {scan_path}")

        clips = service.scan_clips(scan_path)
        clips_map = {c.name: c for c in clips}
        app.state.clips = clips_map
        app.state.clips_dir = scan_path
        worker.set_clips(clips_map)

        await manager.broadcast({"type": "clips_updated"})

        # Auto-queue extraction for newly found EXTRACTING clips
        _auto_queue_extractions(app)

        return [_serialize_clip(c) for c in clips_map.values()]

    @app.post("/api/clips/{name}/extract")
    async def queue_extract(name: str):
        """Queue video-to-frames extraction for a standalone video clip."""
        clip = _get_clip(app, name)
        service: CorridorKeyService = app.state.service

        if clip.state != ClipState.EXTRACTING:
            raise HTTPException(400, "Clip is not in EXTRACTING state")

        if not clip.input_asset or clip.input_asset.asset_type != "video":
            raise HTTPException(400, "Clip has no video to extract")

        video_path = clip.input_asset.path

        # For standalone videos, create a proper clip folder
        # (standalone videos have root_path == clips_dir, not a subfolder)
        clip_folder = os.path.join(app.state.clips_dir, name)
        input_dir = os.path.join(clip_folder, "Input")
        os.makedirs(input_dir, exist_ok=True)

        params = {"video_path": video_path, "out_dir": input_dir}
        job = GPUJob(JobType.VIDEO_EXTRACT, name, params=params)
        ok = service.job_queue.submit(job)
        if not ok:
            raise HTTPException(409, "Duplicate job — already queued or running")
        return {"job_id": job.id, "queued": True}

    @app.post("/api/clips/{name}/inference")
    async def queue_inference(name: str, req: InferenceRequest):
        clip = _get_clip(app, name)
        service: CorridorKeyService = app.state.service

        params_dict = {
            "input_is_linear": req.input_is_linear,
            "despill_strength": req.despill_strength,
            "auto_despeckle": req.auto_despeckle,
            "despeckle_size": req.despeckle_size,
            "refiner_scale": req.refiner_scale,
            "output_config": {
                "fg_enabled": req.fg_enabled,
                "fg_format": req.fg_format,
                "matte_enabled": req.matte_enabled,
                "matte_format": req.matte_format,
                "comp_enabled": req.comp_enabled,
                "comp_format": req.comp_format,
                "processed_enabled": req.processed_enabled,
                "processed_format": req.processed_format,
            },
        }

        job = GPUJob(JobType.INFERENCE, name, params=params_dict)
        ok = service.job_queue.submit(job)
        if not ok:
            raise HTTPException(409, "Duplicate job — already queued or running")
        return {"job_id": job.id, "queued": True}

    @app.post("/api/clips/{name}/gvm")
    async def queue_gvm(name: str):
        clip = _get_clip(app, name)
        service: CorridorKeyService = app.state.service

        job = GPUJob(JobType.GVM_ALPHA, name)
        ok = service.job_queue.submit(job)
        if not ok:
            raise HTTPException(409, "Duplicate job — already queued or running")
        return {"job_id": job.id, "queued": True}

    @app.post("/api/clips/{name}/videomama")
    async def queue_videomama(name: str, req: VideoMaMaRequest | None = None):
        clip = _get_clip(app, name)
        service: CorridorKeyService = app.state.service

        params = {"chunk_size": req.chunk_size if req else 50}
        job = GPUJob(JobType.VIDEOMAMA_ALPHA, name, params=params)
        ok = service.job_queue.submit(job)
        if not ok:
            raise HTTPException(409, "Duplicate job — already queued or running")
        return {"job_id": job.id, "queued": True}

    @app.get("/api/jobs")
    async def get_jobs():
        service: CorridorKeyService = app.state.service
        jobs = service.job_queue.all_jobs_snapshot
        return [_serialize_job(j) for j in jobs]

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: str):
        service: CorridorKeyService = app.state.service
        job = service.job_queue.find_job_by_id(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        service.job_queue.cancel_job(job)
        return {"cancelled": True}

    @app.post("/api/jobs/clear")
    async def clear_job_history():
        service: CorridorKeyService = app.state.service
        service.job_queue.clear_history()
        return {"cleared": True}

    @app.get("/api/clips/{name}/thumbnail")
    async def clip_thumbnail(name: str):
        clip = _get_clip(app, name)
        data = get_or_create_thumbnail(clip)
        if data is None:
            raise HTTPException(404, "No thumbnail available")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/clips/{name}/preview/{frame}")
    async def clip_preview(name: str, frame: int, layer: str = "comp"):
        clip = _get_clip(app, name)
        data = get_frame_preview(clip, frame, layer)
        if data is None:
            raise HTTPException(404, "Frame not available")
        return Response(content=data, media_type="image/jpeg")

    @app.post("/api/clips/delete-outputs")
    async def delete_clip(req: ClipNameRequest):
        """Delete a clip's output files (keeps input + alpha intact)."""
        import shutil

        name = req.name
        clip = _get_clip(app, name)
        service: CorridorKeyService = app.state.service
        worker: GPUWorker = app.state.worker

        # Cancel any running/queued jobs for this clip
        for job in service.job_queue.all_jobs_snapshot:
            if job.clip_name == name and job.status.value in ("queued", "running"):
                service.job_queue.cancel_job(job)

        # Delete the Output directory (keeps input + alpha intact)
        output_dir = os.path.join(clip.root_path, "Output")
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        # Clear thumbnail cache
        thumb_dir = os.path.join(clip.root_path, ".thumbnails")
        if os.path.isdir(thumb_dir):
            shutil.rmtree(thumb_dir)

        # Re-scan to refresh state
        clips = service.scan_clips(app.state.clips_dir)
        clips_map = {c.name: c for c in clips}
        app.state.clips = clips_map
        worker.set_clips(clips_map)

        await manager.broadcast({"type": "clips_updated"})
        return {"deleted": True, "clip": name}

    @app.post("/api/clips/delete-all")
    async def delete_clip_entirely(req: ClipNameRequest):
        """Delete an entire clip folder from disk."""
        import shutil

        name = req.name
        clip = _get_clip(app, name)
        service: CorridorKeyService = app.state.service
        worker: GPUWorker = app.state.worker

        # Cancel any running/queued jobs for this clip
        for job in service.job_queue.all_jobs_snapshot:
            if job.clip_name == name and job.status.value in ("queued", "running"):
                service.job_queue.cancel_job(job)

        # Delete entire clip folder
        if os.path.isdir(clip.root_path):
            shutil.rmtree(clip.root_path)

        # Also delete the standalone video file if it exists
        clips_dir = app.state.clips_dir
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"):
            video_file = os.path.join(clips_dir, name + ext)
            if os.path.isfile(video_file):
                os.remove(video_file)
                break

        # Re-scan
        clips = service.scan_clips(app.state.clips_dir)
        clips_map = {c.name: c for c in clips}
        app.state.clips = clips_map
        worker.set_clips(clips_map)

        await manager.broadcast({"type": "clips_updated"})
        return {"deleted": True, "clip": name}

    @app.post("/api/clips/add-file")
    async def add_file(req: AddFileRequest):
        """Add a single video file as a clip and queue extraction."""
        service: CorridorKeyService = app.state.service
        worker: GPUWorker = app.state.worker
        file_path = req.file_path

        if not os.path.isfile(file_path):
            raise HTTPException(400, f"File not found: {file_path}")
        if not _is_video_or_image(file_path):
            raise HTTPException(400, "Not a supported video/image file")

        # Use the parent directory as clips_dir
        parent_dir = os.path.dirname(file_path)
        stem = os.path.splitext(os.path.basename(file_path))[0]

        # Scan the parent dir (will pick up this file as a standalone video)
        clips = service.scan_clips(parent_dir)
        clips_map = {c.name: c for c in clips}
        app.state.clips = clips_map
        app.state.clips_dir = parent_dir
        worker.set_clips(clips_map)

        # Auto-queue extraction
        _auto_queue_extractions(app)

        await manager.broadcast({"type": "clips_updated"})
        return {
            "added": True,
            "clip": stem,
            "clips_dir": parent_dir,
            "total_clips": len(clips_map),
        }

    @app.post("/api/clips/import-alpha")
    async def import_alpha(req: ImportAlphaRequest):
        """Import an alpha hint from a folder of images or a video file.

        Symlinks (or copies) the frames into the clip's AlphaHint/ directory,
        then rescans so the clip transitions RAW → READY.
        """
        import shutil

        clip = _get_clip(app, req.clip_name)
        service: CorridorKeyService = app.state.service
        worker: GPUWorker = app.state.worker
        alpha_path = req.alpha_path

        if not os.path.exists(alpha_path):
            raise HTTPException(400, f"Path not found: {alpha_path}")

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        if os.path.isdir(alpha_path):
            # It's a folder of images — symlink each file
            files = sorted(f for f in os.listdir(alpha_path) if _is_video_or_image(f))
            if not files:
                raise HTTPException(400, "No image files found in that folder")
            for f in files:
                src = os.path.join(alpha_path, f)
                dst = os.path.join(alpha_dir, f)
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
            count = len(files)
        elif os.path.isfile(alpha_path) and _is_video_or_image(alpha_path):
            # Single video or image — for video, extract; for image, copy
            ext = os.path.splitext(alpha_path)[1].lower()
            if ext in _VIDEO_EXTS:
                # Queue extraction to AlphaHint dir
                from backend.ffmpeg_tools import extract_frames
                count = extract_frames(alpha_path, alpha_dir)
            else:
                # Single image — copy it
                dst = os.path.join(alpha_dir, os.path.basename(alpha_path))
                shutil.copy2(alpha_path, dst)
                count = 1
        else:
            raise HTTPException(400, "Not a valid image/video path")

        # Re-scan to pick up the new alpha
        clips = service.scan_clips(app.state.clips_dir)
        clips_map = {c.name: c for c in clips}
        app.state.clips = clips_map
        worker.set_clips(clips_map)

        await manager.broadcast({"type": "clips_updated"})
        return {"imported": True, "clip": req.clip_name, "alpha_frames": count}

    @app.post("/api/clips/open-output")
    async def open_output(req: ClipNameRequest):
        """Open the clip's Output/FG folder in Finder."""
        import subprocess

        clip = _get_clip(app, req.name)
        fg_dir = os.path.join(clip.root_path, "Output", "FG")
        if not os.path.isdir(fg_dir):
            # Fall back to Output dir
            fg_dir = os.path.join(clip.root_path, "Output")
        if not os.path.isdir(fg_dir):
            raise HTTPException(404, "No output folder found")
        subprocess.Popen(["open", fg_dir])
        return {"opened": fg_dir}

    @app.post("/api/clips/copy-output")
    async def copy_output(req: ClipNameRequest):
        """Copy the FG output to a sibling folder next to the clip for easy access."""
        import shutil

        clip = _get_clip(app, req.name)
        fg_dir = os.path.join(clip.root_path, "Output", "FG")
        if not os.path.isdir(fg_dir):
            raise HTTPException(404, "No FG output found — has processing completed?")

        # Create sibling folder: <clip_name>_KEYED next to the clip folder
        parent = os.path.dirname(clip.root_path)
        dest_name = clip.name + "_KEYED"
        dest_dir = os.path.join(parent, dest_name)
        os.makedirs(dest_dir, exist_ok=True)

        count = 0
        for f in sorted(os.listdir(fg_dir)):
            src = os.path.join(fg_dir, f)
            dst = os.path.join(dest_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                count += 1

        return {"copied": True, "destination": dest_dir, "files_copied": count}

    @app.post("/api/unload")
    async def unload_engines():
        service: CorridorKeyService = app.state.service
        service.unload_engines()
        return {"unloaded": True}

    # --- File Browser ---

    @app.get("/api/browse")
    async def browse_directory(path: str = "~"):
        """List directories and video files at a given path for the folder picker."""
        expanded = os.path.expanduser(path)
        if not os.path.isdir(expanded):
            raise HTTPException(400, f"Not a directory: {path}")

        items = []
        try:
            for name in sorted(os.listdir(expanded)):
                if name.startswith("."):
                    continue
                full = os.path.join(expanded, name)
                if os.path.isdir(full):
                    items.append({"name": name, "type": "dir", "path": full})
                elif _is_video_or_image(name):
                    items.append({"name": name, "type": "file", "path": full})
        except PermissionError:
            raise HTTPException(403, "Permission denied")

        parent = os.path.dirname(expanded)
        return {
            "current": expanded,
            "parent": parent if parent != expanded else None,
            "items": items,
        }

    # --- WebSocket ---

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await manager.connect(ws)
        try:
            while True:
                # Keep connection alive; client can send pings
                await ws.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(ws)

    return app


# --- Helpers ---


def _auto_queue_extractions(app: FastAPI) -> None:
    """Auto-queue VIDEO_EXTRACT jobs for any clips stuck in EXTRACTING state."""
    clips_map: dict = app.state.clips
    service: CorridorKeyService = app.state.service

    for clip in clips_map.values():
        if clip.state != ClipState.EXTRACTING:
            continue
        if not clip.input_asset or clip.input_asset.asset_type != "video":
            continue

        video_path = clip.input_asset.path
        clip_folder = os.path.join(app.state.clips_dir, clip.name)
        input_dir = os.path.join(clip_folder, "Input")
        os.makedirs(input_dir, exist_ok=True)

        params = {"video_path": video_path, "out_dir": input_dir}
        job = GPUJob(JobType.VIDEO_EXTRACT, clip.name, params=params)
        service.job_queue.submit(job)
        logger.info(f"Auto-queued extraction for '{clip.name}'")


def _get_clip(app: FastAPI, name: str):
    clips_map: dict = app.state.clips
    clip = clips_map.get(name)
    if clip is None:
        raise HTTPException(404, f"Clip '{name}' not found")
    return clip


def _serialize_clip(clip) -> dict:
    frame_count = 0
    if clip.input_asset:
        frame_count = clip.input_asset.frame_count

    return {
        "name": clip.name,
        "state": clip.state.value,
        "frame_count": frame_count,
        "has_outputs": clip.has_outputs,
        "completed_frames": clip.completed_frame_count(),
        "warnings": clip.warnings,
        "error_message": clip.error_message,
        "has_alpha": clip.alpha_asset is not None,
        "has_mask": clip.mask_asset is not None,
    }


_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".dpx"}


def _is_video_or_image(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in _VIDEO_EXTS or ext in _IMAGE_EXTS


def _serialize_job(job: GPUJob) -> dict:
    return {
        "id": job.id,
        "job_type": job.job_type.value,
        "clip_name": job.clip_name,
        "status": job.status.value,
        "current_frame": job.current_frame,
        "total_frames": job.total_frames,
        "error_message": job.error_message,
    }
