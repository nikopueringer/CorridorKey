"""Preview endpoint — serves frames as PNG, preview videos as MP4, and downloads as ZIP."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
import threading
import zipfile

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from backend.frame_io import read_image_frame
from backend.natural_sort import natsorted
from backend.project import is_image_file

from ..deps import get_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/preview", tags=["preview"])

_clips_dir: str = ""
# Cache dir for stitched preview videos
_cache_dir: str = ""


def set_clips_dir(path: str) -> None:
    global _clips_dir, _cache_dir
    _clips_dir = path
    _cache_dir = os.path.join(path, ".cache", "preview_videos")
    os.makedirs(_cache_dir, exist_ok=True)


_PASS_MAP = {
    "input": "Input",
    "frames": "Frames",
    "alpha": "AlphaHint",
    "fg": "Output/FG",
    "matte": "Output/Matte",
    "comp": "Output/Comp",
    "processed": "Output/Processed",
}


def _find_clip_root(clip_name: str) -> str | None:
    service = get_service()
    clips = service.scan_clips(_clips_dir)
    for clip in clips:
        if clip.name == clip_name:
            return clip.root_path
    return None


def _resolve_pass_dir(clip_root: str, pass_name: str) -> str:
    """Resolve the directory for a pass, handling input/frames fallback."""
    if pass_name == "input":
        frames_dir = os.path.join(clip_root, "Frames")
        input_dir = os.path.join(clip_root, "Input")
        if os.path.isdir(frames_dir) and os.listdir(frames_dir):
            return frames_dir
        elif os.path.isdir(input_dir):
            return input_dir
        raise HTTPException(status_code=404, detail="No input frames directory found")
    target = os.path.join(clip_root, _PASS_MAP[pass_name])
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail=f"Directory not found: {_PASS_MAP[pass_name]}")
    return target


def _frame_to_png_bytes(img: np.ndarray) -> bytes:
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] >= 3:
        img_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    elif img.ndim == 2:
        img_bgr = img
    else:
        img_bgr = img
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


# --- Single frame preview ---


@router.get("/{clip_name}/{pass_name}/{frame:int}")
def get_preview_frame(clip_name: str, pass_name: str, frame: int):
    if pass_name not in _PASS_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown pass: {pass_name}. Valid: {list(_PASS_MAP.keys())}")

    clip_root = _find_clip_root(clip_name)
    if clip_root is None:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_name}' not found")

    target_dir = _resolve_pass_dir(clip_root, pass_name)
    files = natsorted([f for f in os.listdir(target_dir) if is_image_file(f)])
    if not files:
        raise HTTPException(status_code=404, detail=f"No frames in {pass_name}")
    if frame < 0 or frame >= len(files):
        raise HTTPException(status_code=404, detail=f"Frame {frame} out of range (0-{len(files) - 1})")

    fpath = os.path.join(target_dir, files[frame])

    if pass_name == "matte":
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        img = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=500, detail=f"Failed to read {fpath}")
        if img.ndim == 3:
            img = img[:, :, 0]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        success, buf = cv2.imencode(".png", img)
        if not success:
            raise HTTPException(status_code=500, detail="PNG encode failed")
        return Response(content=buf.tobytes(), media_type="image/png")

    img = read_image_frame(fpath)
    if img is None:
        raise HTTPException(status_code=500, detail=f"Failed to read {fpath}")

    return Response(content=_frame_to_png_bytes(img), media_type="image/png")


# --- Video preview (stitched MP4) ---

# Lock to prevent concurrent ffmpeg encodes for the same cache key
_encode_locks: dict[str, threading.Lock] = {}
_encode_locks_lock = threading.Lock()


def _get_encode_lock(key: str) -> threading.Lock:
    with _encode_locks_lock:
        if key not in _encode_locks:
            _encode_locks[key] = threading.Lock()
        return _encode_locks[key]


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cache_key(clip_root: str, pass_name: str) -> str:
    """Generate a cache key based on directory path and modification time."""
    target_dir = _resolve_pass_dir(clip_root, pass_name)
    files = [f for f in os.listdir(target_dir) if is_image_file(f)]
    # Hash based on dir path, file count, and newest mtime
    newest = max((os.path.getmtime(os.path.join(target_dir, f)) for f in files), default=0)
    raw = f"{target_dir}:{len(files)}:{newest}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


@router.get("/{clip_name}/{pass_name}/video")
def get_preview_video(clip_name: str, pass_name: str, fps: int = 24):
    """Stitch frames into an MP4 for smooth browser playback. Cached."""
    if not _ffmpeg_available():
        raise HTTPException(status_code=503, detail="ffmpeg not available — cannot generate preview video")

    if pass_name not in _PASS_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown pass: {pass_name}")

    clip_root = _find_clip_root(clip_name)
    if clip_root is None:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_name}' not found")

    target_dir = _resolve_pass_dir(clip_root, pass_name)
    files = natsorted([f for f in os.listdir(target_dir) if is_image_file(f)])
    if not files:
        raise HTTPException(status_code=404, detail=f"No frames in {pass_name}")

    # Check cache
    key = _cache_key(clip_root, pass_name)
    cache_path = os.path.join(_cache_dir, f"{clip_name}_{pass_name}_{key}.mp4")

    if os.path.isfile(cache_path):
        return FileResponse(cache_path, media_type="video/mp4", filename=f"{clip_name}_{pass_name}.mp4")

    # Serialize encodes per cache key to prevent duplicate ffmpeg processes
    encode_lock = _get_encode_lock(key)
    if not encode_lock.acquire(timeout=0.1):
        # Another thread is encoding this exact video — wait for it
        encode_lock.acquire()
        encode_lock.release()
        if os.path.isfile(cache_path):
            return FileResponse(cache_path, media_type="video/mp4", filename=f"{clip_name}_{pass_name}.mp4")
        raise HTTPException(status_code=500, detail="Concurrent encode failed")

    concat_path = os.path.join(_cache_dir, f"{key}_concat.txt")
    try:
        with open(concat_path, "w") as f:
            for fname in files:
                fpath = os.path.join(target_dir, fname)
                # For EXR files, we need to convert first — ffmpeg may not handle them well
                # Use a glob pattern if filenames are sequential, otherwise concat
                f.write(f"file '{fpath}'\n")
                f.write(f"duration {1 / fps}\n")

        # Check if files are EXR (ffmpeg needs special handling)
        is_exr = files[0].lower().endswith(".exr")

        if is_exr:
            # Convert via OpenCV → temp PNGs → ffmpeg
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, fname in enumerate(files):
                    fpath = os.path.join(target_dir, fname)
                    img = read_image_frame(fpath)
                    if img is not None:
                        out = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(tmpdir, f"{i:06d}.png"), out_bgr)

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(tmpdir, "%06d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "23",
                    "-movflags",
                    "+faststart",
                    cache_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=300)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode()[-300:])
        else:
            # Direct ffmpeg from image files
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                cache_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode()[-300:])

    except Exception as e:
        logger.error(f"Video stitch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create preview video: {e}") from e
    finally:
        encode_lock.release()
        if os.path.isfile(concat_path):
            os.unlink(concat_path)

    return FileResponse(cache_path, media_type="video/mp4", filename=f"{clip_name}_{pass_name}.mp4")


# --- Download (ZIP) ---


@router.get("/{clip_name}/{pass_name}/download")
def download_pass(clip_name: str, pass_name: str):
    """Download all frames for a pass as a ZIP file."""
    if pass_name not in _PASS_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown pass: {pass_name}")

    clip_root = _find_clip_root(clip_name)
    if clip_root is None:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_name}' not found")

    target_dir = _resolve_pass_dir(clip_root, pass_name)
    files = natsorted(os.listdir(target_dir))
    files = [f for f in files if not f.startswith(".")]
    if not files:
        raise HTTPException(status_code=404, detail=f"No files in {pass_name}")

    zip_name = f"{clip_name}_{pass_name}.zip"

    # Build ZIP to a temp file (streaming partial ZIPs produces corrupt files)
    zip_path = os.path.join(_cache_dir, f"dl_{zip_name}")
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in files:
                fpath = os.path.join(target_dir, fname)
                zf.write(fpath, arcname=os.path.join(pass_name, fname))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {e}") from e

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_name,
        background=None,  # don't delete after send — cached
    )
