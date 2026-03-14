"""FFmpeg subprocess wrapper for video extraction and stitching.

Pure Python, no Qt dependencies. Provides:

- require_ffmpeg / require_ffprobe: locate binaries or raise FFmpegNotFoundError
- check_ffmpeg: return version and path info (used by the doctor command)
- probe_video: get fps, resolution, frame count, codec
- extract_frames: video to image sequence (PNG)
- stitch_video: image sequence to video (H.264)
- write_video_metadata / read_video_metadata: sidecar JSON for roundtrip fidelity

FFmpeg is treated as a system dependency. Install it via your platform package
manager and ensure it is on PATH before using any function in this module.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable

from corridorkey.errors import FFmpegNotFoundError

logger = logging.getLogger(__name__)

# Filename used for the video metadata sidecar JSON.
_METADATA_FILENAME = ".video_metadata.json"


def require_ffmpeg() -> str:
    """Return the path to the ffmpeg binary or raise FFmpegNotFoundError.

    Uses PATH only - no platform-specific search paths. Install FFmpeg
    system-wide and ensure it is on PATH before calling this.

    Returns:
        Absolute path to the ffmpeg binary.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not found on PATH.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError("ffmpeg")
    return path


def require_ffprobe() -> str:
    """Return the path to the ffprobe binary or raise FFmpegNotFoundError.

    Uses PATH only. ffprobe ships alongside ffmpeg in all standard installs.

    Returns:
        Absolute path to the ffprobe binary.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found on PATH.
    """
    path = shutil.which("ffprobe")
    if path is None:
        raise FFmpegNotFoundError("ffprobe")
    return path


def check_ffmpeg() -> dict[str, str | bool]:
    """Return FFmpeg availability and version info for diagnostics.

    Intended for use by the ``corridorkey doctor`` CLI command. Does not
    raise - returns a dict that callers can inspect and display.

    Returns:
        Dict with keys:
            available (bool): True if ffmpeg is on PATH.
            ffmpeg_path (str): Absolute path, or empty string if missing.
            ffprobe_path (str): Absolute path, or empty string if missing.
            version (str): Version string from ``ffmpeg -version``, or
                empty string if unavailable.
    """
    ffmpeg_path = shutil.which("ffmpeg") or ""
    ffprobe_path = shutil.which("ffprobe") or ""
    version = ""

    if ffmpeg_path:
        try:
            result = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            first_line = result.stdout.splitlines()[0] if result.stdout else ""
            # e.g. "ffmpeg version 6.1.1 Copyright ..."
            match = re.search(r"ffmpeg version (\S+)", first_line)
            version = match.group(1) if match else first_line
        except Exception as e:
            logger.debug("FFmpeg version check failed: %s", e)

    return {
        "available": bool(ffmpeg_path),
        "ffmpeg_path": ffmpeg_path,
        "ffprobe_path": ffprobe_path,
        "version": version,
    }


def probe_video(path: str) -> dict:
    """Probe a video file for metadata using ffprobe.

    Args:
        path: Path to the video file.

    Returns:
        Dict with keys: fps (float), width (int), height (int),
        frame_count (int), codec (str), duration (float).

    Raises:
        RuntimeError: If ffprobe is not found or the probe fails.
    """
    ffprobe = require_ffprobe()

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:500]}")

    data = json.loads(result.stdout)

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise RuntimeError(f"No video stream found in {path}")

    fps_str = video_stream.get("r_frame_rate", "24/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 24.0
    else:
        fps = float(fps_str)

    frame_count = 0
    if "nb_frames" in video_stream:
        with contextlib.suppress(ValueError, TypeError):
            frame_count = int(video_stream["nb_frames"])

    if frame_count <= 0:
        duration = float(video_stream.get("duration", 0) or data.get("format", {}).get("duration", 0))
        if duration > 0:
            frame_count = int(duration * fps)

    return {
        "fps": round(fps, 4),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "frame_count": frame_count,
        "codec": video_stream.get("codec_name", "unknown"),
        "duration": float(video_stream.get("duration", 0) or data.get("format", {}).get("duration", 0)),
    }


def extract_frames(
    video_path: str,
    out_dir: str,
    pattern: str = "frame_%06d.png",
    on_progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    total_frames: int = 0,
) -> int:
    """Extract video frames to a PNG image sequence.

    Supports resuming: existing frames are detected and the last few are
    re-extracted (conservative rollback) to guard against partial writes.

    Args:
        video_path: Path to the input video file.
        out_dir: Directory to write frames into (created if needed).
        pattern: Frame filename pattern in FFmpeg style.
        on_progress: Callback(current_frame, total_frames).
        cancel_event: Set this event to cancel extraction.
        total_frames: Expected total frame count. Probed automatically when 0.

    Returns:
        Number of frames present in out_dir after extraction.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not on PATH.
        RuntimeError: If extraction fails or times out.
    """
    ffmpeg = require_ffmpeg()

    os.makedirs(out_dir, exist_ok=True)

    video_info = None
    if total_frames <= 0:
        try:
            video_info = probe_video(video_path)
            total_frames = video_info.get("frame_count", 0)
        except Exception:
            total_frames = 0

    # Number of frames to re-extract at the resume boundary for safety.
    _resume_rollback = 3
    start_frame = 0
    existing = sorted([f for f in os.listdir(out_dir) if f.lower().endswith(".png")])
    if existing:
        remove_count = min(_resume_rollback, len(existing))
        for fname in existing[-remove_count:]:
            os.remove(os.path.join(out_dir, fname))
        start_frame = max(0, len(existing) - remove_count)
        if start_frame > 0:
            logger.info(
                "Resuming extraction from frame %d (%d existed, rolled back %d)",
                start_frame,
                len(existing),
                remove_count,
            )

    if start_frame > 0 and total_frames > 0:
        if video_info is None:
            video_info = probe_video(video_path)
        fps = video_info.get("fps", 24.0)
        seek_sec = start_frame / fps
        cmd = [
            ffmpeg,
            "-ss",
            f"{seek_sec:.4f}",
            "-i",
            video_path,
            "-start_number",
            str(start_frame),
            "-vsync",
            "passthrough",
            os.path.join(out_dir, pattern),
            "-y",
        ]
    else:
        cmd = [
            ffmpeg,
            "-i",
            video_path,
            "-start_number",
            "0",
            "-vsync",
            "passthrough",
            os.path.join(out_dir, pattern),
            "-y",
        ]

    logger.info("Extracting frames: %s -> %s (start_frame=%d)", video_path, out_dir, start_frame)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert proc.stderr is not None  # guaranteed by stderr=subprocess.PIPE

    last_frame = start_frame
    frame_re = re.compile(r"frame=\s*(\d+)")

    import queue as _queue

    line_q: _queue.Queue[str | None] = _queue.Queue()
    stderr = proc.stderr  # capture for closure - ty can't narrow through assert

    def _reader() -> None:
        for ln in stderr:
            line_q.put(ln)
        line_q.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        while True:
            if cancel_event and cancel_event.is_set():
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                logger.info("Extraction cancelled - FFmpeg killed")
                return last_frame

            try:
                line = line_q.get(timeout=0.2)
            except _queue.Empty:
                if proc.poll() is not None:
                    break
                continue

            if line is None:
                break

            match = frame_re.search(line)
            if match:
                last_frame = start_frame + int(match.group(1))
                if on_progress and total_frames > 0:
                    on_progress(last_frame, total_frames)

        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("FFmpeg extraction timed out") from None

    if proc.returncode != 0 and not (cancel_event and cancel_event.is_set()):
        raise RuntimeError(f"FFmpeg extraction failed with code {proc.returncode}")

    extracted = len([f for f in os.listdir(out_dir) if f.lower().endswith(".png")])
    logger.info("Extracted %d frames to %s", extracted, out_dir)
    return extracted


def stitch_video(
    in_dir: str,
    out_path: str,
    fps: float = 24.0,
    pattern: str = "frame_%06d.png",
    codec: str = "libx264",
    crf: int = 18,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Stitch an image sequence back into a video file.

    Args:
        in_dir: Directory containing frame images.
        out_path: Output video file path.
        fps: Frame rate.
        pattern: Frame filename pattern in FFmpeg style.
        codec: Video codec (libx264, libx265, etc.).
        crf: Quality factor (0-51, lower is better).
        on_progress: Callback(current_frame, total_frames).
        cancel_event: Set this event to cancel stitching.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not on PATH.
        RuntimeError: If stitching fails or times out.
    """
    ffmpeg = require_ffmpeg()

    total_frames = len([f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".exr"))])

    cmd = [
        ffmpeg,
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        os.path.join(in_dir, pattern),
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        out_path,
        "-y",
    ]

    logger.info("Stitching video: %s -> %s", in_dir, out_path)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert proc.stderr is not None  # guaranteed by stderr=subprocess.PIPE
    assert proc.stdin is not None  # guaranteed by stdin=subprocess.PIPE

    frame_re = re.compile(r"frame=\s*(\d+)")

    try:
        for line in proc.stderr:
            if cancel_event and cancel_event.is_set():
                try:
                    proc.stdin.write("q\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                proc.wait(timeout=5)
                logger.info("Stitching cancelled")
                return

            match = frame_re.search(line)
            if match:
                current = int(match.group(1))
                if on_progress and total_frames > 0:
                    on_progress(current, total_frames)

        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("FFmpeg stitching timed out") from None

    if proc.returncode != 0 and not (cancel_event and cancel_event.is_set()):
        raise RuntimeError(f"FFmpeg stitching failed with code {proc.returncode}")

    logger.info("Video stitched: %s", out_path)


def write_video_metadata(clip_root: str, metadata: dict) -> None:
    """Write a video metadata sidecar JSON to the clip root.

    Metadata typically includes: source_path, fps, width, height,
    frame_count, codec, duration.

    Args:
        clip_root: Absolute path to the clip folder.
        metadata: Dict to serialise as JSON.
    """
    path = os.path.join(clip_root, _METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.debug("Video metadata written: %s", path)


def read_video_metadata(clip_root: str) -> dict | None:
    """Read the video metadata sidecar from the clip root.

    Args:
        clip_root: Absolute path to the clip folder.

    Returns:
        Parsed metadata dict, or None if the sidecar does not exist.
    """
    path = os.path.join(clip_root, _METADATA_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to read video metadata: %s", e)
        return None
