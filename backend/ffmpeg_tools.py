"""FFmpeg subprocess wrapper for video extraction and stitching.

Pure Python, no Qt deps. Provides:
- find_ffmpeg() / find_ffprobe() — locate binaries
- probe_video() — get fps, resolution, frame count, codec
- extract_frames() — video -> image sequence (PNG)
- stitch_video() — image sequence -> video (H.264)
- write/read_video_metadata() — sidecar JSON for roundtrip fidelity
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
from typing import Callable

logger = logging.getLogger(__name__)

_METADATA_FILENAME = ".video_metadata.json"

# Common install locations on Windows
_FFMPEG_SEARCH_PATHS = [
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
    r"C:\ffmpeg\bin",
]


def find_ffmpeg() -> str | None:
    """Locate ffmpeg binary. Checks PATH then common install dirs."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for d in _FFMPEG_SEARCH_PATHS:
        candidate = os.path.join(d, "ffmpeg.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def find_ffprobe() -> str | None:
    """Locate ffprobe binary. Checks PATH then common install dirs."""
    found = shutil.which("ffprobe")
    if found:
        return found
    for d in _FFMPEG_SEARCH_PATHS:
        candidate = os.path.join(d, "ffprobe.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def probe_video(path: str) -> dict:
    """Probe a video file for metadata.

    Returns dict with keys: fps (float), width (int), height (int),
    frame_count (int), codec (str), duration (float).
    Raises RuntimeError if ffprobe fails.
    """
    ffprobe = find_ffprobe()
    if not ffprobe:
        raise RuntimeError("ffprobe not found")

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

    # Find first video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise RuntimeError(f"No video stream found in {path}")

    # Parse fps from r_frame_rate (e.g. "24000/1001")
    fps_str = video_stream.get("r_frame_rate", "24/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 24.0
    else:
        fps = float(fps_str)

    # Frame count: prefer nb_frames, fall back to duration * fps
    frame_count = 0
    if "nb_frames" in video_stream:
        try:
            frame_count = int(video_stream["nb_frames"])
        except (ValueError, TypeError):
            pass

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
    """Extract video frames to PNG image sequence.

    Args:
        video_path: Path to input video file.
        out_dir: Directory to write frames into (created if needed).
        pattern: Frame filename pattern (FFmpeg style).
        on_progress: Callback(current_frame, total_frames).
        cancel_event: Set to cancel extraction.
        total_frames: Expected total (for progress). Probed if 0.

    Returns:
        Number of frames extracted.

    Raises:
        RuntimeError if ffmpeg is not found or extraction fails.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    os.makedirs(out_dir, exist_ok=True)

    # Probe for total if not provided
    video_info = None
    if total_frames <= 0:
        try:
            video_info = probe_video(video_path)
            total_frames = video_info.get("frame_count", 0)
        except Exception:
            total_frames = 0

    # Resume: detect existing frames and skip ahead with conservative rollback.
    # Delete the last few frames (may be corrupt from mid-write or FFmpeg
    # output buffering) and re-extract from that point.
    _RESUME_ROLLBACK = 3  # frames to re-extract for safety
    start_frame = 0
    existing = sorted([f for f in os.listdir(out_dir) if f.lower().endswith(".png")])
    if existing:
        # Remove the last N frames — they may be corrupt or incomplete
        remove_count = min(_RESUME_ROLLBACK, len(existing))
        for fname in existing[-remove_count:]:
            os.remove(os.path.join(out_dir, fname))
        start_frame = max(0, len(existing) - remove_count)
        if start_frame > 0:
            logger.info(
                f"Resuming extraction from frame {start_frame} ({len(existing)} existed, rolled back {remove_count})"
            )

    if start_frame > 0 and total_frames > 0:
        # Seek to the resume point
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

    logger.info(f"Extracting frames: {video_path} -> {out_dir} (start_frame={start_frame})")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    last_frame = start_frame
    frame_re = re.compile(r"frame=\s*(\d+)")

    # Read stderr in a background thread so cancel checks aren't blocked
    import queue as _queue

    line_q: _queue.Queue[str | None] = _queue.Queue()

    def _reader():
        for ln in proc.stderr:
            line_q.put(ln)
        line_q.put(None)  # sentinel

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        while True:
            # Check cancellation every 0.2s even if no output
            if cancel_event and cancel_event.is_set():
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                logger.info("Extraction cancelled — FFmpeg killed")
                return last_frame

            try:
                line = line_q.get(timeout=0.2)
            except _queue.Empty:
                # No output yet — check if process is still alive
                if proc.poll() is not None:
                    break
                continue

            if line is None:
                break  # stderr closed — process ending

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

    # Count actual extracted frames
    extracted = len([f for f in os.listdir(out_dir) if f.lower().endswith(".png")])
    logger.info(f"Extracted {extracted} frames to {out_dir}")
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
    """Stitch image sequence back into a video file.

    Args:
        in_dir: Directory containing frame images.
        out_path: Output video file path.
        fps: Frame rate.
        pattern: Frame filename pattern.
        codec: Video codec (libx264, libx265, etc.).
        crf: Quality (0-51, lower = better).
        on_progress: Callback(current_frame, total_frames).
        cancel_event: Set to cancel stitching.

    Raises:
        RuntimeError if ffmpeg is not found or stitching fails.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    # Count total frames
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

    logger.info(f"Stitching video: {in_dir} -> {out_path}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

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

    logger.info(f"Video stitched: {out_path}")


def write_video_metadata(clip_root: str, metadata: dict) -> None:
    """Write video metadata sidecar JSON to clip root.

    Metadata typically includes: source_path, fps, width, height,
    frame_count, codec, duration.
    """
    path = os.path.join(clip_root, _METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Video metadata written: {path}")


def read_video_metadata(clip_root: str) -> dict | None:
    """Read video metadata sidecar from clip root. Returns None if not found."""
    path = os.path.join(clip_root, _METADATA_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to read video metadata: {e}")
        return None
