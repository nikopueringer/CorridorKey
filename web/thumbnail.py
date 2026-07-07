"""Thumbnail and frame preview generation for the web UI."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from backend.frame_io import read_image_frame, read_video_frame_at

logger = logging.getLogger(__name__)

THUMB_WIDTH = 320
PREVIEW_MAX_WIDTH = 1280
JPEG_QUALITY = 80


def _resize_to_width(img: np.ndarray, max_width: int) -> np.ndarray:
    """Resize image to fit max_width, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _float_to_uint8_bgr(img_rgb_float: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,1] to uint8 BGR for cv2 encoding."""
    img = np.clip(img_rgb_float, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _encode_jpeg(img_bgr: np.ndarray) -> bytes:
    """Encode BGR uint8 image to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return b""
    return buf.tobytes()


def get_or_create_thumbnail(clip) -> bytes | None:
    """Get or create a cached JPEG thumbnail for a clip.

    Args:
        clip: A ClipEntry with input_asset.

    Returns:
        JPEG bytes, or None if no input available.
    """
    if clip.input_asset is None:
        return None

    # Check cache
    cache_dir = os.path.join(clip.root_path, ".thumbnails")
    cache_path = os.path.join(cache_dir, "thumb.jpg")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return f.read()
        except OSError:
            pass

    # Generate from middle frame
    mid = max(0, clip.input_asset.frame_count // 2)
    img = None

    if clip.input_asset.asset_type == "video":
        img = read_video_frame_at(clip.input_asset.path, mid)
    else:
        files = clip.input_asset.get_frame_files()
        if files:
            idx = min(mid, len(files) - 1)
            fpath = os.path.join(clip.input_asset.path, files[idx])
            img = read_image_frame(fpath)

    if img is None:
        return None

    bgr = _float_to_uint8_bgr(img)
    bgr = _resize_to_width(bgr, THUMB_WIDTH)
    jpeg = _encode_jpeg(bgr)

    if jpeg:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(jpeg)
        except OSError as e:
            logger.debug(f"Failed to cache thumbnail: {e}")

    return jpeg or None


def get_frame_preview(clip, frame_index: int, layer: str = "comp") -> bytes | None:
    """Get a JPEG preview of a specific frame and layer.

    Args:
        clip: A ClipEntry.
        frame_index: Zero-based frame index.
        layer: One of 'input', 'comp', 'fg', 'matte', 'alpha'.

    Returns:
        JPEG bytes, or None if not available.
    """
    img = None

    if layer == "input":
        img = _read_input_frame(clip, frame_index)
    elif layer == "alpha":
        img = _read_alpha_frame(clip, frame_index)
    else:
        # Read from Output directory
        dir_map = {"comp": "Comp", "fg": "FG", "matte": "Matte", "processed": "Processed"}
        subdir = dir_map.get(layer, "Comp")
        output_dir = os.path.join(clip.root_path, "Output", subdir)

        if os.path.isdir(output_dir):
            from backend.natural_sort import natsorted
            from backend.project import is_image_file

            files = natsorted([f for f in os.listdir(output_dir) if is_image_file(f)])
            if frame_index < len(files):
                fpath = os.path.join(output_dir, files[frame_index])
                img = read_image_frame(fpath)

    if img is None:
        return None

    # Grayscale matte → convert to 3-channel for JPEG
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    bgr = _float_to_uint8_bgr(img)
    bgr = _resize_to_width(bgr, PREVIEW_MAX_WIDTH)
    return _encode_jpeg(bgr)


def _read_input_frame(clip, frame_index: int) -> np.ndarray | None:
    """Read an input frame from clip's input asset."""
    if clip.input_asset is None:
        return None
    if clip.input_asset.asset_type == "video":
        return read_video_frame_at(clip.input_asset.path, frame_index)
    files = clip.input_asset.get_frame_files()
    if frame_index >= len(files):
        return None
    fpath = os.path.join(clip.input_asset.path, files[frame_index])
    return read_image_frame(fpath)


def _read_alpha_frame(clip, frame_index: int) -> np.ndarray | None:
    """Read an alpha hint frame."""
    if clip.alpha_asset is None:
        return None
    if clip.alpha_asset.asset_type == "video":
        return read_video_frame_at(clip.alpha_asset.path, frame_index)
    files = clip.alpha_asset.get_frame_files()
    if frame_index >= len(files):
        return None
    fpath = os.path.join(clip.alpha_asset.path, files[frame_index])
    return read_image_frame(fpath)
