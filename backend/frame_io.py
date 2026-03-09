"""Unified frame I/O — read images and video frames as float32 RGB.

All reading functions return float32 arrays in [0, 1] range with RGB channel
order. EXR files are read as-is (linear float); standard formats (PNG, JPG,
etc.) are normalized from uint8.

This module consolidates frame-reading patterns that were previously duplicated
across service.py methods (_read_input_frame, reprocess_single_frame,
_load_frames_for_videomama, _load_mask_frames_for_videomama).
"""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from .validators import normalize_mask_channels, normalize_mask_dtype

logger = logging.getLogger(__name__)

# EXR write flags — PXR24 half-float (smallest working compression)
EXR_WRITE_FLAGS = [
    cv2.IMWRITE_EXR_TYPE,
    cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION,
    cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]


def read_image_frame(fpath: str, gamma_correct_exr: bool = False) -> np.ndarray | None:
    """Read an image file (EXR or standard) as float32 RGB [0, 1].

    Args:
        fpath: Absolute path to image file.
        gamma_correct_exr: If True, apply gamma 1/2.2 to EXR data
            (converts linear → approximate sRGB for models expecting sRGB).

    Returns:
        float32 array [H, W, 3] in RGB order, or None if read fails.
    """
    is_exr = fpath.lower().endswith(".exr")

    if is_exr:
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Could not read frame: %s", fpath)
            return None
        # Strip alpha channel from BGRA EXR
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = np.maximum(img_rgb, 0.0).astype(np.float32)
        if gamma_correct_exr:
            result = np.power(result, 1.0 / 2.2).astype(np.float32)
        return result
    else:
        img = cv2.imread(fpath)
        if img is None:
            logger.warning("Could not read frame: %s", fpath)
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb.astype(np.float32) / 255.0


def read_video_frame_at(
    video_path: str,
    frame_index: int,
) -> np.ndarray | None:
    """Read a single frame from a video by index, as float32 RGB [0, 1].

    Args:
        video_path: Path to video file.
        frame_index: Zero-based frame index to seek to.

    Returns:
        float32 array [H, W, 3] in RGB order, or None if seek/read fails.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    finally:
        cap.release()


def read_video_frames(
    video_path: str,
    processor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Read all frames from a video, optionally applying a processor to each.

    Without a processor, frames are returned as float32 RGB [0, 1].

    Args:
        video_path: Path to video file.
        processor: Optional callable (BGR uint8 frame) → processed array.
            If None, default conversion to float32 RGB [0, 1] is applied.

    Returns:
        List of processed frames.
    """
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if processor is not None:
                frames.append(processor(frame))
            else:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(img_rgb)
    finally:
        cap.release()
    return frames


def read_mask_frame(fpath: str, clip_name: str = "", frame_index: int = 0) -> np.ndarray | None:
    """Read a mask frame as float32 [H, W] in [0, 1].

    Handles any channel count and dtype via normalize_mask_channels/dtype.

    Args:
        fpath: Path to mask image.
        clip_name: For error context in normalization.
        frame_index: For error context in normalization.

    Returns:
        float32 array [H, W] in [0, 1], or None if read fails.
    """
    mask_in = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if mask_in is None:
        return None
    # dtype normalization MUST happen before channel extraction, because
    # normalize_mask_channels casts to float32 — which would make a uint8
    # 255 into float32 255.0, skipping the /255 division in normalize_mask_dtype.
    mask = normalize_mask_dtype(mask_in)
    mask = normalize_mask_channels(mask, clip_name, frame_index)
    return mask


def read_video_mask_at(
    video_path: str,
    frame_index: int,
) -> np.ndarray | None:
    """Read a single mask frame from a video by index, as float32 [H, W] [0, 1].

    Extracts the blue channel (index 2) from BGR, matching the convention
    used by alpha-channel video masks.

    Args:
        video_path: Path to video file.
        frame_index: Zero-based frame index.

    Returns:
        float32 array [H, W] in [0, 1], or None if seek/read fails.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame[:, :, 2].astype(np.float32) / 255.0
    finally:
        cap.release()
