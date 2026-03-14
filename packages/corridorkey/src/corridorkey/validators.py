"""Validation utilities for frame processing.

All validators either return cleaned data or raise typed exceptions from errors.py.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .errors import (
    FrameMismatchError,
    FrameReadError,
    MaskChannelError,
    WriteFailureError,
)

logger = logging.getLogger(__name__)


def validate_frame_counts(
    clip_name: str,
    input_count: int,
    alpha_count: int,
    strict: bool = False,
) -> int:
    """Validate that input and alpha frame counts are compatible.

    Args:
        clip_name: For error messages.
        input_count: Number of input frames.
        alpha_count: Number of alpha frames.
        strict: If True, raises on mismatch. If False, logs warning and returns min.

    Returns:
        The number of frames to process (min of both).

    Raises:
        FrameMismatchError: If strict=True and counts differ.
    """
    if input_count != alpha_count:
        if strict:
            raise FrameMismatchError(clip_name, input_count, alpha_count)
        logger.warning(
            f"Clip '{clip_name}': frame count mismatch - "
            f"input has {input_count}, alpha has {alpha_count}. "
            f"Truncating to {min(input_count, alpha_count)}."
        )
    return min(input_count, alpha_count)


def normalize_mask_channels(
    mask: np.ndarray,
    clip_name: str = "",
    frame_index: int = 0,
) -> np.ndarray:
    """Reduce a mask to a single-channel 2D array.

    Handles any channel count: extracts first channel from multi-channel masks.

    Args:
        mask: Input mask array, any shape [H, W] or [H, W, C].
        clip_name: For error messages.
        frame_index: For error messages.

    Returns:
        2D numpy array [H, W] with float32 values.
    """
    if mask.ndim == 3:
        if mask.shape[2] == 0:
            raise MaskChannelError(clip_name, frame_index, 0)
        # Always extract first channel regardless of channel count
        mask = mask[:, :, 0]
    elif mask.ndim != 2:
        raise MaskChannelError(clip_name, frame_index, mask.ndim)

    return mask.astype(np.float32) if mask.dtype != np.float32 else mask


def normalize_mask_dtype(mask: np.ndarray) -> np.ndarray:
    """Convert mask to float32 [0.0, 1.0] from any common dtype."""
    if mask.dtype == np.uint8:
        return mask.astype(np.float32) / 255.0
    elif mask.dtype == np.uint16:
        return mask.astype(np.float32) / 65535.0
    elif mask.dtype == np.float64:
        return mask.astype(np.float32)
    elif mask.dtype == np.float32:
        return mask
    else:
        return mask.astype(np.float32)


def validate_frame_read(
    frame: np.ndarray | None,
    clip_name: str,
    frame_index: int,
    path: str,
) -> np.ndarray:
    """Validate that a frame was read successfully.

    Args:
        frame: The result of cv2.imread() - None if read failed.
        clip_name: For error messages.
        frame_index: For error messages.
        path: File path that was read.

    Returns:
        The frame array (unchanged).

    Raises:
        FrameReadError: If frame is None.
    """
    if frame is None:
        raise FrameReadError(clip_name, frame_index, path)
    return frame


def validate_write(
    success: bool,
    clip_name: str,
    frame_index: int,
    path: str,
) -> None:
    """Validate that a cv2.imwrite() call succeeded.

    Args:
        success: Return value of cv2.imwrite().
        clip_name: For error messages.
        frame_index: For error messages.
        path: File path that was written.

    Raises:
        WriteFailureError: If success is False.
    """
    if not success:
        raise WriteFailureError(clip_name, frame_index, path)


def ensure_output_dirs(clip_root: str) -> dict[str, str]:
    """Create output subdirectories for a clip and return their paths.

    Returns:
        Dict with keys: 'root', 'fg', 'matte', 'comp', 'processed'
    """
    out_root = os.path.join(clip_root, "Output")
    dirs = {
        "root": out_root,
        "fg": os.path.join(out_root, "FG"),
        "matte": os.path.join(out_root, "Matte"),
        "comp": os.path.join(out_root, "Comp"),
        "processed": os.path.join(out_root, "Processed"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs
