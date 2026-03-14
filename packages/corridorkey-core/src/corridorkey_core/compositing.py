"""Image compositing utilities for the CorridorKey keying pipeline.

Provides color space conversion (linear/sRGB), alpha compositing, green spill
removal, matte cleanup, and preview generation. All functions support both
numpy arrays and PyTorch tensors unless otherwise noted.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import cv2
import numpy as np
import torch


def _is_tensor(x: np.ndarray | torch.Tensor) -> bool:
    return isinstance(x, torch.Tensor)


def _if_tensor(is_tensor: bool, tensor_func: Callable, numpy_func: Callable) -> Callable:
    return tensor_func if is_tensor else numpy_func


def _power(x: np.ndarray | torch.Tensor, exponent: float) -> np.ndarray | torch.Tensor:
    # Dispatches to torch.pow or np.power depending on input type.
    power = _if_tensor(_is_tensor(x), torch.pow, np.power)
    return power(x, exponent)


def _where(
    condition: np.ndarray | torch.Tensor, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    # Dispatches to torch.where or np.where depending on input type.
    where = _if_tensor(_is_tensor(x), torch.where, np.where)
    return where(condition, x, y)


def _clamp(x: np.ndarray | torch.Tensor, min: float) -> np.ndarray | torch.Tensor:
    # Clamps values to the given minimum. Dispatches to torch or numpy.
    if isinstance(x, torch.Tensor):
        return x.clamp(min=min)
    return np.clip(x, min, None)


_torch_stack = functools.partial(torch.stack, dim=-1)
_numpy_stack = functools.partial(np.stack, axis=-1)

# sRGB transfer function constants (IEC 61966-2-1)
# Reference: https://www.color.org/chardata/rgb/srgb.xalter
# Linear values at or below this use the linear segment
_SRGB_LINEAR_THRESHOLD = 0.0031308
# Encoded values at or below this use the linear segment (= _SRGB_LINEAR_THRESHOLD * 12.92)
_SRGB_ENCODED_THRESHOLD = 0.04045
# Slope of the linear segment
_SRGB_LINEAR_SCALE = 12.92
# Exponent for the power curve (encoding: linear -> sRGB)
_SRGB_GAMMA = 1.0 / 2.4
# Scale factor for the power curve
_SRGB_SCALE = 1.055
# Offset for the power curve
_SRGB_OFFSET = 0.055


def linear_to_srgb(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert linear light values to sRGB using the IEC 61966-2-1 piecewise transfer function.

    Supports both numpy arrays and PyTorch tensors. Values below zero are clamped.
    """
    x = _clamp(x, 0.0)
    mask = x <= _SRGB_LINEAR_THRESHOLD
    return _where(mask, x * _SRGB_LINEAR_SCALE, _SRGB_SCALE * _power(x, _SRGB_GAMMA) - _SRGB_OFFSET)


def srgb_to_linear(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert sRGB encoded values to linear light using the IEC 61966-2-1 piecewise transfer function.

    Supports both numpy arrays and PyTorch tensors. Values below zero are clamped.
    """
    x = _clamp(x, 0.0)
    mask = x <= _SRGB_ENCODED_THRESHOLD
    return _where(mask, x / _SRGB_LINEAR_SCALE, _power((x + _SRGB_OFFSET) / _SRGB_SCALE, 2.4))


def premultiply(fg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Multiply foreground color by alpha to produce a premultiplied image.

    Args:
        fg: Color array with shape [..., C] or [C, ...].
        alpha: Alpha array with shape [..., 1] or [1, ...].
    """
    return fg * alpha


def composite_straight(
    fg: np.ndarray | torch.Tensor, bg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Composite a straight (unpremultiplied) foreground over a background.

    Formula: FG * Alpha + BG * (1 - Alpha)
    """
    return fg * alpha + bg * (1.0 - alpha)


def composite_premul(
    fg: np.ndarray | torch.Tensor, bg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Composite a premultiplied foreground over a background.

    Formula: FG + BG * (1 - Alpha)
    """
    return fg + bg * (1.0 - alpha)


_VALID_GREEN_LIMIT_MODES = ("average", "max")


def despill(
    image: np.ndarray | torch.Tensor, green_limit_mode: str = "average", strength: float = 1.0
) -> np.ndarray | torch.Tensor:
    """Remove green spill from an RGB image using a luminance-preserving method.

    Excess green is redistributed equally to red and blue channels to neutralize
    the spill without darkening the subject. Output is clamped to [0, 1].

    Args:
        image: RGB float array in range 0-1.
        green_limit_mode: How to compute the green limit. "average" uses (R+B)/2,
            "max" uses max(R, B). Any other value raises ValueError.
        strength: Blend factor between original and despilled result (0.0 to 1.0).
            Values outside [0, 1] are accepted but may produce out-of-range output.

    Raises:
        ValueError: If green_limit_mode is not "average" or "max".
    """
    if green_limit_mode not in _VALID_GREEN_LIMIT_MODES:
        raise ValueError(f"Invalid green_limit_mode '{green_limit_mode}'. Must be one of: {_VALID_GREEN_LIMIT_MODES}")

    if strength <= 0.0:
        return image

    tensor = _is_tensor(image)
    _maximum = _if_tensor(tensor, torch.max, np.maximum)
    _stack = _if_tensor(tensor, _torch_stack, _numpy_stack)

    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]

    green_limit = _maximum(r, b) if green_limit_mode == "max" else (r + b) / 2.0

    if isinstance(image, torch.Tensor):
        green_excess: torch.Tensor = g - green_limit  # type: ignore[assignment]
        spill_amount = torch.clamp(green_excess, min=0.0)
    else:
        spill_amount = np.maximum(g - green_limit, 0.0)

    g_new = g - spill_amount
    r_new = _clamp(r + (spill_amount * 0.5), 0.0)
    b_new = _clamp(b + (spill_amount * 0.5), 0.0)

    # Clamp to [0, 1] - redistribution can push bright channels above 1.0
    if isinstance(image, torch.Tensor):
        r_new = r_new.clamp(max=1.0)  # ty:ignore[unresolved-attribute]
        b_new = b_new.clamp(max=1.0)  # ty:ignore[unresolved-attribute]
    else:
        r_new = np.clip(r_new, 0.0, 1.0)
        b_new = np.clip(b_new, 0.0, 1.0)

    despilled_image = _stack([r_new, g_new, b_new])

    if strength < 1.0:
        return image * (1.0 - strength) + despilled_image * strength

    return despilled_image


def clean_matte(alpha: np.ndarray, area_threshold: int = 300, dilation: int = 15, blur_size: int = 5) -> np.ndarray:
    """Remove small disconnected regions from a predicted alpha matte.

    Useful for eliminating tracking markers, noise islands, or other small
    artifacts that the model incorrectly classified as foreground.

    Args:
        alpha: Float array with shape [H, W] or [H, W, 1] in range 0.0-1.0.
        area_threshold: Minimum pixel area for a connected component to be kept.
        dilation: Radius in pixels to dilate the cleaned mask before blending.
        blur_size: Radius in pixels for Gaussian blur applied after dilation.
    """
    # Needs to be 2D for connected components analysis
    is_3d = False
    if alpha.ndim == 3:
        is_3d = True
        alpha = alpha[:, :, 0]

    # Binarize at 0.5 to get a uint8 mask for OpenCV
    alpha_binary = (alpha > 0.5).astype(np.uint8) * 255

    # Label each connected foreground region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(alpha_binary, connectivity=8)

    foreground_mask = np.zeros_like(alpha_binary)

    # Keep regions above the area threshold; label 0 is background and is always skipped
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            foreground_mask[labels == i] = 255

    # Dilate to recover edges lost during binarization
    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        foreground_mask = cv2.dilate(foreground_mask, kernel)

    # Blur to soften the hard edges of the cleaned mask
    if blur_size > 0:
        blur_kernel_size = int(blur_size * 2 + 1)
        foreground_mask = cv2.GaussianBlur(foreground_mask, (blur_kernel_size, blur_kernel_size), 0)

    keep_mask = foreground_mask.astype(np.float32) / 255.0

    # Multiply the original soft alpha by the keep_mask to zero out removed regions
    alpha_cleaned = alpha * keep_mask

    if is_3d:
        alpha_cleaned = alpha_cleaned[:, :, np.newaxis]

    return alpha_cleaned


def create_checkerboard(
    width: int, height: int, checker_size: int = 64, color1: float = 0.2, color2: float = 0.4
) -> np.ndarray:
    """Create a grayscale checkerboard pattern for compositing previews.

    Values are in linear light (not gamma-encoded). Convert with srgb_to_linear
    before use if your pipeline expects linear input.

    Args:
        width: Output width in pixels.
        height: Output height in pixels.
        checker_size: Side length of each checker tile in pixels.
        color1: Linear brightness of the dark tiles (0.0-1.0).
        color2: Linear brightness of the light tiles (0.0-1.0).

    Returns:
        Float array with shape [H, W, 3] in range 0.0-1.0.
    """
    x = np.arange(width)
    y = np.arange(height)

    x_tiles = x // checker_size
    y_tiles = y // checker_size

    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)

    # Even sum = color1, odd sum = color2
    checker = (x_grid + y_grid) % 2

    checker_pattern = np.where(checker == 0, color1, color2).astype(np.float32)

    return np.stack([checker_pattern, checker_pattern, checker_pattern], axis=-1)
