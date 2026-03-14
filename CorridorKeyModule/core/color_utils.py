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
    """
    Power function that supports both Numpy arrays and PyTorch tensors.
    """
    power = _if_tensor(_is_tensor(x), torch.pow, np.power)
    return power(x, exponent)


def _where(
    condition: np.ndarray | torch.Tensor, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Where function that supports both Numpy arrays and PyTorch tensors.
    """
    where = _if_tensor(_is_tensor(x), torch.where, np.where)
    return where(condition, x, y)


def _clamp(x: np.ndarray | torch.Tensor, min: float) -> np.ndarray | torch.Tensor:
    """
    Clamp function that supports both Numpy arrays and PyTorch tensors.
    """
    if isinstance(x, torch.Tensor):
        return x.clamp(min=0.0)
    return np.clip(x, 0.0, None)


_torch_stack = functools.partial(torch.stack, dim=-1)
_numpy_stack = functools.partial(np.stack, axis=-1)


_SRGB_LUT_SIZE = 65536


@functools.lru_cache(maxsize=1)
def _linear_to_srgb_lut() -> np.ndarray:
    """Build a LUT for linear→sRGB conversion. Cached on first use."""
    t = np.linspace(0.0, 1.0, _SRGB_LUT_SIZE, dtype=np.float64)
    result = np.where(t <= 0.0031308, t * 12.92, 1.055 * np.power(t, 1.0 / 2.4) - 0.055)
    return result.astype(np.float32)


@functools.lru_cache(maxsize=1)
def _srgb_to_linear_lut() -> np.ndarray:
    """Build a LUT for sRGB→linear conversion. Cached on first use."""
    t = np.linspace(0.0, 1.0, _SRGB_LUT_SIZE, dtype=np.float64)
    result = np.where(t <= 0.04045, t / 12.92, np.power((t + 0.055) / 1.055, 2.4))
    return result.astype(np.float32)


def _apply_lut(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a [0,1]→[0,1] LUT to a float32 array via quantized index lookup.

    Avoids expensive per-pixel np.power calls by quantizing float values into
    LUT indices. Max error vs analytic sRGB: ~0.0002 (imperceptible).
    """
    # Quantize continuous [0,1] floats into discrete LUT bin indices
    quantized_indices = np.clip(values * (_SRGB_LUT_SIZE - 1), 0, _SRGB_LUT_SIZE - 1).astype(np.uint16)
    return lut[quantized_indices]


def linear_to_srgb(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts Linear to sRGB using the official piecewise sRGB transfer function.
    Supports both Numpy arrays and PyTorch tensors.
    Uses LUT acceleration for numpy arrays (avoids np.power).
    """
    if not _is_tensor(x):
        return _apply_lut(np.clip(x, 0.0, 1.0, dtype=np.float32), _linear_to_srgb_lut())
    x = _clamp(x, 0.0)
    mask = x <= 0.0031308
    return _where(mask, x * 12.92, 1.055 * _power(x, 1.0 / 2.4) - 0.055)


def srgb_to_linear(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts sRGB to Linear using the official piecewise sRGB transfer function.
    Supports both Numpy arrays and PyTorch tensors.
    Uses LUT acceleration for numpy arrays (avoids np.power).
    """
    if not _is_tensor(x):
        return _apply_lut(np.clip(x, 0.0, 1.0, dtype=np.float32), _srgb_to_linear_lut())
    x = _clamp(x, 0.0)
    mask = x <= 0.04045
    return _where(mask, x / 12.92, _power((x + 0.055) / 1.055, 2.4))


def premultiply(fg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Premultiplies foreground by alpha.
    fg: Color [..., C] or [C, ...]
    alpha: Alpha [..., 1] or [1, ...]
    """
    return fg * alpha


def unpremultiply(
    fg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor, eps: float = 1e-6
) -> np.ndarray | torch.Tensor:
    """
    Un-premultiplies foreground by alpha.
    Ref: fg_straight = fg_premul / (alpha + eps)
    """
    return fg / (alpha + eps)


def composite_straight(
    fg: np.ndarray | torch.Tensor, bg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Composites Straight FG over BG.
    Formula: FG * Alpha + BG * (1 - Alpha)
    """
    return fg * alpha + bg * (1.0 - alpha)


def composite_premul(
    fg: np.ndarray | torch.Tensor, bg: np.ndarray | torch.Tensor, alpha: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Composites Premultiplied FG over BG.
    Formula: FG + BG * (1 - Alpha)
    """
    return fg + bg * (1.0 - alpha)


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    """
    Converts RGB to YUV (Rec. 601).
    Input: [..., 3, H, W] or [..., 3] depending on layout.
    Supports standard PyTorch BCHW.
    """
    if not _is_tensor(image):
        raise TypeError("rgb_to_yuv only supports dict/tensor inputs currently")

    # Weights for RGB -> Y
    # Rec. 601: 0.299, 0.587, 0.114

    # Assume BCHW layout if 4 dims
    if image.dim() == 4:
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
    elif image.dim() == 3 and image.shape[0] == 3:  # CHW
        r = image[0:1, :, :]
        g = image[1:2, :, :]
        b = image[2:3, :, :]
    else:
        # Last dim conversion
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)

    if image.dim() >= 3 and image.shape[-3] == 3:  # Concatenate along Channel dim
        return torch.cat([y, u, v], dim=-3)
    else:
        return torch.stack([y, u, v], dim=-1)


def dilate_mask(mask: np.ndarray | torch.Tensor, radius: int) -> np.ndarray | torch.Tensor:
    """
    Dilates a mask by a given radius.
    Supports Numpy (using cv2) and PyTorch (using MaxPool).
    radius: Int (pixels). 0 = No change.
    """
    if radius <= 0:
        return mask

    kernel_size = int(radius * 2 + 1)

    if isinstance(mask, torch.Tensor):
        # PyTorch Dilation (using Max Pooling)
        # Expects [B, C, H, W]
        orig_dim = mask.dim()

        if orig_dim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 3:
            mask = mask.unsqueeze(0)

        padding = radius
        dilated = torch.nn.functional.max_pool2d(mask, kernel_size, stride=1, padding=padding)

        if orig_dim == 2:
            return dilated.squeeze()
        elif orig_dim == 3:
            return dilated.squeeze(0)
        return dilated

    # Numpy Dilation (using OpenCV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel)


def apply_garbage_matte(
    predicted_matte: np.ndarray | torch.Tensor,
    garbage_matte_input: np.ndarray | torch.Tensor | None,
    dilation: int = 10,
) -> np.ndarray | torch.Tensor:
    """
    Multiplies predicted matte by a dilated garbage matte to clean up background.
    """
    if garbage_matte_input is None:
        return predicted_matte

    garbage_mask = dilate_mask(garbage_matte_input, dilation)

    # Ensure dimensions match for multiplication
    if _is_tensor(predicted_matte):
        # Handle broadcasting if needed
        pass
    elif garbage_mask.ndim == 2 and predicted_matte.ndim == 3:
        # Numpy
        garbage_mask = garbage_mask[:, :, np.newaxis]

    return predicted_matte * garbage_mask


def despill(
    image: np.ndarray | torch.Tensor, green_limit_mode: str = "average", strength: float = 1.0
) -> np.ndarray | torch.Tensor:
    """
    Removes green spill from an RGB image using a luminance-preserving method.
    image: RGB float (0-1).
    green_limit_mode: 'average' ((R+B)/2) or 'max' (max(R, B)).
    strength: 0.0 to 1.0 multiplier for the despill effect.
    """
    if strength <= 0.0:
        return image

    is_torch = _is_tensor(image)
    _maximum = _if_tensor(is_torch, torch.max, np.maximum)
    _stack = _if_tensor(is_torch, _torch_stack, _numpy_stack)

    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]

    # Green limit: the maximum green value that isn't considered "spill".
    # Anything above this threshold is excess green reflected from the screen.
    if green_limit_mode == "max":
        green_limit = _maximum(r, b)
    else:
        green_limit = (r + b) / 2.0

    # Spill amount: how much green exceeds the limit (clamped to non-negative)
    if isinstance(image, torch.Tensor):
        green_excess: torch.Tensor = g - green_limit  # type: ignore[assignment]
        spill_amount = torch.clamp(green_excess, min=0.0)
    else:
        spill_amount = np.maximum(g - green_limit, 0.0)

    # Redistribute spill energy: subtract from green, split evenly into R and B
    # to preserve overall luminance while removing the green cast
    g_corrected = g - spill_amount
    r_corrected = r + (spill_amount * 0.5)
    b_corrected = b + (spill_amount * 0.5)

    despilled = _stack([r_corrected, g_corrected, b_corrected])

    # Partial strength: blend between original and fully despilled
    if strength < 1.0:
        return image * (1.0 - strength) + despilled * strength

    return despilled


def clean_matte(alpha_np: np.ndarray, area_threshold: int = 300, dilation: int = 15, blur_size: int = 5) -> np.ndarray:
    """
    Cleans up small disconnected components (like tracking markers) from a predicted alpha matte.
    alpha_np: Numpy array [H, W] or [H, W, 1] float (0.0 - 1.0)
    """
    had_channel_dim = False
    if alpha_np.ndim == 3:
        had_channel_dim = True
        alpha_np = alpha_np[:, :, 0]

    # Threshold to binary for connected component analysis
    binary_mask = (alpha_np > 0.5).astype(np.uint8) * 255

    # Find connected components — each isolated region gets a unique label
    num_labels, label_map, stats, _centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Vectorized label filtering: build a lookup table marking which labels to keep.
    # Label 0 is always background; labels with area >= threshold are real subject.
    # Small islands (tracking markers, noise) get zeroed out.
    component_areas = stats[:, cv2.CC_STAT_AREA]
    keep_label = np.zeros(num_labels, dtype=np.uint8)
    keep_label[1:] = (component_areas[1:] >= area_threshold).astype(np.uint8)
    cleaned_mask = (keep_label[label_map] * 255).astype(np.uint8)

    # Dilate the cleaned mask to create a "safe zone" that extends slightly
    # beyond the subject edge — prevents the mask from clipping real detail
    if dilation > 0:
        # Large kernels are O(k²) per pixel. Iterating with small kernels
        # achieves equivalent coverage at 4-5x less cost (benchmarked).
        MAX_SINGLE_KERNEL = 11
        full_kernel_size = int(dilation * 2 + 1)
        if full_kernel_size > MAX_SINGLE_KERNEL:
            small_radius = MAX_SINGLE_KERNEL // 2  # 5px per iteration
            num_iterations = (dilation + small_radius - 1) // small_radius
            small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MAX_SINGLE_KERNEL, MAX_SINGLE_KERNEL))
            for _ in range(num_iterations):
                cleaned_mask = cv2.dilate(cleaned_mask, small_kernel)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (full_kernel_size, full_kernel_size))
            cleaned_mask = cv2.dilate(cleaned_mask, kernel)

    # Soft edge: blur the binary safe zone to avoid hard cutoff artifacts
    if blur_size > 0:
        blur_kernel_size = int(blur_size * 2 + 1)
        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (blur_kernel_size, blur_kernel_size), 0)

    # The safe zone acts as a multiplier: 1.0 inside subject region, 0.0 outside,
    # with a soft gradient at the boundary from the blur step
    safe_zone = cleaned_mask.astype(np.float32) / 255.0
    result_alpha = alpha_np * safe_zone

    if had_channel_dim:
        result_alpha = result_alpha[:, :, np.newaxis]

    return result_alpha


def create_checkerboard(
    width: int, height: int, checker_size: int = 64, color1: float = 0.2, color2: float = 0.4
) -> np.ndarray:
    """
    Creates a linear grayscale checkerboard pattern.
    Returns: Numpy array [H, W, 3] float (0.0-1.0)
    """
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)

    # Determine tile parity
    x_tiles = x // checker_size
    y_tiles = y // checker_size

    # Broadcast to 2D
    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)

    # XOR for checker pattern (1 if odd, 0 if even)
    checker = (x_grid + y_grid) % 2

    # Map 0 to color1 and 1 to color2
    bg_img = np.where(checker == 0, color1, color2).astype(np.float32)

    # Make it 3-channel
    return np.stack([bg_img, bg_img, bg_img], axis=-1)
