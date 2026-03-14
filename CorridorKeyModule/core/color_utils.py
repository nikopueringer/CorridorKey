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


def linear_to_srgb(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts Linear to sRGB using the official piecewise sRGB transfer function.
    Supports both Numpy arrays and PyTorch tensors.
    """
    x = _clamp(x, 0.0)
    mask = x <= 0.0031308
    return _where(mask, x * 12.92, 1.055 * _power(x, 1.0 / 2.4) - 0.055)


def srgb_to_linear(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts sRGB to Linear using the official piecewise sRGB transfer function.
    Supports both Numpy arrays and PyTorch tensors.
    """
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

    tensor = _is_tensor(image)
    _maximum = _if_tensor(tensor, torch.max, np.maximum)
    _stack = _if_tensor(tensor, _torch_stack, _numpy_stack)

    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]

    if green_limit_mode == "max":
        limit = _maximum(r, b)
    else:
        limit = (r + b) / 2.0

    if isinstance(image, torch.Tensor):
        # PyTorch Impl — g/limit are Tensor since image is Tensor
        diff: torch.Tensor = g - limit  # type: ignore[assignment]
        spill_amount = torch.clamp(diff, min=0.0)
    else:
        # Numpy Impl
        spill_amount = np.maximum(g - limit, 0.0)

    g_new = g - spill_amount
    r_new = r + (spill_amount * 0.5)
    b_new = b + (spill_amount * 0.5)

    despilled = _stack([r_new, g_new, b_new])

    if strength < 1.0:
        return image * (1.0 - strength) + despilled * strength

    return despilled


def clean_matte(alpha_np: np.ndarray, area_threshold: int = 300, dilation: int = 15, blur_size: int = 5) -> np.ndarray:
    """
    Cleans up small disconnected components (like tracking markers) from a predicted alpha matte.
    alpha_np: Numpy array [H, W] or [H, W, 1] float (0.0 - 1.0)
    """
    # Needs to be 2D
    is_3d = False
    if alpha_np.ndim == 3:
        is_3d = True
        alpha_np = alpha_np[:, :, 0]

    # Threshold to binary
    mask_8u = (alpha_np > 0.5).astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)

    # Create an empty mask for the cleaned components
    cleaned_mask = np.zeros_like(mask_8u)

    # Keep components larger than the threshold (skip label 0, which is background)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            cleaned_mask[labels == i] = 255

    # Dilate
    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned_mask = cv2.dilate(cleaned_mask, kernel)

    # Blur
    if blur_size > 0:
        b_size = int(blur_size * 2 + 1)
        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (b_size, b_size), 0)

    # Convert back to 0-1 float
    safe_zone = cleaned_mask.astype(np.float32) / 255.0

    # Multiply original alpha by the safe zone
    result_alpha = alpha_np * safe_zone

    if is_3d:
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
