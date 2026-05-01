from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

logger = logging.getLogger(__name__)


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


def despill_opencv(
    image: np.ndarray | torch.Tensor,
    limit_mode: str = "average",
    strength: float = 1.0,
    screen_channel: int = 1,
    *,
    green_limit_mode: str | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Removes screen-color spill from an RGB image using a luminance-preserving method.
    The algorithm is channel-agnostic: it subtracts excess in the screen channel
    relative to the other two channels and redistributes the removed energy.

    image: RGB float (0-1).
    limit_mode: 'average' ((other_a + other_b)/2) or 'max' (max(other_a, other_b)).
    strength: 0.0 to 1.0 multiplier for the despill effect.
    screen_channel: 0=R, 1=G (default — green screen), 2=B (blue screen).
    green_limit_mode: deprecated alias for limit_mode kept for backward compat.
    """
    if green_limit_mode is not None:
        import warnings

        warnings.warn(
            "despill_opencv(green_limit_mode=...) is deprecated; pass limit_mode=... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        limit_mode = green_limit_mode
    if screen_channel not in (0, 1, 2):
        raise ValueError(f"screen_channel must be 0, 1, or 2, got {screen_channel}")
    if strength <= 0.0:
        return image

    tensor = _is_tensor(image)
    _maximum = _if_tensor(tensor, torch.max, np.maximum)
    _stack = _if_tensor(tensor, _torch_stack, _numpy_stack)

    other_a, other_b = (i for i in (0, 1, 2) if i != screen_channel)
    screen = image[..., screen_channel]
    a = image[..., other_a]
    b = image[..., other_b]

    if limit_mode == "max":
        limit = _maximum(a, b)
    else:
        limit = (a + b) / 2.0

    if isinstance(image, torch.Tensor):
        # PyTorch Impl
        diff: torch.Tensor = screen - limit  # type: ignore[assignment]
        spill_amount = torch.clamp(diff, min=0.0)
    else:
        # Numpy Impl
        spill_amount = np.maximum(screen - limit, 0.0)

    screen_new = screen - spill_amount
    a_new = a + (spill_amount * 0.5)
    b_new = b + (spill_amount * 0.5)

    out_channels = [None, None, None]
    out_channels[screen_channel] = screen_new
    out_channels[other_a] = a_new
    out_channels[other_b] = b_new
    despilled = _stack(out_channels)

    if strength < 1.0:
        return image * (1.0 - strength) + despilled * strength

    return despilled


def despill_torch(image: torch.Tensor, strength: float, screen_channel: int = 1) -> torch.Tensor:
    """GPU despill — keeps data on device. screen_channel: 0=R, 1=G, 2=B."""
    if screen_channel not in (0, 1, 2):
        raise ValueError(f"screen_channel must be 0, 1, or 2, got {screen_channel}")
    if strength <= 0.0:
        return image
    other_a, other_b = (i for i in (0, 1, 2) if i != screen_channel)
    screen = image[:, screen_channel]
    a = image[:, other_a]
    b = image[:, other_b]
    limit = (a + b) / 2.0
    spill = torch.clamp(screen - limit, min=0.0)
    screen_new = screen - spill
    a_new = a + spill * 0.5
    b_new = b + spill * 0.5
    out_channels: list[torch.Tensor] = [None, None, None]  # type: ignore[list-item]
    out_channels[screen_channel] = screen_new
    out_channels[other_a] = a_new
    out_channels[other_b] = b_new
    despilled = torch.stack(out_channels, dim=1)
    if strength < 1.0:
        return image * (1.0 - strength) + despilled * strength
    return despilled


# --- Screen color: single source of truth ---------------------------------
# All callers (CLI, settings dataclasses, service layer, checkpoint discovery,
# despill pipelines) import these constants instead of hard-coding the strings.
# Adding a new screen color requires only updating SCREEN_CHANNEL_BY_COLOR.

SCREEN_CHANNEL_BY_COLOR: dict[str, int] = {"green": 1, "blue": 2}
SCREEN_COLOR_CHOICES: tuple[str, ...] = tuple(SCREEN_CHANNEL_BY_COLOR.keys())
SCREEN_COLOR_AUTO: str = "auto"
SCREEN_COLOR_CHOICES_WITH_AUTO: tuple[str, ...] = (SCREEN_COLOR_AUTO,) + SCREEN_COLOR_CHOICES


def screen_channel_for_color(screen_color: str) -> int:
    """Map a screen-color name ("green"/"blue") to its RGB channel index.

    Raises ValueError on unknown values (including "auto" — callers must
    resolve auto to a concrete color before calling this).
    """
    try:
        return SCREEN_CHANNEL_BY_COLOR[screen_color]
    except KeyError:
        raise ValueError(f"Unknown screen_color '{screen_color}'. Valid: {', '.join(SCREEN_COLOR_CHOICES)}") from None


def estimate_screen_color(image_srgb: np.ndarray, alpha_hint: np.ndarray, ambiguity_threshold: float = 0.05) -> str:
    """Detect dominant screen color from an image + coarse alpha hint.

    Looks at pixels where ``alpha_hint < 0.3`` (i.e. the background = the screen)
    and compares mean green vs mean blue. Returns 'green' or 'blue'.

    Falls back to 'green' (with a logger.warning) when:
      - the background region is too small (<1% of pixels), or
      - the green/blue means differ by less than ``ambiguity_threshold``.

    image_srgb: [H, W, 3] float (0-1) sRGB.
    alpha_hint: [H, W] or [H, W, 1] float (0-1), high = foreground subject.
    """
    if image_srgb.ndim != 3 or image_srgb.shape[2] < 3:
        raise ValueError(f"estimate_screen_color expects HxWx3 image, got shape {image_srgb.shape}")
    if alpha_hint.ndim not in (2, 3):
        raise ValueError(f"estimate_screen_color expects HxW or HxWx1 alpha_hint, got shape {alpha_hint.shape}")
    if alpha_hint.ndim == 3:
        alpha_hint = alpha_hint[..., 0]
    if alpha_hint.shape[:2] != image_srgb.shape[:2]:
        raise ValueError(
            f"image_srgb and alpha_hint must agree on H,W: got {image_srgb.shape[:2]} vs {alpha_hint.shape[:2]}"
        )

    bg_mask = alpha_hint < 0.3
    coverage = float(bg_mask.mean()) if bg_mask.size else 0.0
    if coverage < 0.01:
        logger.warning(
            "estimate_screen_color: background region too small (%.2f%% of pixels) — defaulting to green",
            coverage * 100.0,
        )
        return "green"

    bg_pixels = image_srgb[bg_mask]
    mean_g = float(bg_pixels[:, 1].mean())
    mean_b = float(bg_pixels[:, 2].mean())

    if abs(mean_g - mean_b) < ambiguity_threshold:
        logger.warning(
            "estimate_screen_color: green/blue means too close (G=%.3f, B=%.3f) — defaulting to green",
            mean_g,
            mean_b,
        )
        return "green"

    detected = "blue" if mean_b > mean_g else "green"
    logger.info(
        "estimate_screen_color: detected '%s' (background mean G=%.3f, B=%.3f, coverage=%.1f%%)",
        detected,
        mean_g,
        mean_b,
        coverage * 100.0,
    )
    return detected


def connected_components(mask: torch.Tensor, min_component_distance=1, max_iterations=100) -> torch.Tensor:
    """
    Adapted from: https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc
    Args:
        mask: torch Tensor [B, 1, H, W] binary 1 or 0
        min_component_distance: int. Minimum distance between connected components that are separated instead of merged.
        max_iterations: int. Maximum number of flood fill iterations. Adjust based on expected component sizes.
    Returns:
        comp: torch Tensor [B, 1, H, W] with connected component labels (0 = background, 1..N = components)
    """
    bs, _, H, W = mask.shape

    # Reference implementation uses torch.arange instead of torch.randperm
    # torch.randperm converges considerably faster and more uniformly
    # If the batch size is >2 at 4k, float32 can't exactly represent all pixel indices (only up to 2^24)
    # We add 0.1 to ensure all floats get floored to unique integers
    comp = (torch.randperm(bs * W * H, device=mask.device, dtype=torch.float32) + 1.1).view(mask.shape)
    comp[mask != 1] = 0

    for _ in range(max_iterations):
        comp[mask == 1] = F.max_pool2d(
            comp, kernel_size=(2 * min_component_distance) + 1, stride=1, padding=min_component_distance
        )[mask == 1]

    comp = comp.long()
    # Relabel components to have contiguous labels starting from 1
    unique_labels = torch.unique(comp)
    # Add background label (0) if not present
    if unique_labels[0] != 0:
        unique_labels = torch.cat([torch.tensor([0], device=mask.device), unique_labels])
    label_map = torch.zeros(unique_labels.max().item() + 1, dtype=torch.long, device=mask.device)
    label_map[unique_labels] = torch.arange(len(unique_labels), device=mask.device)
    comp = label_map[comp]

    return comp


def clean_matte_opencv(
    alpha_np: np.ndarray, area_threshold: int = 300, dilation: int = 15, blur_size: int = 5
) -> np.ndarray:
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


def clean_matte_torch(alpha: torch.Tensor, area_threshold: int, dilation: int = 15, blur_size: int = 5) -> torch.Tensor:
    """
    Cleans up small disconnected components (like tracking markers) from a predicted alpha matte.
    Supports fully running on the GPU
    alpha_np: torch Tensor [B, 1, H, W] (0.0 - 1.0)
    """
    mask = alpha > 0.25  # [B, 1, H, W]

    # Find the largest connected components in the mask
    # only a limited amount of iterations is needed to find components above the area threshold
    components = connected_components(mask, max_iterations=area_threshold // 20, min_component_distance=4)

    # We can use bincount even for batched inputs because the areas are uniquely labeled across the entire batch
    sizes = torch.bincount(components.flatten())
    big_sizes = torch.nonzero(sizes >= area_threshold)

    mask = torch.zeros_like(mask, dtype=torch.float32)
    # Remove background label (0) if present
    big_sizes = big_sizes[big_sizes > 0]
    mask[torch.isin(components, big_sizes)] = 1.0

    # Dilate back to restore edges of large regions
    if dilation > 0:
        # How many applications with kernel size 5 are needed to achieve the desired dilation radius
        repeats = dilation // 2
        for _ in range(repeats):
            mask = F.max_pool2d(mask, 5, stride=1, padding=2)

    # Blur for soft edges
    if blur_size > 0:
        k = int(blur_size * 2 + 1)
        mask = TF.gaussian_blur(mask, [k, k])

    return alpha * mask


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


@functools.lru_cache(maxsize=4)
def get_checkerboard_linear_torch(w: int, h: int, device: torch.device) -> torch.Tensor:
    """Return a cached checkerboard tensor [3, H, W] on device in linear space."""
    checker_size = 128
    y_coords = torch.arange(h, device=device) // checker_size
    x_coords = torch.arange(w, device=device) // checker_size
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    checker = ((x_grid + y_grid) % 2).float()
    # Map 0 -> 0.15, 1 -> 0.55 (sRGB), then convert to linear before caching
    bg_srgb = checker * 0.4 + 0.15  # [H, W]
    bg_srgb_3 = bg_srgb.unsqueeze(0).expand(3, -1, -1)
    return srgb_to_linear(bg_srgb_3)
