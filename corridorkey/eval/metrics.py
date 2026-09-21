"""Core evaluation metrics for matting quality assessment.

Provides four foundational metrics:
  - alpha_mae: Mean Absolute Error across all pixels for predicted alpha.
  - foreground_mae: Mean Absolute Error for recovered unmultiplied straight foreground RGB.
  - boundary_alpha_mae: Mean Absolute Error evaluated strictly within the transition boundary
    (fractional alpha region plus a narrow morphological band), preventing easy background
    pixels from masking fine-edge degradation.
  - recomposition_mae: Mean Absolute Error between the observed composite and the recomposed
    prediction C_hat = alpha_hat * F_hat + (1 - alpha_hat) * Screen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_tensor(x: torch.Tensor | float | int) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32)


def compute_alpha_mae(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
) -> float:
    """Mean absolute error for alpha matte.

    Args:
        pred_alpha: Predicted alpha tensor, shape [..., 1, H, W] or [..., H, W], in [0, 1].
        target_alpha: Ground truth alpha tensor, matching spatial dimensions, in [0, 1].

    Returns:
        Scalar MAE as float.
    """
    if pred_alpha.shape != target_alpha.shape:
        if pred_alpha.ndim == target_alpha.ndim + 1 and pred_alpha.shape[-3] == 1:
            pred_alpha = pred_alpha.squeeze(-3)
        elif target_alpha.ndim == pred_alpha.ndim + 1 and target_alpha.shape[-3] == 1:
            target_alpha = target_alpha.squeeze(-3)
        else:
            raise ValueError(f"Shape mismatch: pred_alpha {pred_alpha.shape} vs target_alpha {target_alpha.shape}")

    return float(torch.mean(torch.abs(pred_alpha.float() - target_alpha.float())).item())


def compute_foreground_mae(
    pred_fg: torch.Tensor,
    target_fg: torch.Tensor,
    target_alpha: torch.Tensor | None = None,
    alpha_threshold: float = 0.01,
) -> float:
    """Mean absolute error for straight foreground color.

    When target_alpha is provided, evaluation is restricted to pixels where
    target_alpha > alpha_threshold to focus on visible foreground regions.

    Args:
        pred_fg: Predicted foreground RGB [..., 3, H, W], in [0, 1].
        target_fg: Ground truth foreground RGB [..., 3, H, W], in [0, 1].
        target_alpha: Optional ground truth alpha mask [..., 1, H, W] or [..., H, W].
        alpha_threshold: Threshold below which foreground is ignored if mask is supplied.

    Returns:
        Scalar MAE as float.
    """
    diff = torch.abs(pred_fg.float() - target_fg.float())

    if target_alpha is not None:
        mask = target_alpha.float() > alpha_threshold
        if mask.ndim == diff.ndim - 1:
            mask = mask.unsqueeze(-3)
        mask = mask.expand_as(diff)
        if mask.sum() == 0:
            return 0.0
        return float((diff * mask.float()).sum().item() / mask.sum().item())

    return float(torch.mean(diff).item())


def compute_boundary_mask(
    target_alpha: torch.Tensor,
    kernel_size: int = 5,
    fractional_low: float = 0.01,
    fractional_high: float = 0.99,
) -> torch.Tensor:
    """Derive binary evaluation mask isolating difficult boundary/edge pixels.

    The boundary region includes:
      1. All fractional alpha pixels (fractional_low < alpha < fractional_high).
      2. A morphological transition band around the solid foreground boundary
         (dilated binary mask minus eroded binary mask).

    Args:
        target_alpha: Ground truth alpha tensor [..., H, W] or [..., 1, H, W].
        kernel_size: Size of morphological dilation/erosion structuring element (must be odd).
        fractional_low: Lower bound for fractional alpha inclusion.
        fractional_high: Upper bound for fractional alpha inclusion.

    Returns:
        Boolean tensor of shape [..., 1, H, W] indicating boundary pixels.
    """
    orig_ndim = target_alpha.ndim
    if orig_ndim == 2:
        alpha_4d = target_alpha.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        if target_alpha.shape[0] == 1:
            alpha_4d = target_alpha.unsqueeze(0)
        else:
            alpha_4d = target_alpha.unsqueeze(1)
    elif orig_ndim == 4:
        alpha_4d = target_alpha
    else:
        raise ValueError(f"Unsupported target_alpha ndim: {orig_ndim}")

    alpha_4d = alpha_4d.float()

    # 1. Fractional alpha region
    fractional = (alpha_4d > fractional_low) & (alpha_4d < fractional_high)

    # 2. Morphological boundary band
    binary = (alpha_4d >= 0.5).float()
    padding = kernel_size // 2
    dilated = F.max_pool2d(binary, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-binary, kernel_size=kernel_size, stride=1, padding=padding)
    morph_band = (dilated - eroded) > 0.5

    boundary = fractional | morph_band

    if orig_ndim == 2:
        return boundary.squeeze(0).squeeze(0)
    elif orig_ndim == 3 and target_alpha.shape[0] != 1:
        return boundary.squeeze(1)
    elif orig_ndim == 3:
        return boundary.squeeze(0)
    return boundary


def compute_boundary_alpha_mae(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    kernel_size: int = 5,
    fractional_low: float = 0.01,
    fractional_high: float = 0.99,
) -> float:
    """Mean absolute error of alpha restricted exclusively to the boundary region.

    Prevents models from masking poor edge/hair delineation behind large, trivial
    background/foreground areas.

    Args:
        pred_alpha: Predicted alpha [..., H, W] or [..., 1, H, W].
        target_alpha: Ground truth alpha [..., H, W] or [..., 1, H, W].
        kernel_size: Width of morphological edge band.
        fractional_low: Fractional alpha lower threshold.
        fractional_high: Fractional alpha upper threshold.

    Returns:
        Scalar boundary MAE as float.
    """
    if pred_alpha.ndim != target_alpha.ndim:
        if pred_alpha.ndim == target_alpha.ndim + 1 and pred_alpha.shape[-3] == 1:
            pred_alpha = pred_alpha.squeeze(-3)
        elif target_alpha.ndim == pred_alpha.ndim + 1 and target_alpha.shape[-3] == 1:
            target_alpha = target_alpha.squeeze(-3)
        else:
            raise ValueError(f"Shape mismatch: {pred_alpha.shape} vs {target_alpha.shape}")

    boundary_mask = compute_boundary_mask(
        target_alpha,
        kernel_size=kernel_size,
        fractional_low=fractional_low,
        fractional_high=fractional_high,
    )

    diff = torch.abs(pred_alpha.float() - target_alpha.float())
    boundary_diff = diff[boundary_mask]

    if boundary_diff.numel() == 0:
        return float(torch.mean(diff).item())

    return float(torch.mean(boundary_diff).item())


def compute_recomposition_mae(
    pred_alpha: torch.Tensor,
    pred_fg: torch.Tensor,
    composite: torch.Tensor,
    screen_rgb: torch.Tensor | None = None,
) -> float:
    """Mean absolute error between observed composite and recomposed prediction.

    Physical recomposition formula:
        C_hat = alpha_hat * F_hat + (1 - alpha_hat) * Screen

    When screen_rgb is provided, it serves as the constant backing color B.
    If screen_rgb is None, B is estimated from the composite where alpha == 0,
    or directly from composite background pixels.

    Args:
        pred_alpha: Predicted alpha [..., 1, H, W] or [..., H, W].
        pred_fg: Predicted straight foreground [..., 3, H, W].
        composite: Observed ground truth composite image [..., 3, H, W].
        screen_rgb: Backing screen color [3], [3, 1, 1], [3, H, W], or None.

    Returns:
        Scalar recomposition MAE as float.
    """
    alpha = pred_alpha.float()
    if alpha.ndim == pred_fg.ndim - 1:
        alpha = alpha.unsqueeze(-3)
    alpha = alpha.clamp(0.0, 1.0)
    fg = pred_fg.float().clamp(0.0, 1.0)
    comp = composite.float()

    if screen_rgb is not None:
        screen = _ensure_tensor(screen_rgb).float()
        if screen.ndim == 1 and screen.shape[0] == 3:
            screen = screen.view(3, 1, 1).expand_as(fg)
        elif screen.ndim == 3 and screen.shape[0] == 3:
            screen = screen.expand_as(fg)
    else:
        # If screen_rgb not provided, treat background pixels in composite as backing
        screen = comp

    recomposed = alpha * fg + (1.0 - alpha) * screen
    return float(torch.mean(torch.abs(recomposed - comp)).item())
