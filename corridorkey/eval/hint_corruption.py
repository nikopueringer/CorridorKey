"""Parameterized alpha-hint corruption model and asymmetry benchmark.

Formalizes hint degradation into measurable undersegmentation and
oversegmentation regimes, quantifying how model recovery degrades as
guide hints deviate from ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from .cases import MattingEvalSample
from .metrics import compute_alpha_mae, compute_boundary_alpha_mae
from .runner import predict_sample


@dataclass(frozen=True)
class HintCorruptionSpec:
    """Parameters governing synthetic alpha-hint corruption.

    Attributes:
        erosion_px: Number of pixels to morphologically erode (undersegmentation).
        dilation_px: Number of pixels to morphologically dilate (oversegmentation).
        blur_sigma: Gaussian blur standard deviation.
        holes: Number of interior dropout holes to punch into solid foreground.
        hole_radius_px: Radius of interior dropout holes.
        false_islands: Number of false-positive foreground islands in background.
        false_island_radius_px: Radius of false-positive islands.
        edge_jitter_px: Maximum pixel displacement for high-frequency edge noise.
        translation_px: Rigid (dx, dy) spatial offset of guide mask.
        downsample_factor: Resolution degradation factor (e.g. 2.0 = half res).
        seed: Random seed for stochastic operations.
    """

    erosion_px: int = 0
    dilation_px: int = 0
    blur_sigma: float = 0.0
    holes: int = 0
    hole_radius_px: int = 0
    false_islands: int = 0
    false_island_radius_px: int = 0
    edge_jitter_px: int = 0
    translation_px: tuple[int, int] = (0, 0)
    downsample_factor: float = 1.0
    seed: int | None = None


def corrupt_alpha(
    alpha: torch.Tensor | np.ndarray,
    spec: HintCorruptionSpec,
) -> torch.Tensor:
    """Apply a parameterized corruption spec to a ground truth alpha matte.

    Args:
        alpha: Ground truth alpha tensor [..., 1, H, W] or [H, W], in [0, 1].
        spec: HintCorruptionSpec defining the corruption operations.

    Returns:
        Corrupted hint tensor [1, H, W] in [0, 1], float32.
    """
    if isinstance(alpha, torch.Tensor):
        alpha_np = alpha.detach().cpu().float().numpy()
    else:
        alpha_np = np.asarray(alpha, dtype=np.float32)

    # Squeeze to 2D for OpenCV morphological operations
    while alpha_np.ndim > 2:
        alpha_np = alpha_np[0]

    h, w = alpha_np.shape
    rng = np.random.default_rng(spec.seed)
    hint = alpha_np.copy()

    # 1. Translation / rigid displacement
    dx, dy = spec.translation_px
    if dx != 0 or dy != 0:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        hint = cv2.warpAffine(hint, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # 2. Morphological erosion (undersegmentation)
    if spec.erosion_px > 0:
        k_size = spec.erosion_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        hint = cv2.erode(hint, kernel)

    # 3. Morphological dilation (oversegmentation)
    if spec.dilation_px > 0:
        k_size = spec.dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        hint = cv2.dilate(hint, kernel)

    # 4. Interior holes (dropout within solid foreground)
    if spec.holes > 0 and spec.hole_radius_px > 0:
        solid_mask = (hint >= 0.8).astype(np.uint8)
        # Erode solid mask so holes do not touch outer perimeter
        safe_margin = spec.hole_radius_px * 2 + 1
        k_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_margin, safe_margin))
        valid_centers = cv2.erode(solid_mask, k_safe)
        y_indices, x_indices = np.where(valid_centers > 0)
        if len(x_indices) > 0:
            count = min(spec.holes, len(x_indices))
            selected = rng.choice(len(x_indices), size=count, replace=False)
            for idx in selected:
                cx, cy = int(x_indices[idx]), int(y_indices[idx])
                cv2.circle(hint, (cx, cy), spec.hole_radius_px, 0.0, -1)

    # 5. False islands (spurious foreground in background)
    if spec.false_islands > 0 and spec.false_island_radius_px > 0:
        bg_mask = (hint <= 0.05).astype(np.uint8)
        margin = spec.false_island_radius_px * 2 + 1
        k_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
        valid_bg = cv2.erode(bg_mask, k_margin)
        y_indices, x_indices = np.where(valid_bg > 0)
        if len(x_indices) > 0:
            count = min(spec.false_islands, len(x_indices))
            selected = rng.choice(len(x_indices), size=count, replace=False)
            for idx in selected:
                cx, cy = int(x_indices[idx]), int(y_indices[idx])
                cv2.circle(hint, (cx, cy), spec.false_island_radius_px, 1.0, -1)

    # 6. Edge jitter (high-frequency perimeter displacement)
    if spec.edge_jitter_px > 0:
        edge_band = ((hint > 0.05) & (hint < 0.95)) | (
            cv2.dilate((hint >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
            - cv2.erode((hint >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
        )
        jitter = rng.uniform(-spec.edge_jitter_px, spec.edge_jitter_px, size=hint.shape).astype(np.float32)
        hint = np.where(edge_band > 0, np.clip(hint + jitter / (spec.edge_jitter_px * 2), 0.0, 1.0), hint)

    # 7. Resolution degradation
    if spec.downsample_factor > 1.0:
        low_w = max(4, int(w / spec.downsample_factor))
        low_h = max(4, int(h / spec.downsample_factor))
        hint = cv2.resize(hint, (low_w, low_h), interpolation=cv2.INTER_LINEAR)
        hint = cv2.resize(hint, (w, h), interpolation=cv2.INTER_LINEAR)

    # 8. Gaussian blur
    if spec.blur_sigma > 0.0:
        k_blur = int(spec.blur_sigma * 4) | 1
        hint = cv2.GaussianBlur(hint, (k_blur, k_blur), spec.blur_sigma)

    hint = np.clip(hint, 0.0, 1.0)
    return torch.from_numpy(hint).unsqueeze(0).float()


def compute_hint_iou(
    hint: torch.Tensor,
    target_alpha: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection-over-Union (IoU) between binarized hint and target alpha.

    Args:
        hint: Alpha hint tensor [..., H, W].
        target_alpha: Target alpha tensor [..., H, W].
        threshold: Binarization threshold (default 0.5).

    Returns:
        IoU scalar in [0.0, 1.0].
    """
    if hint.shape != target_alpha.shape:
        if hint.ndim == target_alpha.ndim + 1 and hint.shape[-3] == 1:
            hint = hint.squeeze(-3)
        elif target_alpha.ndim == hint.ndim + 1 and target_alpha.shape[-3] == 1:
            target_alpha = target_alpha.squeeze(-3)
        else:
            raise ValueError(f"Shape mismatch: hint {hint.shape} vs target_alpha {target_alpha.shape}")

    h_bin = hint.float() >= threshold
    t_bin = target_alpha.float() >= threshold
    intersection = (h_bin & t_bin).sum().item()
    union = (h_bin | t_bin).sum().item()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_asymmetric_error(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute directional false-positive vs false-negative alpha error.

    False Positive (FP): Background erroneously included as foreground (A < threshold).
    False Negative (FN): Foreground erroneously omitted as background (A >= threshold).

    Args:
        pred_alpha: Predicted alpha tensor [..., H, W].
        target_alpha: Ground truth alpha tensor [..., H, W].
        threshold: Separation threshold between background and foreground.

    Returns:
        dict containing 'fp_alpha' and 'fn_alpha'.
    """
    if pred_alpha.shape != target_alpha.shape:
        if pred_alpha.ndim == target_alpha.ndim + 1 and pred_alpha.shape[-3] == 1:
            pred_alpha = pred_alpha.squeeze(-3)
        elif target_alpha.ndim == pred_alpha.ndim + 1 and target_alpha.shape[-3] == 1:
            target_alpha = target_alpha.squeeze(-3)
        else:
            raise ValueError(f"Shape mismatch: pred_alpha {pred_alpha.shape} vs target_alpha {target_alpha.shape}")

    diff = torch.abs(pred_alpha.float() - target_alpha.float())
    bg_mask = target_alpha.float() < threshold
    fg_mask = target_alpha.float() >= threshold

    fp_alpha = float(diff[bg_mask].mean().item()) if bg_mask.sum() > 0 else 0.0
    fn_alpha = float(diff[fg_mask].mean().item()) if fg_mask.sum() > 0 else 0.0

    return {
        "fp_alpha": fp_alpha,
        "fn_alpha": fn_alpha,
    }


CANONICAL_HINT_PROFILES: dict[str, HintCorruptionSpec] = {
    "clean": HintCorruptionSpec(),
    "erosion_4px": HintCorruptionSpec(erosion_px=4),
    "erosion_8px": HintCorruptionSpec(erosion_px=8),
    "erosion_16px": HintCorruptionSpec(erosion_px=16),
    "dilation_4px": HintCorruptionSpec(dilation_px=4),
    "dilation_8px": HintCorruptionSpec(dilation_px=8),
    "dilation_16px": HintCorruptionSpec(dilation_px=16),
    "blur_sigma_4": HintCorruptionSpec(blur_sigma=4.0),
    "blur_sigma_8": HintCorruptionSpec(blur_sigma=8.0),
    "interior_holes": HintCorruptionSpec(holes=3, hole_radius_px=8, seed=42),
    "false_islands": HintCorruptionSpec(false_islands=3, false_island_radius_px=8, seed=42),
    "edge_jitter": HintCorruptionSpec(edge_jitter_px=3, seed=42),
}


@dataclass
class HintProfileResult:
    """Metrics for a single corruption profile across a benchmark set."""

    profile_name: str
    hint_iou: float
    alpha_mae: float
    boundary_alpha_mae: float
    fp_alpha: float
    fn_alpha: float


@dataclass
class HintSweepReport:
    """Full matrix report evaluating model response across hint corruption profiles."""

    results: list[HintProfileResult]
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "profiles": [
                {
                    "profile": r.profile_name,
                    "hint_iou": round(r.hint_iou, 4),
                    "alpha_mae": round(r.alpha_mae, 6),
                    "boundary_alpha_mae": round(r.boundary_alpha_mae, 6),
                    "fp_alpha": round(r.fp_alpha, 6),
                    "fn_alpha": round(r.fn_alpha, 6),
                }
                for r in self.results
            ],
        }


def evaluate_hint_profile(
    engine: Any,
    samples: Sequence[MattingEvalSample],
    profile_name: str,
    spec: HintCorruptionSpec,
) -> HintProfileResult:
    """Evaluate a benchmark suite under a specific hint corruption profile."""
    if not samples:
        raise ValueError("Cannot evaluate an empty sample list.")

    iou_list: list[float] = []
    alpha_mae_list: list[float] = []
    boundary_mae_list: list[float] = []
    fp_list: list[float] = []
    fn_list: list[float] = []

    for sample in samples:
        # Generate corrupted hint from ground truth alpha
        corrupted_hint = corrupt_alpha(sample.target_alpha, spec)

        # Create evaluation sample with the corrupted hint
        eval_sample = MattingEvalSample(
            composite=sample.composite,
            alpha_hint=corrupted_hint,
            target_alpha=sample.target_alpha,
            target_foreground=sample.target_foreground,
            screen_rgb=sample.screen_rgb,
            tags=sample.tags,
            name=sample.name,
        )

        iou = compute_hint_iou(corrupted_hint, sample.target_alpha)
        iou_list.append(iou)

        pred_alpha, _ = predict_sample(engine, eval_sample)
        pred_alpha = pred_alpha.clamp(0.0, 1.0)

        a_mae = compute_alpha_mae(pred_alpha, sample.target_alpha)
        b_mae = compute_boundary_alpha_mae(pred_alpha, sample.target_alpha)
        asym = compute_asymmetric_error(pred_alpha, sample.target_alpha)

        alpha_mae_list.append(a_mae)
        boundary_mae_list.append(b_mae)
        fp_list.append(asym["fp_alpha"])
        fn_list.append(asym["fn_alpha"])

    n = len(samples)
    return HintProfileResult(
        profile_name=profile_name,
        hint_iou=sum(iou_list) / n,
        alpha_mae=sum(alpha_mae_list) / n,
        boundary_alpha_mae=sum(boundary_mae_list) / n,
        fp_alpha=sum(fp_list) / n,
        fn_alpha=sum(fn_list) / n,
    )


def run_hint_sweep(
    engine: Any,
    samples: Sequence[MattingEvalSample],
    profiles: dict[str, HintCorruptionSpec] | None = None,
    model_name: str = "CorridorKey",
) -> HintSweepReport:
    """Run full hint corruption benchmark across all profiles.

    Args:
        engine: Model or callable.
        samples: Benchmark evaluation samples.
        profiles: Dictionary of profile names to specs (defaults to CANONICAL_HINT_PROFILES).
        model_name: Name of evaluated model.

    Returns:
        HintSweepReport containing the complete matrix.
    """
    if not samples:
        raise ValueError("Cannot evaluate an empty sample list.")

    if profiles is None:
        profiles = CANONICAL_HINT_PROFILES

    results: list[HintProfileResult] = []
    for name, spec in profiles.items():
        res = evaluate_hint_profile(engine, samples, name, spec)
        results.append(res)

    return HintSweepReport(results=results, model_name=model_name)
