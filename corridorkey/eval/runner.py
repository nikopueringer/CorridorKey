"""Evaluation runner executing matting inference and aggregating metrics.

Runs models against MattingEvalSample instances, computes core metrics,
and compiles overall, per-case, and per-tag evaluation summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .cases import MattingEvalSample
from .metrics import (
    compute_alpha_mae,
    compute_boundary_alpha_mae,
    compute_foreground_mae,
    compute_recomposition_mae,
)


@dataclass
class CaseResult:
    """Evaluation metrics for a single test case."""

    name: str
    tags: list[str]
    alpha_mae: float
    foreground_mae: float
    boundary_alpha_mae: float
    recomposition_mae: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_mae": round(self.alpha_mae, 6),
            "foreground_mae": round(self.foreground_mae, 6),
            "boundary_alpha_mae": round(self.boundary_alpha_mae, 6),
            "recomposition_mae": round(self.recomposition_mae, 6),
            "tags": sorted(self.tags),
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation results over an entire benchmark suite."""

    alpha_mae: float
    foreground_mae: float
    boundary_alpha_mae: float
    recomposition_mae: float
    cases: dict[str, CaseResult] = field(default_factory=dict)
    tag_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    model_name: str = ""
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary conforming to benchmark JSON schema."""
        return {
            "model": self.model_name,
            "sample_count": self.sample_count,
            "alpha_mae": round(self.alpha_mae, 6),
            "foreground_mae": round(self.foreground_mae, 6),
            "boundary_alpha_mae": round(self.boundary_alpha_mae, 6),
            "recomposition_mae": round(self.recomposition_mae, 6),
            "cases": {name: res.to_dict() for name, res in self.cases.items()},
            "tags": {tag: {k: round(v, 6) for k, v in metrics.items()} for tag, metrics in self.tag_breakdown.items()},
        }


def predict_sample(
    engine: Any,
    sample: MattingEvalSample,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute model prediction for a sample and extract (pred_alpha, pred_fg).

    Supports:
      1. CorridorKeyEngine instance (calls process_frame).
      2. PyTorch nn.Module (forward pass with 4-channel input [C, H]).
      3. Callable (composite, hint) -> (pred_alpha, pred_fg).

    Returns:
        tuple of (pred_alpha [1, H, W], pred_fg [3, H, W]) float32 tensors in [0, 1].
    """
    # 1. CorridorKeyEngine instance
    if hasattr(engine, "process_frame"):
        comp_np = sample.composite.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        hint_np = sample.alpha_hint.squeeze(0).cpu().numpy().astype(np.float32)

        screen_channel = 1
        if sample.screen_rgb is not None and sample.screen_rgb[2] > sample.screen_rgb[1]:
            screen_channel = 2

        out = engine.process_frame(
            comp_np,
            hint_np,
            screen_channel=screen_channel,
            auto_despeckle=False,
            generate_comp=False,
            post_process_on_gpu=False,
        )
        if isinstance(out, list):
            out = out[0]

        alpha_np = out["alpha"]
        if alpha_np.ndim == 2:
            alpha_np = alpha_np[np.newaxis, :, :]
        elif alpha_np.ndim == 3 and alpha_np.shape[-1] == 1:
            alpha_np = np.transpose(alpha_np, (2, 0, 1))

        fg_np = out["fg"]
        if fg_np.ndim == 3 and fg_np.shape[-1] == 3:
            fg_np = np.transpose(fg_np, (2, 0, 1))

        pred_alpha = torch.from_numpy(alpha_np).float()
        pred_fg = torch.from_numpy(fg_np).float()
        return pred_alpha, pred_fg

    # 2. PyTorch nn.Module
    if isinstance(engine, torch.nn.Module):
        engine.eval()
        device = next(engine.parameters()).device
        inp = torch.cat([sample.composite, sample.alpha_hint], dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = engine(inp)

        if isinstance(out, dict):
            pred_alpha = out["alpha"].detach().cpu()
            pred_fg = out["fg"].detach().cpu()
        elif isinstance(out, (tuple, list)):
            pred_alpha, pred_fg = out[0].detach().cpu(), out[1].detach().cpu()
        else:
            raise TypeError(f"Unexpected module output type: {type(out)}")

        if pred_alpha.ndim == 4:
            pred_alpha = pred_alpha.squeeze(0)
        if pred_fg.ndim == 4:
            pred_fg = pred_fg.squeeze(0)
        return pred_alpha, pred_fg

    # 3. Arbitrary callable
    if callable(engine):
        res = engine(sample.composite, sample.alpha_hint)
        if isinstance(res, dict):
            pred_alpha, pred_fg = res["alpha"], res["fg"]
        else:
            pred_alpha, pred_fg = res[0], res[1]

        if not isinstance(pred_alpha, torch.Tensor):
            pred_alpha = torch.from_numpy(np.asarray(pred_alpha))
        if not isinstance(pred_fg, torch.Tensor):
            pred_fg = torch.from_numpy(np.asarray(pred_fg))

        if pred_alpha.ndim == 2:
            pred_alpha = pred_alpha.unsqueeze(0)
        elif pred_alpha.ndim == 3 and pred_alpha.shape[-1] == 1:
            pred_alpha = pred_alpha.permute(2, 0, 1)

        if pred_fg.ndim == 3 and pred_fg.shape[-1] == 3:
            pred_fg = pred_fg.permute(2, 0, 1)

        return pred_alpha.float(), pred_fg.float()

    raise TypeError(f"Unsupported engine type: {type(engine)}")


def evaluate_sample(engine: Any, sample: MattingEvalSample) -> CaseResult:
    """Evaluate a single sample and return individual metrics."""
    pred_alpha, pred_fg = predict_sample(engine, sample)

    # Ensure predictions are clamped to valid physical range [0, 1]
    pred_alpha = pred_alpha.clamp(0.0, 1.0)
    pred_fg = pred_fg.clamp(0.0, 1.0)

    alpha_mae = compute_alpha_mae(pred_alpha, sample.target_alpha)
    fg_mae = compute_foreground_mae(pred_fg, sample.target_foreground, sample.target_alpha)
    boundary_mae = compute_boundary_alpha_mae(pred_alpha, sample.target_alpha)
    recomp_mae = compute_recomposition_mae(pred_alpha, pred_fg, sample.composite, sample.screen_rgb)

    return CaseResult(
        name=sample.name or "unnamed_sample",
        tags=list(sample.tags),
        alpha_mae=alpha_mae,
        foreground_mae=fg_mae,
        boundary_alpha_mae=boundary_mae,
        recomposition_mae=recomp_mae,
    )


def evaluate_dataset(
    engine: Any,
    samples: Sequence[MattingEvalSample],
    model_name: str = "CorridorKey",
) -> EvaluationReport:
    """Run evaluation across all provided benchmark samples and aggregate statistics.

    Args:
        engine: Model runner or callable.
        samples: Sequence of MattingEvalSample fixtures.
        model_name: Identifier for model/checkpoint being evaluated.

    Returns:
        EvaluationReport containing global, per-case, and per-tag metrics.
    """
    if not samples:
        raise ValueError("Cannot evaluate an empty sample list.")

    case_results: dict[str, CaseResult] = {}
    tag_accum: dict[str, list[CaseResult]] = {}

    for sample in samples:
        res = evaluate_sample(engine, sample)
        case_results[res.name] = res

        for tag in sample.tags:
            tag_accum.setdefault(tag, []).append(res)

    # Aggregate global means
    n = len(samples)
    global_alpha_mae = sum(c.alpha_mae for c in case_results.values()) / n
    global_fg_mae = sum(c.foreground_mae for c in case_results.values()) / n
    global_boundary_mae = sum(c.boundary_alpha_mae for c in case_results.values()) / n
    global_recomp_mae = sum(c.recomposition_mae for c in case_results.values()) / n

    # Aggregate tag means
    tag_breakdown: dict[str, dict[str, float]] = {}
    for tag, c_list in sorted(tag_accum.items()):
        k = len(c_list)
        tag_breakdown[tag] = {
            "count": float(k),
            "alpha_mae": sum(c.alpha_mae for c in c_list) / k,
            "foreground_mae": sum(c.foreground_mae for c in c_list) / k,
            "boundary_alpha_mae": sum(c.boundary_alpha_mae for c in c_list) / k,
            "recomposition_mae": sum(c.recomposition_mae for c in c_list) / k,
        }

    return EvaluationReport(
        model_name=model_name,
        sample_count=n,
        alpha_mae=global_alpha_mae,
        foreground_mae=global_fg_mae,
        boundary_alpha_mae=global_boundary_mae,
        recomposition_mae=global_recomp_mae,
        cases=case_results,
        tag_breakdown=tag_breakdown,
    )
