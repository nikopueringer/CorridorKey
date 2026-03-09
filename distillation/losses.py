"""
Custom torchdistill mid-level losses for green-screen matting distillation.

Two losses are registered here and can be referenced by name in the
torchdistill YAML config (``distillation/configs/distill_greenformer.yaml``).

MatteTaskLoss
    Compares the **student** model's predictions to ground-truth targets.
    Uses MSE on both alpha and foreground channels, weighted independently.

MatteResponseKDLoss
    Response-based knowledge distillation loss.  Compares the **student**
    model's soft predictions to the **teacher** model's soft predictions,
    encouraging the student to mimic the teacher's full output distribution
    (not just a hard ground-truth).  Uses MSE (L2) on each output channel.

Both classes are decorated with ``@register_mid_level_loss`` so that they
can be instantiated by key name inside a torchdistill YAML config's
``criterion.kwargs.sub_terms`` block.

Example YAML snippet
--------------------
.. code-block:: yaml

    criterion:
      key: 'WeightedSumLoss'
      kwargs:
        sub_terms:
          task:
            criterion:
              key: 'MatteTaskLoss'
              kwargs:
                alpha_weight: 1.0
                fg_weight: 1.0
            weight: 1.0
          kd:
            criterion:
              key: 'MatteResponseKDLoss'
              kwargs:
                alpha_weight: 0.5
                fg_weight: 0.5
            weight: 0.5
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdistill.losses.registry import register_mid_level_loss


@register_mid_level_loss
class MatteTaskLoss(nn.Module):
    """
    Task loss: MSE between student predictions and ground-truth targets.

    Parameters
    ----------
    alpha_weight:
        Multiplier applied to the alpha-channel MSE component.
    fg_weight:
        Multiplier applied to the foreground-channel MSE component.
    """

    def __init__(self, alpha_weight: float = 1.0, fg_weight: float = 1.0) -> None:
        super().__init__()
        self.alpha_weight = alpha_weight
        self.fg_weight = fg_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_io_dict: dict,
        teacher_io_dict: dict,  # noqa: ARG002 — unused; required by torchdistill interface
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        student_output = student_io_dict["."]["output"]
        alpha_loss = self.mse(student_output["alpha"], targets["alpha"])
        fg_loss = self.mse(student_output["fg"], targets["fg"])
        return self.alpha_weight * alpha_loss + self.fg_weight * fg_loss


@register_mid_level_loss
class MatteResponseKDLoss(nn.Module):
    """
    Response-based KD loss: MSE between student and teacher soft predictions.

    The teacher model is run without gradients by torchdistill
    (``requires_grad: false`` in YAML) before this loss is invoked, so no
    explicit ``torch.no_grad`` is needed here.

    Parameters
    ----------
    alpha_weight:
        Multiplier applied to the alpha-channel MSE component.
    fg_weight:
        Multiplier applied to the foreground-channel MSE component.
    """

    def __init__(self, alpha_weight: float = 0.5, fg_weight: float = 0.5) -> None:
        super().__init__()
        self.alpha_weight = alpha_weight
        self.fg_weight = fg_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_io_dict: dict,
        teacher_io_dict: dict,
        targets: dict[str, torch.Tensor],  # noqa: ARG002 — unused; required by interface
    ) -> torch.Tensor:
        student_output = student_io_dict["."]["output"]
        teacher_output = teacher_io_dict["."]["output"]
        alpha_kd = self.mse(student_output["alpha"], teacher_output["alpha"])
        fg_kd = self.mse(student_output["fg"], teacher_output["fg"])
        return self.alpha_weight * alpha_kd + self.fg_weight * fg_kd
