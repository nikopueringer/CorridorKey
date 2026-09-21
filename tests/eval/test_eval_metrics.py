"""Tests for matting evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from corridorkey.eval.metrics import (
    compute_alpha_mae,
    compute_boundary_alpha_mae,
    compute_boundary_mask,
    compute_foreground_mae,
    compute_recomposition_mae,
)


def test_alpha_mae_identical():
    a = torch.rand(1, 64, 64)
    assert compute_alpha_mae(a, a) == pytest.approx(0.0)


def test_alpha_mae_known_difference():
    a = torch.zeros(1, 32, 32)
    b = torch.full((1, 32, 32), 0.25)
    assert compute_alpha_mae(a, b) == pytest.approx(0.25)


def test_alpha_mae_dimension_squeezing():
    a_3d = torch.zeros(1, 16, 16)
    b_2d = torch.full((16, 16), 0.5)
    assert compute_alpha_mae(a_3d, b_2d) == pytest.approx(0.5)
    assert compute_alpha_mae(b_2d, a_3d) == pytest.approx(0.5)


def test_alpha_mae_shape_mismatch_raises():
    a = torch.zeros(1, 16, 16)
    b = torch.zeros(1, 32, 32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_alpha_mae(a, b)


def test_foreground_mae_identical():
    fg = torch.rand(3, 32, 32)
    assert compute_foreground_mae(fg, fg) == pytest.approx(0.0)


def test_foreground_mae_with_alpha_mask():
    fg_pred = torch.ones(3, 10, 10)
    fg_target = torch.zeros(3, 10, 10)

    # Only first column has foreground alpha > threshold
    alpha = torch.zeros(1, 10, 10)
    alpha[0, :, 0] = 1.0

    # If evaluated across whole frame, error would be 1.0.
    # If evaluated strictly where alpha > 0.01, it is also 1.0 on that column.
    assert compute_foreground_mae(fg_pred, fg_target, alpha) == pytest.approx(1.0)

    # Now make prediction match target on the masked column, but differ on background
    fg_pred[0, :, 0] = 0.0
    fg_pred[1, :, 0] = 0.0
    fg_pred[2, :, 0] = 0.0
    # Masked region has zero error, background has error 1.0. With mask, result must be 0.0!
    assert compute_foreground_mae(fg_pred, fg_target, alpha) == pytest.approx(0.0)


def test_foreground_mae_empty_mask():
    fg1 = torch.rand(3, 10, 10)
    fg2 = torch.rand(3, 10, 10)
    alpha_zero = torch.zeros(1, 10, 10)
    # Mask sum is 0 -> should return 0.0 cleanly without zero division
    assert compute_foreground_mae(fg1, fg2, alpha_zero) == 0.0


def test_compute_boundary_mask_fractional():
    # Test that fractional alpha pixels (0.01 < alpha < 0.99) are marked as boundary
    alpha = torch.zeros(1, 20, 20)
    alpha[0, 10, 10] = 0.5  # Fractional pixel
    mask = compute_boundary_mask(alpha, kernel_size=3)
    assert mask[0, 10, 10].item() is True
    assert mask[0, 0, 0].item() is False


def test_compute_boundary_mask_step_edge():
    # Pure binary step: left half 0, right half 1
    alpha = torch.zeros(1, 20, 20)
    alpha[:, :, 10:] = 1.0

    mask = compute_boundary_mask(alpha, kernel_size=5)
    # The morphological band around column 10 should be True
    assert mask[0, 5, 10].item() is True
    assert mask[0, 5, 9].item() is True
    # Pixels far from transition should be False
    assert mask[0, 5, 0].item() is False
    assert mask[0, 5, 19].item() is False


def test_boundary_alpha_mae_isolates_edges():
    # Ground truth: 100x100 image, 10x10 square in center
    target_alpha = torch.zeros(1, 100, 100)
    target_alpha[0, 45:55, 45:55] = 1.0

    # Prediction 1: perfect everywhere except 1px error right on the edge
    pred_edge_err = target_alpha.clone()
    pred_edge_err[0, 45, 45] = 0.0

    # Overall MAE is tiny because of 10000 pixels
    global_mae = compute_alpha_mae(pred_edge_err, target_alpha)
    boundary_mae = compute_boundary_alpha_mae(pred_edge_err, target_alpha, kernel_size=3)

    assert boundary_mae > global_mae


def test_recomposition_mae_perfect_recomposition():
    # C = alpha * F + (1 - alpha) * B
    alpha = torch.full((1, 32, 32), 0.6)
    fg = torch.full((3, 32, 32), 0.8)
    screen = torch.tensor([0.0, 1.0, 0.0])
    screen_field = screen.view(3, 1, 1).expand(3, 32, 32)
    comp = 0.6 * fg + 0.4 * screen_field

    # Perfect prediction should yield 0.0 recomposition error
    mae = compute_recomposition_mae(alpha, fg, comp, screen_rgb=screen)
    assert mae == pytest.approx(0.0, abs=1e-6)


def test_recomposition_mae_wrong_foreground():
    alpha = torch.ones(1, 16, 16)
    fg_truth = torch.ones(3, 16, 16)
    fg_pred = torch.zeros(3, 16, 16)
    comp = fg_truth

    mae = compute_recomposition_mae(alpha, fg_pred, comp)
    assert mae == pytest.approx(1.0)
