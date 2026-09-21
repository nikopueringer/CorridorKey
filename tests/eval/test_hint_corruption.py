"""Unit tests for parameterized alpha-hint corruption and asymmetry sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from corridorkey.eval.__main__ import main
from corridorkey.eval.cases import MattingEvalSample
from corridorkey.eval.hint_corruption import (
    CANONICAL_HINT_PROFILES,
    HintCorruptionSpec,
    compute_asymmetric_error,
    compute_hint_iou,
    corrupt_alpha,
    run_hint_sweep,
)
from corridorkey.eval.report import format_hint_sweep_text_table


def _make_binary_disk_alpha(size: int = 64) -> torch.Tensor:
    h, w = size, size
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist = torch.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    return (dist <= size * 0.35).float().unsqueeze(0)


def test_corrupt_alpha_clean():
    alpha = _make_binary_disk_alpha(64)
    spec = HintCorruptionSpec()
    corrupted = corrupt_alpha(alpha, spec)
    assert corrupted.shape == (1, 64, 64)
    assert torch.allclose(alpha, corrupted)


def test_corrupt_alpha_erosion():
    alpha = _make_binary_disk_alpha(64)
    spec = HintCorruptionSpec(erosion_px=4)
    eroded = corrupt_alpha(alpha, spec)
    assert eroded.sum() < alpha.sum()
    iou = compute_hint_iou(eroded, alpha)
    assert 0.4 < iou < 1.0


def test_corrupt_alpha_dilation():
    alpha = _make_binary_disk_alpha(64)
    spec = HintCorruptionSpec(dilation_px=4)
    dilated = corrupt_alpha(alpha, spec)
    assert dilated.sum() > alpha.sum()
    iou = compute_hint_iou(dilated, alpha)
    assert 0.4 < iou < 1.0


def test_corrupt_alpha_holes():
    alpha = _make_binary_disk_alpha(64)
    spec = HintCorruptionSpec(holes=2, hole_radius_px=4, seed=123)
    holed = corrupt_alpha(alpha, spec)
    assert holed.sum() < alpha.sum()


def test_corrupt_alpha_false_islands():
    alpha = _make_binary_disk_alpha(64)
    spec = HintCorruptionSpec(false_islands=2, false_island_radius_px=3, seed=123)
    islanded = corrupt_alpha(alpha, spec)
    # Check that pixels outside the subject circle received 1.0
    bg_mask = alpha == 0.0
    assert (islanded[bg_mask] > 0.5).sum() > 0


def test_compute_hint_iou_identity():
    alpha = _make_binary_disk_alpha(32)
    assert compute_hint_iou(alpha, alpha) == pytest.approx(1.0)


def test_compute_hint_iou_disjoint():
    a = torch.zeros(1, 32, 32)
    a[0, :16, :16] = 1.0
    b = torch.zeros(1, 32, 32)
    b[0, 16:, 16:] = 1.0
    assert compute_hint_iou(a, b) == pytest.approx(0.0)


def test_compute_asymmetric_error():
    # Ground truth: left half 1, right half 0
    target = torch.zeros(1, 10, 10)
    target[0, :, :5] = 1.0

    # Prediction 1: false positive on background (included extra background)
    pred_fp = target.clone()
    pred_fp[0, :, 5:7] = 1.0
    err_fp = compute_asymmetric_error(pred_fp, target)
    assert err_fp["fp_alpha"] > 0.0
    assert err_fp["fn_alpha"] == 0.0

    # Prediction 2: false negative on foreground (missing foreground)
    pred_fn = target.clone()
    pred_fn[0, :, :2] = 0.0
    err_fn = compute_asymmetric_error(pred_fn, target)
    assert err_fn["fp_alpha"] == 0.0
    assert err_fn["fn_alpha"] > 0.0


def test_hint_sweep_report_and_formatting():
    alpha = _make_binary_disk_alpha(32)
    sample = MattingEvalSample(
        composite=torch.zeros(3, 32, 32),
        alpha_hint=alpha,
        target_alpha=alpha,
        target_foreground=torch.ones(3, 32, 32),
        screen_rgb=torch.tensor([0.0, 1.0, 0.0]),
        tags=frozenset({"test"}),
        name="test_disk",
    )

    def mock_engine(comp, hint):
        return hint, comp

    subset_profiles = {
        "clean": CANONICAL_HINT_PROFILES["clean"],
        "erosion_4px": CANONICAL_HINT_PROFILES["erosion_4px"],
        "dilation_4px": CANONICAL_HINT_PROFILES["dilation_4px"],
    }

    report = run_hint_sweep(mock_engine, [sample], profiles=subset_profiles, model_name="MockEngine")
    assert len(report.results) == 3
    assert report.model_name == "MockEngine"

    # Verify table formatting
    table_str = format_hint_sweep_text_table(report)
    assert "MockEngine" in table_str
    assert "erosion_4px" in table_str
    assert "dilation_4px" in table_str

    # Verify dict serialization
    d = report.to_dict()
    assert len(d["profiles"]) == 3
    assert d["profiles"][0]["profile"] == "clean"


def test_cli_hint_sweep(tmp_path: Path):
    dataset_dir = tmp_path / "fixtures"
    out_file = tmp_path / "hint_sweep_receipt.json"

    exit_code = main(
        [
            "--checkpoint",
            "mock",
            "--dataset",
            str(dataset_dir),
            "--generate-fixtures",
            "--hint-sweep",
            "--output",
            str(out_file),
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert out_file.is_file()

    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)

    assert "profiles" in data
    assert len(data["profiles"]) == len(CANONICAL_HINT_PROFILES)
    profile_names = [p["profile"] for p in data["profiles"]]
    assert "erosion_4px" in profile_names
    assert "dilation_4px" in profile_names
    assert "false_islands" in profile_names


def test_hint_metrics_dimension_squeezing():
    alpha_3d = _make_binary_disk_alpha(32)  # [1, 32, 32]
    alpha_2d = alpha_3d.squeeze(0)  # [32, 32]

    # compute_hint_iou should work across 2D and 3D
    iou = compute_hint_iou(alpha_3d, alpha_2d)
    assert iou == pytest.approx(1.0)

    # compute_asymmetric_error should work across 2D and 3D
    err = compute_asymmetric_error(alpha_2d, alpha_3d)
    assert err["fp_alpha"] == pytest.approx(0.0)
    assert err["fn_alpha"] == pytest.approx(0.0)


def test_hint_metrics_shape_mismatch_raises():
    t1 = torch.zeros(1, 32, 32)
    t2 = torch.zeros(1, 16, 16)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_hint_iou(t1, t2)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_asymmetric_error(t1, t2)


def test_run_hint_sweep_empty_samples_raises():
    def mock_engine(comp, hint):
        return hint, comp

    with pytest.raises(ValueError, match="Cannot evaluate an empty sample list"):
        run_hint_sweep(mock_engine, [])
