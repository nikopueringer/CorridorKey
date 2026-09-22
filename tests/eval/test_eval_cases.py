"""Tests for dataset sample contracts and fixture loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from corridorkey.eval.cases import (
    MattingEvalSample,
    load_dataset,
    load_sample_from_directory,
)
from corridorkey.eval.fixtures import (
    _save_png_gray,
    _save_png_rgb,
    create_diagnostic_fixtures,
    generate_opaque_subject,
)


def test_matting_eval_sample_valid():
    comp = torch.rand(3, 64, 64)
    hint = torch.rand(1, 64, 64)
    alpha = torch.rand(1, 64, 64)
    fg = torch.rand(3, 64, 64)
    screen = torch.tensor([0.0, 1.0, 0.0])

    sample = MattingEvalSample(
        composite=comp,
        alpha_hint=hint,
        target_alpha=alpha,
        target_foreground=fg,
        screen_rgb=screen,
        tags=frozenset({"test", "hair"}),
        name="test_case",
    )
    assert sample.name == "test_case"
    assert "hair" in sample.tags
    assert sample.composite.shape == (3, 64, 64)
    assert sample.target_alpha.shape == (1, 64, 64)


def test_matting_eval_sample_2d_hint_squeezing():
    comp = torch.rand(3, 32, 32)
    hint_2d = torch.rand(32, 32)
    alpha_2d = torch.rand(32, 32)
    fg = torch.rand(3, 32, 32)

    sample = MattingEvalSample(
        composite=comp,
        alpha_hint=hint_2d,
        target_alpha=alpha_2d,
        target_foreground=fg,
        screen_rgb=None,
        tags=frozenset(),
    )
    assert sample.alpha_hint.shape == (1, 32, 32)
    assert sample.target_alpha.shape == (1, 32, 32)


def test_matting_eval_sample_dimension_mismatch():
    comp = torch.rand(3, 32, 32)
    hint = torch.rand(1, 64, 64)  # Mismatched spatial size
    alpha = torch.rand(1, 32, 32)
    fg = torch.rand(3, 32, 32)

    with pytest.raises(ValueError, match="Spatial dimension mismatch"):
        MattingEvalSample(
            composite=comp,
            alpha_hint=hint,
            target_alpha=alpha,
            target_foreground=fg,
            screen_rgb=None,
            tags=frozenset(),
        )


def test_load_sample_from_directory(tmp_path: Path):
    case_dir = tmp_path / "sample_case"
    case_dir.mkdir()

    data = generate_opaque_subject(size=64)
    _save_png_rgb(case_dir / "composite.png", data["composite"])
    _save_png_gray(case_dir / "alpha_hint.png", data["alpha_hint"])
    _save_png_gray(case_dir / "target_alpha.png", data["target_alpha"])
    _save_png_rgb(case_dir / "target_foreground.png", data["target_foreground"])
    with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data["metadata"], f)

    sample = load_sample_from_directory(case_dir)
    assert sample.name == "sample_case"
    assert "opaque" in sample.tags
    assert sample.composite.shape == (3, 64, 64)
    assert sample.target_alpha.shape == (1, 64, 64)


def test_load_dataset_canonical_fixtures(tmp_path: Path):
    fixtures_dir = tmp_path / "fixtures"
    create_diagnostic_fixtures(fixtures_dir, size=64)
    samples = load_dataset(fixtures_dir)

    assert len(samples) == 7
    names = {s.name for s in samples}
    expected_names = {
        "opaque_subject",
        "soft_hair",
        "motion_blur",
        "green_foreground",
        "blue_foreground",
        "bad_hint_eroded",
        "bad_hint_dilated",
    }
    assert names == expected_names

    for s in samples:
        assert s.composite.shape == (3, 64, 64)
        assert s.alpha_hint.shape == (1, 64, 64)
        assert s.target_alpha.shape == (1, 64, 64)
        assert s.target_foreground.shape == (3, 64, 64)
        assert len(s.tags) > 0
