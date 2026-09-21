"""Tests for evaluation runner, reporting, and CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from corridorkey.eval.__main__ import main
from corridorkey.eval.cases import MattingEvalSample
from corridorkey.eval.report import format_json_report, format_text_table, write_json_report
from corridorkey.eval.runner import evaluate_dataset, evaluate_sample, predict_sample


def _create_dummy_sample(name: str = "dummy", tags: tuple[str, ...] = ("test",)) -> MattingEvalSample:
    h, w = 32, 32
    comp = torch.zeros(3, h, w)
    hint = torch.zeros(1, h, w)
    alpha = torch.zeros(1, h, w)
    alpha[0, 10:20, 10:20] = 1.0
    fg = torch.full((3, h, w), 0.8)
    screen = torch.tensor([0.0, 1.0, 0.0])
    return MattingEvalSample(
        composite=comp,
        alpha_hint=hint,
        target_alpha=alpha,
        target_foreground=fg,
        screen_rgb=screen,
        tags=frozenset(tags),
        name=name,
    )


def test_predict_sample_callable():
    sample = _create_dummy_sample()

    def dummy_fn(comp, hint):
        return hint, comp

    pred_alpha, pred_fg = predict_sample(dummy_fn, sample)
    assert pred_alpha.shape == (1, 32, 32)
    assert pred_fg.shape == (3, 32, 32)


def test_predict_sample_engine_mock():
    sample = _create_dummy_sample()
    mock_engine = MagicMock()
    mock_engine.process_frame.return_value = {
        "alpha": np.zeros((32, 32, 1), dtype=np.float32),
        "fg": np.zeros((32, 32, 3), dtype=np.float32),
    }

    pred_alpha, pred_fg = predict_sample(mock_engine, sample)
    assert pred_alpha.shape == (1, 32, 32)
    assert pred_fg.shape == (3, 32, 32)
    assert mock_engine.process_frame.called


def test_predict_sample_torch_module():
    sample = _create_dummy_sample()

    class TinyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, 1)

        def forward(self, x):
            out = self.conv(x)
            return {"alpha": out[:, :1], "fg": out[:, 1:]}

    mod = TinyModule()
    pred_alpha, pred_fg = predict_sample(mod, sample)
    assert pred_alpha.shape == (1, 32, 32)
    assert pred_fg.shape == (3, 32, 32)


def test_evaluate_sample():
    sample = _create_dummy_sample(name="sample_a")

    # Oracle engine that returns ground truth
    def oracle_engine(comp, hint):
        return sample.target_alpha, sample.target_foreground

    res = evaluate_sample(oracle_engine, sample)
    assert res.name == "sample_a"
    assert res.alpha_mae == pytest.approx(0.0)
    assert res.foreground_mae == pytest.approx(0.0)
    assert res.boundary_alpha_mae == pytest.approx(0.0)


def test_evaluate_dataset_aggregation():
    sample1 = _create_dummy_sample(name="case_1", tags=("hair", "fine"))
    sample2 = _create_dummy_sample(name="case_2", tags=("hair", "motion"))

    def stub_engine(comp, hint):
        # Off-by-one everywhere
        return sample1.target_alpha + 0.1, sample1.target_foreground + 0.2

    report = evaluate_dataset(stub_engine, [sample1, sample2], model_name="StubModel")
    assert report.sample_count == 2
    assert report.model_name == "StubModel"
    assert "case_1" in report.cases
    assert "case_2" in report.cases
    assert "hair" in report.tag_breakdown
    assert report.tag_breakdown["hair"]["count"] == 2.0


def test_report_formatting(tmp_path: Path):
    sample = _create_dummy_sample(name="case_1")
    report = evaluate_dataset(lambda c, h: (h, c), [sample], model_name="TestModel")

    # Test text formatting
    text_tbl = format_text_table(report)
    assert "TestModel" in text_tbl
    assert "case_1" in text_tbl

    # Test JSON formatting
    json_str = format_json_report(report)
    data = json.loads(json_str)
    assert data["model"] == "TestModel"
    assert "alpha_mae" in data
    assert "cases" in data

    # Test file writing
    out_file = tmp_path / "report.json"
    write_json_report(report, str(out_file))
    assert out_file.is_file()


def test_cli_execution_mock(tmp_path: Path):
    dataset_dir = tmp_path / "fixtures"
    out_file = tmp_path / "cli_eval.json"
    exit_code = main(
        [
            "--checkpoint",
            "mock",
            "--dataset",
            str(dataset_dir),
            "--generate-fixtures",
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

    assert data["sample_count"] == 7
    assert "alpha_mae" in data
    assert "cases" in data
    assert "soft_hair" in data["cases"]
    assert "opaque_subject" in data["cases"]
