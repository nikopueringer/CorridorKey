"""
Tests for poc_distill.py — distillation proof-of-concept.

All tests run on CPU with tiny synthetic tensors.  No GPU, no checkpoint,
and no torchdistill installation required.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest
import torch

import poc_distill as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")
_TINY = 32  # img_size for fast CPU tests


def _run(steps: int = 5, img_size: int = _TINY, **kwargs) -> list[pd.StepRecord]:
    teacher = pd._build_teacher(img_size, use_refiner=False, device=CPU)
    student = pd._build_student(img_size, use_refiner=False, device=CPU)
    return pd.run_distillation(
        teacher=teacher,
        student=student,
        device=CPU,
        img_size=img_size,
        batch_size=1,
        steps=steps,
        lr=1e-3,
        task_weight=1.0,
        kd_weight=0.5,
        use_amp=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class TestMatteTaskLoss:
    def test_zero_when_identical(self):
        crit = pd._MatteTaskLoss()
        pred = {"alpha": torch.ones(1, 1, 8, 8) * 0.5, "fg": torch.ones(1, 3, 8, 8) * 0.5}
        gt = {"alpha": torch.ones(1, 1, 8, 8) * 0.5, "fg": torch.ones(1, 3, 8, 8) * 0.5}
        assert crit(pred, gt).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_different(self):
        crit = pd._MatteTaskLoss()
        pred = {"alpha": torch.zeros(1, 1, 8, 8), "fg": torch.zeros(1, 3, 8, 8)}
        gt = {"alpha": torch.ones(1, 1, 8, 8), "fg": torch.ones(1, 3, 8, 8)}
        assert crit(pred, gt).item() > 0

    def test_scalar_output(self):
        crit = pd._MatteTaskLoss()
        pred = {"alpha": torch.rand(1, 1, 8, 8), "fg": torch.rand(1, 3, 8, 8)}
        gt = {"alpha": torch.rand(1, 1, 8, 8), "fg": torch.rand(1, 3, 8, 8)}
        loss = crit(pred, gt)
        assert loss.ndim == 0

    def test_weights_scale_loss(self):
        """Doubling alpha_weight should increase loss when alpha error is non-zero."""
        pred = {"alpha": torch.zeros(1, 1, 8, 8), "fg": torch.zeros(1, 3, 8, 8)}
        gt = {"alpha": torch.ones(1, 1, 8, 8), "fg": torch.zeros(1, 3, 8, 8)}
        loss_base = pd._MatteTaskLoss(alpha_weight=1.0)(pred, gt)
        loss_2x = pd._MatteTaskLoss(alpha_weight=2.0)(pred, gt)
        assert loss_2x.item() > loss_base.item()


class TestMatteResponseKDLoss:
    def test_zero_when_student_matches_teacher(self):
        crit = pd._MatteResponseKDLoss()
        shared = {"alpha": torch.rand(1, 1, 8, 8), "fg": torch.rand(1, 3, 8, 8)}
        assert crit(shared, shared).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_different(self):
        crit = pd._MatteResponseKDLoss()
        s_out = {"alpha": torch.zeros(1, 1, 8, 8), "fg": torch.zeros(1, 3, 8, 8)}
        t_out = {"alpha": torch.ones(1, 1, 8, 8), "fg": torch.ones(1, 3, 8, 8)}
        assert crit(s_out, t_out).item() > 0

    def test_scalar_output(self):
        crit = pd._MatteResponseKDLoss()
        s_out = {"alpha": torch.rand(1, 1, 8, 8), "fg": torch.rand(1, 3, 8, 8)}
        t_out = {"alpha": torch.rand(1, 1, 8, 8), "fg": torch.rand(1, 3, 8, 8)}
        assert crit(s_out, t_out).ndim == 0


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


class TestSyntheticBatch:
    def test_input_shape(self):
        inp, _ = pd._synthetic_batch(2, _TINY, CPU)
        assert inp.shape == (2, 4, _TINY, _TINY)

    def test_alpha_range(self):
        _, targets = pd._synthetic_batch(1, _TINY, CPU)
        assert targets["alpha"].min() >= 0.0
        assert targets["alpha"].max() <= 1.0

    def test_fg_range(self):
        _, targets = pd._synthetic_batch(1, _TINY, CPU)
        assert targets["fg"].min() >= 0.0
        assert targets["fg"].max() <= 1.0


# ---------------------------------------------------------------------------
# run_distillation
# ---------------------------------------------------------------------------


class TestRunDistillation:
    def test_returns_one_record_per_step(self):
        records = _run(steps=4)
        assert len(records) == 4

    def test_step_numbers_are_sequential(self):
        records = _run(steps=5)
        assert [r.step for r in records] == list(range(1, 6))

    def test_losses_are_positive(self):
        records = _run(steps=3)
        for r in records:
            assert r.task_loss > 0
            assert r.kd_loss >= 0
            assert r.total_loss > 0

    def test_total_equals_weighted_sum(self):
        """total = 1.0*task + 0.5*kd (the default weights used in _run)."""
        records = _run(steps=2)
        for r in records:
            expected = 1.0 * r.task_loss + 0.5 * r.kd_loss
            assert r.total_loss == pytest.approx(expected, rel=1e-5)

    def test_loss_decreases_over_many_steps(self):
        """With a reasonably high LR and enough steps, total loss should trend down."""
        records = _run(steps=60, img_size=32)
        # Compare first 5 steps avg vs last 5 steps avg
        first_avg = sum(r.total_loss for r in records[:5]) / 5
        last_avg = sum(r.total_loss for r in records[-5:]) / 5
        assert last_avg < first_avg, f"Expected loss to decrease: first={first_avg:.4f}, last={last_avg:.4f}"

    def test_teacher_params_unchanged(self):
        """Teacher parameters must stay frozen (no gradient updates)."""
        teacher = pd._build_teacher(_TINY, use_refiner=False, device=CPU)
        student = pd._build_student(_TINY, use_refiner=False, device=CPU)

        # Snapshot teacher weights before training
        before = {k: v.clone() for k, v in teacher.state_dict().items()}

        pd.run_distillation(
            teacher=teacher,
            student=student,
            device=CPU,
            img_size=_TINY,
            batch_size=1,
            steps=3,
            lr=1e-3,
            task_weight=1.0,
            kd_weight=0.5,
            use_amp=False,
        )

        after = teacher.state_dict()
        for k in before:
            assert torch.allclose(before[k], after[k]), f"Teacher param {k} changed!"

    def test_student_params_updated(self):
        """At least some student parameters must change after training."""
        teacher = pd._build_teacher(_TINY, use_refiner=False, device=CPU)
        student = pd._build_student(_TINY, use_refiner=False, device=CPU)

        before = {k: v.clone() for k, v in student.state_dict().items()}
        pd.run_distillation(
            teacher=teacher,
            student=student,
            device=CPU,
            img_size=_TINY,
            batch_size=1,
            steps=3,
            lr=1e-3,
            task_weight=1.0,
            kd_weight=0.5,
            use_amp=False,
        )
        after = student.state_dict()

        changed = any(not torch.allclose(before[k], after[k]) for k in before)
        assert changed, "No student parameters were updated during distillation!"

    def test_kd_weight_zero_eliminates_kd_loss_from_total(self):
        """Setting kd_weight=0 means kd_loss has no influence on total."""
        teacher = pd._build_teacher(_TINY, use_refiner=False, device=CPU)
        student = pd._build_student(_TINY, use_refiner=False, device=CPU)
        zero_kd = pd.run_distillation(
            teacher=teacher,
            student=student,
            device=CPU,
            img_size=_TINY,
            batch_size=1,
            steps=2,
            lr=1e-3,
            task_weight=1.0,
            kd_weight=0.0,
            use_amp=False,
        )
        for r in zero_kd:
            assert r.total_loss == pytest.approx(r.task_loss, rel=1e-5)


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------


class TestBenchmarkSpeed:
    @pytest.fixture(scope="class")
    def speed_results(self):
        teacher = pd._build_teacher(_TINY, use_refiner=False, device=CPU)
        student = pd._build_student(_TINY, use_refiner=False, device=CPU)
        return pd.benchmark_speed(teacher, student, _TINY, CPU, warmup=1, runs=2, use_amp=False)

    def test_returns_two_records(self, speed_results):
        t_rec, s_rec = speed_results
        assert t_rec is not None
        assert s_rec is not None

    def test_teacher_larger_than_student(self, speed_results):
        t_rec, s_rec = speed_results
        assert t_rec.params_m > s_rec.params_m

    def test_fps_positive(self, speed_results):
        t_rec, s_rec = speed_results
        assert t_rec.fps > 0
        assert s_rec.fps > 0

    def test_timings_populated(self, speed_results):
        t_rec, s_rec = speed_results
        assert len(t_rec.timings_ms) == 2
        assert len(s_rec.timings_ms) == 2


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


class TestWriteCsv:
    def test_csv_has_headers(self, tmp_path):
        records = _run(steps=2)
        out = str(tmp_path / "distill.csv")
        pd.write_csv(records, out)
        content = Path(out).read_text()
        reader = csv.DictReader(StringIO(content))
        assert "step" in reader.fieldnames
        assert "task_loss" in reader.fieldnames
        assert "kd_loss" in reader.fieldnames
        assert "total_loss" in reader.fieldnames

    def test_csv_row_count(self, tmp_path):
        records = _run(steps=3)
        out = str(tmp_path / "distill.csv")
        pd.write_csv(records, out)
        rows = list(csv.DictReader(StringIO(Path(out).read_text())))
        assert len(rows) == 3

    def test_csv_values_match_records(self, tmp_path):
        records = _run(steps=2)
        out = str(tmp_path / "distill.csv")
        pd.write_csv(records, out)
        rows = list(csv.DictReader(StringIO(Path(out).read_text())))
        for rec, row in zip(records, rows, strict=True):
            assert int(row["step"]) == rec.step
            assert float(row["total_loss"]) == pytest.approx(rec.total_loss, rel=1e-4)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_defaults(self):
        args = pd._parse_args([])
        assert args.img_size == 64
        assert args.steps == 20
        assert args.batch_size == 1
        assert args.lr == pytest.approx(1e-4)
        assert args.task_weight == pytest.approx(1.0)
        assert args.kd_weight == pytest.approx(0.5)
        assert args.device == "auto"
        assert not args.no_refiner
        assert not args.no_amp

    def test_custom_flags(self):
        args = pd._parse_args(
            [
                "--img-size",
                "128",
                "--steps",
                "10",
                "--kd-weight",
                "0.8",
                "--device",
                "cpu",
                "--no-refiner",
            ]
        )
        assert args.img_size == 128
        assert args.steps == 10
        assert args.kd_weight == pytest.approx(0.8)
        assert args.device == "cpu"
        assert args.no_refiner is True

    def test_output_flag(self):
        args = pd._parse_args(["--output", "out.csv"])
        assert args.output == "out.csv"


# ---------------------------------------------------------------------------
# End-to-end main()
# ---------------------------------------------------------------------------


class TestMainCli:
    def test_main_returns_zero(self):
        rc = pd.main(
            [
                "--img-size",
                "32",
                "--steps",
                "5",
                "--device",
                "cpu",
                "--no-refiner",
                "--no-amp",
                "--warmup",
                "1",
                "--time-runs",
                "2",
            ]
        )
        assert rc == 0

    def test_main_prints_convergence_info(self, capsys):
        pd.main(
            [
                "--img-size",
                "32",
                "--steps",
                "40",
                "--device",
                "cpu",
                "--no-refiner",
                "--no-amp",
                "--warmup",
                "1",
                "--time-runs",
                "2",
            ]
        )
        out = capsys.readouterr().out
        assert "Training summary" in out
        assert "Loss decreased" in out or "Loss did not decrease" in out

    def test_main_writes_csv(self, tmp_path):
        out = str(tmp_path / "distill.csv")
        pd.main(
            [
                "--img-size",
                "32",
                "--steps",
                "3",
                "--device",
                "cpu",
                "--no-refiner",
                "--no-amp",
                "--warmup",
                "1",
                "--time-runs",
                "2",
                "--output",
                out,
            ]
        )
        rows = list(csv.DictReader(StringIO(Path(out).read_text())))
        assert len(rows) == 3

    def test_main_shows_model_comparison(self, capsys):
        pd.main(
            [
                "--img-size",
                "32",
                "--steps",
                "3",
                "--device",
                "cpu",
                "--no-refiner",
                "--no-amp",
                "--warmup",
                "1",
                "--time-runs",
                "2",
            ]
        )
        out = capsys.readouterr().out
        assert "Model comparison" in out
        assert "GreenFormer" in out
        assert "GreenFormerSmall" in out
        assert "smaller" in out
