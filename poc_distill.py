#!/usr/bin/env python3
"""
poc_distill.py — CorridorKey knowledge distillation proof-of-concept.

Demonstrates the full teacher→student distillation loop on purely synthetic
data.  No real footage, no pre-trained weights, and no GPU are required to
run this script — it works on any machine including the CPU fallback path.

What this PoC shows
-------------------
1. **Model comparison** — parameter counts, estimated VRAM, and inference
   latency for GreenFormer (teacher, 70 M params) vs GreenFormerSmall
   (student, ~35 M params, ~2× smaller).

2. **Distillation training loop** — ``N`` optimiser steps where the frozen
   teacher's soft predictions guide the student via two losses:

   * ``task_loss``  — MSE between the student output and synthetic GT targets
     (represents real-data supervision; analogous to ``MatteTaskLoss``).
   * ``kd_loss``    — MSE between the student and teacher soft predictions
     (response-based KD; analogous to ``MatteResponseKDLoss``).
   * ``total``      — ``task_weight * task_loss + kd_weight * kd_loss``

3. **Convergence signal** — total loss should decrease monotonically,
   confirming gradients flow correctly through the student.

4. **Speed comparison** — wall-clock inference latency (teacher vs student)
   measured after training.

Production path
---------------
Once you are happy with the concept, swap the synthetic data for real footage
using ``distillation/train_distill.py`` (which wraps ``DistillationBox`` from
torchdistill and reads ``distillation/configs/distill_greenformer.yaml``)::

    python -m distillation.train_distill \\
        --config  distillation/configs/distill_greenformer.yaml \\
        --data-root /path/to/dataset \\
        --teacher-ckpt /path/to/teacher.pth \\
        --output-dir ./checkpoints/student

Usage
-----
::

    # Instant smoke-test — CPU, tiny size, 10 steps:
    python poc_distill.py

    # CUDA — more realistic resolution and more steps:
    python poc_distill.py --device cuda --img-size 256 --steps 50

    # Save the per-step losses to CSV for plotting:
    python poc_distill.py --device cuda --steps 100 --output distill_run.csv

    # Tune distillation balance (higher kd-weight = more teacher mimicry):
    python poc_distill.py --kd-weight 1.0 --task-weight 0.5

Options
-------
--img-size    Resolution for teacher + student forward passes (default: 64).
--steps       Number of distillation optimiser steps (default: 20).
--batch-size  Synthetic mini-batch size (default: 1).
--lr          Student optimiser learning rate (default: 1e-4).
--task-weight Multiplier for the GT task loss component (default: 1.0).
--kd-weight   Multiplier for the teacher KD loss component (default: 0.5).
--device      cpu | cuda | auto  (default: auto).
--no-refiner  Disable CNN refiner in both teacher and student.
--no-amp      Disable automatic mixed-precision (fp16) on CUDA.
--output      Write per-step CSV to FILE (optional).
--warmup      Warmup forward passes before timing (default: 2).
--time-runs   Timed passes per model for the speed comparison (default: 5).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field, fields

import torch
import torch.nn as nn

from CorridorKeyModule.core.model_transformer import GreenFormer, GreenFormerSmall

# ---------------------------------------------------------------------------
# Loss functions
# (Pure nn.Module implementations — no torchdistill dependency needed.
#  The production versions in distillation/losses.py add the @register_mid_level_loss
#  decorator so torchdistill can look them up by name from the YAML config.)
# ---------------------------------------------------------------------------


class _MatteTaskLoss(nn.Module):
    """MSE between student output and synthetic GT targets."""

    def __init__(self, alpha_weight: float = 1.0, fg_weight: float = 1.0) -> None:
        super().__init__()
        self.alpha_weight = alpha_weight
        self.fg_weight = fg_weight
        self._mse = nn.MSELoss()

    def forward(
        self,
        student_out: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        alpha_loss = self._mse(student_out["alpha"], targets["alpha"])
        fg_loss = self._mse(student_out["fg"], targets["fg"])
        return self.alpha_weight * alpha_loss + self.fg_weight * fg_loss


class _MatteResponseKDLoss(nn.Module):
    """MSE between student soft predictions and frozen teacher soft predictions."""

    def __init__(self, alpha_weight: float = 0.5, fg_weight: float = 0.5) -> None:
        super().__init__()
        self.alpha_weight = alpha_weight
        self.fg_weight = fg_weight
        self._mse = nn.MSELoss()

    def forward(
        self,
        student_out: dict[str, torch.Tensor],
        teacher_out: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        alpha_kd = self._mse(student_out["alpha"], teacher_out["alpha"])
        fg_kd = self._mse(student_out["fg"], teacher_out["fg"])
        return self.alpha_weight * alpha_kd + self.fg_weight * fg_kd


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    task_loss: float
    kd_loss: float
    total_loss: float


@dataclass
class SpeedRecord:
    model: str
    params_m: float
    mean_ms: float
    fps: float
    vram_gb: float
    timings_ms: list[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _synthetic_batch(
    batch_size: int,
    img_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Returns (inp, targets) where:
      inp     : [B, 4, H, W]  — normalised RGB (channels 0-2) + alpha hint (channel 3)
      targets : alpha [B,1,H,W] and fg [B,3,H,W] in [0,1]
    """
    inp = torch.randn(batch_size, 4, img_size, img_size, device=device)
    targets = {
        "alpha": torch.rand(batch_size, 1, img_size, img_size, device=device),
        "fg": torch.rand(batch_size, 3, img_size, img_size, device=device),
    }
    return inp, targets


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _build_teacher(img_size: int, use_refiner: bool, device: torch.device) -> GreenFormer:
    teacher = GreenFormer(img_size=img_size, use_refiner=use_refiner)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _build_student(img_size: int, use_refiner: bool, device: torch.device) -> GreenFormerSmall:
    student = GreenFormerSmall(img_size=img_size, use_refiner=use_refiner)
    student = student.to(device)
    student.train()
    return student


# ---------------------------------------------------------------------------
# Distillation training loop
# ---------------------------------------------------------------------------


def run_distillation(
    teacher: GreenFormer,
    student: GreenFormerSmall,
    device: torch.device,
    img_size: int,
    batch_size: int,
    steps: int,
    lr: float,
    task_weight: float,
    kd_weight: float,
    use_amp: bool,
) -> list[StepRecord]:
    """
    Run ``steps`` distillation optimiser steps on synthetic data.

    Returns a list of :class:`StepRecord` (one per step).
    """
    task_criterion = _MatteTaskLoss()
    kd_criterion = _MatteResponseKDLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    # GradScaler is created once and reused across all steps (CUDA AMP only)
    scaler = torch.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    records: list[StepRecord] = []

    for step in range(1, steps + 1):
        inp, targets = _synthetic_batch(batch_size, img_size, device)

        # 1. Teacher soft labels (no gradients)
        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    teacher_out = teacher(inp)
            else:
                teacher_out = teacher(inp)

        # 2. Student forward (gradients enabled)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                student_out = student(inp)
                task_loss = task_criterion(student_out, targets)
                kd_loss = kd_criterion(student_out, teacher_out)
                total = task_weight * task_loss + kd_weight * kd_loss
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            student_out = student(inp)
            task_loss = task_criterion(student_out, targets)
            kd_loss = kd_criterion(student_out, teacher_out)
            total = task_weight * task_loss + kd_weight * kd_loss
            total.backward()
            optimizer.step()

        records.append(
            StepRecord(
                step=step,
                task_loss=task_loss.item(),
                kd_loss=kd_loss.item(),
                total_loss=total.item(),
            )
        )

    return records


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------


def _time_model(
    model: nn.Module,
    img_size: int,
    device: torch.device,
    warmup: int,
    runs: int,
    use_amp: bool,
) -> list[float]:
    """Return per-run wall-clock times (ms)."""
    inp = torch.randn(1, 4, img_size, img_size, device=device)
    model.eval()

    def _forward() -> None:
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(inp)
        else:
            model(inp)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            _forward()
            _sync()

    timings = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _forward()
            _sync()
            timings.append((time.perf_counter() - t0) * 1000.0)

    return timings


def benchmark_speed(
    teacher: GreenFormer,
    student: GreenFormerSmall,
    img_size: int,
    device: torch.device,
    warmup: int,
    runs: int,
    use_amp: bool,
) -> tuple[SpeedRecord, SpeedRecord]:
    """Benchmark inference latency for teacher and student."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    teacher_times = _time_model(teacher, img_size, device, warmup, runs, use_amp)
    teacher_vram = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    student_times = _time_model(student, img_size, device, warmup, runs, use_amp)
    student_vram = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0

    def _sr(name: str, model: nn.Module, timings: list[float], vram: float) -> SpeedRecord:
        mean = statistics.mean(timings)
        return SpeedRecord(
            model=name,
            params_m=_count_params(model) / 1e6,
            mean_ms=mean,
            fps=1000.0 / mean if mean > 0 else 0.0,
            vram_gb=vram,
            timings_ms=timings,
        )

    return (
        _sr("teacher (GreenFormer)", teacher, teacher_times, teacher_vram),
        _sr("student (GreenFormerSmall)", student, student_times, student_vram),
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 70


def _header(title: str) -> None:
    print()
    print(_SEP)
    print(f"  {title}")
    print(_SEP)


def print_comparison(teacher_rec: SpeedRecord, student_rec: SpeedRecord) -> None:
    """Print model size and speed comparison."""
    _header("Model comparison")
    fmt = "  {:<35} {:>10} {:>12} {:>10}"
    print(fmt.format("", "params (M)", "mean (ms)", "vram (GB)"))
    print(fmt.format("─" * 35, "─" * 10, "─" * 12, "─" * 10))

    def _row(rec: SpeedRecord) -> None:
        vram = f"{rec.vram_gb:.2f}" if rec.vram_gb > 0 else "n/a (CPU)"
        print(fmt.format(rec.model, f"{rec.params_m:.1f}", f"{rec.mean_ms:.1f}", vram))

    _row(teacher_rec)
    _row(student_rec)

    speedup = teacher_rec.mean_ms / student_rec.mean_ms if student_rec.mean_ms > 0 else 0.0
    size_ratio = teacher_rec.params_m / student_rec.params_m if student_rec.params_m > 0 else 0.0
    print()
    print(f"  Student is {size_ratio:.2f}× smaller  |  {speedup:.2f}× faster inference")
    if teacher_rec.vram_gb > 0 and student_rec.vram_gb > 0:
        vram_save = teacher_rec.vram_gb - student_rec.vram_gb
        print(f"  VRAM saved: {vram_save:.2f} GB  ({vram_save / teacher_rec.vram_gb * 100:.0f}% reduction)")


def print_training_header() -> None:
    _header("Distillation training — per-step losses")
    print(f"  {'step':>6}  {'task_loss':>12}  {'kd_loss':>12}  {'total':>12}  {'Δtotal':>10}")
    print(f"  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 10}")


def print_step(rec: StepRecord, prev_total: float | None) -> None:
    delta = ""
    if prev_total is not None:
        diff = rec.total_loss - prev_total
        delta = f"{diff:+.6f}"
    print(f"  {rec.step:>6}  {rec.task_loss:>12.6f}  {rec.kd_loss:>12.6f}  {rec.total_loss:>12.6f}  {delta:>10}")


def print_training_summary(records: list[StepRecord]) -> None:
    if len(records) < 2:
        return
    first = records[0].total_loss
    last = records[-1].total_loss
    reduced = first - last
    pct = reduced / first * 100 if first > 0 else 0.0
    _header("Training summary")
    print(f"  Initial total loss : {first:.6f}")
    print(f"  Final   total loss : {last:.6f}")
    print(f"  Reduction          : {reduced:.6f}  ({pct:.1f}%)")
    if last < first:
        print("  ✓ Loss decreased — distillation gradients flowing correctly.")
    else:
        print("  ⚠ Loss did not decrease (expected on random data with very few steps).")


def write_csv(records: list[StepRecord], path: str) -> None:
    col_names = [f.name for f in fields(StepRecord)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=col_names)
        writer.writeheader()
        for r in records:
            writer.writerow({c: getattr(r, c) for c in col_names})
    print(f"\n  Per-step losses saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorridorKey distillation PoC — teacher→student KD on synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--img-size", type=int, default=64, metavar="N", help="Square resolution for synthetic tensors (default: 64)"
    )
    p.add_argument(
        "--steps", type=int, default=20, metavar="N", help="Number of distillation optimiser steps (default: 20)"
    )
    p.add_argument("--batch-size", type=int, default=1, metavar="N", help="Synthetic mini-batch size (default: 1)")
    p.add_argument("--lr", type=float, default=1e-4, help="Student learning rate (default: 1e-4)")
    p.add_argument("--task-weight", type=float, default=1.0, help="Weight for the GT task loss (default: 1.0)")
    p.add_argument("--kd-weight", type=float, default=0.5, help="Weight for the teacher KD loss (default: 0.5)")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto  (default: auto)")
    p.add_argument("--no-refiner", action="store_true", help="Disable CNN refiner in both models")
    p.add_argument("--no-amp", action="store_true", help="Disable automatic mixed-precision on CUDA")
    p.add_argument("--warmup", type=int, default=2, metavar="N", help="Warmup passes for speed benchmark (default: 2)")
    p.add_argument(
        "--time-runs", type=int, default=5, metavar="N", help="Timed passes per model for speed comparison (default: 5)"
    )
    p.add_argument("--output", metavar="FILE", help="Write per-step CSV to FILE (optional)")
    p.add_argument("--log-every", type=int, default=1, metavar="N", help="Print loss every N steps (default: 1)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    use_refiner = not args.no_refiner
    use_amp = not args.no_amp

    # ── Banner ──────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        CorridorKey  —  Distillation  PoC                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  device      : {device}")
    print(f"  img_size    : {args.img_size}×{args.img_size}")
    print(f"  steps       : {args.steps}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  lr          : {args.lr}")
    print(f"  task_weight : {args.task_weight}   kd_weight: {args.kd_weight}")
    print(f"  refiner     : {'enabled' if use_refiner else 'disabled (--no-refiner)'}")
    print(f"  amp         : {'fp16 on CUDA' if use_amp else 'disabled (--no-amp)'}")
    print()
    print("  Losses used:")
    print("    task_loss — MSE(student output, synthetic GT)  ← MatteTaskLoss")
    print("    kd_loss   — MSE(student output, teacher output) ← MatteResponseKDLoss")
    print(f"    total     = {args.task_weight}×task_loss + {args.kd_weight}×kd_loss")

    # ── Build models ─────────────────────────────────────────────────────────
    print()
    print("Building teacher (GreenFormer) and student (GreenFormerSmall)...")
    teacher = _build_teacher(args.img_size, use_refiner, device)
    student = _build_student(args.img_size, use_refiner, device)

    t_params = _count_params(teacher)
    s_params = _count_params(student)
    print(f"  Teacher params : {t_params:,}  ({t_params / 1e6:.1f} M)")
    print(f"  Student params : {s_params:,}  ({s_params / 1e6:.1f} M)  [{t_params / s_params:.2f}× smaller]")

    # ── Distillation loop ─────────────────────────────────────────────────────
    print_training_header()

    records: list[StepRecord] = []
    prev_total: float | None = None

    for rec in run_distillation(
        teacher=teacher,
        student=student,
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        task_weight=args.task_weight,
        kd_weight=args.kd_weight,
        use_amp=use_amp,
    ):
        records.append(rec)
        if rec.step == 1 or rec.step % args.log_every == 0 or rec.step == args.steps:
            print_step(rec, prev_total)
        prev_total = rec.total_loss

    print_training_summary(records)

    # ── Speed comparison ──────────────────────────────────────────────────────
    _header("Inference speed comparison  (random weights — pure latency)")
    print("  Timing forward passes …")
    teacher_sr, student_sr = benchmark_speed(
        teacher=teacher,
        student=student,
        img_size=args.img_size,
        device=device,
        warmup=args.warmup,
        runs=args.time_runs,
        use_amp=use_amp,
    )
    print_comparison(teacher_sr, student_sr)

    # ── CSV export ────────────────────────────────────────────────────────────
    if args.output:
        write_csv(records, args.output)

    # ── Footer ────────────────────────────────────────────────────────────────
    print()
    print(_SEP)
    print("  Next step: train on real data with")
    print("    python -m distillation.train_distill \\")
    print("        --config  distillation/configs/distill_greenformer.yaml \\")
    print("        --data-root /path/to/your/dataset \\")
    print("        --teacher-ckpt /path/to/teacher.pth \\")
    print("        --output-dir ./checkpoints/student")
    print(_SEP)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
