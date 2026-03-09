#!/usr/bin/env python3
"""
poc_scale.py — CorridorKey inference scaling proof-of-concept.

Purpose
-------
An RTX 3090 has 24 GB of VRAM, but CorridorKey at its native 2048×2048 resolution
uses ~22.7 GB, leaving barely 1.3 GB of headroom.  When the GPU is run at a reduced
power limit (underpowering), the lower memory-bandwidth throughput means that even if
VRAM fits, latency per frame increases.

This script sweeps a configurable range of ``img_size`` values, runs a small number of
timed forward passes for each size with synthetic inputs (no real footage or checkpoint
required), and prints a summary table so you can choose the right resolution for your
hardware budget.

Usage
-----
::

    # Quick PoC on any machine (CPU, no weights needed):
    python poc_scale.py

    # CUDA — test a specific size range:
    python poc_scale.py --device cuda --sizes 512 768 1024 1280 1536 2048

    # Disable the CNN refiner (saves ~1-2 GB VRAM):
    python poc_scale.py --device cuda --no-refiner

    # Save results to CSV:
    python poc_scale.py --device cuda --output results.csv

    # Increase warmup/runs for more stable timing:
    python poc_scale.py --device cuda --warmup 3 --runs 10

Output columns
--------------
``size``        — img_size fed to the model (square, pixels per side)
``vram_gb``     — peak VRAM allocated during inference (CUDA only; 0.00 on CPU/MPS)
``mean_ms``     — mean wall-clock time per forward pass (ms)
``p95_ms``      — 95th-percentile latency (ms)
``fps``         — estimated throughput (frames/second) based on mean latency
``speedup``     — relative speedup vs the baseline size (first entry in --sizes)
``status``      — ``OK`` or ``OOM`` (out-of-memory)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field, fields
from typing import Optional

import torch
import torch.nn as nn

from CorridorKeyModule.core.model_transformer import GreenFormer

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    size: int
    vram_gb: float
    mean_ms: float
    p95_ms: float
    fps: float
    speedup: float
    status: str
    # Raw per-run timings kept for programmatic consumers (not printed in table)
    timings_ms: list[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def _build_model(img_size: int, use_refiner: bool, device: torch.device) -> nn.Module:
    """Instantiate GreenFormer with random weights — no checkpoint needed."""
    model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
        img_size=img_size,
        use_refiner=use_refiner,
    )
    model = model.to(device)
    model.eval()
    return model


def _make_input(img_size: int, device: torch.device) -> torch.Tensor:
    """Create a random [1, 4, img_size, img_size] float32 tensor."""
    return torch.randn(1, 4, img_size, img_size, device=device)


def _run_single(
    img_size: int,
    use_refiner: bool,
    device: torch.device,
    warmup: int,
    runs: int,
    use_amp: bool,
) -> BenchResult:
    """Benchmark one img_size. Returns BenchResult (status='OOM' on failure)."""
    # Build model and input — catch OOM at construction time
    model: nn.Module | None = None
    try:
        model = _build_model(img_size, use_refiner, device)
    except torch.cuda.OutOfMemoryError:
        return BenchResult(img_size, 0.0, 0.0, 0.0, 0.0, 0.0, "OOM")

    inp = _make_input(img_size, device)

    def _forward() -> None:
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(inp)  # type: ignore[misc]
        else:
            model(inp)  # type: ignore[misc]

    # --- Warmup ---
    with torch.no_grad():
        for _ in range(warmup):
            try:
                _forward()
                _sync(device)
            except torch.cuda.OutOfMemoryError:
                return BenchResult(img_size, 0.0, 0.0, 0.0, 0.0, 0.0, "OOM")

    # --- Reset VRAM stats before timed runs ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            try:
                t0 = _now(device)
                _forward()
                _sync(device)
                timings.append((_now(device) - t0) * 1000.0)  # → ms
            except torch.cuda.OutOfMemoryError:
                break

    if not timings:
        return BenchResult(img_size, 0.0, 0.0, 0.0, 0.0, 0.0, "OOM")

    # --- Collect VRAM ---
    vram_gb = 0.0
    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    mean_ms = statistics.mean(timings)
    p95_ms = sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 1 else timings[0]

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return BenchResult(
        size=img_size,
        vram_gb=vram_gb,
        mean_ms=mean_ms,
        p95_ms=p95_ms,
        fps=1000.0 / mean_ms if mean_ms > 0 else 0.0,
        speedup=0.0,  # filled in by caller
        status="OK",
        timings_ms=timings,
    )


def run_benchmark(
    sizes: list[int],
    device: torch.device,
    warmup: int = 2,
    runs: int = 5,
    use_refiner: bool = True,
    use_amp: bool = True,
) -> list[BenchResult]:
    """Run the full sweep and return results with speedups normalised to the first OK entry."""
    results: list[BenchResult] = []
    baseline_ms: Optional[float] = None

    for size in sizes:
        print(f"  Benchmarking img_size={size} … ", end="", flush=True)
        r = _run_single(size, use_refiner, device, warmup, runs, use_amp)
        if r.status == "OK":
            if baseline_ms is None:
                baseline_ms = r.mean_ms
            r.speedup = baseline_ms / r.mean_ms if r.mean_ms > 0 else 0.0
        print(r.status)
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


_COL_WIDTH = {
    "size": 6,
    "vram_gb": 9,
    "mean_ms": 9,
    "p95_ms": 9,
    "fps": 7,
    "speedup": 9,
    "status": 8,
}

_HEADERS = {
    "size": "size",
    "vram_gb": "vram(GB)",
    "mean_ms": "mean(ms)",
    "p95_ms": "p95(ms)",
    "fps": "fps",
    "speedup": "speedup",
    "status": "status",
}


def print_table(results: list[BenchResult]) -> None:
    """Print a neat fixed-width summary table to stdout."""
    cols = list(_HEADERS.keys())
    header = "  ".join(h.rjust(_COL_WIDTH[c]) for c, h in zip(cols, _HEADERS.values(), strict=True))
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        row_vals = {
            "size": str(r.size),
            "vram_gb": f"{r.vram_gb:.2f}" if r.status == "OK" else "—",
            "mean_ms": f"{r.mean_ms:.1f}" if r.status == "OK" else "—",
            "p95_ms": f"{r.p95_ms:.1f}" if r.status == "OK" else "—",
            "fps": f"{r.fps:.3f}" if r.status == "OK" else "—",
            "speedup": f"{r.speedup:.2f}×" if r.status == "OK" else "—",
            "status": r.status,
        }
        print("  ".join(row_vals[c].rjust(_COL_WIDTH[c]) for c in cols))
    print(sep)


def write_csv(results: list[BenchResult], path: str) -> None:
    """Write results to a CSV file."""
    col_names = [f.name for f in fields(BenchResult) if f.name != "timings_ms"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=col_names)
        writer.writeheader()
        for r in results:
            writer.writerow({c: getattr(r, c) for c in col_names})
    print(f"Results saved → {path}")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _now(device: torch.device) -> float:
    """Return current time in seconds, synchronised to the device clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# NVIDIA power-limit introspection (optional)
# ---------------------------------------------------------------------------


def _query_gpu_info() -> Optional[str]:
    """
    Try to read current GPU power limit via nvidia-smi.
    Returns a human-readable string or None if unavailable.
    """
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,power.limit,power.draw,clocks.sm,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            text=True,
        )
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        # First GPU
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) >= 5:
            name, pl, pd, sm, mem = parts[:5]
            return (
                f"{name}  |  power_limit={pl} W  power_draw={pd} W  "
                f"sm_clock={sm} MHz  vram={mem} MiB"
            )
        return lines[0]
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorridorKey inference scaling PoC — benchmark VRAM & throughput at different resolutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[256, 512, 768, 1024, 1280, 1536, 2048],
        metavar="N",
        help="img_size values to benchmark (default: 256 512 768 1024 1280 1536 2048)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto | cpu | cuda | cuda:0 (default: auto)",
    )
    p.add_argument("--warmup", type=int, default=2, metavar="N", help="Warmup passes per size (default: 2)")
    p.add_argument("--runs", type=int, default=5, metavar="N", help="Timed passes per size (default: 5)")
    p.add_argument(
        "--no-refiner",
        action="store_true",
        help="Disable CNN refiner module (~saves 1-2 GB VRAM)",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (fp16 on CUDA, default: on)",
    )
    p.add_argument("--output", metavar="FILE", help="Write CSV results to FILE (optional)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # --- Resolve device ---
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

    # --- Header ---
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       CorridorKey  —  Inference Scaling  PoC                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  device   : {device}")
    print(f"  sizes    : {args.sizes}")
    print(f"  warmup   : {args.warmup}   runs: {args.runs}")
    print(f"  refiner  : {'enabled' if use_refiner else 'DISABLED  (--no-refiner)'}")
    print(f"  amp      : {'fp16 (CUDA)' if use_amp else 'DISABLED  (--no-amp)'}")

    gpu_info = _query_gpu_info()
    if gpu_info:
        print(f"  GPU      : {gpu_info}")

    print()

    # --- Benchmark ---
    sizes = sorted(set(args.sizes))
    results = run_benchmark(
        sizes=sizes,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
        use_refiner=use_refiner,
        use_amp=use_amp,
    )

    # --- Display ---
    print()
    print_table(results)
    print()

    # --- Recommendation ---
    ok_results = [r for r in results if r.status == "OK"]
    if ok_results and device.type == "cuda":
        max_vram = max(r.vram_gb for r in ok_results)
        recommended = max(ok_results, key=lambda r: r.size)
        print(
            f"  Largest fitting size  : {recommended.size}×{recommended.size}  "
            f"({recommended.vram_gb:.2f} GB peak VRAM,  {recommended.fps:.3f} FPS)"
        )
        print(
            f"  Total VRAM headroom   : {max_vram:.2f} GB used  "
            f"(24 GB 3090 → ~{24.0 - max_vram:.1f} GB free)"
        )
        if ok_results[-1].speedup > 1.0:
            fastest = min(ok_results, key=lambda r: r.mean_ms)
            print(
                f"  Fastest setting       : {fastest.size}×{fastest.size}  "
                f"({fastest.speedup:.2f}× speedup vs {ok_results[0].size}×{ok_results[0].size})"
            )
        print()

    # --- CSV ---
    if args.output:
        write_csv(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
