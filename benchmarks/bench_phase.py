"""Benchmark script for CorridorKey optimization phases.

Measures three metrics per run:
  1. Execution time (per-frame, excluding warmup)
  2. Memory usage (MPS unified / CUDA peak)
  3. Pixel difference from baseline (per-channel MAE, max error, PSNR)

Usage:
  # Generate baseline (run once before any optimizations)
  uv run python benchmarks/bench_phase.py --generate-baseline --clip <path_to_input_video> --alpha <path_to_alpha_video>

  # Benchmark current code against baseline
  uv run python benchmarks/bench_phase.py --clip <input_video> --alpha <alpha_video>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np
import torch

# Ensure EXR support before any other cv2 usage
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

MAX_BENCHMARK_FRAMES = 20


def load_video_frames(video_path: str, max_frames: int = MAX_BENCHMARK_FRAMES) -> list[np.ndarray]:
    """Load up to max_frames from a video file as float32 RGB [0,1]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb.astype(np.float32) / 255.0)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    print(f"Loaded {len(frames)} frames from {os.path.basename(video_path)}")
    return frames


def load_mask_frames(video_path: str, max_frames: int = MAX_BENCHMARK_FRAMES) -> list[np.ndarray]:
    """Load up to max_frames from a mask video as float32 [H,W] [0,1]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open mask video: {video_path}")

    masks = []
    while len(masks) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Take blue channel (same as clip_manager.py convention)
        mask = frame[:, :, 2].astype(np.float32) / 255.0
        masks.append(mask)
    cap.release()

    if not masks:
        raise ValueError(f"No frames read from {video_path}")
    print(f"Loaded {len(masks)} mask frames from {os.path.basename(video_path)}")
    return masks


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_checkpoint() -> str:
    """Find first .pth checkpoint in standard locations."""
    search_dirs = [
        os.path.join(PROJECT_ROOT, "CorridorKeyModule", "checkpoints"),
        os.path.join(PROJECT_ROOT, "CorridorKeyModule", "IgnoredCheckpoints"),
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".pth"):
                    return os.path.join(d, f)
    raise FileNotFoundError("No .pth checkpoint found in CorridorKeyModule/checkpoints/ or IgnoredCheckpoints/")


def create_engine(device: str, checkpoint: str | None = None, backbone_size: int | None = None):
    """Create a CorridorKeyEngine instance."""
    from CorridorKeyModule.inference_engine import CorridorKeyEngine

    ckpt = checkpoint or find_checkpoint()
    print(f"Device: {device}")
    print(f"Checkpoint: {os.path.basename(ckpt)}")
    if backbone_size:
        print(f"Backbone size: {backbone_size}")
    return CorridorKeyEngine(ckpt, device=device, backbone_size=backbone_size)


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------


def measure_memory_before(device: str) -> int:
    """Capture memory baseline before inference."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    elif device == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory()
    return 0


def measure_memory_after(device: str) -> dict:
    """Capture memory after inference."""
    if device == "cuda":
        torch.cuda.synchronize()
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "peak_bytes": torch.cuda.max_memory_allocated(),
        }
    elif device == "mps":
        torch.mps.synchronize()
        return {
            "allocated_bytes": torch.mps.driver_allocated_memory(),
            "peak_bytes": torch.mps.driver_allocated_memory(),  # MPS has no true peak tracker
        }
    return {"allocated_bytes": 0, "peak_bytes": 0}


# ---------------------------------------------------------------------------
# Pixel difference reporting
# ---------------------------------------------------------------------------


def pixel_diff_report(baseline: np.ndarray, result: np.ndarray, label: str) -> dict:
    """Per-channel divergence report between baseline and current result."""
    abs_diff = np.abs(baseline.astype(np.float64) - result.astype(np.float64))
    max_err = float(abs_diff.max())
    mae = float(abs_diff.mean())
    pct_above_1e4 = float((abs_diff > 1e-4).mean() * 100)
    pct_above_1e2 = float((abs_diff > 1e-2).mean() * 100)

    # PSNR (peak = 1.0 for [0,1] data)
    mse = float(np.mean(abs_diff**2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))

    print(f"  [{label}] Max err: {max_err:.6f}, MAE: {mae:.6f}, PSNR: {psnr:.1f} dB")
    print(f"  [{label}] Pixels > 1e-4: {pct_above_1e4:.2f}%, > 1e-2: {pct_above_1e2:.2f}%")

    return {
        "max_err": max_err,
        "mae": mae,
        "psnr": psnr,
        "pct_above_1e4": pct_above_1e4,
        "pct_above_1e2": pct_above_1e2,
    }


def generate_diff_heatmap(baseline: np.ndarray, result: np.ndarray, output_path: str):
    """Generate amplified difference heatmap for visual inspection."""
    diff = np.abs(baseline.astype(np.float64) - result.astype(np.float64))
    # Average across channels if multi-channel
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    # Amplify 10x and clamp
    diff_vis = np.clip(diff * 10.0, 0.0, 1.0)
    cv2.imwrite(output_path, (diff_vis * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def run_benchmark(engine, frames, masks, device: str) -> tuple[list[dict], dict, list[float]]:
    """Run inference on all frames, return (outputs, memory_info, frame_times).

    Returns per-frame outputs as list of dicts with numpy arrays.
    """
    outputs = []
    frame_times = []

    mem_before = measure_memory_before(device)

    for i, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        # Sync before timing
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        result = engine.process_frame(frame, mask)

        # Sync after inference for accurate timing
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        t1 = time.perf_counter()

        frame_times.append(t1 - t0)
        outputs.append(result)

        print(f"  Frame {i + 1}/{len(frames)}: {t1 - t0:.3f}s", end="\r")

    print()

    mem_info = measure_memory_after(device)
    mem_info["before_bytes"] = mem_before
    mem_info["delta_bytes"] = mem_info["allocated_bytes"] - mem_before

    return outputs, mem_info, frame_times


def print_timing_summary(frame_times: list[float]):
    """Print timing stats, excluding first frame (warmup)."""
    if len(frame_times) <= 1:
        print(f"  Only {len(frame_times)} frame(s) — insufficient for stats after warmup exclusion")
        return

    steady = frame_times[1:]  # Exclude warmup
    print(f"  Warmup frame: {frame_times[0]:.3f}s")
    print(f"  Mean:   {statistics.mean(steady):.3f}s")
    print(f"  Median: {statistics.median(steady):.3f}s")
    if len(steady) > 1:
        print(f"  Stdev:  {statistics.stdev(steady):.4f}s")
    print(f"  Min:    {min(steady):.3f}s")
    print(f"  Max:    {max(steady):.3f}s")


def print_memory_summary(mem_info: dict):
    """Print memory usage stats."""
    gb = 1e9
    print(f"  Before inference: {mem_info['before_bytes'] / gb:.2f} GB")
    print(f"  After inference:  {mem_info['allocated_bytes'] / gb:.2f} GB")
    print(f"  Delta:            {mem_info['delta_bytes'] / gb:.2f} GB")
    print(f"  Peak:             {mem_info['peak_bytes'] / gb:.2f} GB")


# ---------------------------------------------------------------------------
# Baseline save / load
# ---------------------------------------------------------------------------


def save_baseline(outputs: list[dict], frame_times: list[float], mem_info: dict, baseline_dir: str):
    """Save baseline outputs as .npy files + timing/memory as JSON."""
    os.makedirs(baseline_dir, exist_ok=True)

    for i, result in enumerate(outputs):
        frame_id = f"frame_{i + 1:03d}"
        np.save(os.path.join(baseline_dir, f"{frame_id}_alpha.npy"), result["alpha"])
        np.save(os.path.join(baseline_dir, f"{frame_id}_fg.npy"), result["fg"])
        np.save(os.path.join(baseline_dir, f"{frame_id}_processed.npy"), result["processed"])
        np.save(os.path.join(baseline_dir, f"{frame_id}_comp.npy"), result["comp"])

    # Save timing
    steady = frame_times[1:] if len(frame_times) > 1 else frame_times
    timing = {
        "warmup_s": frame_times[0] if frame_times else 0,
        "mean_s": statistics.mean(steady) if steady else 0,
        "median_s": statistics.median(steady) if steady else 0,
        "stdev_s": statistics.stdev(steady) if len(steady) > 1 else 0,
        "all_frame_times_s": frame_times,
    }
    with open(os.path.join(baseline_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    # Save memory
    with open(os.path.join(baseline_dir, "memory.json"), "w") as f:
        json.dump(mem_info, f, indent=2)

    print(f"\nBaseline saved to {baseline_dir}/ ({len(outputs)} frames)")


def load_baseline(baseline_dir: str) -> list[dict]:
    """Load baseline .npy files."""
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    outputs = []
    i = 0
    while True:
        i += 1
        frame_id = f"frame_{i:03d}"
        alpha_path = os.path.join(baseline_dir, f"{frame_id}_alpha.npy")
        if not os.path.exists(alpha_path):
            break
        outputs.append(
            {
                "alpha": np.load(os.path.join(baseline_dir, f"{frame_id}_alpha.npy")),
                "fg": np.load(os.path.join(baseline_dir, f"{frame_id}_fg.npy")),
                "processed": np.load(os.path.join(baseline_dir, f"{frame_id}_processed.npy")),
                "comp": np.load(os.path.join(baseline_dir, f"{frame_id}_comp.npy")),
            }
        )

    if not outputs:
        raise ValueError(f"No baseline frames found in {baseline_dir}")
    print(f"Loaded {len(outputs)} baseline frames from {baseline_dir}")
    return outputs


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

OUTPUT_CHANNELS = {
    "alpha": "Alpha (linear)",
    "fg": "FG (sRGB)",
    "processed": "Processed RGBA (linear premul)",
    "comp": "Composite (sRGB)",
}


def compare_to_baseline(current: list[dict], baseline: list[dict], diff_dir: str | None = None):
    """Compare current outputs to baseline, print per-channel report."""
    num_frames = min(len(current), len(baseline))
    print(f"\nComparing {num_frames} frames against baseline...")

    all_diffs: dict[str, list[dict]] = {k: [] for k in OUTPUT_CHANNELS}

    for i in range(num_frames):
        for key in OUTPUT_CHANNELS:
            diffs = pixel_diff_report(baseline[i][key], current[i][key], f"Frame {i + 1} {key}")
            all_diffs[key].append(diffs)

            # Generate heatmap if difference exceeds threshold
            if diff_dir and diffs["max_err"] > 1e-4:
                os.makedirs(diff_dir, exist_ok=True)
                heatmap_path = os.path.join(diff_dir, f"frame_{i + 1:03d}_{key}_diff.png")
                generate_diff_heatmap(baseline[i][key], current[i][key], heatmap_path)
                print(f"    -> Heatmap: {heatmap_path}")

    # Aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY (across all frames)")
    print("=" * 70)
    for key, label in OUTPUT_CHANNELS.items():
        diffs = all_diffs[key]
        if not diffs:
            continue
        avg_mae = statistics.mean(d["mae"] for d in diffs)
        worst_max = max(d["max_err"] for d in diffs)
        avg_psnr = statistics.mean(d["psnr"] for d in diffs)
        avg_pct_1e4 = statistics.mean(d["pct_above_1e4"] for d in diffs)
        print(f"  {label}:")
        print(f"    Avg MAE: {avg_mae:.6f}  |  Worst max err: {worst_max:.6f}")
        print(f"    Avg PSNR: {avg_psnr:.1f} dB  |  Avg pixels > 1e-4: {avg_pct_1e4:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CorridorKey phase benchmark")
    parser.add_argument("--clip", required=True, help="Path to input RGB video")
    parser.add_argument("--alpha", required=True, help="Path to alpha hint video")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (auto-detected if omitted)")
    parser.add_argument("--device", default=None, help="Device override (cuda/mps/cpu, auto-detected if omitted)")
    parser.add_argument("--max-frames", type=int, default=MAX_BENCHMARK_FRAMES, help="Max frames to benchmark")
    parser.add_argument(
        "--generate-baseline", action="store_true", help="Run unoptimized pipeline and save baseline outputs"
    )
    parser.add_argument(
        "--baseline", default=os.path.join(PROJECT_ROOT, "benchmarks", "baseline"), help="Path to baseline directory"
    )
    parser.add_argument("--diff-dir", default=None, help="Directory for diff heatmaps (auto if comparing)")
    parser.add_argument(
        "--backbone-size", type=int, default=None, help="Backbone resolution (e.g. 1024). None = same as img_size"
    )

    args = parser.parse_args()

    device = args.device or get_device()
    engine = create_engine(device, args.checkpoint, backbone_size=args.backbone_size)

    # Load frames
    frames = load_video_frames(args.clip, max_frames=args.max_frames)
    masks = load_mask_frames(args.alpha, max_frames=args.max_frames)

    # Truncate to matching count
    num_frames = min(len(frames), len(masks))
    frames = frames[:num_frames]
    masks = masks[:num_frames]

    # Run benchmark
    print(f"\nRunning benchmark ({num_frames} frames)...")
    outputs, mem_info, frame_times = run_benchmark(engine, frames, masks, device)

    # Report timing
    print("\n--- Timing ---")
    print_timing_summary(frame_times)

    # Report memory
    print("\n--- Memory ---")
    print_memory_summary(mem_info)

    if args.generate_baseline:
        save_baseline(outputs, frame_times, mem_info, args.baseline)
        print("\nBaseline generation complete. Re-run without --generate-baseline to compare.")
    else:
        # Compare against baseline
        try:
            baseline_outputs = load_baseline(args.baseline)
        except FileNotFoundError:
            print(f"\nNo baseline found at {args.baseline}. Run with --generate-baseline first.")
            sys.exit(1)

        diff_dir = args.diff_dir or os.path.join(args.baseline, "diffs")
        compare_to_baseline(outputs, baseline_outputs, diff_dir=diff_dir)


if __name__ == "__main__":
    main()
