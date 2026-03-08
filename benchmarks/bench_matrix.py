"""Benchmark matrix — runs all optimization flag combinations and outputs a comparison table.

Usage:
  uv run python benchmarks/bench_matrix.py --clip <input_video> --alpha <alpha_video>
  uv run python benchmarks/bench_matrix.py --clip <input_video> --alpha <alpha_video> --baseline benchmarks/baseline/
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bench_phase import (  # noqa: E402
    compare_to_baseline,
    create_engine,
    get_device,
    load_baseline,
    load_mask_frames,
    load_video_frames,
    print_memory_summary,
    print_timing_summary,
    run_benchmark,
)

# Each preset: (name, engine kwargs)
PRESETS = {
    "Quality": {
        "fp16": True,
        "backbone_size": None,
        "refiner_tile_size": 512,
        "refiner_tile_overlap": 96,
    },
    "Fast Preview": {
        "fp16": True,
        "backbone_size": 1024,
        "refiner_tile_size": 512,
        "refiner_tile_overlap": 96,
    },
    "Low VRAM": {
        "fp16": True,
        "backbone_size": 1024,
        "refiner_tile_size": 256,
        "refiner_tile_overlap": 96,
    },
    "Legacy": {
        "fp16": False,
        "backbone_size": None,
        "refiner_tile_size": None,
        "refiner_tile_overlap": 96,
    },
}


def main():
    parser = argparse.ArgumentParser(description="CorridorKey benchmark matrix")
    parser.add_argument("--clip", required=True, help="Path to input RGB video")
    parser.add_argument("--alpha", required=True, help="Path to alpha hint video")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--device", default=None, help="Device override")
    parser.add_argument("--max-frames", type=int, default=5, help="Max frames per preset (default 5)")
    parser.add_argument(
        "--baseline", default=os.path.join(PROJECT_ROOT, "benchmarks", "baseline"), help="Baseline directory"
    )
    args = parser.parse_args()

    device = args.device or get_device()
    frames = load_video_frames(args.clip, max_frames=args.max_frames)
    masks = load_mask_frames(args.alpha, max_frames=args.max_frames)

    num_frames = min(len(frames), len(masks))
    frames = frames[:num_frames]
    masks = masks[:num_frames]

    # Load baseline if available
    baseline_outputs = None
    try:
        baseline_outputs = load_baseline(args.baseline)
    except (FileNotFoundError, ValueError):
        print(f"No baseline at {args.baseline} — skipping quality comparison.\n")

    # Results table
    results = []

    for preset_name, kwargs in PRESETS.items():
        print(f"\n{'=' * 70}")
        print(f"PRESET: {preset_name}")
        print(
            f"  fp16={kwargs['fp16']}, backbone={kwargs['backbone_size']}, "
            f"tile={kwargs['refiner_tile_size']}, overlap={kwargs['refiner_tile_overlap']}"
        )
        print("=" * 70)

        engine = create_engine(
            device,
            args.checkpoint,
            backbone_size=kwargs["backbone_size"],
            refiner_tile_size=kwargs["refiner_tile_size"],
            refiner_tile_overlap=kwargs["refiner_tile_overlap"],
        )

        outputs, mem_info, frame_times = run_benchmark(engine, frames, masks, device)

        print("\n--- Timing ---")
        print_timing_summary(frame_times)

        print("\n--- Memory ---")
        print_memory_summary(mem_info)

        import statistics

        steady = frame_times[1:] if len(frame_times) > 1 else frame_times
        row = {
            "preset": preset_name,
            "median_s": statistics.median(steady) if steady else 0,
            "peak_gb": mem_info["peak_bytes"] / 1e9,
        }

        if baseline_outputs:
            compare_to_baseline(outputs, baseline_outputs)

        results.append(row)

        # Free engine to release VRAM before next preset
        del engine
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Summary table
    print(f"\n{'=' * 70}")
    print("MATRIX SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Preset':<20} {'Median (s)':>12} {'Peak Mem (GB)':>14}")
    print("-" * 48)
    for row in results:
        print(f"{row['preset']:<20} {row['median_s']:>12.3f} {row['peak_gb']:>14.2f}")


if __name__ == "__main__":
    main()
