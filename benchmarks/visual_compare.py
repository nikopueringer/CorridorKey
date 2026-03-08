"""Visual comparison — run each preset on a single frame and save outputs for manual review.

Usage:
  uv run python benchmarks/visual_compare.py --clip <input_video> --alpha <alpha_video>
  uv run python benchmarks/visual_compare.py --clip <input> --alpha <alpha> --output-dir benchmarks/visual_review
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from benchmarks.bench_phase import create_engine, get_device, load_mask_frames, load_video_frames  # noqa: E402


@dataclass(frozen=True)
class Preset:
    fp16: bool
    backbone_size: int | None
    refiner_tile_size: int | None
    refiner_tile_overlap: int


# Same presets as bench_matrix.py
PRESETS: dict[str, Preset] = {
    "Quality": Preset(fp16=True, backbone_size=None, refiner_tile_size=512, refiner_tile_overlap=96),
    "Fast_Preview": Preset(fp16=True, backbone_size=1024, refiner_tile_size=512, refiner_tile_overlap=96),
    "Low_VRAM": Preset(fp16=True, backbone_size=1024, refiner_tile_size=256, refiner_tile_overlap=96),
    "Legacy": Preset(fp16=False, backbone_size=None, refiner_tile_size=None, refiner_tile_overlap=96),
}

# Output passes to save
OUTPUT_PASSES = {
    "alpha": "Alpha (linear)",
    "fg": "FG (sRGB)",
    "processed": "Processed RGBA (linear premul)",
    "comp": "Composite (sRGB)",
}


def save_pass_as_png(data: np.ndarray, path: str):
    """Save a [0,1] float array as 8-bit PNG (BGR for cv2)."""
    if data.ndim == 2:
        # Single channel (alpha) — save as grayscale
        img = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)
    elif data.shape[2] == 4:
        # RGBA — save as BGRA
        bgra = np.copy(data)
        bgra[:, :, :3] = bgra[:, :, 2::-1]  # RGB -> BGR
        img = (np.clip(bgra, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        # RGB -> BGR
        bgr = data[:, :, ::-1]
        img = (np.clip(bgr, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser(description="CorridorKey visual preset comparison")
    parser.add_argument("--clip", required=True, help="Path to input RGB video")
    parser.add_argument("--alpha", required=True, help="Path to alpha hint video")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--device", default=None, help="Device override")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "benchmarks", "visual_review"),
        help="Output directory (default: benchmarks/visual_review)",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="Which frame to use (0-based, default: 0)")
    args = parser.parse_args()

    device = args.device or get_device()

    # Load single frame
    frames = load_video_frames(args.clip, max_frames=args.frame_index + 1)
    masks = load_mask_frames(args.alpha, max_frames=args.frame_index + 1)

    if args.frame_index >= len(frames) or args.frame_index >= len(masks):
        print(f"Error: frame index {args.frame_index} out of range (have {len(frames)} frames)")
        sys.exit(1)

    frame = frames[args.frame_index]
    mask = masks[args.frame_index]

    # Save input for reference
    os.makedirs(args.output_dir, exist_ok=True)
    save_pass_as_png(frame, os.path.join(args.output_dir, "input_rgb.png"))
    save_pass_as_png(mask, os.path.join(args.output_dir, "input_alpha_hint.png"))
    print(f"Saved input reference to {args.output_dir}/")

    import torch

    for preset_name, preset in PRESETS.items():
        print(f"\n{'=' * 50}")
        print(f"PRESET: {preset_name}")
        print(
            f"  fp16={preset.fp16}, backbone={preset.backbone_size}, "
            f"tile={preset.refiner_tile_size}, overlap={preset.refiner_tile_overlap}"
        )
        print("=" * 50)

        engine = create_engine(
            device,
            args.checkpoint,
            backbone_size=preset.backbone_size,
            refiner_tile_size=preset.refiner_tile_size,
            refiner_tile_overlap=preset.refiner_tile_overlap,
            fp16=preset.fp16,
        )

        result = engine.process_frame(frame, mask)

        # Save each pass
        preset_dir = os.path.join(args.output_dir, preset_name)
        os.makedirs(preset_dir, exist_ok=True)

        for pass_name in OUTPUT_PASSES:
            data = result[pass_name]
            out_path = os.path.join(preset_dir, f"{pass_name}.png")
            save_pass_as_png(data, out_path)

        print(f"  Saved {len(OUTPUT_PASSES)} passes to {preset_dir}/")

        # Free engine VRAM
        del engine
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    print(f"\nDone. Visual review outputs in: {args.output_dir}/")
    print("Directory structure:")
    print(f"  {args.output_dir}/")
    print("    input_rgb.png")
    print("    input_alpha_hint.png")
    for preset_name in PRESETS:
        print(f"    {preset_name}/")
        for pass_name in OUTPUT_PASSES:
            print(f"      {pass_name}.png")


if __name__ == "__main__":
    main()
