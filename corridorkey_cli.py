"""CorridorKey command-line interface and interactive wizard.

This module handles CLI argument parsing, environment setup, and the
interactive wizard workflow. The pipeline logic lives in clip_manager.py,
which can be imported independently as a library.

Usage (via launcher scripts):
    uv run python corridorkey_cli.py --action wizard --win_path "V:\\..."
    uv run python corridorkey_cli.py --action run_inference
    uv run python corridorkey_cli.py --action generate_alphas
    uv run python corridorkey_cli.py --action list
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import sys
import warnings

from clip_manager import (
    LINUX_MOUNT_ROOT,
    ClipEntry,
    generate_alphas,
    is_video_file,
    map_path,
    organize_target,
    run_inference,
    run_videomama,
    scan_clips,
)
from device_utils import resolve_device

logger = logging.getLogger(__name__)


def _configure_environment() -> None:
    """Set up logging and warnings for interactive CLI use.

    This is called once at startup when running from the command line.
    It is NOT called when importing clip_manager as a library.
    """
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _prompt_optimization_preset() -> dict:
    """Prompt user to select an optimization preset for inference."""
    print("\n--- Optimization Presets ---")
    print("  [1] Quality (default) — FP16 on, GPU post on, full backbone, tiled refiner")
    print("  [2] Fast Preview       — FP16 on, GPU post on, backbone 1024, tiled refiner")
    print("  [3] Low VRAM           — FP16 on, GPU post on, backbone 1024, tile 256")
    print("  [4] Legacy (no opts)   — FP16 off, GPU post off, full backbone, no tiling")
    print("  [5] Custom             — Configure each flag individually")

    choice = input("Select preset [1]: ").strip()

    _base = {"fp16": True, "gpu_postprocess": True, "refiner_tile_overlap": 96}
    presets = {
        "1": {**_base, "backbone_size": None, "refiner_tile_size": 512},
        "2": {**_base, "backbone_size": 1024, "refiner_tile_size": 512},
        "3": {**_base, "backbone_size": 1024, "refiner_tile_size": 256},
        "4": {
            "fp16": False,
            "gpu_postprocess": False,
            "backbone_size": None,
            "refiner_tile_size": None,
            "refiner_tile_overlap": 96,
        },
    }

    if choice in presets:
        return presets[choice]
    if choice == "5":
        return _prompt_custom_optimizations()
    # Default to Quality preset
    return presets["1"]


def _prompt_custom_optimizations() -> dict:
    """Prompt user to configure each optimization flag individually."""
    fp16 = input("  FP16 weight casting? [Y/n]: ").strip().lower() != "n"
    gpu_post = input("  GPU post-processing? [Y/n]: ").strip().lower() != "n"

    bb_val = input("  Backbone size (blank = full res, e.g. 1024): ").strip()
    backbone_size = int(bb_val) if bb_val else None

    tile_val = input("  Refiner tile size (0 = disabled, blank = 512): ").strip()
    if tile_val == "0":
        refiner_tile_size = None
    elif tile_val:
        refiner_tile_size = int(tile_val)
    else:
        refiner_tile_size = 512

    overlap_val = input("  Refiner tile overlap (blank = 96): ").strip()
    refiner_tile_overlap = int(overlap_val) if overlap_val else 96

    return {
        "fp16": fp16,
        "gpu_postprocess": gpu_post,
        "backbone_size": backbone_size,
        "refiner_tile_size": refiner_tile_size,
        "refiner_tile_overlap": refiner_tile_overlap,
    }


def interactive_wizard(win_path: str, device: str | None = None) -> None:
    print("\n" + "=" * 60)
    print(" CORRIDOR KEY - SMART WIZARD")
    print("=" * 60)

    # 1. Resolve Path
    print(f"Windows Path: {win_path}")

    # Check if we are running locally where the Windows path exists
    if os.path.exists(win_path):
        process_path = win_path
        print(f"Running locally dynamically using path: {process_path}")
    else:
        # Fallback to mapping for remote linux execution
        process_path = map_path(win_path)
        print(f"Linux/Remote Path:   {process_path}")

        if not os.path.exists(process_path):
            print("\n[ERROR] Path does not exist locally OR on Linux mount!")
            print(f"Expected Linux Mount Root: {LINUX_MOUNT_ROOT}")
            return

    # 2. Analyze
    # We treat process_path as the ROOT containing SHOTS
    # Or is process_path the SHOT itself?
    # HEURISTIC: If it contains "Input", it's a shot. If it contains folders, it's a project.

    # Let's assume it's a folder containing CLIPS (Batch Mode)
    # But if the user drops it IN a shot folder, we should handle that too.

    target_is_shot = False
    if os.path.exists(os.path.join(process_path, "Input")) or glob.glob(os.path.join(process_path, "Input.*")):
        target_is_shot = True

    work_dirs = []
    if target_is_shot:
        work_dirs = [process_path]
    else:
        # Scan subfolders
        work_dirs = [
            os.path.join(process_path, d)
            for d in os.listdir(process_path)
            if os.path.isdir(os.path.join(process_path, d))
        ]
        # Filter out output/hints
        work_dirs = [
            d
            for d in work_dirs
            if os.path.basename(d) not in ["Output", "AlphaHint", "VideoMamaMaskHint", ".ipynb_checkpoints"]
        ]

    print(f"\nFound {len(work_dirs)} potential clip folders.")

    # Check for loose videos in root
    loose_videos = [
        f for f in os.listdir(process_path) if is_video_file(f) and os.path.isfile(os.path.join(process_path, f))
    ]

    # Check if existing folders need organization
    dirs_needing_org = []
    for d in work_dirs:
        # Check for Input
        has_input = os.path.exists(os.path.join(d, "Input")) or glob.glob(os.path.join(d, "Input.*"))
        # Check for hints
        has_alpha = os.path.exists(os.path.join(d, "AlphaHint"))
        has_mask = os.path.exists(os.path.join(d, "VideoMamaMaskHint"))

        if not has_input or not has_alpha or not has_mask:
            dirs_needing_org.append(d)

    if loose_videos or dirs_needing_org:
        if loose_videos:
            print(f"Found {len(loose_videos)} loose video files that need organization:")
            for v in loose_videos:
                print(f"  - {v}")

        if dirs_needing_org:
            print(f"Found {len(dirs_needing_org)} folders that might need setup (Hints/Input):")
            # Limit output if too many
            if len(dirs_needing_org) < 10:
                for d in dirs_needing_org:
                    print(f"  - {os.path.basename(d)}")
            else:
                print(f"  - ...and {len(dirs_needing_org)} others.")

        # 3. Organize Loop
        yn = input("\n[1] Organize Clips & Create Hint Folders? [y/N]: ").strip().lower()
        if yn == "y":
            # Organize loose videos first
            for v in loose_videos:
                clip_name = os.path.splitext(v)[0]
                ext = os.path.splitext(v)[1]
                target_folder = os.path.join(process_path, clip_name)

                if os.path.exists(target_folder):
                    logger.warning(f"Skipping loose video '{v}': Target folder '{clip_name}' already exists.")
                    continue

                try:
                    os.makedirs(target_folder)
                    target_file = os.path.join(target_folder, f"Input{ext}")
                    shutil.move(os.path.join(process_path, v), target_file)
                    logger.info(f"Organized: Moved '{v}' to '{clip_name}/Input{ext}'")

                    # Also initialize hints immediately
                    for hint in ["AlphaHint", "VideoMamaMaskHint"]:
                        os.makedirs(os.path.join(target_folder, hint), exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to organize video '{v}': {e}")

            # Organize existing folders
            for d in work_dirs:
                organize_target(d)
            print("Organization Complete.")

            # Re-scan in case structure changed
            # If it was a shot, it's still a shot (unless we messed it up)
            # If it wasn't a shot, we re-scan subdirs
            if not target_is_shot:
                work_dirs = [
                    os.path.join(process_path, d)
                    for d in os.listdir(process_path)
                    if os.path.isdir(os.path.join(process_path, d))
                ]
                work_dirs = [
                    d for d in work_dirs if os.path.basename(d) not in ["Output", "AlphaHint", "VideoMamaMaskHint"]
                ]

    # 4. Status Check Loop
    while True:
        ready = []
        masked = []
        raw = []

        for d in work_dirs:
            entry = ClipEntry(os.path.basename(d), d)
            try:
                entry.find_assets()  # This checks Input and AlphaHint
            except (FileNotFoundError, ValueError, OSError):
                pass  # Might act up if Input missing

            # Check VideoMamaMaskHint (Strict: videomamamaskhint.ext or VideoMamaMaskHint/)
            has_mask = False
            mask_dir = os.path.join(d, "VideoMamaMaskHint")

            # 1. Directory Check
            if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
                has_mask = True

            # 2. File Check (Strict Stem Match)
            if not has_mask:
                for f in os.listdir(d):
                    stem, _ = os.path.splitext(f)
                    if stem.lower() == "videomamamaskhint" and is_video_file(f):
                        has_mask = True
                        break

            if entry.alpha_asset:
                ready.append(entry)
            elif has_mask:
                masked.append(entry)
            else:
                raw.append(entry)

        print("\n" + "-" * 40)
        print("STATUS REPORT:")
        print(f"  READY (AlphaHint found): {len(ready)}")
        for c in ready:
            print(f"    - {c.name}")

        print(f"  MASKED (VideoMamaMaskHint found): {len(masked)}")
        for c in masked:
            print(f"    - {c.name}")

        print(f"  RAW (Input only):        {len(raw)}")
        for c in raw:
            print(f"    - {c.name}")
        print("-" * 40 + "\n")

        # Combine checks for actions
        missing_alpha = masked + raw

        print("\nACTIONS:")
        if missing_alpha:
            print(f"  [v] Run VideoMaMa (Found {len(masked)} ready with masks)")
            print(f"  [g] Run GVM (Auto-Matte on {len(raw)} clips without Mask Hint)")

        if ready:
            print(f"  [i] Run Inference (on {len(ready)} ready clips)")

        print("  [r] Re-Scan Folders")
        print("  [q] Quit")

        choice = input("\nSelect Action: ").strip().lower()

        if choice == "v":
            # VideoMaMa
            print("\n--- VideoMaMa ---")
            print("Scanning for VideoMamaMaskHints...")
            # We pass ALL missing alpha clips. run_videomama checks for the actual files.
            run_videomama(missing_alpha, chunk_size=50, device=device)
            input("VideoMaMa batch complete. Press Enter to Re-Scan...")
            continue

        elif choice == "g":
            # GVM
            print("\n--- GVM Auto-Matte ---")
            print(f"This will generate alphas for {len(raw)} clips that have NO Mask Hint.")

            yn = input("Proceed with GVM? [y/N]: ").strip().lower()
            if yn == "y":
                generate_alphas(raw, device=device)
                input("GVM batch complete. Press Enter to Re-Scan...")
            continue

        elif choice == "i":
            # Inference
            print("\n--- Corridor Key Inference ---")
            opt_config = _prompt_optimization_preset()
            try:
                run_inference(ready, device=device, **opt_config)
            except (RuntimeError, FileNotFoundError) as e:
                logger.error(f"Inference failed: {e}")
            input("Inference batch complete. Press Enter to Re-Scan...")
            continue

        elif choice == "r":
            print("\nRe-scanning...")
            continue

        elif choice == "q":
            break

        else:
            print("Invalid selection.")
            continue

    print("\nWizard Complete. Goodbye!")


def main() -> None:
    _configure_environment()

    parser = argparse.ArgumentParser(description="CorridorKey Clip Manager")
    parser.add_argument("--action", choices=["generate_alphas", "run_inference", "list", "wizard"], required=True)
    parser.add_argument("--win_path", help=r"Windows Path (example: V:\...) for Wizard Mode", default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto-detect CUDA > MPS > CPU)",
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="FP16 weight casting")
    parser.add_argument(
        "--gpu-postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run color math on GPU with cached assets",
    )
    parser.add_argument(
        "--backbone-size", type=int, default=None, help="Backbone resolution (e.g. 1024). None = full res"
    )
    parser.add_argument(
        "--refiner-tile-size", type=int, default=512, help="Refiner tile size (0 = disabled, default 512)"
    )
    parser.add_argument("--refiner-tile-overlap", type=int, default=96, help="Refiner tile overlap pixels (default 96)")

    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")

    refiner_tile_size = args.refiner_tile_size or None  # 0 -> None (disabled)

    try:
        if args.action == "list":
            scan_clips()
        elif args.action == "generate_alphas":
            clips = scan_clips()
            generate_alphas(clips, device=device)
        elif args.action == "run_inference":
            clips = scan_clips()
            run_inference(
                clips,
                device=device,
                backbone_size=args.backbone_size,
                refiner_tile_size=refiner_tile_size,
                refiner_tile_overlap=args.refiner_tile_overlap,
                fp16=args.fp16,
                gpu_postprocess=args.gpu_postprocess,
            )
        elif args.action == "wizard":
            if not args.win_path:
                print("Error: --win_path required for wizard.")
            else:
                interactive_wizard(args.win_path, device=device)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
