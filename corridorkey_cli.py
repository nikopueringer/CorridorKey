#!/usr/bin/env -S uv run python

"""CorridorKey command-line interface and interactive wizard.

This module handles CLI argument parsing, environment setup, and the
interactive wizard workflow. The pipeline logic lives in clip_manager.py,
which can be imported independently as a library.

Usage (via launcher scripts):
    uv run python corridorkey_cli.py --help
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
    run_videomama
)
from device_utils import resolve_device

def wizard_print(action, msg):
    if action == 'wizard':
        print(msg)
    else:
        logging.debug(msg)

def do_process(
    despill_strength,
    despeckle,
    despeckle_size,
    device,
    gamma_encoding,
    max_frames,
    organize_clips,
    refiner_strength,
    action,
    input_path,
    **kwargs
) -> None:
    wizard_print(action, "\n" + "=" * 60)
    wizard_print(action, " CORRIDOR KEY - SMART WIZARD")
    wizard_print(action, "=" * 60)

    # 1. Map Path
    linux_path = map_path(input_path)
    wizard_print(action, f"Windows Path: {input_path}")
    wizard_print(action, f"Linux Path:   {linux_path}")

    if not os.path.exists(linux_path):
        logging.error("\n[ERROR] Path does not exist on Linux mount!")
        logging.error(f"Expected: {LINUX_MOUNT_ROOT}")
        return

    # 2. Analyze
    # We treat linux_path as the ROOT containing SHOTS
    # Or is linux_path the SHOT itself?
    # HEURISTIC: If it contains "Input", it's a shot. If it contains folders, it's a project.

    # Let's assume it's a folder containing CLIPS (Batch Mode)
    # But if the user drops it IN a shot folder, we should handle that too.

    target_is_shot = False
    if os.path.exists(os.path.join(linux_path, "Input")) or glob.glob(os.path.join(linux_path, "Input.*")):
        target_is_shot = True

    work_dirs = []
    if target_is_shot:
        work_dirs = [linux_path]
    else:
        # Scan subfolders
        work_dirs = [
            os.path.join(linux_path, d) for d in os.listdir(linux_path) if os.path.isdir(os.path.join(linux_path, d))
        ]
        # Filter out output/hints
        work_dirs = [
            d
            for d in work_dirs
            if os.path.basename(d) not in ["Output", "AlphaHint", "VideoMamaMaskHint", ".ipynb_checkpoints"]
        ]

    wizard_print(action, f"\nFound {len(work_dirs)} potential clip folders.")

    # Check for loose videos in root
    loose_videos = [
        f for f in os.listdir(linux_path) if is_video_file(f) and os.path.isfile(os.path.join(linux_path, f))
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
            wizard_print(action, f"Found {len(loose_videos)} loose video files that need organization:")
            for v in loose_videos:
                wizard_print(action, f"  - {v}")

        if dirs_needing_org:
            wizard_print(action, f"Found {len(dirs_needing_org)} folders that might need setup (Hints/Input):")
            # Limit output if too many
            if len(dirs_needing_org) < 10:
                for d in dirs_needing_org:
                    wizard_print(action, f"  - {os.path.basename(d)}")
            else:
                wizard_print(action, f"  - ...and {len(dirs_needing_org)} others.")

        # 3. Organize Loop
        if (action == "wizard"):
            yn = input("\n[1] Organize Clips & Create Hint Folders? [y/N]: ").strip().lower() == 'y'
        else:
            yn = organize_clips
            
        if yn:
            # Organize loose videos first
            for v in loose_videos:
                clip_name = os.path.splitext(v)[0]
                ext = os.path.splitext(v)[1]
                target_folder = os.path.join(linux_path, clip_name)

                if os.path.exists(target_folder):
                    logging.warning(f"Skipping loose video '{v}': Target folder '{clip_name}' already exists.")
                    continue

                try:
                    os.makedirs(target_folder)
                    target_file = os.path.join(target_folder, f"Input{ext}")
                    shutil.move(os.path.join(linux_path, v), target_file)
                    logging.info(f"Organized: Moved '{v}' to '{clip_name}/Input{ext}'")

                    # Also initialize hints immediately
                    for hint in ["AlphaHint", "VideoMamaMaskHint"]:
                        os.makedirs(os.path.join(target_folder, hint), exist_ok=True)
                except Exception as e:
                    logging.error(f"Failed to organize video '{v}': {e}")

            # Organize existing folders
            for d in work_dirs:
                organize_target(d)
            wizard_print(action, "Organization Complete.")

            # Re-scan in case structure changed
            # If it was a shot, it's still a shot (unless we messed it up)
            # If it wasn't a shot, we re-scan subdirs
            if not target_is_shot:
                work_dirs = [
                    os.path.join(linux_path, d)
                    for d in os.listdir(linux_path)
                    if os.path.isdir(os.path.join(linux_path, d))
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

        wizard_print(action, "\n" + "-" * 40)
        wizard_print(action, "STATUS REPORT:")
        wizard_print(action, f"  READY (AlphaHint found): {len(ready)}")
        for c in ready:
            wizard_print(action, f"    - {c.name}")

        wizard_print(action, f"  MASKED (VideoMamaMaskHint found): {len(masked)}")
        for c in masked:
            wizard_print(action, f"    - {c.name}")

        wizard_print(action, f"  RAW (Input only):        {len(raw)}")
        for c in raw:
            wizard_print(action, f"    - {c.name}")
        wizard_print(action, "-" * 40 + "\n")

        # Combine checks for actions
        missing_alpha = masked + raw

        if action == 'wizard':
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
                run_videomama(missing_alpha, chunk_size=50, **locals())
                input("VideoMaMa batch complete. Press Enter to Re-Scan...")
                continue

            elif choice == "g":
                # GVM
                print("\n--- GVM Auto-Matte ---")
                print(f"This will generate alphas for {len(raw)} clips that have NO Mask Hint.")

                yn = input("Proceed with GVM? [y/N]: ").strip().lower()
                if yn == "y":
                    generate_alphas(raw, **locals())
                    input("GVM batch complete. Press Enter to Re-Scan...")
                continue

            elif choice == "i":
                # Inference
                print("\n--- Corridor Key Inference ---")
                try:
                    run_inference(ready, **locals())
                except (RuntimeError, FileNotFoundError) as e:
                    logging.error(f"Inference failed: {e}")
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
        elif action == "run_inference":
            run_inference(ready, **locals())
            break
        elif action == "run_gvm":
            generate_alphas(raw, **locals())
            break
        elif action == "run_videomama":
            run_videomama(missing_alpha, chunk_size=50, **locals())
            break

def main() -> None:
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="CorridorKey")
    parser.add_argument("--despill-strength", help=r"0-[10]", type=int, default=10)
    parser.add_argument("--despeckle", help=r"[True]/False Removes tracking dots in Processed/Comp", type=bool, default=True)
    parser.add_argument("--despeckle-size", help=r"Default: 400 (min pixels for spot)", type=int, default=400)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Compute device (default: auto-detect CUDA > MPS > CPU)")
    parser.add_argument("--gamma-encoding", help=r"Default: srgb", choices=["linear", "srgb"], default="srgb")
    parser.add_argument("--max-frames", help=r"Number of frames to process", type=int, default=None)
    parser.add_argument("--organize-clips", help=r"[True]/False Organizes files/folders", type=bool, default=True)
    parser.add_argument("--refiner-strength", help=r"Default: 1.0 (multiplier) (experimental)", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    parser.add_argument("--action", choices=["run_videomama", "run_gvm", "run_inference", "wizard"], required=True)
    parser.add_argument("input_path", help=r"File/Folder path to scan for shots/projects", default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = resolve_device(args.device)
    logging.debug(f"Using device: {device}")

    try:
        do_process(**vars(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)

if __name__ == "__main__":
    main()
