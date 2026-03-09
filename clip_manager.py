from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import sys
from typing import TYPE_CHECKING

# Enable OpenEXR support in OpenCV — needed for EXR I/O throughout the pipeline.
# Must be set before any cv2.imread/imwrite calls on .exr files.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

from device_utils import resolve_device

if TYPE_CHECKING:
    from gvm_core import GVMProcessor

logger = logging.getLogger(__name__)


# Core Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "ClipsForInference")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

# Network Mapping
# Windows Drive -> Linux Mount Point
WIN_DRIVE_ROOT = os.getenv("CORRIDORKEY_WIN_DRIVE_ROOT", "V:\\")
LINUX_MOUNT_ROOT = os.getenv("CORRIDORKEY_LINUX_MOUNT_ROOT", "/mnt/ssd-storage")


# --- Helpers ---
def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp"))


def is_video_file(filename: str) -> bool:
    return filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def map_path(source_path: str) -> str:
    r"""
    Converts a Windows path (example: V:\Projects\Shot1) to the local Linux path.
    Non-Windows paths are normalized and returned unchanged.
    """
    normalized_path = os.path.expanduser(source_path.strip())

    # Check if it starts with the drive letter
    if normalized_path.upper().startswith(WIN_DRIVE_ROOT.upper()):
        # Remove drive letter
        rel_path = normalized_path[len(WIN_DRIVE_ROOT) :]
        # Combine and flip slashes
        linux_path = os.path.join(LINUX_MOUNT_ROOT, rel_path).replace("\\", "/")
        return linux_path

    return os.path.normpath(normalized_path)


# --- Classes ---
class ClipAsset:
    def __init__(self, path: str, asset_type: str) -> None:
        self.path = path
        self.type = asset_type  # 'sequence' or 'video'
        self.frame_count = 0
        self._calculate_length()

    def _calculate_length(self) -> None:
        if self.type == "sequence":
            files = sorted([f for f in os.listdir(self.path) if is_image_file(f)])
            self.frame_count = len(files)
        elif self.type == "video":
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                self.frame_count = 0
            else:
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()


class ClipEntry:
    def __init__(self, name: str, root_path: str) -> None:
        self.name = name
        self.root_path = root_path
        self.input_asset: ClipAsset | None = None
        self.alpha_asset: ClipAsset | None = None

    def find_assets(self) -> None:
        # 1. Look for Input
        input_dir = os.path.join(self.root_path, "Input")

        # Check for directory first
        if os.path.isdir(input_dir):
            if not os.listdir(input_dir):
                raise ValueError(f"Clip '{self.name}': 'Input' directory is empty.")
            self.input_asset = ClipAsset(input_dir, "sequence")
        else:
            # Check for video file (Case-Insensitive)
            candidates = glob.glob(os.path.join(self.root_path, "[Ii]nput.*"))
            candidates = [c for c in candidates if is_video_file(c)]

            if candidates:
                self.input_asset = ClipAsset(candidates[0], "video")
            else:
                # Fallback: Look for ANY video file in the directory
                all_files = glob.glob(os.path.join(self.root_path, "*"))
                video_files = [f for f in all_files if is_video_file(f)]

                if video_files:
                    logger.info(f"Clip '{self.name}': Using '{os.path.basename(video_files[0])}' as Input.")
                    self.input_asset = ClipAsset(video_files[0], "video")
                else:
                    raise ValueError(f"Clip '{self.name}': No 'Input' directory or video file found.")

        if self.input_asset.frame_count == 0:
            raise ValueError(f"Clip '{self.name}': Input asset has 0 frames or could not be read.")

        # 2. Look for Alpha
        # Check for 'AlphaHint' or 'alphahint' directory
        alpha_dir_upper = os.path.join(self.root_path, "AlphaHint")
        alpha_dir_lower = os.path.join(self.root_path, "alphahint")

        target_alpha_dir = None
        if os.path.isdir(alpha_dir_upper):
            target_alpha_dir = alpha_dir_upper
        elif os.path.isdir(alpha_dir_lower):
            target_alpha_dir = alpha_dir_lower

        if target_alpha_dir:
            if not os.listdir(target_alpha_dir):
                logging.warning(f"Clip '{self.name}': AlphaHint directory exists but is empty. Marking for generation.")
                self.alpha_asset = None
            else:
                # Check for image sequence first
                self.alpha_asset = ClipAsset(target_alpha_dir, "sequence")
                if self.alpha_asset.frame_count == 0:
                    # Fallback: check for video file inside the AlphaHint directory
                    video_candidates = [f for f in os.listdir(target_alpha_dir) if is_video_file(f)]
                    if video_candidates:
                        self.alpha_asset = ClipAsset(os.path.join(target_alpha_dir, video_candidates[0]), "video")
                    else:
                        logging.warning(
                            f"Clip '{self.name}': AlphaHint directory has no valid image or video files."
                            " Marking for generation."
                        )
                        self.alpha_asset = None
        else:
            # Check for video file (Case-Insensitive)
            # Match AlphaHint.* or alphahint.*
            candidates = glob.glob(os.path.join(self.root_path, "[Aa]lpha[Hh]int.*"))
            candidates = [c for c in candidates if is_video_file(c)]

            if candidates:
                self.alpha_asset = ClipAsset(candidates[0], "video")
            else:
                self.alpha_asset = None  # Missing, needs generation

    def validate_pair(self) -> None:
        if self.input_asset and self.alpha_asset:
            if self.input_asset.frame_count != self.alpha_asset.frame_count:
                raise ValueError(
                    f"Clip '{self.name}': Frame count mismatch! "
                    f"Input: {self.input_asset.frame_count}, Alpha: {self.alpha_asset.frame_count}"
                )


# --- Logic ---


def get_gvm_processor(device: str = "cpu") -> GVMProcessor:
    try:
        from gvm_core import GVMProcessor

        return GVMProcessor(device=device)
    except ImportError:
        raise ImportError(
            "Could not import gvm_core. Please ensure 'gvm_core' is in the project root and requirements are installed."
        ) from None
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GVM Processor: {e}") from e


def generate_alphas(clips, device=None):
    clips_to_process = [c for c in clips if c.alpha_asset is None]

    if not clips_to_process:
        logger.info("All clips have valid Alpha assets. No generation needed.")
        return

    logger.info(f"Found {len(clips_to_process)} clips missing Alpha.")

    if device is None:
        device = resolve_device()

    try:
        processor = get_gvm_processor(device=device)
    except ImportError as e:
        logger.error(f"GVM Import Error: {e}")
        logger.error("Skipping GVM generation. Please install GVM requirements if you wish to use this feature.")
        return
    except Exception as e:
        logger.error(f"GVM Initialization Error: {e}")
        return

    for clip in clips_to_process:
        logger.info(f"Generating Alpha for: {clip.name}")

        alpha_output_dir = os.path.join(clip.root_path, "AlphaHint")
        if os.path.exists(alpha_output_dir):
            shutil.rmtree(alpha_output_dir)
        os.makedirs(alpha_output_dir, exist_ok=True)

        try:
            processor.process_sequence(
                input_path=clip.input_asset.path,
                output_dir=None,
                num_frames_per_batch=1,
                decode_chunk_size=1,
                denoise_steps=1,
                mode="matte",
                write_video=False,
                direct_output_dir=alpha_output_dir,
            )

            # Post-Process: Naming Convention
            generated_files = sorted([f for f in os.listdir(alpha_output_dir) if f.endswith(".png")])

            if not generated_files:
                logger.error(f"GVM finished but no PNGs found in {alpha_output_dir}")
                continue

            if clip.input_asset.type == "sequence":
                in_files = sorted([f for f in os.listdir(clip.input_asset.path) if is_image_file(f)])
                stems = [os.path.splitext(f)[0] for f in in_files]
            else:
                base_name = os.path.splitext(os.path.basename(clip.input_asset.path))[0]
                stems = [base_name] * len(generated_files)

            for i, gvm_file in enumerate(generated_files):
                if i >= len(stems):
                    break

                stem = stems[i]
                new_name = f"{stem}_alphaHint_{i:04d}.png"

                old_path = os.path.join(alpha_output_dir, gvm_file)
                new_path = os.path.join(alpha_output_dir, new_name)

                if old_path != new_path:
                    os.rename(old_path, new_path)

            logger.info(f"Saved {len(generated_files)} alpha frames to {alpha_output_dir}")

        except Exception as e:
            logger.error(f"Error generating alpha for {clip.name}: {e}")
            import traceback

            traceback.print_exc()


def run_videomama(clips: list[ClipEntry], chunk_size: int = 50, device: str | None = None) -> None:
    """
    Runs VideoMaMa on clips that have VideoMamaMaskHint but NO AlphaHint.
    """
    # Process if:
    # 1. Has VideoMamaMaskHint (File or Folder, Case-Insensitive)
    # 2. AND (Alpha is Missing OR Alpha is a Video File we want to upgrade)

    clips_to_process = []
    clip_mask_paths = {}  # Store the resolved mask path for each clip

    for c in clips:
        # Search for 'videomamamaskhint' asset (Strict: videomamamaskhint.ext or VideoMamaMaskHint/)
        candidates = []
        for f in os.listdir(c.root_path):
            stem, _ = os.path.splitext(f)
            if stem.lower() == "videomamamaskhint":
                candidates.append(f)

        mask_asset_path = None
        has_mask = False

        # Priority: Directory > Video File
        # Check directories first
        for cand in candidates:
            path = os.path.join(c.root_path, cand)
            if os.path.isdir(path) and len(os.listdir(path)) > 0:
                has_mask = True
                mask_asset_path = path
                break

        # If no directory, check files
        if not has_mask:
            for cand in candidates:
                path = os.path.join(c.root_path, cand)
                if os.path.isfile(path) and is_video_file(path):
                    has_mask = True
                    mask_asset_path = path
                    break

        if not has_mask:
            continue

        # Store for later
        clip_mask_paths[c.name] = mask_asset_path

        if c.alpha_asset is None:
            clips_to_process.append(c)
        elif c.alpha_asset.type == "video":
            clips_to_process.append(c)

    if not clips_to_process:
        logger.info("No candidates for VideoMaMa (looking for VideoMamaMaskHint + [NoAlpha OR VideoAlpha]).")
        return

    logger.info(f"Found {len(clips_to_process)} clips for VideoMaMa processing.")

    # Import locally — sys.path mutation is needed because VideoMaMaInferenceModule
    # uses intra-package imports that assume its directory is on the path.
    try:
        sys.path.append(os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import load_videomama_model
        from VideoMaMaInferenceModule.inference import run_inference as run_videomama_frames
    except ImportError as e:
        logger.error(f"Failed to import VideoMaMa: {e}")
        return

    if device is None:
        device = resolve_device()

    logger.info("Loading VideoMaMa Pipeline...")
    pipeline = load_videomama_model(device=device)

    for clip in clips_to_process:
        logger.info(f"Running VideoMaMa on: {clip.name}")

        # Retrieve resolved path
        mask_hint_path = clip_mask_paths[clip.name]
        logger.info(f"  Using VideoMamaMaskHint: {os.path.basename(mask_hint_path)}")

        alpha_output_dir = os.path.join(clip.root_path, "AlphaHint")

        # Load Inputs
        # 1. Input Frames (RGB)
        input_frames = []
        if clip.input_asset.type == "video":
            cap = cv2.VideoCapture(clip.input_asset.path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                input_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            files = sorted([f for f in os.listdir(clip.input_asset.path) if is_image_file(f)])
            for f in files:
                fpath = os.path.join(clip.input_asset.path, f)
                # Handle EXR (Float 0-1) vs Standard (Int 0-255)
                if f.lower().endswith(".exr"):
                    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Normalize Float 0-1
                        img = np.clip(img, 0.0, 1.0)
                        # Linear -> sRGB (Gamma 2.2 Approximation) for VideoMaMa
                        img = img ** (1.0 / 2.2)
                        # 0-255 uint8
                        img = (img * 255.0).astype(np.uint8)
                        # Ensure 3 channels
                        if img.ndim == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    img = cv2.imread(fpath)

                if img is not None:
                    input_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 2. Mask Frames
        mask_frames = []

        # Check if VideoMamaMaskHint is a directory or a file (video)
        if os.path.isdir(mask_hint_path):
            # Directory of Images
            mask_files = sorted([f for f in os.listdir(mask_hint_path) if is_image_file(f)])
            for f in mask_files:
                fpath = os.path.join(mask_hint_path, f)
                m = None

                # Handle EXR Masks
                if f.lower().endswith(".exr"):
                    m = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if m is not None:
                        if m.ndim == 3:
                            m = m[:, :, 0]
                        m = np.clip(m, 0.0, 1.0)
                        m = (m * 255.0).astype(np.uint8)
                else:
                    # Standard Masks
                    m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

                if m is not None:
                    # Force Binary Thresholding
                    _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
                    mask_frames.append(m)

        elif os.path.isfile(mask_hint_path):
            # Handle Video File
            cap = cv2.VideoCapture(mask_hint_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to Grayscale
                m = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Force Binary Thresholding
                _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
                mask_frames.append(m)
            cap.release()

        # Validate Lengths
        num_frames = min(len(input_frames), len(mask_frames))
        input_frames = input_frames[:num_frames]
        mask_frames = mask_frames[:num_frames]

        if num_frames == 0:
            logger.error(f"Skipping {clip.name}: No valid frame pairs found.")
            continue

        # Run Inference
        try:
            # Prepare Output Directory First
            # Logic: If it exists as a FILE (legacy/error), delete it.
            if os.path.exists(alpha_output_dir) and not os.path.isdir(alpha_output_dir):
                logger.warning(f"Removing file '{alpha_output_dir}' to create directory.")
                os.remove(alpha_output_dir)

            # If there was a Video Alpha Asset (e.g. AlphaHint.mp4), rename it to backup so it doesn't conflict
            if clip.alpha_asset and clip.alpha_asset.type == "video":
                old_path = clip.alpha_asset.path
                if os.path.exists(old_path):
                    dir_name = os.path.dirname(old_path)
                    base, ext = os.path.splitext(os.path.basename(old_path))
                    backup_path = os.path.join(dir_name, f"{base}_backup{ext}")
                    logger.info(
                        f"Backing up existing Alpha Video: "
                        f"{os.path.basename(old_path)} -> {os.path.basename(backup_path)}"
                    )
                    os.rename(old_path, backup_path)
                    # Clear it from memory so we rely on the new one
                    clip.alpha_asset = None

            os.makedirs(alpha_output_dir, exist_ok=True)

            # Name setup
            if clip.input_asset.type == "sequence":
                in_names = sorted(
                    [os.path.splitext(f)[0] for f in os.listdir(clip.input_asset.path) if is_image_file(f)]
                )
            else:
                stem = os.path.splitext(os.path.basename(clip.input_asset.path))[0]
                in_names = [f"{stem}_{i:05d}" for i in range(num_frames)]

            total_saved = 0

            # Iterate generator
            for chunk_frames in run_videomama_frames(pipeline, input_frames, mask_frames, chunk_size=chunk_size):
                for frame in chunk_frames:
                    if total_saved >= len(in_names):
                        break

                    name = in_names[total_saved]
                    out_path = os.path.join(alpha_output_dir, f"{name}.png")

                    # Convert to BGR and Save
                    cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    total_saved += 1

                logger.info(f"  Saved {total_saved}/{num_frames} frames...")

            logger.info(f"VideoMaMa Complete: Saved {total_saved} frames to AlphaHint.")

            # Update clip state in memory (dummy) - re-scan will pick it up properly
            clip.alpha_asset = ClipAsset(alpha_output_dir, "sequence")

        except Exception as e:
            logger.error(f"VideoMaMa failed for {clip.name}: {e}")
            import traceback

            traceback.print_exc()


def run_inference(clips, device=None, backend=None, max_frames=None, img_size=2048, precision="auto"):

    ready_clips = [c for c in clips if c.input_asset and c.alpha_asset]

    if not ready_clips:
        logger.info("No clips found with both Input and Alpha assets. Run generate_coarse_alpha first?")
        return

    logger.info(f"Found {len(ready_clips)} clips ready for inference.")

    # --- User Prompts ---
    print("\n--- Inference Settings ---")

    # 1. Gamma Prompt
    user_input_is_linear = False
    gamma_choice = input("Is the input sequence Linear (l) or sRGB (s)? [l/s]: ").strip().lower()
    if gamma_choice == "l":
        user_input_is_linear = True
        logger.info("User selected: Linear Input")
    else:
        logger.info("User selected: sRGB Input (or default)")

    # 2. Despill Prompt
    despill_val = input("Enter Despill Strength (0-10, 10 is max despill) [default 5]: ").strip()
    try:
        despill_int = int(despill_val)
        despill_int = max(0, min(10, despill_int))
    except ValueError:
        despill_int = 5

    despill_strength = despill_int / 10.0
    logger.info(f"User selected: Despill Strength {despill_int}/10 ({despill_strength})")
    # 3. Auto-Despeckle Prompt
    auto_despeckle = True
    despeckle_size = 400
    despeckle_choice = input("Enable Auto-Despeckle (removes tracking dots in Processed/Comp)? [Y/n]: ").strip().lower()
    if despeckle_choice == "n":
        auto_despeckle = False
        logger.info("User selected: Auto-Despeckle OFF")
    else:
        logger.info("User selected: Auto-Despeckle ON (default)")
        size_val = input("Enter Auto-Despeckle Size (min pixels for a spot) [default 400]: ").strip()
        try:
            val_int = int(size_val)
            despeckle_size = max(0, val_int)
        except ValueError:
            despeckle_size = 400
        logger.info(f"User selected: Auto-Despeckle Size {despeckle_size}px")

    # 4. Refiner Strength Prompt
    refiner_val = input("Enter Refiner Strength (multiplier) [default 1.0] (experimental): ").strip()
    if refiner_val == "":
        refiner_scale = 1.0
    else:
        try:
            refiner_scale = float(refiner_val)
        except ValueError:
            refiner_scale = 1.0
    logger.info(f"User selected: Refiner Strength {refiner_scale}")

    print("--------------------------\n")

    # Ensure Output Directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    import numpy as np

    if device is None:
        device = resolve_device()
    from CorridorKeyModule.backend import create_engine

    engine = create_engine(backend=backend, device=device, img_size=img_size, precision=precision)

    for clip in ready_clips:
        logger.info(f"Running Inference on: {clip.name}")

        # Setup Outputs in ClipFolder/Output/...
        clip_out_root = os.path.join(clip.root_path, "Output")
        fg_dir = os.path.join(clip_out_root, "FG")
        matte_dir = os.path.join(clip_out_root, "Matte")
        comp_dir = os.path.join(clip_out_root, "Comp")
        proc_dir = os.path.join(clip_out_root, "Processed")

        for d in [fg_dir, matte_dir, comp_dir, proc_dir]:
            os.makedirs(d, exist_ok=True)

        num_frames = min(clip.input_asset.frame_count, clip.alpha_asset.frame_count)
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
        logger.info(
            f"  Input frames: {clip.input_asset.frame_count},"
            f" Alpha frames: {clip.alpha_asset.frame_count} -> Processing {num_frames} frames"
        )

        if num_frames == 0:
            logger.warning(f"Clip '{clip.name}': 0 frames to process, skipping.")
            continue

        input_cap = None
        alpha_cap = None
        input_files = []
        alpha_files = []

        if clip.input_asset.type == "video":
            input_cap = cv2.VideoCapture(clip.input_asset.path)
        else:
            input_files = sorted([f for f in os.listdir(clip.input_asset.path) if is_image_file(f)])

        if clip.alpha_asset.type == "video":
            alpha_cap = cv2.VideoCapture(clip.alpha_asset.path)
        else:
            alpha_files = sorted([f for f in os.listdir(clip.alpha_asset.path) if is_image_file(f)])

        for i in range(num_frames):
            if i % 10 == 0:
                print(f"  Frame {i}/{num_frames}...", end="\r")

            # 1. Read Input
            img_srgb = None
            input_stem = f"{i:05d}"

            # Use the user-defined gamma
            input_is_linear = user_input_is_linear

            if input_cap:
                ret, frame = input_cap.read()
                if not ret:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_srgb = img_rgb.astype(np.float32) / 255.0
                input_stem = f"{i:05d}"
            else:
                fpath = os.path.join(clip.input_asset.path, input_files[i])
                input_stem = os.path.splitext(input_files[i])[0]

                is_exr = fpath.lower().endswith(".exr")
                if is_exr:
                    img_linear = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if img_linear is None:
                        continue
                    img_linear_rgb = cv2.cvtColor(img_linear, cv2.COLOR_BGR2RGB)
                    # Support overriding EXR behavior if user picked 's'
                    img_srgb = np.maximum(img_linear_rgb, 0.0)
                else:
                    img_bgr = cv2.imread(fpath)
                    if img_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_srgb = img_rgb.astype(np.float32) / 255.0

            # 2. Read Alpha (Mask)
            mask_linear = None
            if alpha_cap:
                ret, frame = alpha_cap.read()
                if not ret:
                    break
                mask_linear = frame[:, :, 2].astype(np.float32) / 255.0
            else:
                fpath = os.path.join(clip.alpha_asset.path, alpha_files[i])
                mask_in = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

                if mask_in is None:
                    continue

                if mask_in.ndim == 3:
                    if mask_in.shape[2] == 3:
                        mask_linear = mask_in[:, :, 0]
                    else:
                        mask_linear = mask_in
                else:
                    mask_linear = mask_in

                if mask_linear.dtype == np.uint8:
                    mask_linear = mask_linear.astype(np.float32) / 255.0
                elif mask_linear.dtype == np.uint16:
                    mask_linear = mask_linear.astype(np.float32) / 65535.0
                else:
                    mask_linear = mask_linear.astype(np.float32)

            if mask_linear.shape[:2] != img_srgb.shape[:2]:
                mask_linear = cv2.resize(
                    mask_linear, (img_srgb.shape[1], img_srgb.shape[0]), interpolation=cv2.INTER_LINEAR
                )

            # 3. Process
            USE_STRAIGHT_MODEL = True
            res = engine.process_frame(
                img_srgb,
                mask_linear,
                input_is_linear=input_is_linear,
                fg_is_straight=USE_STRAIGHT_MODEL,
                despill_strength=despill_strength,
                auto_despeckle=auto_despeckle,
                despeckle_size=despeckle_size,
                refiner_scale=refiner_scale,
            )

            pred_fg = res["fg"]  # sRGB
            pred_alpha = res["alpha"]  # Linear

            # 4. Save (EXR DWAB Half-Float)

            # Compression Params
            exr_flags = [
                cv2.IMWRITE_EXR_TYPE,
                cv2.IMWRITE_EXR_TYPE_HALF,
                # DWAB fails. PXR24 verified as smallest working format (46KB vs ZIP 56KB vs B44A 688KB)
                cv2.IMWRITE_EXR_COMPRESSION,
                cv2.IMWRITE_EXR_COMPRESSION_PXR24,
            ]

            # Save FG
            # pred_fg is RGB 0-1 float. Convert to BGR for OpenCV
            fg_bgr = cv2.cvtColor(pred_fg, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(fg_dir, f"{input_stem}.exr"), fg_bgr, exr_flags)

            # Save Matte
            if pred_alpha.ndim == 3:
                pred_alpha = pred_alpha[:, :, 0]
            # Matte is single channel linear float
            cv2.imwrite(os.path.join(matte_dir, f"{input_stem}.exr"), pred_alpha, exr_flags)

            # 5. Generate Reference Comp
            comp_srgb = res["comp"]
            # Save Comp (PNG 8-bit)
            comp_bgr = cv2.cvtColor((np.clip(comp_srgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(comp_dir, f"{input_stem}.png"), comp_bgr)

            # 6. Save Processed (RGBA EXR)
            if "processed" in res:
                # Result is RGBA
                proc_rgba = res["processed"]
                # Convert to BGRA for OpenCV
                proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(os.path.join(proc_dir, f"{input_stem}.exr"), proc_bgra, exr_flags)

        print("")
        if input_cap:
            input_cap.release()
        if alpha_cap:
            alpha_cap.release()
        logger.info(f"Clip {clip.name} Complete.")


def organize_target(target_dir: str) -> None:
    """
    Organizes a specific folder.
    1. If loose video -> Rename to Input.ext (if safe).
    2. If sequence -> Move to Input/.
    3. Ensure AlphaHint and VideoMamaMaskHint folders exist.
    """
    logger.info(f"Organizing Target: {target_dir}")

    if not os.path.exists(target_dir):
        logger.error(f"Target directory not found: {target_dir}")
        return

    # Check for loose video
    # Strategy: Find largest video file that ISN'T named Input.*
    candidates = [f for f in os.listdir(target_dir) if is_video_file(f)]
    candidates = [f for f in candidates if not os.path.splitext(f)[0].lower() == "input"]

    if candidates and not os.path.exists(os.path.join(target_dir, "Input")):
        # If multiple, pick largest (heuristic for 'Main Plate')
        candidates.sort(key=lambda f: os.path.getsize(os.path.join(target_dir, f)), reverse=True)
        main_clip = candidates[0]
        ext = os.path.splitext(main_clip)[1]

        try:
            shutil.move(os.path.join(target_dir, main_clip), os.path.join(target_dir, f"Input{ext}"))
            logger.info(f"Renamed '{main_clip}' to 'Input{ext}'")
        except Exception as e:
            logger.error(f"Failed to rename '{main_clip}': {e}")

    # Check for Image Sequence (Flat)
    # Only if Input folder doesn't exist and Input video doesn't exist
    has_input_dir = os.path.isdir(os.path.join(target_dir, "Input"))
    has_input_video = any(
        is_video_file(f) and os.path.basename(f).lower().startswith("input") for f in os.listdir(target_dir)
    )

    if not has_input_dir and not has_input_video:
        all_files = sorted(glob.glob(os.path.join(target_dir, "*")))
        image_files = [f for f in all_files if is_image_file(f)]

        if len(image_files) > 0:
            try:
                input_subdir = os.path.join(target_dir, "Input")
                os.makedirs(input_subdir)
                for img in image_files:
                    shutil.move(img, os.path.join(input_subdir, os.path.basename(img)))
                logger.info(
                    f"Organized: Moved {len(image_files)} images in '{os.path.basename(target_dir)}' to 'Input/'"
                )
            except Exception as e:
                logger.error(f"Failed to organize sequence in '{target_dir}': {e}")

    # Create Hints
    for hint in ["AlphaHint", "VideoMamaMaskHint"]:
        hint_path = os.path.join(target_dir, hint)
        if not os.path.exists(hint_path):
            os.makedirs(hint_path)


def organize_clips(clips_dir: str) -> None:
    """
    Legacy wrapper for backward compatibility with 'ClipsForInference' folder.
    Organizes all subfolders in the given directory using the new logic.
    """
    if not os.path.exists(clips_dir):
        logger.warning(f"Clips directory not found: {clips_dir}")
        return

    logger.info(f"Organizing Clips Directory: {clips_dir}")

    # Check for loose videos in root
    loose_videos = [f for f in os.listdir(clips_dir) if is_video_file(f) and os.path.isfile(os.path.join(clips_dir, f))]

    # Organize loose videos first
    for v in loose_videos:
        clip_name = os.path.splitext(v)[0]
        ext = os.path.splitext(v)[1]
        target_folder = os.path.join(clips_dir, clip_name)

        if os.path.exists(target_folder):
            logger.warning(f"Skipping loose video '{v}': Target folder '{clip_name}' already exists.")
            continue

        try:
            os.makedirs(target_folder)
            target_file = os.path.join(target_folder, f"Input{ext}")
            shutil.move(os.path.join(clips_dir, v), target_file)
            logger.info(f"Organized: Moved '{v}' to '{clip_name}/Input{ext}'")

            # Also initialize hints immediately
            for hint in ["AlphaHint", "VideoMamaMaskHint"]:
                os.makedirs(os.path.join(target_folder, hint), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to organize video '{v}': {e}")

    # Now iterate all subdirectories and run organize_target
    for entry in os.listdir(clips_dir):
        full_path = os.path.join(clips_dir, entry)
        if os.path.isdir(full_path) and entry not in ["IgnoredClips", "Output"]:
            organize_target(full_path)


def scan_clips() -> list[ClipEntry]:
    if not os.path.exists(CLIPS_DIR):
        os.makedirs(CLIPS_DIR, exist_ok=True)
        return []

    # Auto-organize first
    organize_clips(CLIPS_DIR)

    clip_dirs = [d for d in os.listdir(CLIPS_DIR) if os.path.isdir(os.path.join(CLIPS_DIR, d))]

    valid_clips = []
    invalid_clips = []

    for d in clip_dirs:
        if d.startswith(".") or d.startswith("_") or d == "IgnoredClips":
            continue

        full_path = os.path.join(CLIPS_DIR, d)

        try:
            entry = ClipEntry(d, full_path)
            entry.find_assets()
            entry.validate_pair()
            valid_clips.append(entry)
        except ValueError as ve:
            invalid_clips.append((d, str(ve)))
        except Exception as e:
            invalid_clips.append((d, f"Unexpected error: {e}"))

    if invalid_clips:
        print("\n" + "=" * 60)
        print(" INVALID OR SKIPPED CLIPS")
        print("=" * 60)
        for name, reason in invalid_clips:
            print(f"- {name}: {reason}")
        print("=" * 60 + "\n")
    else:
        print("\nAll clip folders appear valid.\n")

    return valid_clips


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CorridorKey Clip Manager")
    parser.add_argument("--action", choices=["generate_alphas", "run_inference", "list", "wizard"], required=True)
    parser.add_argument(
        "--path",
        help="Clip source path for wizard mode. Accepts Linux/Mac paths and mapped Windows paths.",
        default=None,
    )
    parser.add_argument(
        "--win_path",
        help=r"Deprecated alias for --path (example: V:\...)",
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto-detect CUDA > MPS > CPU)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "mlx"],
        default="auto",
        help="Inference backend (default: auto-detect MLX on Apple Silicon, else Torch)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to process per clip (e.g. 1 for first frame only)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=2048,
        help="Model inference resolution. Lower values reduce VRAM usage (default: 2048).",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "fp32"],
        default="auto",
        help="Torch precision mode. auto uses fp16 on CUDA and fp32 elsewhere.",
    )

    args = parser.parse_args()
    source_path = args.path or args.win_path

    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")

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
            backend=args.backend,
            max_frames=args.max_frames,
            img_size=args.img_size,
            precision=args.precision,
        )
    elif args.action == "wizard":
        if not source_path:
            print("Error: --path (or deprecated --win_path) required for wizard.")
        else:
            raise NotImplementedError("interactive_wizard is not yet implemented")
