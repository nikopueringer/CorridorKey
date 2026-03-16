"""CorridorKeyService — clean backend API for the UI and CLI.

This module wraps all processing logic from clip_manager.py into a
service layer. The UI never calls inference engines directly — it
calls methods here which handle validation, state transitions, and
error reporting.

Model Residency Policy:
    Only ONE heavy model is loaded at a time. Before loading a new
    model type, the previous is unloaded and VRAM freed via
    device_utils.clear_device_cache(). This prevents OOM on 24GB cards.
"""

from __future__ import annotations

import glob as glob_module
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

# Enable OpenEXR support (must be before cv2 import)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from .clip_state import (
    ClipAsset,
    ClipEntry,
    ClipState,
    scan_clips_dir,
)
from .errors import (
    CorridorKeyError,
    FrameReadError,
    JobCancelledError,
    WriteFailureError,
)
from .frame_io import (
    EXR_WRITE_FLAGS,
    read_image_frame,
    read_mask_frame,
    read_video_frame_at,
    read_video_frames,
    read_video_mask_at,
)
from .job_queue import GPUJob, GPUJobQueue
from .validators import (
    ensure_output_dirs,
    validate_frame_counts,
    validate_frame_read,
    validate_write,
)

logger = logging.getLogger(__name__)

# Project paths — frozen-build aware
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _ActiveModel(Enum):
    """Tracks which heavy model is currently loaded in VRAM."""

    NONE = "none"
    INFERENCE = "inference"
    GVM = "gvm"
    VIDEOMAMA = "videomama"


@dataclass
class InferenceParams:
    """Frozen parameters for a single inference job."""

    input_is_linear: bool = False
    despill_strength: float = 1.0  # 0.0 to 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
    refiner_scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceParams":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class OutputConfig:
    """Which output types to produce and their format."""

    fg_enabled: bool = True
    fg_format: str = "exr"  # "exr" or "png"
    matte_enabled: bool = True
    matte_format: str = "exr"
    comp_enabled: bool = True
    comp_format: str = "png"
    processed_enabled: bool = True
    processed_format: str = "exr"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OutputConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def enabled_outputs(self) -> list[str]:
        """Return list of enabled output names for manifest."""
        out = []
        if self.fg_enabled:
            out.append("fg")
        if self.matte_enabled:
            out.append("matte")
        if self.comp_enabled:
            out.append("comp")
        if self.processed_enabled:
            out.append("processed")
        return out


@dataclass
class FrameResult:
    """Result summary for a single processed frame (no numpy in this struct)."""

    frame_index: int
    input_stem: str
    success: bool
    warning: str | None = None


class CorridorKeyService:
    """Main backend service — scan, validate, process, write.

    Usage:
        service = CorridorKeyService()
        clips = service.scan_clips("/path/to/ClipsForInference")
        ready = service.get_clips_by_state(clips, ClipState.READY)

        for clip in ready:
            params = InferenceParams(despill_strength=0.8)
            service.run_inference(clip, params, on_progress=my_callback)
    """

    def __init__(self):
        self._engine = None
        self._gvm_processor = None
        self._videomama_pipeline = None
        self._active_model = _ActiveModel.NONE
        self._device: str = "cpu"
        self._job_queue: GPUJobQueue | None = None
        # GPU mutex — serializes ALL model operations (Codex: thread safety)
        self._gpu_lock = threading.Lock()

    @property
    def job_queue(self) -> GPUJobQueue:
        """Lazy-init GPU job queue (only needed when UI is running)."""
        if self._job_queue is None:
            self._job_queue = GPUJobQueue()
        return self._job_queue

    # --- Device & Engine Management ---

    def detect_device(self) -> str:
        """Detect best available compute device using centralized device_utils."""
        try:
            from device_utils import resolve_device

            self._device = resolve_device()
        except ImportError:
            self._device = "cpu"
            logger.warning("device_utils not available — using CPU")
        logger.info(f"Compute device: {self._device}")
        return self._device

    def get_vram_info(self) -> dict[str, float]:
        """Get GPU VRAM info in GB. Returns empty dict if not CUDA."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            props = torch.cuda.get_device_properties(0)
            total_bytes = props.total_memory
            reserved = torch.cuda.memory_reserved(0)
            return {
                "total": total_bytes / (1024**3),
                "reserved": reserved / (1024**3),
                "allocated": torch.cuda.memory_allocated(0) / (1024**3),
                "free": (total_bytes - reserved) / (1024**3),
                "name": torch.cuda.get_device_name(0),
            }
        except Exception as e:
            logger.debug(f"VRAM query failed: {e}")
            return {}

    @staticmethod
    def _vram_allocated_mb() -> float:
        """Return current VRAM allocated in MB, or 0 if unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024**2)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _safe_offload(obj: object) -> None:
        """Move a model's GPU tensors to CPU before dropping the reference.

        Handles diffusers pipelines (.to('cpu')), plain nn.Modules (.cpu()),
        and objects with an explicit unload() method.
        """
        if obj is None:
            return
        logger.debug(f"Offloading model: {type(obj).__name__}")
        try:
            if hasattr(obj, "unload"):
                obj.unload()
            elif hasattr(obj, "to"):
                obj.to("cpu")
            elif hasattr(obj, "cpu"):
                obj.cpu()
        except Exception as e:
            logger.debug(f"Model offload warning: {e}")

    def _ensure_model(self, needed: _ActiveModel) -> None:
        """Model residency manager — unload current model if switching types.

        Only ONE heavy model stays in VRAM at a time. Before loading a
        different model, the previous is moved to CPU and dereferenced.
        """
        if self._active_model == needed:
            return

        # Unload whatever is currently loaded
        if self._active_model != _ActiveModel.NONE:
            # Snapshot VRAM before unload for leak diagnosis
            vram_before_mb = self._vram_allocated_mb()
            logger.info(
                f"Unloading {self._active_model.value} model for {needed.value} (VRAM before: {vram_before_mb:.0f}MB)"
            )

            if self._active_model == _ActiveModel.INFERENCE:
                self._safe_offload(self._engine)
                self._engine = None
            elif self._active_model == _ActiveModel.GVM:
                self._safe_offload(self._gvm_processor)
                self._gvm_processor = None
            elif self._active_model == _ActiveModel.VIDEOMAMA:
                self._safe_offload(self._videomama_pipeline)
                self._videomama_pipeline = None

            import gc

            gc.collect()

            try:
                from device_utils import clear_device_cache

                clear_device_cache(self._device)
            except ImportError:
                logger.debug("device_utils not available for cache clear during model switch")

            vram_after_mb = self._vram_allocated_mb()
            freed = vram_before_mb - vram_after_mb
            logger.info(f"VRAM after unload: {vram_after_mb:.0f}MB (freed {freed:.0f}MB)")

        self._active_model = needed

    def _get_engine(self):
        """Lazy-load the CorridorKey inference engine."""
        self._ensure_model(_ActiveModel.INFERENCE)

        if self._engine is not None:
            return self._engine

        try:
            from CorridorKeyModule.inference_engine import CorridorKeyEngine
        except ImportError as exc:
            raise RuntimeError("CorridorKeyModule is not installed. Run: uv sync") from exc

        ckpt_dir = os.path.join(BASE_DIR, "CorridorKeyModule", "checkpoints")
        ckpt_files = glob_module.glob(os.path.join(ckpt_dir, "*.pth"))

        if len(ckpt_files) == 0:
            raise FileNotFoundError(f"No .pth checkpoint found in {ckpt_dir}")
        elif len(ckpt_files) > 1:
            raise ValueError(
                f"Multiple checkpoints found in {ckpt_dir}. "
                f"Please ensure only one exists: {[os.path.basename(f) for f in ckpt_files]}"
            )

        ckpt_path = ckpt_files[0]
        logger.info(f"Loading checkpoint: {os.path.basename(ckpt_path)}")
        t0 = time.monotonic()
        self._engine = CorridorKeyEngine(
            checkpoint_path=ckpt_path,
            device=self._device,
            img_size=2048,
        )
        logger.info(f"Engine loaded in {time.monotonic() - t0:.1f}s")
        return self._engine

    def _get_gvm(self):
        """Lazy-load the GVM processor."""
        self._ensure_model(_ActiveModel.GVM)

        if self._gvm_processor is not None:
            return self._gvm_processor

        from gvm_core import GVMProcessor

        logger.info("Loading GVM processor...")
        t0 = time.monotonic()
        self._gvm_processor = GVMProcessor(device=self._device)
        logger.info(f"GVM loaded in {time.monotonic() - t0:.1f}s")
        return self._gvm_processor

    def _get_videomama_pipeline(self):
        """Lazy-load the VideoMaMa inference pipeline."""
        self._ensure_model(_ActiveModel.VIDEOMAMA)

        if self._videomama_pipeline is not None:
            return self._videomama_pipeline

        sys.path.insert(0, os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import load_videomama_model

        logger.info("Loading VideoMaMa pipeline...")
        t0 = time.monotonic()
        self._videomama_pipeline = load_videomama_model(device=self._device)
        logger.info(f"VideoMaMa loaded in {time.monotonic() - t0:.1f}s")
        return self._videomama_pipeline

    def unload_engines(self) -> None:
        """Free GPU memory by unloading all engines."""
        self._safe_offload(self._engine)
        self._safe_offload(self._gvm_processor)
        self._safe_offload(self._videomama_pipeline)
        self._engine = None
        self._gvm_processor = None
        self._videomama_pipeline = None
        self._active_model = _ActiveModel.NONE
        try:
            from device_utils import clear_device_cache

            clear_device_cache(self._device)
        except ImportError:
            logger.debug("device_utils not available for cache clear during unload")
        logger.info("All engines unloaded, VRAM freed")

    # --- Clip Scanning ---

    def scan_clips(
        self,
        clips_dir: str,
        allow_standalone_videos: bool = True,
    ) -> list[ClipEntry]:
        """Scan a directory for clip folders."""
        return scan_clips_dir(clips_dir, allow_standalone_videos=allow_standalone_videos)

    def get_clips_by_state(
        self,
        clips: list[ClipEntry],
        state: ClipState,
    ) -> list[ClipEntry]:
        """Filter clips by state."""
        return [c for c in clips if c.state == state]

    # --- Frame I/O Helpers ---

    def _read_input_frame(
        self,
        clip: ClipEntry,
        frame_index: int,
        input_files: list[str],
        input_cap: Any | None,
        input_is_linear: bool,
    ) -> tuple[np.ndarray | None, str, bool]:
        """Read a single input frame.

        Returns:
            (image_float32, stem_name, is_linear_override)
        """
        logger.debug(f"Reading input frame {frame_index} for '{clip.name}'")
        input_stem = f"{frame_index:05d}"

        if input_cap:
            ret, frame = input_cap.read()
            if not ret:
                return None, input_stem, False
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return img_rgb.astype(np.float32) / 255.0, input_stem, input_is_linear
        else:
            if frame_index >= len(input_files):
                logger.warning(
                    f"Clip '{clip.name}': frame_index {frame_index} out of range (have {len(input_files)} frames)"
                )
                return None, f"{frame_index:05d}", input_is_linear
            fpath = os.path.join(clip.input_asset.path, input_files[frame_index])
            input_stem = os.path.splitext(input_files[frame_index])[0]
            img = read_image_frame(fpath)
            validate_frame_read(img, clip.name, frame_index, fpath)
            return img, input_stem, input_is_linear

    def _read_alpha_frame(
        self,
        clip: ClipEntry,
        frame_index: int,
        alpha_files: list[str],
        alpha_cap: Any | None,
    ) -> np.ndarray | None:
        """Read a single alpha/mask frame and normalize to [H, W] float32."""
        if alpha_cap:
            ret, frame = alpha_cap.read()
            if not ret:
                return None
            return frame[:, :, 2].astype(np.float32) / 255.0
        else:
            fpath = os.path.join(clip.alpha_asset.path, alpha_files[frame_index])
            mask = read_mask_frame(fpath, clip.name, frame_index)
            validate_frame_read(mask, clip.name, frame_index, fpath)
            return mask

    def _write_image(
        self,
        img: np.ndarray,
        path: str,
        fmt: str,
        clip_name: str,
        frame_index: int,
    ) -> None:
        """Write a single image in the requested format."""
        if fmt == "exr":
            # EXR requires float32 — convert if uint8 (e.g. pre-converted comp)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype != np.float32:
                img = img.astype(np.float32)
            validate_write(cv2.imwrite(path, img, EXR_WRITE_FLAGS), clip_name, frame_index, path)
        else:
            # PNG 8-bit
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            validate_write(cv2.imwrite(path, img), clip_name, frame_index, path)

    def _write_manifest(
        self,
        output_root: str,
        output_config: OutputConfig,
        params: InferenceParams,
    ) -> None:
        """Write run manifest recording expected outputs/extensions per run.

        Codex: base resume on manifest, not hardcoded FG/Matte intersection.
        Uses atomic write (tmp + rename) to prevent corruption.
        """
        manifest = {
            "version": 1,
            "enabled_outputs": output_config.enabled_outputs,
            "formats": {
                "fg": output_config.fg_format,
                "matte": output_config.matte_format,
                "comp": output_config.comp_format,
                "processed": output_config.processed_format,
            },
            "params": params.to_dict(),
        }
        manifest_path = os.path.join(output_root, ".corridorkey_manifest.json")
        tmp_path = manifest_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(manifest, f, indent=2)
            # Atomic replace (os.replace is atomic on both POSIX and Windows)
            os.replace(tmp_path, manifest_path)
        except Exception as e:
            logger.warning(f"Failed to write manifest: {e}")

    def _write_outputs(
        self,
        res: dict,
        dirs: dict[str, str],
        input_stem: str,
        clip_name: str,
        frame_index: int,
        output_config: OutputConfig | None = None,
    ) -> None:
        """Write output types for a single frame respecting OutputConfig."""
        cfg = output_config or OutputConfig()
        logger.debug(f"Writing outputs for '{clip_name}' frame {frame_index} stem='{input_stem}'")

        pred_fg = res["fg"]
        pred_alpha = res["alpha"]

        # FG
        if cfg.fg_enabled:
            fg_bgr = cv2.cvtColor(pred_fg, cv2.COLOR_RGB2BGR)
            fg_path = os.path.join(dirs["fg"], f"{input_stem}.{cfg.fg_format}")
            self._write_image(fg_bgr, fg_path, cfg.fg_format, clip_name, frame_index)

        # Matte
        if cfg.matte_enabled:
            alpha = pred_alpha
            if alpha.ndim == 3:
                alpha = alpha[:, :, 0]
            matte_path = os.path.join(dirs["matte"], f"{input_stem}.{cfg.matte_format}")
            self._write_image(alpha, matte_path, cfg.matte_format, clip_name, frame_index)

        # Comp
        if cfg.comp_enabled:
            comp_srgb = res["comp"]
            comp_bgr = cv2.cvtColor(
                (np.clip(comp_srgb, 0.0, 1.0) * 255.0).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            comp_path = os.path.join(dirs["comp"], f"{input_stem}.{cfg.comp_format}")
            self._write_image(comp_bgr, comp_path, cfg.comp_format, clip_name, frame_index)

        # Processed (RGBA premultiplied)
        if cfg.processed_enabled and "processed" in res:
            proc_rgba = res["processed"]
            proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)
            proc_path = os.path.join(dirs["processed"], f"{input_stem}.{cfg.processed_format}")
            self._write_image(proc_bgra, proc_path, cfg.processed_format, clip_name, frame_index)

    # --- Processing ---

    def run_inference(
        self,
        clip: ClipEntry,
        params: InferenceParams,
        job: GPUJob | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
        skip_stems: set[str] | None = None,
        output_config: OutputConfig | None = None,
        frame_range: tuple[int, int] | None = None,
    ) -> list[FrameResult]:
        """Run CorridorKey inference on a single clip.

        Args:
            clip: Must be in READY or COMPLETE state with both input_asset and alpha_asset.
            params: Frozen inference parameters.
            job: Optional GPUJob for cancel checking.
            on_progress: Called with (clip_name, current_frame, total_frames).
            on_warning: Called with warning messages for non-fatal issues.
            skip_stems: Set of frame stems to skip (for resume support).
            output_config: Which outputs to write and their formats.

        Returns:
            List of FrameResult for each frame.

        Raises:
            JobCancelledError: If job.is_cancelled becomes True.
            Various CorridorKeyError subclasses for fatal issues.
        """
        if clip.input_asset is None or clip.alpha_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input or alpha asset")

        t_start = time.monotonic()

        with self._gpu_lock:
            engine = self._get_engine()
        dirs = ensure_output_dirs(clip.root_path)
        cfg = output_config or OutputConfig()

        # Write run manifest (Codex: resume must know which outputs were enabled)
        self._write_manifest(dirs["root"], cfg, params)

        num_frames = validate_frame_counts(
            clip.name,
            clip.input_asset.frame_count,
            clip.alpha_asset.frame_count,
        )

        # Open video captures or get file lists
        input_cap = None
        alpha_cap = None
        input_files: list[str] = []
        alpha_files: list[str] = []

        if clip.input_asset.asset_type == "video":
            input_cap = cv2.VideoCapture(clip.input_asset.path)
        else:
            input_files = clip.input_asset.get_frame_files()

        if clip.alpha_asset.asset_type == "video":
            alpha_cap = cv2.VideoCapture(clip.alpha_asset.path)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()

        results: list[FrameResult] = []
        skipped: list[int] = []
        skip_stems = skip_stems or set()

        # Determine frame range (in/out markers or full clip)
        if frame_range is not None:
            range_start = max(0, frame_range[0])
            range_end = min(num_frames - 1, frame_range[1])
            frame_indices = range(range_start, range_end + 1)
            range_count = range_end - range_start + 1
        else:
            frame_indices = range(num_frames)
            range_count = num_frames

        try:
            for progress_i, i in enumerate(frame_indices):
                # Check cancellation between frames
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, i)

                # Report progress every frame (enables responsive cancel + timer)
                if on_progress:
                    on_progress(clip.name, progress_i, range_count)

                try:
                    # Read input
                    img, input_stem, is_linear = self._read_input_frame(
                        clip,
                        i,
                        input_files,
                        input_cap,
                        params.input_is_linear,
                    )
                    if img is None:
                        skipped.append(i)
                        results.append(FrameResult(i, f"{i:05d}", False, "video read failed"))
                        continue

                    # Resume: skip frames that already have outputs
                    if input_stem in skip_stems:
                        results.append(FrameResult(i, input_stem, True, "resumed (skipped)"))
                        continue

                    # Read alpha
                    mask = self._read_alpha_frame(clip, i, alpha_files, alpha_cap)
                    if mask is None:
                        skipped.append(i)
                        results.append(FrameResult(i, input_stem, False, "alpha read failed"))
                        continue

                    # Resize mask if dimensions don't match input
                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

                    # Process (GPU-locked — process_frame mutates model hooks)
                    t_frame = time.monotonic()
                    with self._gpu_lock:
                        res = engine.process_frame(
                            img,
                            mask,
                            input_is_linear=is_linear,
                            fg_is_straight=True,
                            despill_strength=params.despill_strength,
                            auto_despeckle=params.auto_despeckle,
                            despeckle_size=params.despeckle_size,
                            refiner_scale=params.refiner_scale,
                        )
                    logger.debug(f"Clip '{clip.name}' frame {i}: process_frame {time.monotonic() - t_frame:.3f}s")

                    # Write outputs
                    self._write_outputs(res, dirs, input_stem, clip.name, i, cfg)
                    results.append(FrameResult(i, input_stem, True))

                except FrameReadError as e:
                    logger.warning(str(e))
                    skipped.append(i)
                    results.append(FrameResult(i, f"{i:05d}", False, str(e)))
                    if on_warning:
                        on_warning(str(e))

                except WriteFailureError as e:
                    logger.error(str(e))
                    results.append(FrameResult(i, f"{i:05d}", False, str(e)))
                    if on_warning:
                        on_warning(str(e))

            # Final progress
            if on_progress:
                on_progress(clip.name, range_count, range_count)

        finally:
            if input_cap:
                input_cap.release()
            if alpha_cap:
                alpha_cap.release()

        # Summary
        processed = sum(1 for r in results if r.success)
        if skipped:
            msg = (
                f"Clip '{clip.name}': {len(skipped)} frame(s) skipped: "
                f"{skipped[:20]}{'...' if len(skipped) > 20 else ''}"
            )
            logger.warning(msg)
            if on_warning:
                on_warning(msg)

        t_total = time.monotonic() - t_start
        range_label = f" (range {frame_range[0]}-{frame_range[1]})" if frame_range else ""
        logger.info(
            f"Clip '{clip.name}': inference complete{range_label}. {processed}/{range_count} frames "
            f"in {t_total:.1f}s ({t_total / max(processed, 1):.2f}s/frame avg)"
        )

        # State transition — only set COMPLETE if full clip was processed
        is_full_clip = frame_range is None or (frame_range[0] == 0 and frame_range[1] >= num_frames - 1)
        if processed == range_count and is_full_clip:
            try:
                clip.transition_to(ClipState.COMPLETE)
            except Exception as e:
                logger.warning(f"Clip '{clip.name}': state transition to COMPLETE failed: {e}")

        return results

    # --- Single-Frame Reprocess (Preview) ---

    def is_engine_loaded(self) -> bool:
        """True if the inference engine is already loaded in VRAM."""
        return self._active_model == _ActiveModel.INFERENCE and self._engine is not None

    def reprocess_single_frame(
        self,
        clip: ClipEntry,
        params: InferenceParams,
        frame_index: int,
        job: GPUJob | None = None,
    ) -> dict | None:
        """Reprocess a single frame with current params.

        Returns the result dict (fg, alpha, comp, processed) or None.
        This runs through the GPU lock for thread safety.
        Does NOT write to disk — returns in-memory results for preview.
        """
        t_start = time.monotonic()
        if clip.input_asset is None or clip.alpha_asset is None:
            return None

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            engine = self._get_engine()

        # Read the specific input frame
        if clip.input_asset.asset_type == "video":
            img = read_video_frame_at(clip.input_asset.path, frame_index)
        else:
            input_files = clip.input_asset.get_frame_files()
            if frame_index >= len(input_files):
                return None
            img = read_image_frame(os.path.join(clip.input_asset.path, input_files[frame_index]))
        if img is None:
            return None

        # Read the specific alpha frame
        if clip.alpha_asset.asset_type == "video":
            mask = read_video_mask_at(clip.alpha_asset.path, frame_index)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()
            if frame_index >= len(alpha_files):
                return None
            mask = read_mask_frame(
                os.path.join(clip.alpha_asset.path, alpha_files[frame_index]),
                clip.name,
                frame_index,
            )
        if mask is None:
            return None

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            res = engine.process_frame(
                img,
                mask,
                input_is_linear=params.input_is_linear,
                fg_is_straight=True,
                despill_strength=params.despill_strength,
                auto_despeckle=params.auto_despeckle,
                despeckle_size=params.despeckle_size,
                refiner_scale=params.refiner_scale,
            )
        logger.debug(f"Clip '{clip.name}' frame {frame_index}: reprocess {time.monotonic() - t_start:.3f}s")
        return res

    # --- GVM Alpha Generation ---

    def run_gvm(
        self,
        clip: ClipEntry,
        job: GPUJob | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        """Run GVM auto alpha generation for a clip.

        Transitions clip: RAW → READY (creates AlphaHint directory).

        Args:
            clip: Must be in RAW state with input_asset.
            job: Optional GPUJob for cancel checking.
            on_progress: Progress callback (GVM is monolithic, reports start/end).
            on_warning: Warning callback.
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for GVM")

        t_start = time.monotonic()

        with self._gpu_lock:
            gvm = self._get_gvm()

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        if on_progress:
            on_progress(clip.name, 0, 1)

        # Check cancel before starting
        if job and job.is_cancelled:
            raise JobCancelledError(clip.name, 0)

        # Per-batch progress callback — GVM iterates over frames internally
        def _gvm_progress(batch_idx: int, total_batches: int) -> None:
            if on_progress:
                on_progress(clip.name, batch_idx, total_batches)
            # Check cancel between batches
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, batch_idx)

        try:
            gvm.process_sequence(
                input_path=clip.input_asset.path,
                output_dir=clip.root_path,
                num_frames_per_batch=1,
                decode_chunk_size=1,
                denoise_steps=1,
                mode="matte",
                write_video=False,
                direct_output_dir=alpha_dir,
                progress_callback=_gvm_progress,
            )
        except JobCancelledError:
            raise
        except Exception as e:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0) from None
            raise CorridorKeyError(f"GVM failed for '{clip.name}': {e}") from e

        # Refresh alpha asset
        clip.alpha_asset = ClipAsset(alpha_dir, "sequence")

        if on_progress:
            on_progress(clip.name, 1, 1)

        # Transition RAW → READY
        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after GVM: {e}")

        elapsed = time.monotonic() - t_start
        logger.info(f"GVM complete for '{clip.name}': {clip.alpha_asset.frame_count} alpha frames in {elapsed:.1f}s")

    # --- VideoMaMa Alpha Generation ---

    def run_videomama(
        self,
        clip: ClipEntry,
        job: GPUJob | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        chunk_size: int = 50,
    ) -> None:
        """Run VideoMaMa guided alpha generation for a clip.

        Transitions clip: MASKED → READY (creates AlphaHint directory).

        Args:
            clip: Must be in MASKED state with input_asset and mask_asset.
            job: Optional GPUJob for cancel checking.
            on_progress: Progress callback with per-chunk updates.
            on_warning: Warning callback.
            on_status: Phase status callback (e.g. "Loading model...").
            chunk_size: Frames per chunk (lower = less RAM, default 50).
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for VideoMaMa")
        if clip.mask_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing mask asset for VideoMaMa")

        def _status(msg: str) -> None:
            logger.info(f"VideoMaMa [{clip.name}]: {msg}")
            if on_status:
                on_status(msg)

        def _check_cancel(phase: str = "") -> None:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0)

        t_start = time.monotonic()

        # ── Phase 1: Load model ──
        _status("Loading model...")
        with self._gpu_lock:
            pipeline = self._get_videomama_pipeline()
        _check_cancel("model load")

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        # Don't report progress yet — phase status is showing in the status bar.
        # Sending on_progress(0, N) would switch the status bar to frame-counter
        # mode and overwrite the phase text on every tick.

        # ── Phase 2: Load input frames ──
        _status("Loading frames...")
        input_frames = self._load_frames_for_videomama(
            clip.input_asset,
            clip.name,
            job=job,
            on_status=on_status,
        )
        _check_cancel("frame load")

        # ── Phase 3: Load + stem-match masks ──
        _status("Loading masks...")
        mask_stems: dict[str, np.ndarray] = {}
        if clip.mask_asset.asset_type == "sequence":
            mask_files = clip.mask_asset.get_frame_files()
            for _i, fname in enumerate(mask_files):
                _check_cancel("mask load")
                fpath = os.path.join(clip.mask_asset.path, fname)
                m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    _, binary = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
                    stem = os.path.splitext(fname)[0]
                    mask_stems[stem] = binary
        else:
            raw_masks = self._load_mask_frames_for_videomama(clip.mask_asset, clip.name)
            for i, m in enumerate(raw_masks):
                mask_stems[f"frame_{i:06d}"] = m

        # Build output filenames from input stems
        if clip.input_asset and clip.input_asset.asset_type == "sequence":
            input_names = clip.input_asset.get_frame_files()
        else:
            input_names = [f"frame_{i:06d}.png" for i in range(len(input_frames))]

        # Align masks to input frames by stem, defaulting to all-black
        num_frames = len(input_frames)
        mask_frames = []
        for fname in input_names:
            stem = os.path.splitext(fname)[0]
            if stem in mask_stems:
                mask_frames.append(mask_stems[stem])
            else:
                h_m, w_m = input_frames[0].shape[:2] if input_frames else (4, 4)
                mask_frames.append(np.zeros((h_m, w_m), dtype=np.uint8))

        # ── Resume logic ──
        existing_alpha = []
        if os.path.isdir(alpha_dir):
            existing_alpha = [f for f in os.listdir(alpha_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        n_existing = len(existing_alpha)
        completed_chunks = n_existing // chunk_size
        start_chunk = max(0, completed_chunks - 1)
        start_frame = start_chunk * chunk_size
        if start_frame > 0:
            keep = set()
            for i in range(start_frame):
                if i < len(input_names):
                    stem = os.path.splitext(input_names[i])[0]
                    keep.add(f"{stem}.png")
            for fname in existing_alpha:
                if fname not in keep:
                    os.remove(os.path.join(alpha_dir, fname))
            logger.info(
                f"VideoMaMa resuming for '{clip.name}': {n_existing} alpha frames existed, "
                f"rolling back to chunk {start_chunk} (frame {start_frame})"
            )

        # ── Phase 4: Inference (per-chunk) ──
        sys.path.insert(0, os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import run_inference

        total_chunks = (num_frames + chunk_size - 1) // chunk_size
        _status(f"Running inference (chunk 1/{total_chunks})...")
        frames_written = start_frame
        for chunk_idx, chunk_output in enumerate(
            run_inference(pipeline, input_frames, mask_frames, chunk_size=chunk_size)
        ):
            _check_cancel("inference")

            # Skip already-completed chunks (resume)
            if chunk_idx < start_chunk:
                frames_written += len(chunk_output)
                if on_progress:
                    on_progress(clip.name, frames_written, num_frames)
                continue

            _status(f"Processing chunk {chunk_idx + 1}/{total_chunks}...")

            # Write chunk frames
            t_chunk = time.monotonic()
            for frame in chunk_output:
                out_bgr = cv2.cvtColor(
                    (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
                if frames_written < len(input_names):
                    stem = os.path.splitext(input_names[frames_written])[0]
                    out_name = f"{stem}.png"
                else:
                    out_name = f"frame_{frames_written:06d}.png"
                out_path = os.path.join(alpha_dir, out_name)
                cv2.imwrite(out_path, out_bgr)
                frames_written += 1
            chunk_elapsed = time.monotonic() - t_chunk
            logger.debug(f"Clip '{clip.name}' chunk {chunk_idx}: {len(chunk_output)} frames in {chunk_elapsed:.3f}s")

            if on_progress:
                on_progress(clip.name, frames_written, num_frames)

        # Refresh alpha asset
        clip.alpha_asset = ClipAsset(alpha_dir, "sequence")

        # Transition MASKED → READY
        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after VideoMaMa: {e}")

        elapsed = time.monotonic() - t_start
        logger.info(f"VideoMaMa complete for '{clip.name}': {frames_written} alpha frames in {elapsed:.1f}s")

    def _load_frames_for_videomama(
        self,
        asset: ClipAsset,
        clip_name: str,
        job: GPUJob | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> list[np.ndarray]:
        """Load input frames for VideoMaMa as uint8 RGB [0, 255].

        The VideoMaMa inference code expects uint8 arrays for PIL conversion.
        Reports loading progress via on_status and checks cancel via job.
        """
        if asset.asset_type == "video":
            raw = read_video_frames(asset.path)
            return [(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8) for f in raw]
        frames = []
        files = asset.get_frame_files()
        total = len(files)
        for i, fname in enumerate(files):
            if job and job.is_cancelled:
                from .errors import JobCancelledError

                raise JobCancelledError(clip_name, i)
            fpath = os.path.join(asset.path, fname)
            img = read_image_frame(fpath, gamma_correct_exr=True)
            if img is not None:
                frames.append((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
            if on_status and i % 20 == 0 and i > 0:
                on_status(f"Loading frames ({i}/{total})...")
        return frames

    def _load_mask_frames_for_videomama(self, asset: ClipAsset, clip_name: str) -> list[np.ndarray]:
        """Load mask frames for VideoMaMa as uint8 grayscale [0, 255].

        The VideoMaMa inference code expects uint8 arrays for PIL conversion.
        Binary threshold at 10: anything above → 255 (foreground), else → 0.
        """

        def _threshold_mask(bgr_frame: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            return binary  # uint8

        if asset.asset_type == "video":
            return read_video_frames(asset.path, processor=_threshold_mask)
        masks = []
        for fname in asset.get_frame_files():
            fpath = os.path.join(asset.path, fname)
            mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            masks.append(binary)  # uint8
        return masks
