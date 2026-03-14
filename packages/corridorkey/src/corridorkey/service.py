"""CorridorKeyService - clean backend API for the UI and CLI.

This module wraps all processing logic into a service layer. The UI never
calls inference engines directly - it calls methods here which handle
validation, state transitions, and error reporting.

Model Residency Policy:
    Only ONE heavy model is loaded at a time. Before loading a new
    model type, the previous is unloaded and VRAM freed via
    device_utils.clear_device_cache(). This prevents OOM on 24GB cards.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

# Enable OpenEXR support (must be before cv2 import)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from . import device_utils
from .clip_state import ClipAsset, ClipEntry, ClipState, scan_clips_dir
from .engine_factory import create_engine
from .errors import CorridorKeyError, FrameReadError, JobCancelledError, WriteFailureError
from .frame_io import (
    EXR_WRITE_FLAGS,
    read_image_frame,
    read_mask_frame,
    read_video_frame_at,
    read_video_frames,
    read_video_mask_at,
)
from .job_queue import GPUJob, GPUJobQueue
from .validators import ensure_output_dirs, validate_frame_counts, validate_frame_read, validate_write

logger = logging.getLogger(__name__)

# Project paths - frozen-build aware
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS  # type: ignore[attr-defined]
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
    def from_dict(cls, d: dict) -> InferenceParams:
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
    def from_dict(cls, d: dict) -> OutputConfig:
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
    """Main backend service - scan, validate, process, write.

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
        self._gpu_lock = threading.Lock()

    @property
    def job_queue(self) -> GPUJobQueue:
        """Lazy-init GPU job queue (only needed when UI is running)."""
        if self._job_queue is None:
            self._job_queue = GPUJobQueue()
        return self._job_queue

    def detect_device(self) -> str:
        """Detect best available compute device."""
        self._device = device_utils.resolve_device()
        logger.info("Compute device: %s", self._device)
        return self._device

    def get_vram_info(self) -> dict[str, float | str]:
        """Get GPU VRAM info in GB. Returns empty dict if not CUDA."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            props = torch.cuda.get_device_properties(0)
            total_bytes = props.total_mem
            reserved = torch.cuda.memory_reserved(0)
            return {
                "total": total_bytes / (1024**3),
                "reserved": reserved / (1024**3),
                "allocated": torch.cuda.memory_allocated(0) / (1024**3),
                "free": (total_bytes - reserved) / (1024**3),
                "name": torch.cuda.get_device_name(0),
            }
        except Exception as e:
            logger.debug("VRAM query failed: %s", e)
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
    def _safe_offload(obj: Any) -> None:
        """Move a model's GPU tensors to CPU before dropping the reference."""
        if obj is None:
            return
        logger.debug("Offloading model: %s", type(obj).__name__)
        try:
            if hasattr(obj, "unload"):
                obj.unload()
            elif hasattr(obj, "to"):
                obj.to("cpu")
            elif hasattr(obj, "cpu"):
                obj.cpu()
        except Exception as e:
            logger.debug("Model offload warning: %s", e)

    def _ensure_model(self, needed: _ActiveModel) -> None:
        """Model residency manager - unload current model if switching types."""
        if self._active_model == needed:
            return

        if self._active_model != _ActiveModel.NONE:
            vram_before_mb = self._vram_allocated_mb()
            logger.info(
                "Unloading %s model for %s (VRAM before: %.0fMB)",
                self._active_model.value,
                needed.value,
                vram_before_mb,
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
            device_utils.clear_device_cache(self._device)

            vram_after_mb = self._vram_allocated_mb()
            logger.info("VRAM after unload: %.0fMB (freed %.0fMB)", vram_after_mb, vram_before_mb - vram_after_mb)

        self._active_model = needed

    def _get_engine(self):
        """Lazy-load the CorridorKey inference engine via engine_factory."""
        self._ensure_model(_ActiveModel.INFERENCE)
        if self._engine is not None:
            return self._engine
        logger.info("Loading inference engine (device=%s)...", self._device)
        t0 = time.monotonic()
        self._engine = create_engine(device=self._device)
        logger.info("Engine loaded in %.1fs", time.monotonic() - t0)
        return self._engine

    def _get_gvm(self):
        """Lazy-load the GVM processor."""
        self._ensure_model(_ActiveModel.GVM)
        if self._gvm_processor is not None:
            return self._gvm_processor
        from gvm_core import GVMProcessor  # type: ignore[import-not-found]

        logger.info("Loading GVM processor...")
        t0 = time.monotonic()
        self._gvm_processor = GVMProcessor(device=self._device)
        logger.info("GVM loaded in %.1fs", time.monotonic() - t0)
        return self._gvm_processor

    def _get_videomama_pipeline(self):
        """Lazy-load the VideoMaMa inference pipeline."""
        self._ensure_model(_ActiveModel.VIDEOMAMA)
        if self._videomama_pipeline is not None:
            return self._videomama_pipeline
        sys.path.insert(0, os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import load_videomama_model  # type: ignore[import-not-found]

        logger.info("Loading VideoMaMa pipeline...")
        t0 = time.monotonic()
        self._videomama_pipeline = load_videomama_model(device=self._device)
        logger.info("VideoMaMa loaded in %.1fs", time.monotonic() - t0)
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
        device_utils.clear_device_cache(self._device)
        logger.info("All engines unloaded, VRAM freed")

    def scan_clips(self, clips_dir: str, allow_standalone_videos: bool = True) -> list[ClipEntry]:
        """Scan a directory for clip folders."""
        return scan_clips_dir(clips_dir, allow_standalone_videos=allow_standalone_videos)

    def get_clips_by_state(self, clips: list[ClipEntry], state: ClipState) -> list[ClipEntry]:
        """Filter clips by state."""
        return [c for c in clips if c.state == state]

    def _read_input_frame(
        self,
        clip: ClipEntry,
        frame_index: int,
        input_files: list[str],
        input_cap: Any | None,
        input_is_linear: bool,
    ) -> tuple[np.ndarray | None, str, bool]:
        """Read a single input frame. Returns (image_float32, stem_name, is_linear)."""
        logger.debug("Reading input frame %d for '%s'", frame_index, clip.name)
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
                    "Clip '%s': frame_index %d out of range (have %d frames)",
                    clip.name,
                    frame_index,
                    len(input_files),
                )
                return None, f"{frame_index:05d}", input_is_linear
            fpath = os.path.join(clip.input_asset.path, input_files[frame_index])  # type: ignore[union-attr]
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
            fpath = os.path.join(clip.alpha_asset.path, alpha_files[frame_index])  # type: ignore[union-attr]
            mask = read_mask_frame(fpath, clip.name, frame_index)
            validate_frame_read(mask, clip.name, frame_index, fpath)
            return mask

    def _write_image(self, img: np.ndarray, path: str, fmt: str, clip_name: str, frame_index: int) -> None:
        """Write a single image in the requested format."""
        if fmt == "exr":
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype != np.float32:
                img = img.astype(np.float32)
            validate_write(cv2.imwrite(path, img, EXR_WRITE_FLAGS), clip_name, frame_index, path)
        else:
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            validate_write(cv2.imwrite(path, img), clip_name, frame_index, path)

    def _write_manifest(self, output_root: str, output_config: OutputConfig, params: InferenceParams) -> None:
        """Write run manifest. Uses atomic write (tmp + rename) to prevent corruption."""
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
            os.replace(tmp_path, manifest_path)
        except Exception as e:
            logger.warning("Failed to write manifest: %s", e)

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
        logger.debug("Writing outputs for '%s' frame %d stem='%s'", clip_name, frame_index, input_stem)
        pred_fg = res["fg"]
        pred_alpha = res["alpha"]
        if cfg.fg_enabled:
            fg_bgr = cv2.cvtColor(pred_fg, cv2.COLOR_RGB2BGR)
            self._write_image(
                fg_bgr, os.path.join(dirs["fg"], f"{input_stem}.{cfg.fg_format}"), cfg.fg_format, clip_name, frame_index
            )
        if cfg.matte_enabled:
            alpha = pred_alpha[:, :, 0] if pred_alpha.ndim == 3 else pred_alpha
            self._write_image(
                alpha,
                os.path.join(dirs["matte"], f"{input_stem}.{cfg.matte_format}"),
                cfg.matte_format,
                clip_name,
                frame_index,
            )
        if cfg.comp_enabled:
            comp_bgr = cv2.cvtColor((np.clip(res["comp"], 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            self._write_image(
                comp_bgr,
                os.path.join(dirs["comp"], f"{input_stem}.{cfg.comp_format}"),
                cfg.comp_format,
                clip_name,
                frame_index,
            )
        if cfg.processed_enabled and "processed" in res:
            proc_bgra = cv2.cvtColor(res["processed"], cv2.COLOR_RGBA2BGRA)
            self._write_image(
                proc_bgra,
                os.path.join(dirs["processed"], f"{input_stem}.{cfg.processed_format}"),
                cfg.processed_format,
                clip_name,
                frame_index,
            )

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
            frame_range: Optional (start, end) inclusive frame indices.

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
        self._write_manifest(dirs["root"], cfg, params)

        num_frames = validate_frame_counts(clip.name, clip.input_asset.frame_count, clip.alpha_asset.frame_count)

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
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, i)
                if on_progress:
                    on_progress(clip.name, progress_i, range_count)
                try:
                    img, input_stem, is_linear = self._read_input_frame(
                        clip, i, input_files, input_cap, params.input_is_linear
                    )
                    if img is None:
                        skipped.append(i)
                        results.append(FrameResult(i, f"{i:05d}", False, "video read failed"))
                        continue
                    if input_stem in skip_stems:
                        results.append(FrameResult(i, input_stem, True, "resumed (skipped)"))
                        continue
                    mask = self._read_alpha_frame(clip, i, alpha_files, alpha_cap)
                    if mask is None:
                        skipped.append(i)
                        results.append(FrameResult(i, input_stem, False, "alpha read failed"))
                        continue
                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
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
                    logger.debug("Clip '%s' frame %d: process_frame %.3fs", clip.name, i, time.monotonic() - t_frame)
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
            if on_progress:
                on_progress(clip.name, range_count, range_count)
        finally:
            if input_cap:
                input_cap.release()
            if alpha_cap:
                alpha_cap.release()

        processed = sum(1 for r in results if r.success)
        if skipped:
            msg = f"Clip '{clip.name}': {len(skipped)} frame(s) skipped: {skipped[:20]}{'...' if len(skipped) > 20 else ''}"
            logger.warning(msg)
            if on_warning:
                on_warning(msg)

        t_total = time.monotonic() - t_start
        range_label = f" (range {frame_range[0]}-{frame_range[1]})" if frame_range else ""
        logger.info(
            "Clip '%s': inference complete%s. %d/%d frames in %.1fs (%.2fs/frame avg)",
            clip.name,
            range_label,
            processed,
            range_count,
            t_total,
            t_total / max(processed, 1),
        )

        is_full_clip = frame_range is None or (frame_range[0] == 0 and frame_range[1] >= num_frames - 1)
        if processed == range_count and is_full_clip:
            try:
                clip.transition_to(ClipState.COMPLETE)
            except Exception as e:
                logger.warning("Clip '%s': state transition to COMPLETE failed: %s", clip.name, e)

        return results

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
        """Reprocess a single frame. Does NOT write to disk - returns in-memory results for preview."""
        t_start = time.monotonic()
        if clip.input_asset is None or clip.alpha_asset is None:
            return None
        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            engine = self._get_engine()

        if clip.input_asset.asset_type == "video":
            img = read_video_frame_at(clip.input_asset.path, frame_index)
        else:
            input_files = clip.input_asset.get_frame_files()
            if frame_index >= len(input_files):
                return None
            img = read_image_frame(os.path.join(clip.input_asset.path, input_files[frame_index]))
        if img is None:
            return None

        if clip.alpha_asset.asset_type == "video":
            mask = read_video_mask_at(clip.alpha_asset.path, frame_index)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()
            if frame_index >= len(alpha_files):
                return None
            mask = read_mask_frame(
                os.path.join(clip.alpha_asset.path, alpha_files[frame_index]), clip.name, frame_index
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
        logger.debug("Clip '%s' frame %d: reprocess %.3fs", clip.name, frame_index, time.monotonic() - t_start)
        return res

    def run_gvm(
        self,
        clip: ClipEntry,
        job: GPUJob | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        """Run GVM auto alpha generation for a clip. Transitions: RAW -> READY."""
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for GVM")

        t_start = time.monotonic()
        with self._gpu_lock:
            gvm = self._get_gvm()

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        if on_progress:
            on_progress(clip.name, 0, 1)
        if job and job.is_cancelled:
            raise JobCancelledError(clip.name, 0)

        def _gvm_progress(batch_idx: int, total_batches: int) -> None:
            if on_progress:
                on_progress(clip.name, batch_idx, total_batches)
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

        clip.alpha_asset = ClipAsset(alpha_dir, "sequence")
        if on_progress:
            on_progress(clip.name, 1, 1)
        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after GVM: {e}")

        logger.info(
            "GVM complete for '%s': %d alpha frames in %.1fs",
            clip.name,
            clip.alpha_asset.frame_count,
            time.monotonic() - t_start,
        )

    def run_videomama(
        self,
        clip: ClipEntry,
        job: GPUJob | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        chunk_size: int = 50,
    ) -> None:
        """Run VideoMaMa guided alpha generation for a clip. Transitions: MASKED -> READY."""
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for VideoMaMa")
        if clip.mask_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing mask asset for VideoMaMa")

        def _status(msg: str) -> None:
            logger.info("VideoMaMa [%s]: %s", clip.name, msg)
            if on_status:
                on_status(msg)

        def _check_cancel(phase: str = "") -> None:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0)

        t_start = time.monotonic()

        _status("Loading model...")
        with self._gpu_lock:
            pipeline = self._get_videomama_pipeline()
        _check_cancel("model load")

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        _status("Loading frames...")
        input_frames = self._load_frames_for_videomama(clip.input_asset, clip.name, job=job, on_status=on_status)
        _check_cancel("frame load")

        _status("Loading masks...")
        mask_stems: dict[str, np.ndarray] = {}
        if clip.mask_asset.asset_type == "sequence":
            for fname in clip.mask_asset.get_frame_files():
                _check_cancel("mask load")
                fpath = os.path.join(clip.mask_asset.path, fname)
                m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    _, binary = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
                    mask_stems[os.path.splitext(fname)[0]] = binary
        else:
            for i, m in enumerate(self._load_mask_frames_for_videomama(clip.mask_asset, clip.name)):
                mask_stems[f"frame_{i:06d}"] = m

        input_names = (
            clip.input_asset.get_frame_files()
            if clip.input_asset.asset_type == "sequence"
            else [f"frame_{i:06d}.png" for i in range(len(input_frames))]
        )
        num_frames = len(input_frames)
        mask_frames = []
        for fname in input_names:
            stem = os.path.splitext(fname)[0]
            if stem in mask_stems:
                mask_frames.append(mask_stems[stem])
            else:
                h_m, w_m = input_frames[0].shape[:2] if input_frames else (4, 4)
                mask_frames.append(np.zeros((h_m, w_m), dtype=np.uint8))

        # Resume logic
        existing_alpha = (
            [f for f in os.listdir(alpha_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if os.path.isdir(alpha_dir)
            else []
        )
        n_existing = len(existing_alpha)
        completed_chunks = n_existing // chunk_size
        start_chunk = max(0, completed_chunks - 1)
        start_frame = start_chunk * chunk_size
        if start_frame > 0:
            keep = {f"{os.path.splitext(input_names[i])[0]}.png" for i in range(start_frame) if i < len(input_names)}
            for fname in existing_alpha:
                if fname not in keep:
                    os.remove(os.path.join(alpha_dir, fname))
            logger.info(
                "VideoMaMa resuming for '%s': rolling back to chunk %d (frame %d)", clip.name, start_chunk, start_frame
            )

        sys.path.insert(0, os.path.join(BASE_DIR, "VideoMaMaInferenceModule"))
        from VideoMaMaInferenceModule.inference import run_inference  # type: ignore[import-not-found]

        total_chunks = (num_frames + chunk_size - 1) // chunk_size
        _status(f"Running inference (chunk 1/{total_chunks})...")
        frames_written = start_frame
        for chunk_idx, chunk_output in enumerate(
            run_inference(pipeline, input_frames, mask_frames, chunk_size=chunk_size)
        ):
            _check_cancel("inference")
            if chunk_idx < start_chunk:
                frames_written += len(chunk_output)
                if on_progress:
                    on_progress(clip.name, frames_written, num_frames)
                continue
            _status(f"Processing chunk {chunk_idx + 1}/{total_chunks}...")
            t_chunk = time.monotonic()
            for frame in chunk_output:
                out_bgr = cv2.cvtColor((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                out_name = (
                    f"{os.path.splitext(input_names[frames_written])[0]}.png"
                    if frames_written < len(input_names)
                    else f"frame_{frames_written:06d}.png"
                )
                cv2.imwrite(os.path.join(alpha_dir, out_name), out_bgr)
                frames_written += 1
            logger.debug(
                "Clip '%s' chunk %d: %d frames in %.3fs",
                clip.name,
                chunk_idx,
                len(chunk_output),
                time.monotonic() - t_chunk,
            )
            if on_progress:
                on_progress(clip.name, frames_written, num_frames)

        clip.alpha_asset = ClipAsset(alpha_dir, "sequence")
        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after VideoMaMa: {e}")

        logger.info(
            "VideoMaMa complete for '%s': %d alpha frames in %.1fs",
            clip.name,
            frames_written,
            time.monotonic() - t_start,
        )

    def _load_frames_for_videomama(
        self,
        asset: ClipAsset,
        clip_name: str,
        job: GPUJob | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> list[np.ndarray]:
        """Load input frames for VideoMaMa as uint8 RGB [0, 255]."""
        if asset.asset_type == "video":
            raw = read_video_frames(asset.path)
            return [(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8) for f in raw]
        frames = []
        files = asset.get_frame_files()
        total = len(files)
        for i, fname in enumerate(files):
            if job and job.is_cancelled:
                raise JobCancelledError(clip_name, i)
            img = read_image_frame(os.path.join(asset.path, fname), gamma_correct_exr=True)
            if img is not None:
                frames.append((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
            if on_status and i % 20 == 0 and i > 0:
                on_status(f"Loading frames ({i}/{total})...")
        return frames

    def _load_mask_frames_for_videomama(self, asset: ClipAsset, clip_name: str) -> list[np.ndarray]:
        """Load mask frames for VideoMaMa as uint8 grayscale [0, 255].

        Binary threshold at 10: anything above -> 255 (foreground), else -> 0.
        """

        def _threshold_mask(bgr_frame: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            return binary

        if asset.asset_type == "video":
            return read_video_frames(asset.path, processor=_threshold_mask)
        masks = []
        for fname in asset.get_frame_files():
            mask = cv2.imread(os.path.join(asset.path, fname), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            masks.append(binary)
        return masks
