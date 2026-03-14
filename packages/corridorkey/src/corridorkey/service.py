"""CorridorKeyService - backend API for CLI and GUI consumers.

This module is the single entry point for all processing. Consumers never
call inference engines directly - they call methods here which handle
validation, state transitions, and error reporting.

Model Residency Policy:
    Only ONE engine is loaded at a time. Unload before loading a new one
    to free VRAM via device_utils.clear_device_cache().
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Enable OpenEXR support (must be before cv2 import)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from corridorkey_core.engine_factory import create_engine

from corridorkey import device_utils
from corridorkey.clip_state import ClipEntry, ClipState, scan_clips_dir
from corridorkey.config import CorridorKeyConfig, load_config
from corridorkey.errors import CorridorKeyError, FrameReadError, JobCancelledError, WriteFailureError
from corridorkey.frame_io import (
    EXR_WRITE_FLAGS,
    read_image_frame,
    read_mask_frame,
    read_video_frame_at,
    read_video_mask_at,
)
from corridorkey.job_queue import GPUJob, GPUJobQueue
from corridorkey.protocols import AlphaGenerator
from corridorkey.validators import ensure_output_dirs, validate_frame_counts, validate_frame_read, validate_write

logger = logging.getLogger(__name__)


@dataclass
class InferenceParams:
    """Frozen parameters for a single inference job.

    Attributes:
        input_is_linear: True if the input frames are in linear light (e.g. EXR).
        despill_strength: Strength of the green-spill suppression (0.0-1.0).
        auto_despeckle: Enable automatic matte despeckling.
        despeckle_size: Maximum speckle area in pixels to remove.
        refiner_scale: Scale factor passed to the optional refiner stage.
    """

    input_is_linear: bool = False
    despill_strength: float = 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
    refiner_scale: float = 1.0

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for manifest storage).

        Returns:
            Dict representation of all fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> InferenceParams:
        """Deserialise from a plain dict, ignoring unknown keys.

        Args:
            d: Dict that may contain extra keys from older manifest versions.

        Returns:
            InferenceParams instance with known fields populated.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class OutputConfig:
    """Which output types to produce and their format.

    Attributes:
        fg_enabled: Write foreground (RGBA) frames.
        fg_format: File format for FG frames ("exr" or "png").
        matte_enabled: Write alpha matte frames.
        matte_format: File format for matte frames ("exr" or "png").
        comp_enabled: Write composited preview frames.
        comp_format: File format for comp frames ("exr" or "png").
        processed_enabled: Write pre-processed input frames.
        processed_format: File format for processed frames ("exr" or "png").
    """

    fg_enabled: bool = True
    fg_format: str = "exr"  # "exr" or "png"
    matte_enabled: bool = True
    matte_format: str = "exr"
    comp_enabled: bool = True
    comp_format: str = "png"
    processed_enabled: bool = True
    processed_format: str = "exr"

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for manifest storage).

        Returns:
            Dict representation of all fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OutputConfig:
        """Deserialise from a plain dict, ignoring unknown keys.

        Args:
            d: Dict that may contain extra keys from older manifest versions.

        Returns:
            OutputConfig instance with known fields populated.
        """
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
    """Result summary for a single processed frame (no numpy arrays).

    Attributes:
        frame_index: Zero-based index of the frame within the clip.
        input_stem: Filename stem of the input frame (e.g. "frame_000001").
        success: True if the frame was processed and written successfully.
        warning: Non-fatal message if the frame was skipped or had issues.
    """

    frame_index: int
    input_stem: str
    success: bool
    warning: str | None = None


class CorridorKeyService:
    """Main backend service - scan, validate, process, write.

    Holds the loaded inference engine and GPU lock. Everything else
    (project management, frame I/O, validation) is handled by pure
    functions in their respective modules.

    Usage (CLI):
        config = load_config()
        service = CorridorKeyService(config)
        clips = service.scan_clips("/path/to/clips")
        for clip in service.get_clips_by_state(clips, ClipState.READY):
            service.run_inference(clip, InferenceParams())

    Usage (GUI):
        config = load_config(overrides={"device": "cuda"})
        service = CorridorKeyService(config)
        queue = service.job_queue  # GPUJobQueue for async job management
    """

    def __init__(self, config: CorridorKeyConfig | None = None) -> None:
        self._config = config or load_config()
        self._engine = None
        self._engine_loaded = False
        self._device: str = self._config.device
        self._job_queue: GPUJobQueue | None = None
        self._gpu_lock = threading.Lock()

    def default_inference_params(self) -> InferenceParams:
        """Build InferenceParams seeded from the loaded config.

        Callers that want config-driven defaults should use this rather
        than constructing InferenceParams() directly.

        Returns:
            InferenceParams with values from CorridorKeyConfig.
        """
        return InferenceParams(
            input_is_linear=self._config.input_is_linear,
            despill_strength=self._config.despill_strength,
            auto_despeckle=self._config.auto_despeckle,
            despeckle_size=self._config.despeckle_size,
            refiner_scale=self._config.refiner_scale,
        )

    def default_output_config(self) -> OutputConfig:
        """Build OutputConfig seeded from the loaded config.

        Returns:
            OutputConfig with format values from CorridorKeyConfig.
        """
        return OutputConfig(
            fg_format=self._config.fg_format,
            matte_format=self._config.matte_format,
            comp_format=self._config.comp_format,
            processed_format=self._config.processed_format,
        )

    @property
    def job_queue(self) -> GPUJobQueue:
        """Lazy-init GPU job queue."""
        if self._job_queue is None:
            self._job_queue = GPUJobQueue()
        return self._job_queue

    def detect_device(self, requested: str | None = None) -> str:
        """Resolve and store the compute device.

        Args:
            requested: Explicit device string ("cuda", "mps", "cpu", or "auto").
                None uses the device from config, falling back to auto-detection.

        Returns:
            Resolved device string stored on this service instance.
        """
        self._device = device_utils.resolve_device(requested or self._config.device)
        logger.info("Compute device: %s", self._device)
        return self._device

    def get_vram_info(self) -> dict[str, float | str]:
        """Return GPU VRAM info in GB.

        Returns:
            Dict with keys "total", "reserved", "allocated", "free" (all GB),
            and "name" (device name string). Empty dict if CUDA is unavailable.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            props = torch.cuda.get_device_properties(0)
            total = props.total_mem
            reserved = torch.cuda.memory_reserved(0)
            return {
                "total": total / (1024**3),
                "reserved": reserved / (1024**3),
                "allocated": torch.cuda.memory_allocated(0) / (1024**3),
                "free": (total - reserved) / (1024**3),
                "name": torch.cuda.get_device_name(0),
            }
        except Exception as e:
            logger.debug("VRAM query failed: %s", e)
            return {}

    @staticmethod
    def _vram_allocated_mb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024**2)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _safe_offload(obj: Any) -> None:
        """Move model tensors to CPU before dropping the reference."""
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

    def _get_engine(self):
        """Lazy-load the inference engine."""
        if self._engine is not None:
            return self._engine
        logger.info("Loading inference engine (device=%s)...", self._device)
        t0 = time.monotonic()
        self._engine = create_engine(
            checkpoint_dir=self._config.checkpoint_dir,
            device=self._device,
        )
        self._engine_loaded = True
        logger.info("Engine loaded in %.1fs", time.monotonic() - t0)
        return self._engine

    def unload_engine(self) -> None:
        """Free GPU memory by unloading the inference engine."""
        vram_before = self._vram_allocated_mb()
        self._safe_offload(self._engine)
        self._engine = None
        self._engine_loaded = False
        import gc

        gc.collect()
        device_utils.clear_device_cache(self._device)
        logger.info(
            "Engine unloaded. VRAM before: %.0fMB, after: %.0fMB",
            vram_before,
            self._vram_allocated_mb(),
        )

    def is_engine_loaded(self) -> bool:
        """True if the inference engine is loaded in VRAM."""
        return self._engine_loaded and self._engine is not None

    def run_alpha_generator(
        self,
        clip: ClipEntry,
        generator: AlphaGenerator,
        job: GPUJob | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        """Run any AlphaGenerator implementation against a clip.

        The generator is responsible for writing frames to AlphaHint/
        and transitioning the clip to READY state.

        Args:
            clip: Clip in RAW or MASKED state.
            generator: Any object implementing the AlphaGenerator protocol.
            job: Optional GPUJob for cancel checking.
            on_progress: Called with (clip_name, current, total).
            on_warning: Called with non-fatal warning messages.
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' has no input asset")

        logger.info("Running alpha generator '%s' for clip '%s'", generator.name, clip.name)

        def _progress(clip_name: str, current: int, total: int) -> None:
            if job and job.is_cancelled:
                raise JobCancelledError(clip_name, current)
            if on_progress:
                on_progress(clip_name, current, total)

        try:
            generator.generate(clip, on_progress=_progress, on_warning=on_warning)
        except JobCancelledError:
            raise
        except Exception as e:
            if job and job.is_cancelled:
                raise JobCancelledError(clip.name, 0) from None
            raise CorridorKeyError(f"Alpha generator '{generator.name}' failed for '{clip.name}': {e}") from e

    def scan_clips(self, clips_dir: str, allow_standalone_videos: bool = True) -> list[ClipEntry]:
        """Scan a directory for clip folders.

        Args:
            clips_dir: Path to the directory to scan.
            allow_standalone_videos: When False, loose video files at the top
                level are ignored.

        Returns:
            List of ClipEntry objects found in the directory.
        """
        return scan_clips_dir(clips_dir, allow_standalone_videos=allow_standalone_videos)

    def get_clips_by_state(self, clips: list[ClipEntry], state: ClipState) -> list[ClipEntry]:
        """Filter clips by state.

        Args:
            clips: Full list of ClipEntry objects to filter.
            state: Target ClipState to match.

        Returns:
            Subset of clips whose state equals the requested state.
        """
        return [c for c in clips if c.state == state]

    def _read_input_frame(
        self,
        clip: ClipEntry,
        frame_index: int,
        input_files: list[str],
        input_cap: Any | None,
        input_is_linear: bool,
    ) -> tuple[np.ndarray | None, str, bool]:
        """Read one input frame from either a video capture or an image sequence.

        Args:
            clip: Clip being processed (used for logging).
            frame_index: Zero-based frame index.
            input_files: Sorted list of image filenames (empty for video assets).
            input_cap: OpenCV VideoCapture for video assets, None for sequences.
            input_is_linear: Whether the source is in linear light.

        Returns:
            Tuple of (image array or None, stem string, is_linear flag).
        """
        input_stem = f"{frame_index:05d}"
        if input_cap:
            ret, frame = input_cap.read()
            if not ret:
                return None, input_stem, False
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return img_rgb.astype(np.float32) / 255.0, input_stem, input_is_linear
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
        """Read one alpha hint frame from either a video capture or an image sequence.

        Args:
            clip: Clip being processed (used for path resolution).
            frame_index: Zero-based frame index.
            alpha_files: Sorted list of alpha image filenames (empty for video assets).
            alpha_cap: OpenCV VideoCapture for video alpha assets, None for sequences.

        Returns:
            Float32 alpha array in [0, 1], or None if the read failed.
        """
        if alpha_cap:
            ret, frame = alpha_cap.read()
            if not ret:
                return None
            return frame[:, :, 2].astype(np.float32) / 255.0
        fpath = os.path.join(clip.alpha_asset.path, alpha_files[frame_index])  # type: ignore[union-attr]
        mask = read_mask_frame(fpath, clip.name, frame_index)
        validate_frame_read(mask, clip.name, frame_index, fpath)
        return mask

    def _write_image(self, img: np.ndarray, path: str, fmt: str, clip_name: str, frame_index: int) -> None:
        """Write a single image to disk in the requested format.

        Handles dtype conversion: EXR expects float32, PNG expects uint8.

        Args:
            img: Image array to write.
            path: Absolute destination path including filename and extension.
            fmt: Target format string ("exr" or "png").
            clip_name: Clip name used in error messages.
            frame_index: Frame index used in error messages.
        """
        if fmt == "exr":
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
            validate_write(cv2.imwrite(path, img, EXR_WRITE_FLAGS), clip_name, frame_index, path)
        else:
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            validate_write(cv2.imwrite(path, img), clip_name, frame_index, path)

    def _write_manifest(self, output_root: str, output_config: OutputConfig, params: InferenceParams) -> None:
        """Write a JSON run manifest to the output directory.

        Uses atomic write (tmp file + os.replace) to avoid partial reads.
        Failures are logged as warnings and do not abort processing.

        Args:
            output_root: Absolute path to the Output directory.
            output_config: Output configuration to record.
            params: Inference parameters to record.
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
        cfg: OutputConfig,
    ) -> None:
        """Write all enabled output images for a single processed frame.

        Args:
            res: Engine result dict with keys "fg", "alpha", "comp", "processed".
            dirs: Output subdirectory paths keyed by output name.
            input_stem: Filename stem used for output filenames.
            clip_name: Clip name used in error messages.
            frame_index: Frame index used in error messages.
            cfg: Output configuration controlling which outputs to write.
        """
        if cfg.fg_enabled:
            fg_bgr = cv2.cvtColor(res["fg"], cv2.COLOR_RGB2BGR)
            self._write_image(
                fg_bgr, os.path.join(dirs["fg"], f"{input_stem}.{cfg.fg_format}"), cfg.fg_format, clip_name, frame_index
            )
        if cfg.matte_enabled:
            alpha = res["alpha"]
            alpha = alpha[:, :, 0] if alpha.ndim == 3 else alpha
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
        """Run inference on a single clip.

        Args:
            clip: Must be READY or COMPLETE with input_asset and alpha_asset.
            params: Inference parameters.
            job: Optional GPUJob for cancel checking.
            on_progress: Called with (clip_name, current, total).
            on_warning: Called with non-fatal warning messages.
            skip_stems: Frame stems to skip (resume support).
            output_config: Which outputs to write and their formats.
            frame_range: Optional (start, end) inclusive frame indices.

        Returns:
            List of FrameResult per frame.

        Raises:
            JobCancelledError, CorridorKeyError subclasses.
        """
        if clip.input_asset is None or clip.alpha_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input or alpha asset")

        t_start = time.monotonic()
        with self._gpu_lock:
            engine = self._get_engine()

        cfg = output_config or OutputConfig()
        dirs = ensure_output_dirs(clip.root_path)
        self._write_manifest(dirs["root"], cfg, params)

        num_frames = validate_frame_counts(clip.name, clip.input_asset.frame_count, clip.alpha_asset.frame_count)

        input_cap = alpha_cap = None
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
                    logger.debug("Clip '%s' frame %d: %.3fs", clip.name, i, time.monotonic() - t_frame)
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

    def reprocess_single_frame(
        self,
        clip: ClipEntry,
        params: InferenceParams,
        frame_index: int,
        job: GPUJob | None = None,
    ) -> dict | None:
        """Reprocess a single frame in memory. Does not write to disk.

        Used for live preview in the GUI.

        Args:
            clip: Clip with valid input_asset and alpha_asset.
            params: Inference parameters to apply.
            frame_index: Zero-based index of the frame to reprocess.
            job: Optional GPUJob for cancel checking.

        Returns:
            Engine result dict with keys "fg", "alpha", "comp", "processed",
            or None if the frame could not be read or the job was cancelled.
        """
        if clip.input_asset is None or clip.alpha_asset is None:
            return None
        if job and job.is_cancelled:
            return None

        t_start = time.monotonic()
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
        logger.debug("Clip '%s' frame %d: reprocess %.3fs", clip.name, frame_index, time.monotonic() - t_start)
        return res
