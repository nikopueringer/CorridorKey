"""High-level pipeline operations for CLI and batch processing.

Wraps CorridorKeyService into simple top-level functions that process
entire directories without requiring callers to manage ClipEntry state.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from corridorkey.clip_state import ClipEntry, ClipState
from corridorkey.config import CorridorKeyConfig
from corridorkey.errors import CorridorKeyError, JobCancelledError
from corridorkey.protocols import AlphaGenerator
from corridorkey.service import CorridorKeyService, InferenceParams, OutputConfig

logger = logging.getLogger(__name__)


@dataclass
class ClipSummary:
    """Processing result for a single clip."""

    name: str
    state: str
    frames_processed: int = 0
    frames_total: int = 0
    error: str | None = None
    skipped: bool = False


@dataclass
class PipelineResult:
    """Aggregate result for a full pipeline run."""

    clips: list[ClipSummary] = field(default_factory=list)

    @property
    def succeeded(self) -> list[ClipSummary]:
        return [c for c in self.clips if c.error is None and not c.skipped]

    @property
    def failed(self) -> list[ClipSummary]:
        return [c for c in self.clips if c.error is not None]

    @property
    def skipped(self) -> list[ClipSummary]:
        return [c for c in self.clips if c.skipped]


def process_directory(
    clips_dir: str,
    params: InferenceParams | None = None,
    output_config: OutputConfig | None = None,
    alpha_generator: AlphaGenerator | None = None,
    config: CorridorKeyConfig | None = None,
    device: str | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_warning: Callable[[str], None] | None = None,
    on_clip_start: Callable[[str, str], None] | None = None,
    on_clip_done: Callable[[ClipSummary], None] | None = None,
) -> PipelineResult:
    """Scan a directory and process all eligible clips end-to-end.

    Pipeline per clip:
      RAW + alpha_generator provided  -> generate alpha -> run inference
      RAW + no alpha_generator        -> skip (logged as warning)
      MASKED + alpha_generator        -> generate alpha -> run inference
      READY                           -> run inference directly
      COMPLETE                        -> skip (already done)
      ERROR                           -> skip (logged as warning)

    Args:
        clips_dir: Path to scan for clips.
        params: Inference parameters. Defaults to InferenceParams().
        output_config: Output format config. Defaults to OutputConfig().
        alpha_generator: Optional AlphaGenerator for RAW/MASKED clips.
        config: CorridorKeyConfig instance. Loaded from disk if None.
        device: Compute device override ("cuda", "mps", "cpu", "auto").
            Overrides config.device when provided.
        on_progress: Called with (clip_name, current_frame, total_frames).
        on_warning: Called with non-fatal warning messages.
        on_clip_start: Called with (clip_name, state) before processing each clip.
        on_clip_done: Called with ClipSummary after each clip completes.

    Returns:
        PipelineResult with per-clip summaries.
    """
    service = CorridorKeyService(config)
    service.detect_device(device)

    params = params or service.default_inference_params()
    output_config = output_config or service.default_output_config()
    result = PipelineResult()

    try:
        clips = service.scan_clips(clips_dir)
        if not clips:
            logger.warning("No clips found in: %s", clips_dir)
            return result

        logger.info("Found %d clip(s) in %s", len(clips), clips_dir)

        for clip in clips:
            summary = _process_clip(
                clip=clip,
                service=service,
                params=params,
                output_config=output_config,
                alpha_generator=alpha_generator,
                on_progress=on_progress,
                on_warning=on_warning,
                on_clip_start=on_clip_start,
            )
            result.clips.append(summary)
            if on_clip_done:
                on_clip_done(summary)
    finally:
        service.unload_engine()

    logger.info(
        "Pipeline complete: %d succeeded, %d failed, %d skipped",
        len(result.succeeded),
        len(result.failed),
        len(result.skipped),
    )
    return result


def _process_clip(
    clip: ClipEntry,
    service: CorridorKeyService,
    params: InferenceParams,
    output_config: OutputConfig,
    alpha_generator: AlphaGenerator | None,
    on_progress: Callable[[str, int, int], None] | None,
    on_warning: Callable[[str], None] | None,
    on_clip_start: Callable[[str, str], None] | None,
) -> ClipSummary:
    """Process a single clip through the pipeline. Returns a ClipSummary."""
    state_name = clip.state.value

    if on_clip_start:
        on_clip_start(clip.name, state_name)

    # COMPLETE - nothing to do
    if clip.state == ClipState.COMPLETE:
        logger.info("Clip '%s': already COMPLETE, skipping", clip.name)
        return ClipSummary(name=clip.name, state=state_name, skipped=True)

    # ERROR - can't process without user intervention
    if clip.state == ClipState.ERROR:
        msg = f"Clip '{clip.name}' is in ERROR state: {clip.error_message}"
        logger.warning(msg)
        if on_warning:
            on_warning(msg)
        return ClipSummary(name=clip.name, state=state_name, skipped=True)

    # RAW/MASKED - need alpha generation first
    if clip.state in (ClipState.RAW, ClipState.MASKED):
        if alpha_generator is None:
            msg = f"Clip '{clip.name}' is {state_name} but no alpha generator provided - skipping"
            logger.warning(msg)
            if on_warning:
                on_warning(msg)
            return ClipSummary(name=clip.name, state=state_name, skipped=True)

        try:
            service.run_alpha_generator(clip, alpha_generator, on_progress=on_progress, on_warning=on_warning)
        except JobCancelledError:
            return ClipSummary(name=clip.name, state=state_name, skipped=True)
        except CorridorKeyError as e:
            clip.set_error(str(e))
            return ClipSummary(name=clip.name, state=state_name, error=str(e))

    # EXTRACTING - can't process yet
    if clip.state == ClipState.EXTRACTING:
        msg = f"Clip '{clip.name}' is still EXTRACTING - skipping"
        logger.warning(msg)
        if on_warning:
            on_warning(msg)
        return ClipSummary(name=clip.name, state=state_name, skipped=True)

    # READY - run inference
    try:
        frame_results = service.run_inference(
            clip,
            params,
            on_progress=on_progress,
            on_warning=on_warning,
            output_config=output_config,
        )
        processed = sum(1 for r in frame_results if r.success)
        return ClipSummary(
            name=clip.name,
            state=clip.state.value,
            frames_processed=processed,
            frames_total=len(frame_results),
        )
    except JobCancelledError:
        return ClipSummary(name=clip.name, state=state_name, skipped=True)
    except CorridorKeyError as e:
        clip.set_error(str(e))
        return ClipSummary(name=clip.name, state=state_name, error=str(e))
