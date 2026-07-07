"""Background GPU worker thread for the web UI.

Pulls jobs from GPUJobQueue and dispatches to CorridorKeyService methods.
Progress is broadcast to WebSocket clients via an async callback.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from backend.errors import JobCancelledError
from backend.ffmpeg_tools import extract_frames
from backend.job_queue import GPUJob, JobType
from backend.service import CorridorKeyService, InferenceParams, OutputConfig

logger = logging.getLogger(__name__)


class GPUWorker:
    """Background thread that consumes GPU jobs one at a time."""

    def __init__(
        self,
        service: CorridorKeyService,
        broadcast,
        loop: asyncio.AbstractEventLoop,
    ):
        self.service = service
        self._broadcast = broadcast
        self._loop = loop
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clips: dict[str, object] = {}  # name -> ClipEntry
        self._rescan_callback = None  # set by app to trigger rescan after extraction

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-worker")
        self._thread.start()
        logger.info("GPU worker started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("GPU worker stopped")

    def set_clips(self, clips: dict[str, object]) -> None:
        self._clips = clips

    def _broadcast_sync(self, data: dict) -> None:
        """Schedule an async broadcast from the worker thread."""
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)
        except RuntimeError:
            pass  # event loop closed

    def _run(self) -> None:
        queue = self.service.job_queue

        while not self._stop.is_set():
            job = queue.next_job()
            if job is None:
                self._stop.wait(0.5)
                continue

            queue.start_job(job)
            clip = self._clips.get(job.clip_name)
            if clip is None:
                queue.fail_job(job, f"Clip '{job.clip_name}' not found")
                self._broadcast_sync({
                    "type": "job_failed",
                    "job_id": job.id,
                    "clip_name": job.clip_name,
                    "error": f"Clip '{job.clip_name}' not found",
                })
                continue

            self._broadcast_sync({
                "type": "job_started",
                "job_id": job.id,
                "clip_name": job.clip_name,
                "job_type": job.job_type.value,
            })

            try:
                self._dispatch(job, clip)
                queue.complete_job(job)
                self._broadcast_sync({
                    "type": "job_completed",
                    "job_id": job.id,
                    "clip_name": job.clip_name,
                })
                # After extraction, rescan so the clip transitions from EXTRACTING → RAW
                if job.job_type == JobType.VIDEO_EXTRACT and self._rescan_callback:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._rescan_callback(), self._loop
                        )
                    except RuntimeError:
                        pass
                self._broadcast_sync({"type": "clips_updated"})
            except JobCancelledError:
                queue.mark_cancelled(job)
                self._broadcast_sync({
                    "type": "job_cancelled",
                    "job_id": job.id,
                    "clip_name": job.clip_name,
                })
            except Exception as e:
                queue.fail_job(job, str(e))
                logger.exception(f"Job [{job.id}] failed")
                self._broadcast_sync({
                    "type": "job_failed",
                    "job_id": job.id,
                    "clip_name": job.clip_name,
                    "error": str(e),
                })

    def _dispatch(self, job: GPUJob, clip) -> None:
        """Route a job to the appropriate service method."""

        def on_progress(clip_name: str, current: int, total: int) -> None:
            job.current_frame = current
            job.total_frames = total
            self._broadcast_sync({
                "type": "progress",
                "job_id": job.id,
                "clip_name": clip_name,
                "current": current,
                "total": total,
            })

        def on_warning(msg: str) -> None:
            self._broadcast_sync({
                "type": "warning",
                "job_id": job.id,
                "clip_name": job.clip_name,
                "message": msg,
            })

        def on_status(msg: str) -> None:
            self._broadcast_sync({
                "type": "status",
                "job_id": job.id,
                "clip_name": job.clip_name,
                "message": msg,
            })

        if job.job_type == JobType.INFERENCE:
            params = InferenceParams.from_dict(job.params)
            output_config = None
            if "output_config" in job.params:
                output_config = OutputConfig.from_dict(job.params["output_config"])
            skip_stems = clip.completed_stems() if hasattr(clip, "completed_stems") else None
            self.service.run_inference(
                clip,
                params,
                job=job,
                on_progress=on_progress,
                on_warning=on_warning,
                skip_stems=skip_stems,
                output_config=output_config,
            )

        elif job.job_type == JobType.GVM_ALPHA:
            self.service.run_gvm(
                clip,
                job=job,
                on_progress=on_progress,
                on_warning=on_warning,
            )

        elif job.job_type == JobType.VIDEOMAMA_ALPHA:
            chunk_size = job.params.get("chunk_size", 50)
            self.service.run_videomama(
                clip,
                job=job,
                on_progress=on_progress,
                on_warning=on_warning,
                on_status=on_status,
                chunk_size=chunk_size,
            )

        elif job.job_type == JobType.VIDEO_EXTRACT:
            video_path = job.params["video_path"]
            out_dir = job.params["out_dir"]

            def on_extract_progress(current: int, total: int) -> None:
                job.current_frame = current
                job.total_frames = total
                self._broadcast_sync({
                    "type": "progress",
                    "job_id": job.id,
                    "clip_name": job.clip_name,
                    "current": current,
                    "total": total,
                })

            cancel_event = threading.Event()
            # Store cancel_event on job so it can be signalled
            job._cancel_event = cancel_event
            extract_frames(
                video_path,
                out_dir,
                on_progress=on_extract_progress,
                cancel_event=cancel_event,
            )

        elif job.job_type == JobType.PREVIEW_REPROCESS:
            params = InferenceParams.from_dict(job.params)
            frame_index = job.params.get("frame_index", 0)
            self.service.reprocess_single_frame(
                clip,
                params,
                frame_index,
                job=job,
            )
