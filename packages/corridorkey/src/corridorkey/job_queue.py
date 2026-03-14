"""GPU job queue with mutual exclusion.

Ensures only ONE GPU job runs at a time across all job types
(inference, GVM alpha gen, VideoMaMa alpha gen). This prevents VRAM
contention - CorridorKey alone needs ~22.7GB of 24GB.

Design:
    - Thread-safe queue of GPUJob dataclasses
    - Single consumer loop (designed to be driven by a QThread in the UI,
      or called directly in CLI mode)
    - Jobs carry a cancel flag checked between frames
    - Callbacks for progress, warnings, completion, errors
    - Jobs have stable IDs assigned at creation time
    - Deduplication prevents double-submit of same clip+job_type
    - Job history preserved for UI display (cancelled/completed/failed)
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import JobCancelledError

logger = logging.getLogger(__name__)


class JobType(Enum):
    INFERENCE = "inference"
    GVM_ALPHA = "gvm_alpha"
    VIDEOMAMA_ALPHA = "videomama_alpha"
    PREVIEW_REPROCESS = "preview_reprocess"
    VIDEO_EXTRACT = "video_extract"
    VIDEO_STITCH = "video_stitch"


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class GPUJob:
    """A single GPU job to be executed."""

    job_type: JobType
    clip_name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    params: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    _cancel_requested: bool = field(default=False, repr=False)
    error_message: str | None = None

    # Progress tracking
    current_frame: int = 0
    total_frames: int = 0

    def request_cancel(self) -> None:
        """Signal that this job should stop at the next frame boundary."""
        self._cancel_requested = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def check_cancelled(self) -> None:
        """Raise JobCancelledError if cancel was requested. Call between frames."""
        if self._cancel_requested:
            raise JobCancelledError(self.clip_name, self.current_frame)


# Callback type aliases
ProgressCallback = Callable[[str, int, int], None]  # clip_name, current, total
WarningCallback = Callable[[str], None]  # message
CompletionCallback = Callable[[str], None]  # clip_name
ErrorCallback = Callable[[str, str], None]  # clip_name, error_message


class GPUJobQueue:
    """Thread-safe GPU job queue with mutual exclusion.

    Usage (CLI mode):
        queue = GPUJobQueue()
        queue.submit(GPUJob(JobType.INFERENCE, "shot1", params={...}))
        queue.submit(GPUJob(JobType.GVM_ALPHA, "shot2", params={...}))

        # Process all jobs sequentially
        while queue.has_pending:
            job = queue.next_job()
            if job:
                queue.start_job(job)
                try:
                    run_the_job(job)  # your processing function
                    queue.complete_job(job)
                except Exception as e:
                    queue.fail_job(job, str(e))

    Usage (GUI mode):
        The GPU worker QThread calls next_job() / start_job() / complete_job()
        in its run loop. The UI submits jobs from the main thread.
    """

    def __init__(self):
        self._queue: deque[GPUJob] = deque()
        self._lock = threading.Lock()
        self._current_job: GPUJob | None = None
        self._history: list[GPUJob] = []  # completed/cancelled/failed jobs for UI display

        # Callbacks (set by UI or CLI)
        self.on_progress: ProgressCallback | None = None
        self.on_warning: WarningCallback | None = None
        self.on_completion: CompletionCallback | None = None
        self.on_error: ErrorCallback | None = None

    def submit(self, job: GPUJob) -> bool:
        """Add a job to the queue. Returns False if duplicate detected.

        PREVIEW_REPROCESS uses replacement semantics - any existing preview
        reprocess in the queue is replaced by the new one (latest-only).
        """
        with self._lock:
            # PREVIEW_REPROCESS: replace existing queued preview jobs (latest-only)
            if job.job_type == JobType.PREVIEW_REPROCESS:
                replaced = [j for j in self._queue if j.job_type == JobType.PREVIEW_REPROCESS]
                for old in replaced:
                    self._queue.remove(old)
                    old.status = JobStatus.CANCELLED
                    logger.debug(f"Preview reprocess [{old.id}] replaced by [{job.id}]")
            else:
                # Deduplication: reject if same clip+job_type already queued or running
                for existing in self._queue:
                    if existing.clip_name == job.clip_name and existing.job_type == job.job_type:
                        logger.warning(
                            f"Duplicate job rejected: {job.job_type.value} for '{job.clip_name}' "
                            f"(already queued as {existing.id})"
                        )
                        return False
                if (
                    self._current_job
                    and self._current_job.clip_name == job.clip_name
                    and self._current_job.job_type == job.job_type
                    and self._current_job.status == JobStatus.RUNNING
                ):
                    logger.warning(
                        f"Duplicate job rejected: {job.job_type.value} for '{job.clip_name}' "
                        f"(already running as {self._current_job.id})"
                    )
                    return False

            job.status = JobStatus.QUEUED
            self._queue.append(job)
            logger.info(f"Job queued [{job.id}]: {job.job_type.value} for '{job.clip_name}'")
            return True

    def next_job(self) -> GPUJob | None:
        """Get the next pending job without starting it. Returns None if empty."""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None

    def start_job(self, job: GPUJob) -> None:
        """Mark a job as running. Must be called before processing."""
        with self._lock:
            if job in self._queue:
                self._queue.remove(job)
            job.status = JobStatus.RUNNING
            self._current_job = job
            logger.info(f"Job started [{job.id}]: {job.job_type.value} for '{job.clip_name}'")

    def complete_job(self, job: GPUJob) -> None:
        """Mark a job as successfully completed."""
        with self._lock:
            job.status = JobStatus.COMPLETED
            if self._current_job is job:
                self._current_job = None
            self._history.append(job)
            logger.info(f"Job completed [{job.id}]: {job.job_type.value} for '{job.clip_name}'")
        # Emit AFTER lock release (Codex: no deadlock risk)
        if self.on_completion:
            self.on_completion(job.clip_name)

    def fail_job(self, job: GPUJob, error: str) -> None:
        """Mark a job as failed."""
        with self._lock:
            job.status = JobStatus.FAILED
            job.error_message = error
            if self._current_job is job:
                self._current_job = None
            self._history.append(job)
            logger.error(f"Job failed [{job.id}]: {job.job_type.value} for '{job.clip_name}': {error}")
        # Emit AFTER lock release
        if self.on_error:
            self.on_error(job.clip_name, error)

    def mark_cancelled(self, job: GPUJob) -> None:
        """Mark a running job as cancelled AND clear _current_job.

        This is the cancel-safe path that was missing - calling
        job.request_cancel() alone doesn't clear _current_job, which
        poisons queue state for subsequent jobs.
        """
        with self._lock:
            job.status = JobStatus.CANCELLED
            if self._current_job is job:
                self._current_job = None
            self._history.append(job)
            logger.info(f"Job cancelled [{job.id}]: {job.job_type.value} for '{job.clip_name}'")

    def cancel_job(self, job: GPUJob) -> None:
        """Request cancellation of a specific job."""
        with self._lock:
            if job.status == JobStatus.QUEUED:
                if job in self._queue:
                    self._queue.remove(job)
                job.status = JobStatus.CANCELLED
                self._history.append(job)
                logger.info(f"Job removed from queue [{job.id}]: {job.job_type.value} for '{job.clip_name}'")
            elif job.status == JobStatus.RUNNING:
                # Signal cancel - worker calls mark_cancelled() after catching JobCancelledError
                job.request_cancel()
                logger.info(f"Job cancel requested [{job.id}]: {job.job_type.value} for '{job.clip_name}'")

    def cancel_current(self) -> None:
        """Cancel the currently running job, if any."""
        with self._lock:
            if self._current_job and self._current_job.status == JobStatus.RUNNING:
                self._current_job.request_cancel()

    def cancel_all(self) -> None:
        """Cancel current job and clear the queue."""
        with self._lock:
            # Cancel current
            if self._current_job and self._current_job.status == JobStatus.RUNNING:
                self._current_job.request_cancel()
            # Clear queue - preserve in history
            for job in self._queue:
                job.status = JobStatus.CANCELLED
                self._history.append(job)
            self._queue.clear()
            logger.info("All jobs cancelled")

    def report_progress(self, clip_name: str, current: int, total: int) -> None:
        """Report progress for the current job. Called by processing code."""
        if self._current_job:
            self._current_job.current_frame = current
            self._current_job.total_frames = total
        if self.on_progress:
            self.on_progress(clip_name, current, total)

    def report_warning(self, message: str) -> None:
        """Report a non-fatal warning. Called by processing code."""
        logger.warning(message)
        if self.on_warning:
            self.on_warning(message)

    def find_job_by_id(self, job_id: str) -> GPUJob | None:
        """Find a job by ID in queue, current, or history."""
        with self._lock:
            if self._current_job and self._current_job.id == job_id:
                return self._current_job
            for job in self._queue:
                if job.id == job_id:
                    return job
            for job in self._history:
                if job.id == job_id:
                    return job
        return None

    def clear_history(self) -> None:
        """Clear job history (for UI reset)."""
        with self._lock:
            self._history.clear()

    def remove_job(self, job_id: str) -> None:
        """Remove a single finished job from history."""
        with self._lock:
            self._history = [j for j in self._history if j.id != job_id]

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return len(self._queue) > 0

    @property
    def current_job(self) -> GPUJob | None:
        with self._lock:
            return self._current_job

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def queue_snapshot(self) -> list[GPUJob]:
        """Return a copy of the current queue for display purposes."""
        with self._lock:
            return list(self._queue)

    @property
    def history_snapshot(self) -> list[GPUJob]:
        """Return a copy of job history for display purposes."""
        with self._lock:
            return list(self._history)

    @property
    def all_jobs_snapshot(self) -> list[GPUJob]:
        """Return current + queued + history for full queue panel display."""
        with self._lock:
            result = []
            if self._current_job:
                result.append(self._current_job)
            result.extend(self._queue)
            result.extend(self._history)
            return result
