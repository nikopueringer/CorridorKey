"""Unit tests for GPUJobQueue."""

from __future__ import annotations

import pytest
from corridorkey.errors import JobCancelledError
from corridorkey.job_queue import GPUJob, GPUJobQueue, JobStatus, JobType


def _job(clip: str = "shot1", job_type: JobType = JobType.INFERENCE) -> GPUJob:
    return GPUJob(job_type=job_type, clip_name=clip)


class TestGPUJob:
    def test_cancel_sets_flag(self):
        job = _job()
        assert not job.is_cancelled
        job.request_cancel()
        assert job.is_cancelled

    def test_check_cancelled_raises(self):
        job = _job()
        job.request_cancel()
        with pytest.raises(JobCancelledError):
            job.check_cancelled()

    def test_check_cancelled_no_raise_when_not_cancelled(self):
        job = _job()
        job.check_cancelled()  # should not raise


class TestGPUJobQueue:
    def test_submit_and_next(self):
        q = GPUJobQueue()
        job = _job()
        assert q.submit(job)
        assert q.next_job() is job

    def test_duplicate_rejected(self):
        q = GPUJobQueue()
        j1 = _job("shot1", JobType.INFERENCE)
        j2 = _job("shot1", JobType.INFERENCE)
        assert q.submit(j1)
        assert not q.submit(j2)

    def test_different_clip_not_duplicate(self):
        q = GPUJobQueue()
        assert q.submit(_job("shot1"))
        assert q.submit(_job("shot2"))
        assert q.pending_count == 2

    def test_start_job_removes_from_queue(self):
        q = GPUJobQueue()
        job = _job()
        q.submit(job)
        q.start_job(job)
        assert q.pending_count == 0
        assert q.current_job is job

    def test_complete_job_clears_current(self):
        q = GPUJobQueue()
        job = _job()
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)
        assert q.current_job is None
        assert job.status == JobStatus.COMPLETED

    def test_fail_job(self):
        q = GPUJobQueue()
        job = _job()
        q.submit(job)
        q.start_job(job)
        q.fail_job(job, "something broke")
        assert job.status == JobStatus.FAILED
        assert job.error_message == "something broke"
        assert q.current_job is None

    def test_cancel_queued_job(self):
        q = GPUJobQueue()
        job = _job()
        q.submit(job)
        q.cancel_job(job)
        assert job.status == JobStatus.CANCELLED
        assert q.pending_count == 0

    def test_cancel_running_job_sets_flag(self):
        q = GPUJobQueue()
        job = _job()
        q.submit(job)
        q.start_job(job)
        q.cancel_job(job)
        assert job.is_cancelled

    def test_mark_cancelled_clears_current(self):
        q = GPUJobQueue()
        job = _job()
        q.submit(job)
        q.start_job(job)
        q.mark_cancelled(job)
        assert q.current_job is None
        assert job.status == JobStatus.CANCELLED

    def test_cancel_all_clears_queue(self):
        q = GPUJobQueue()
        q.submit(_job("a"))
        q.submit(_job("b"))
        q.cancel_all()
        assert q.pending_count == 0

    def test_preview_reprocess_replaces_existing(self):
        q = GPUJobQueue()
        j1 = _job("shot1", JobType.PREVIEW_REPROCESS)
        j2 = _job("shot1", JobType.PREVIEW_REPROCESS)
        q.submit(j1)
        q.submit(j2)
        assert q.pending_count == 1
        assert q.next_job() is j2

    def test_has_pending(self):
        q = GPUJobQueue()
        assert not q.has_pending
        q.submit(_job())
        assert q.has_pending

    def test_find_job_by_id(self):
        q = GPUJobQueue()
        job = _job()
        q.submit(job)
        found = q.find_job_by_id(job.id)
        assert found is job

    def test_on_completion_callback(self):
        completed: list[str] = []
        q = GPUJobQueue()
        q.on_completion = lambda name: completed.append(name)
        job = _job("shot1")
        q.submit(job)
        q.start_job(job)
        q.complete_job(job)
        assert completed == ["shot1"]

    def test_on_error_callback(self):
        errors: list[tuple] = []
        q = GPUJobQueue()
        q.on_error = lambda name, msg: errors.append((name, msg))
        job = _job("shot1")
        q.submit(job)
        q.start_job(job)
        q.fail_job(job, "oops")
        assert errors == [("shot1", "oops")]

    def test_alpha_gen_job_type(self):
        job = GPUJob(job_type=JobType.ALPHA_GEN, clip_name="shot1")
        assert job.job_type == JobType.ALPHA_GEN

    def test_queue_snapshot(self):
        q = GPUJobQueue()
        q.submit(_job("a"))
        q.submit(_job("b"))
        snap = q.queue_snapshot
        assert len(snap) == 2
