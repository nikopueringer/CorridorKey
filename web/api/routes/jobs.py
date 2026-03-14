"""Job submission, listing, and cancellation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.job_queue import GPUJob, JobType

from ..deps import get_queue, get_service
from ..schemas import (
    ExtractJobRequest,
    GVMJobRequest,
    InferenceJobRequest,
    JobListResponse,
    JobSchema,
    PipelineJobRequest,
    VideoMaMaJobRequest,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _job_to_schema(job: GPUJob) -> JobSchema:
    return JobSchema(
        id=job.id,
        job_type=job.job_type.value,
        clip_name=job.clip_name,
        status=job.status.value,
        current_frame=job.current_frame,
        total_frames=job.total_frames,
        error_message=job.error_message,
    )


@router.get("", response_model=JobListResponse)
def list_jobs():
    queue = get_queue()
    current = queue.current_job
    return JobListResponse(
        current=_job_to_schema(current) if current else None,
        queued=[_job_to_schema(j) for j in queue.queue_snapshot],
        history=[_job_to_schema(j) for j in queue.history_snapshot],
    )


@router.post("/inference", response_model=list[JobSchema])
def submit_inference(req: InferenceJobRequest):
    queue = get_queue()
    submitted = []
    for clip_name in req.clip_names:
        job = GPUJob(
            job_type=JobType.INFERENCE,
            clip_name=clip_name,
            params={
                "inference_params": req.params.model_dump(),
                "output_config": req.output_config.model_dump(),
                "frame_range": list(req.frame_range) if req.frame_range else None,
            },
        )
        if queue.submit(job):
            submitted.append(_job_to_schema(job))
    if not submitted:
        raise HTTPException(status_code=409, detail="All jobs rejected (duplicates)")
    return submitted


@router.post("/gvm", response_model=list[JobSchema])
def submit_gvm(req: GVMJobRequest):
    queue = get_queue()
    submitted = []
    for clip_name in req.clip_names:
        job = GPUJob(job_type=JobType.GVM_ALPHA, clip_name=clip_name)
        if queue.submit(job):
            submitted.append(_job_to_schema(job))
    if not submitted:
        raise HTTPException(status_code=409, detail="All jobs rejected (duplicates)")
    return submitted


@router.post("/videomama", response_model=list[JobSchema])
def submit_videomama(req: VideoMaMaJobRequest):
    queue = get_queue()
    submitted = []
    for clip_name in req.clip_names:
        job = GPUJob(
            job_type=JobType.VIDEOMAMA_ALPHA,
            clip_name=clip_name,
            params={"chunk_size": req.chunk_size},
        )
        if queue.submit(job):
            submitted.append(_job_to_schema(job))
    if not submitted:
        raise HTTPException(status_code=409, detail="All jobs rejected (duplicates)")
    return submitted


@router.post("/pipeline", response_model=list[JobSchema])
def submit_pipeline(req: PipelineJobRequest):
    """Submit the first step of a full pipeline.

    Only queues the NEXT needed step for each clip. When that step
    completes, the worker auto-chains the following step (via the
    pipeline params stored on the job). This ensures each step finishes
    before the next begins.
    """
    from ..routes.clips import _clips_dir

    queue = get_queue()
    service = get_service()
    clips = service.scan_clips(_clips_dir)
    clip_map = {c.name: c for c in clips}

    # Pipeline params stored on each job so the worker can chain the next step
    pipeline_params = {
        "pipeline": True,
        "alpha_method": req.alpha_method,
        "inference_params": req.params.model_dump(),
        "output_config": req.output_config.model_dump(),
    }

    submitted = []
    for clip_name in req.clip_names:
        clip = clip_map.get(clip_name)
        if not clip:
            continue

        state = clip.state.value

        if state == "EXTRACTING":
            job = GPUJob(job_type=JobType.VIDEO_EXTRACT, clip_name=clip_name, params=pipeline_params)
        elif state == "RAW":
            if req.alpha_method == "videomama":
                job = GPUJob(
                    job_type=JobType.VIDEOMAMA_ALPHA,
                    clip_name=clip_name,
                    params={**pipeline_params, "chunk_size": 50},
                )
            else:
                job = GPUJob(job_type=JobType.GVM_ALPHA, clip_name=clip_name, params=pipeline_params)
        elif state == "MASKED":
            # MASKED clips already have a mask — run VideoMaMa to generate alpha, then inference
            job = GPUJob(
                job_type=JobType.VIDEOMAMA_ALPHA,
                clip_name=clip_name,
                params={**pipeline_params, "chunk_size": 50},
            )
        elif state in ("READY", "COMPLETE"):
            job = GPUJob(job_type=JobType.INFERENCE, clip_name=clip_name, params=pipeline_params)
        else:
            continue

        if queue.submit(job):
            submitted.append(_job_to_schema(job))

    if not submitted:
        raise HTTPException(status_code=409, detail="No jobs submitted (clips may already be complete or duplicates)")
    return submitted


@router.post("/extract", response_model=list[JobSchema])
def submit_extract(req: ExtractJobRequest):
    queue = get_queue()
    submitted = []
    for clip_name in req.clip_names:
        job = GPUJob(job_type=JobType.VIDEO_EXTRACT, clip_name=clip_name)
        if queue.submit(job):
            submitted.append(_job_to_schema(job))
    if not submitted:
        raise HTTPException(status_code=409, detail="All jobs rejected (duplicates)")
    return submitted


@router.delete("/{job_id}")
def cancel_job(job_id: str):
    queue = get_queue()
    job = queue.find_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    queue.cancel_job(job)
    return {"status": "cancelled", "job_id": job_id}


@router.get("/{job_id}/log")
def get_job_log(job_id: str):
    """Get detailed error/log info for a job."""
    queue = get_queue()
    job = queue.find_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {
        "id": job.id,
        "job_type": job.job_type.value,
        "clip_name": job.clip_name,
        "status": job.status.value,
        "error_message": job.error_message,
        "current_frame": job.current_frame,
        "total_frames": job.total_frames,
        "params": job.params,
    }


@router.delete("")
def cancel_all():
    queue = get_queue()
    queue.cancel_all()
    return {"status": "all_cancelled"}
