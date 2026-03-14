"""Singleton dependencies for the FastAPI application."""

from __future__ import annotations

from backend.job_queue import GPUJobQueue
from backend.service import CorridorKeyService

_service: CorridorKeyService | None = None
_queue: GPUJobQueue | None = None


def get_service() -> CorridorKeyService:
    global _service
    if _service is None:
        _service = CorridorKeyService()
    return _service


def get_queue() -> GPUJobQueue:
    global _queue
    if _queue is None:
        _queue = GPUJobQueue()
    return _queue
