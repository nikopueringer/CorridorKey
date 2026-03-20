"""WebSocket endpoint and connection manager for real-time updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info(f"WebSocket connected ({len(self._connections)} total)")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info(f"WebSocket disconnected ({len(self._connections)} total)")

    async def _broadcast(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_sync(self, message: dict[str, Any]) -> None:
        """Thread-safe broadcast from the worker thread."""
        if not self._connections or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)
        except RuntimeError:
            pass

    def send_job_progress(self, job_id: str, clip_name: str, current: int, total: int) -> None:
        self.broadcast_sync(
            {
                "type": "job:progress",
                "data": {"job_id": job_id, "clip_name": clip_name, "current": current, "total": total},
            }
        )

    def send_job_status(self, job_id: str, status: str, error: str | None = None) -> None:
        self.broadcast_sync(
            {
                "type": "job:status",
                "data": {"job_id": job_id, "status": status, "error": error},
            }
        )

    def send_job_warning(self, job_id: str, message: str) -> None:
        self.broadcast_sync(
            {
                "type": "job:warning",
                "data": {"job_id": job_id, "message": message},
            }
        )

    def send_clip_state_changed(self, clip_name: str, new_state: str) -> None:
        self.broadcast_sync(
            {
                "type": "clip:state_changed",
                "data": {"clip_name": clip_name, "new_state": new_state},
            }
        )

    def send_vram_update(self, vram: dict) -> None:
        self.broadcast_sync(
            {
                "type": "vram:update",
                "data": vram,
            }
        )


manager = ConnectionManager()


async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            # Keep connection alive; we don't expect client messages
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
