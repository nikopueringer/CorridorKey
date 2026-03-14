"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.project import projects_root

from .deps import get_queue, get_service
from .routes import clips, jobs, preview, projects, system, upload
from .worker import start_worker
from .ws import manager, websocket_endpoint

logger = logging.getLogger(__name__)

# Resolve clips directory from env or default to Projects/
CLIPS_DIR = os.environ.get("CK_CLIPS_DIR", "")


def _resolve_clips_dir() -> str:
    if CLIPS_DIR:
        return os.path.abspath(CLIPS_DIR)
    return projects_root()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: detect device, start worker. Shutdown: stop worker."""
    clips_dir = _resolve_clips_dir()
    os.makedirs(clips_dir, exist_ok=True)

    clips.set_clips_dir(clips_dir)
    preview.set_clips_dir(clips_dir)

    service = get_service()
    device = service.detect_device()
    logger.info(f"Device: {device}, Clips dir: {clips_dir}")

    loop = asyncio.get_running_loop()
    manager.set_loop(loop)

    queue = get_queue()
    worker_thread, stop_event = start_worker(service, queue, clips_dir)

    app.state.clips_dir = clips_dir
    app.state.worker_thread = worker_thread
    app.state.stop_event = stop_event

    yield

    stop_event.set()
    worker_thread.join(timeout=5)
    logger.info("Worker thread joined")


def create_app() -> FastAPI:
    """Application factory — call with uvicorn --factory."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    app = FastAPI(
        title="CorridorKey WebUI",
        version="1.0.0",
        lifespan=lifespan,
    )

    # API routes
    app.include_router(clips.router)
    app.include_router(jobs.router)
    app.include_router(system.router)
    app.include_router(preview.router)
    app.include_router(projects.router)
    app.include_router(upload.router)

    # WebSocket
    app.websocket("/ws")(websocket_endpoint)

    # Serve built Svelte SPA from web/frontend/build/
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "build")
    if os.path.isdir(static_dir):
        # Mount static assets (JS, CSS, images) — but NOT as catch-all
        app.mount("/_app", StaticFiles(directory=os.path.join(static_dir, "_app")), name="spa-assets")

        index_html = os.path.join(static_dir, "index.html")

        # SPA catch-all: any non-API, non-asset GET request serves index.html
        @app.get("/{path:path}", include_in_schema=False)
        async def spa_fallback(request: Request, path: str):
            # Don't intercept API or WebSocket paths
            if path.startswith("api/") or path == "ws":
                return
            # Serve actual static files if they exist (favicon, etc.)
            file_path = os.path.join(static_dir, path)
            if path and os.path.isfile(file_path):
                return FileResponse(file_path)
            return FileResponse(index_html)
    else:
        logger.warning(f"SPA build directory not found at {static_dir} — serving API only")

    return app
