"""GridPythia FastAPI application factory."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import GridPythia.server.state as state
from GridPythia.server import services
from GridPythia.server.routers.appliance import router as appliance_router
from GridPythia.server.routers.config import router as config_router
from GridPythia.server.routers.inverters import router as inverters_router
from GridPythia.server.routers.optimization import router as optimization_router
from GridPythia.server.routers.predictions import router as predictions_router
from GridPythia.server.routers.realtime import router as realtime_router

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start background tasks on startup; cancel them on shutdown."""
    mqtt_task: asyncio.Task | None = None
    scheduler_task: asyncio.Task | None = None

    try:
        cfg, _ = services.load_config()
    except Exception:
        cfg = None

    if cfg is not None and cfg.server.mqtt.enabled:
        from GridPythia.server.mqtt_gateway import run_gateway

        mqtt_task = asyncio.create_task(run_gateway(cfg.server.mqtt), name="mqtt-gateway")

    from GridPythia.server.scheduler import run_scheduler

    scheduler_task = asyncio.create_task(run_scheduler(), name="server-scheduler")

    try:
        yield
    finally:
        if mqtt_task is not None and not mqtt_task.done():
            mqtt_task.cancel()
            try:
                await mqtt_task
            except asyncio.CancelledError:
                pass
        if scheduler_task is not None and not scheduler_task.done():
            scheduler_task.cancel()
            try:
                await scheduler_task
            except asyncio.CancelledError:
                pass
        state.mqtt_connected = False


def create_app(config_path: Path) -> FastAPI:
    """Create and configure the GridPythia FastAPI application.

    Args:
        config_path: Absolute path to the YAML configuration file.

    Returns:
        Configured :class:`~fastapi.FastAPI` instance ready for uvicorn.
    """
    state.config_path = config_path

    app = FastAPI(
        title="GridPythia API",
        description=(
            "REST API for GridPythia home energy management: "
            "prediction forecasts and MILP optimisation."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )

    app.include_router(config_router, prefix="/api")
    app.include_router(predictions_router, prefix="/api/predictions")
    app.include_router(inverters_router, prefix="/api")
    app.include_router(optimization_router, prefix="/api")
    app.include_router(realtime_router, prefix="/api")
    app.include_router(appliance_router, prefix="/api")

    # Static frontend – mounted last so all /api/* routes take precedence.
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app
