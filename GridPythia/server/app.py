"""GridPythia FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import GridPythia.server.state as state
from GridPythia.server.routers.config import router as config_router
from GridPythia.server.routers.optimization import router as optimization_router
from GridPythia.server.routers.predictions import router as predictions_router

_STATIC_DIR = Path(__file__).parent / "static"


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
    )

    app.include_router(config_router, prefix="/api")
    app.include_router(predictions_router, prefix="/api/predictions")
    app.include_router(optimization_router, prefix="/api")

    # Static frontend – mounted last so all /api/* routes take precedence.
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app
