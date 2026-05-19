"""Load learning plugin – adaptive load forecast with MQTT ingestion.

This plugin adds:
* REST endpoints for data ingestion, vacation mode, and statistics.
* MQTT subscription for real-time power measurements.
* Periodic TSDB maintenance task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from structlog import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

    from GridPythia.config import AppConfig

from GridPythia.server.plugins.load_learning.router import create_router
from GridPythia.server.plugins.load_learning.service import LoadLearningService

logger = get_logger(__name__)


class LoadLearningPlugin:
    """Server plugin for adaptive load learning."""

    def __init__(self) -> None:
        self._service: LoadLearningService | None = None

    @property
    def name(self) -> str:
        return "load_learning"

    def register(self, app: FastAPI, config: AppConfig) -> None:
        """Register the load learning router and store config."""
        self._service = LoadLearningService(config)
        router = create_router(self._service)
        app.include_router(router, prefix="/api/plugins/load-learning")
        # Store service on app state for access from MQTT gateway
        app.state.load_learning_service = self._service

    async def startup(self) -> None:
        """Start MQTT subscription and maintenance loop."""
        if self._service is not None:
            await self._service.start()

    async def shutdown(self) -> None:
        """Stop background tasks."""
        if self._service is not None:
            await self._service.stop()
