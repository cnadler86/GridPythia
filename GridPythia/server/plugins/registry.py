"""Plugin registry and lifecycle management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from structlog import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

    from GridPythia.config import AppConfig

logger = get_logger(__name__)


class ServerPlugin(Protocol):
    """Protocol for server plugins."""

    @property
    def name(self) -> str: ...

    def register(self, app: FastAPI, config: AppConfig) -> None:
        """Register routes, background tasks, etc."""
        ...

    async def startup(self) -> None:
        """Called during app lifespan startup."""
        ...

    async def shutdown(self) -> None:
        """Called during app lifespan shutdown."""
        ...


class PluginRegistry:
    """Manages plugin discovery and lifecycle."""

    def __init__(self) -> None:
        self._plugins: list[ServerPlugin] = []

    def discover(self, config: AppConfig) -> None:
        """Discover and instantiate plugins based on config flags."""
        # Load learning plugin (adaptive load)
        if config.prediction.load.adaptive.enabled:
            from GridPythia.server.plugins.load_learning import LoadLearningPlugin

            self._plugins.append(LoadLearningPlugin())
            logger.info("plugin_discovered", plugin="load_learning")

    def register_all(self, app: FastAPI, config: AppConfig) -> None:
        """Register all discovered plugins with the FastAPI app."""
        for plugin in self._plugins:
            plugin.register(app, config)
            logger.info("plugin_registered", plugin=plugin.name)

    async def startup_all(self) -> None:
        """Run startup hooks for all plugins."""
        for plugin in self._plugins:
            await plugin.startup()

    async def shutdown_all(self) -> None:
        """Run shutdown hooks for all plugins."""
        for plugin in self._plugins:
            await plugin.shutdown()

    @property
    def plugins(self) -> list[ServerPlugin]:
        return list(self._plugins)
