"""Server plugin system for GridPythia.

Plugins are optional modules that register additional FastAPI routers,
MQTT subscriptions, and background tasks.  They are lazy-loaded based on
configuration flags to minimize RAM usage on embedded systems.
"""

from GridPythia.server.plugins.registry import PluginRegistry

__all__ = ["PluginRegistry"]
