"""GridPythia service layer.

This package provides application services that encapsulate business logic
and state management. Services are designed for dependency injection and
can be shared across web servers, MQTT handlers, and CLI tools.

Classes
-------
GridPythiaService
    Central application service managing config, providers, optimizer, and cache.
PredictionCache
    TTL-based cache for prediction data.
"""

from GridPythia.services.application import GridPythiaService, PredictionCache

__all__ = ["GridPythiaService", "PredictionCache"]
