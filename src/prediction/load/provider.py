"""Load forecast provider interface."""

from abc import abstractmethod
from array import array
from datetime import datetime

from src.prediction.base import PredictionProvider


class LoadProvider(PredictionProvider):
    """Returns electrical load power in W per time step."""

    @abstractmethod
    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        """Return ``array('f', ...)`` of watts with ``n_steps`` entries."""
        ...
