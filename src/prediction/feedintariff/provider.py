"""Feed-in tariff provider interface."""

from abc import abstractmethod
from array import array
from datetime import datetime

from src.prediction.base import PredictionProvider


class FeedInTariffProvider(PredictionProvider):
    """Returns feed-in tariff in EUR / Wh per time step."""

    @abstractmethod
    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        """Return ``array('f', ...)`` of EUR/Wh with ``n_steps`` entries."""
        ...
