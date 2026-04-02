"""Feed-in tariff provider interface."""

from abc import abstractmethod

import numpy as np

from GridPythia.prediction.base import PredictionProvider


class FeedInTariffProvider(PredictionProvider):
    """Returns feed-in tariff in EUR/Wh per time step."""

    @abstractmethod
    async def fetch(self, timestamps: list) -> np.ndarray:
        """Return float32 ndarray of EUR/Wh, same length as *timestamps*."""
        ...
