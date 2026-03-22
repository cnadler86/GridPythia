"""Electric-price provider interface."""

from abc import abstractmethod

import polars as pl

from src.prediction.base import PredictionProvider


class ElecPriceProvider(PredictionProvider):
    """Returns electricity market price in EUR/Wh per time step."""

    @abstractmethod
    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        """Return Float32 Series of EUR/Wh, same length as *timestamps*."""
        ...
