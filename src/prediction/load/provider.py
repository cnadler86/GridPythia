"""Load forecast provider interface."""

from abc import abstractmethod

import polars as pl

from src.prediction.base import PredictionProvider


class LoadProvider(PredictionProvider):
    """Returns electrical load power in W per time step."""

    @abstractmethod
    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        """Return Float32 Series of watts, same length as *timestamps*."""
        ...
