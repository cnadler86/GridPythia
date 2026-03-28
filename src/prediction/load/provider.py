"""Load forecast provider interface."""

from abc import abstractmethod

import polars as pl

from src.prediction.base import PredictionProvider


class LoadProvider(PredictionProvider):
    """Returns electrical load energy in Wh per time step."""

    @abstractmethod
    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        """Return Float32 Series of Wh, same length as *timestamps*."""
        ...
