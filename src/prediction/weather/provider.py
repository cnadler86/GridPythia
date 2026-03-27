"""Weather provider interface."""

from abc import abstractmethod

import polars as pl

from src.prediction.base import PredictionProvider


class WeatherProvider(PredictionProvider):
    """Returns multi-channel weather data for a time window.

    The returned ``pl.DataFrame`` contains one column per channel.  The two
    mandatory channels are ``temperature_c`` and ``cloud_cover_pct``;
    optional ones (``wind_speed_kmh``, ``humidity_pct``, ``precipitation_mm``,
    ``pressure_hpa``, ``ghi_wm2``, ``dni_wm2``, ``dhi_wm2``) are only present
    when the data source provides them.  All columns are ``Float32``.
    """

    @abstractmethod
    async def fetch(self, timestamps: pl.Series) -> pl.DataFrame:
        """Return a DataFrame with one row per timestamp."""
        ...
