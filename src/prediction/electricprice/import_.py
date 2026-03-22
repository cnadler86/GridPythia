"""Electricity price import from a pre-built list."""

import polars as pl

from src.prediction.base import resample_to_timestamps
from src.prediction.electricprice.provider import ElecPriceProvider


class ElecPriceImport(ElecPriceProvider):
    """Provide electricity prices from an explicit list of EUR/Wh values.

    If the source step differs from the timestamps spacing the data is
    linearly resampled.  If the list is shorter than the window the last
    value is repeated.
    """

    def __init__(self, prices_wh: list[float], source_dt_hours: float = 1.0) -> None:
        self._prices: list[float] = list(prices_wh)
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "ElecPriceImport"

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        pad = self._prices[-1] if self._prices else 0.0
        return resample_to_timestamps(self._prices, self._source_dt, timestamps, pad_value=pad)
