"""Feed-in tariff import from a pre-built list."""

import polars as pl

from GridPythia.prediction.base import resample_to_timestamps
from GridPythia.prediction.feedintariff.provider import FeedInTariffProvider


class FeedInTariffImport(FeedInTariffProvider):
    """Provide feed-in tariffs from an explicit list of EUR/Wh values."""

    def __init__(self, tariffs_wh: list[float], source_dt_hours: float = 1.0) -> None:
        self._tariffs: list[float] = list(tariffs_wh)
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "FeedInTariffImport"

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        pad = self._tariffs[-1] if self._tariffs else 0.0
        return resample_to_timestamps(self._tariffs, self._source_dt, timestamps, pad_value=pad)
