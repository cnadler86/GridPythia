"""PV forecast import provider."""

import polars as pl

from src.prediction.base import resample_to_timestamps
from src.prediction.pvforecast.provider import PVForecastProvider


class PVForecastImport(PVForecastProvider):
    """Provide PV power output from an explicit list of watt values.

    Missing trailing steps default to 0 (night-time assumption).
    """

    def __init__(self, power_w: list[float], source_dt_hours: float = 1.0) -> None:
        self._power: list[float] = list(power_w)
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "PVForecastImport"

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        # PV pads with 0 (night) beyond the provided data
        return resample_to_timestamps(self._power, self._source_dt, timestamps, pad_value=0.0)
