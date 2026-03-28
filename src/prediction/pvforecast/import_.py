"""PV forecast import provider."""

import polars as pl

from src.prediction.base import resample_to_timestamps
from src.prediction.pvforecast.provider import PVForecastProvider


class PVForecastImport(PVForecastProvider):
    """Provide PV forecast from an explicit list of watt values.

    Missing trailing steps default to 0 (night-time assumption).
    """

    def __init__(self, power_w: list[float], source_dt_hours: float = 1.0) -> None:
        self._power: list[float] = list(power_w)
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "PVForecastImport"

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        # Resample in power-space and convert to Wh per target step.
        power_w = resample_to_timestamps(self._power, self._source_dt, timestamps, pad_value=0.0)
        ts_list: list = timestamps.to_list()
        if len(ts_list) >= 2:
            dt_hours = (ts_list[1] - ts_list[0]).total_seconds() / 3600.0
        else:
            dt_hours = 1.0
        return power_w * float(dt_hours)
