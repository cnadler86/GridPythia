"""Electricity price import from a pre-built list."""

from array import array
from datetime import datetime

from src.prediction.base import make_array, n_steps, resample
from src.prediction.electricprice.provider import ElecPriceProvider


class ElecPriceImport(ElecPriceProvider):
    """Provide electricity prices from an explicit list of EUR / Wh values.

    If the requested *dt_hours* differs from *source_dt_hours* the data is
    linearly resampled.  If the list is shorter than the requested window the
    last known value is repeated.
    """

    def __init__(self, prices_wh: list[float], source_dt_hours: float = 1.0) -> None:
        self._prices = array("f", prices_wh)
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "ElecPriceImport"

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        data = self._prices
        if abs(self._source_dt - dt_hours) > 1e-9:
            data = resample(data, self._source_dt, dt_hours)

        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)
        result = make_array(size=steps)
        for i in range(min(steps, len(data))):
            result[i] = data[i]
        # Pad with last known value
        if len(data) < steps and len(data) > 0:
            pad = data[-1]
            for i in range(len(data), steps):
                result[i] = pad
        return result
