"""PV forecast import provider."""

from array import array
from datetime import datetime

from src.prediction.base import make_array, n_steps, resample
from src.prediction.pvforecast.provider import PVForecastProvider


class PVForecastImport(PVForecastProvider):
    """Provide PV power output from an explicit list of watt values.

    Missing trailing steps default to 0 (night-time assumption).
    """

    def __init__(self, power_w: list[float], source_dt_hours: float = 1.0) -> None:
        self._power = array("f", power_w)
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "PVForecastImport"

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        data = self._power
        if abs(self._source_dt - dt_hours) > 1e-9:
            data = resample(data, self._source_dt, dt_hours)

        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)
        result = make_array(size=steps)
        for i in range(min(steps, len(data))):
            result[i] = data[i]
        # PV defaults to 0 for missing steps (night)
        return result
