"""Load forecast import from a pre-built list."""

from array import array
from datetime import datetime

from src.prediction.base import make_array, n_steps, resample
from src.prediction.load.provider import LoadProvider


class LoadImport(LoadProvider):
    """Provide load forecast from an explicit list of watt values."""

    def __init__(self, load_w: list[float], source_dt_hours: float = 1.0) -> None:
        self._load = array("f", load_w)
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "LoadImport"

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        data = self._load
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
