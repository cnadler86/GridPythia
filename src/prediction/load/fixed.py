"""Fixed / schedule-based load provider."""

from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.prediction.base import make_array, n_steps
from src.prediction.load.provider import LoadProvider


@dataclass
class LoadTimeWindow:
    """A daily-recurring load window.

    *start_hour* / *end_hour* in ``[0, 24)``; *power_w* in watts.
    """

    start_hour: float
    end_hour: float
    power_w: float


class LoadFixed(LoadProvider):
    """Constant or time-of-day load.

    Without a *schedule* every step returns *power_w*.
    """

    def __init__(
        self,
        power_w: float = 500.0,
        schedule: list[LoadTimeWindow] | None = None,
    ) -> None:
        self._power_w = power_w
        self._schedule = schedule

    @property
    def provider_id(self) -> str:
        return "LoadFixed"

    def _power_at(self, hour_of_day: float) -> float:
        if self._schedule:
            for w in self._schedule:
                if w.start_hour <= hour_of_day < w.end_hour:
                    return w.power_w
        return self._power_w

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)
        result = make_array(size=steps)
        for i in range(steps):
            t = start + timedelta(hours=i * dt_hours)
            result[i] = self._power_at(t.hour + t.minute / 60.0)
        return result
