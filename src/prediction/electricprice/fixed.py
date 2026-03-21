"""Fixed / time-of-use electricity price provider."""

from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.prediction.base import make_array, n_steps
from src.prediction.electricprice.provider import ElecPriceProvider


@dataclass
class TimeWindow:
    """A daily-recurring price window.

    *start_hour* and *end_hour* are floats in ``[0, 24)``.
    *value* is in EUR / kWh.
    """

    start_hour: float
    end_hour: float
    value: float


class ElecPriceFixed(ElecPriceProvider):
    """Constant or time-of-use electricity price.

    When *schedule* is ``None`` every step gets the flat *price_kwh*.
    Otherwise, the first matching :class:`TimeWindow` wins.
    """

    def __init__(
        self,
        price_kwh: float = 0.30,
        charges_kwh: float = 0.0,
        vat_rate: float = 1.0,
        schedule: list[TimeWindow] | None = None,
    ) -> None:
        self._price_kwh = price_kwh
        self._charges_kwh = charges_kwh
        self._vat_rate = vat_rate
        self._schedule = schedule

    @property
    def provider_id(self) -> str:
        return "ElecPriceFixed"

    def _price_at(self, hour_of_day: float) -> float:
        """EUR / Wh at a given fractional hour of day."""
        if self._schedule:
            for w in self._schedule:
                if w.start_hour <= hour_of_day < w.end_hour:
                    return (
                        w.value / 1000.0 + self._charges_kwh / 1000.0
                    ) * self._vat_rate
        return (self._price_kwh / 1000.0 + self._charges_kwh / 1000.0) * self._vat_rate

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)
        result = make_array(size=steps)
        for i in range(steps):
            t = start + timedelta(hours=i * dt_hours)
            result[i] = self._price_at(t.hour + t.minute / 60.0)
        return result
