"""Fixed feed-in tariff provider."""

from array import array
from datetime import datetime

from src.prediction.base import make_array, n_steps
from src.prediction.feedintariff.provider import FeedInTariffProvider


class FeedInTariffFixed(FeedInTariffProvider):
    """Constant feed-in tariff for every time step.

    *tariff_kwh* is specified in EUR / kWh and stored internally as EUR / Wh.
    """

    def __init__(self, tariff_kwh: float = 0.082) -> None:
        self._tariff_wh = tariff_kwh / 1000.0

    @property
    def provider_id(self) -> str:
        return "FeedInTariffFixed"

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)
        return make_array([self._tariff_wh] * steps)
