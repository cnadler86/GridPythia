"""Fixed feed-in tariff provider."""

import numpy as np

from GridPythia.prediction.feedintariff.provider import FeedInTariffProvider


class FeedInTariffFixed(FeedInTariffProvider):
    """Constant feed-in tariff for every time step.

    *tariff_kwh* is specified in EUR/kWh and stored internally as EUR/Wh.
    """

    def __init__(self, tariff_kwh: float = 0.0) -> None:
        self._tariff_wh = tariff_kwh / 1000.0

    @property
    def provider_id(self) -> str:
        return "FeedInTariffFixed"

    async def fetch(self, timestamps: list) -> np.ndarray:
        return np.full(len(timestamps), self._tariff_wh, dtype=np.float32)
