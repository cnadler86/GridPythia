"""Energy-Charts electricity price provider."""

import logging
from array import array
from datetime import datetime, timedelta, timezone
from statistics import median

from src.prediction.base import make_array, n_steps
from src.prediction.electricprice.provider import ElecPriceProvider

logger = logging.getLogger(__name__)


class ElecPriceEnergyCharts(ElecPriceProvider):
    """Fetch day-ahead electricity prices from the Energy-Charts API.

    Prices beyond the last available API timestamp are extended using
    Exponential Smoothing (requires ``statsmodels``, optional) or a simple
    median fallback.
    """

    def __init__(
        self,
        bidding_zone: str = "DE-LU",
        charges_kwh: float = 0.0,
        vat_rate: float = 1.0,
    ) -> None:
        self._bidding_zone = bidding_zone
        self._charges_kwh = charges_kwh
        self._vat_rate = vat_rate

    @property
    def provider_id(self) -> str:
        return "EnergyCharts"

    # ── API request ───────────────────────────────────────────────────

    def _request_prices(
        self, start: datetime, end: datetime
    ) -> list[tuple[datetime, float]]:
        """Call ``https://api.energy-charts.info/price``.

        Returns a list of *(aware-datetime, price_eur_per_wh)* tuples.
        """
        import requests

        url = "https://api.energy-charts.info/price"
        params = {
            "bzn": self._bidding_zone,
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "end": end.strftime("%Y-%m-%dT%H:%M"),
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        timestamps: list[int] = data.get("unix_seconds", [])
        prices_mwh: list[float | None] = data.get("price", [])
        charges_wh = self._charges_kwh / 1000.0

        result: list[tuple[datetime, float]] = []
        for ts, price_mwh in zip(timestamps, prices_mwh):
            if price_mwh is None:
                continue
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            price_wh = price_mwh / 1_000_000.0  # EUR/MWh → EUR/Wh
            if charges_wh > 0:
                price_wh = (price_wh + charges_wh) * self._vat_rate
            result.append((dt, price_wh))

        return result

    # ── ETS / fallback ────────────────────────────────────────────────

    @staticmethod
    def _cap_outliers(values: list[float], sigma: int = 2) -> list[float]:
        n = len(values)
        if n < 2:
            return list(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = var**0.5
        lo, hi = mean - sigma * std, mean + sigma * std
        return [max(lo, min(hi, v)) for v in values]

    def _forecast(self, history: list[float], steps: int) -> list[float]:
        """Extend *history* by *steps* hours using ETS or median."""
        capped = self._cap_outliers(history)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            sp = 168 if len(capped) > 800 else 24
            if len(capped) > sp * 2:
                model = ExponentialSmoothing(
                    capped, seasonal="add", seasonal_periods=sp
                ).fit(optimized=True)
                return [float(v) for v in model.forecast(steps)]
        except Exception:
            pass
        med = median(capped) if capped else 0.0
        return [med] * steps

    # ── public API ────────────────────────────────────────────────────

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)

        # Pull extra 5 weeks of history for ETS
        hist_start = start - timedelta(days=35)
        raw = self._request_prices(hist_start, end)
        if not raw:
            return make_array(size=steps)

        # Separate history (before start) from future
        history: list[float] = []
        lookup: dict[int, float] = {}
        for dt, price in raw:
            offset_h = (dt - start).total_seconds() / 3600.0
            idx = round(offset_h / dt_hours)
            if idx < 0:
                history.append(price)
            elif 0 <= idx < steps:
                lookup[idx] = price
                history.append(price)

        result = make_array(size=steps)
        last_known = -1
        for idx in sorted(lookup):
            result[idx] = lookup[idx]
            last_known = idx

        # Fill remaining steps with ETS / median
        if last_known < steps - 1 and history:
            gap = steps - last_known - 1
            forecast = self._forecast(history, gap)
            for i, val in enumerate(forecast):
                result[last_known + 1 + i] = val

        return result
