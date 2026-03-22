"""Energy-Charts electricity price provider."""

import logging
from datetime import datetime, timedelta, timezone

import aiohttp
import polars as pl

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
        vat_rate: float = 1.19,
    ) -> None:
        self._bidding_zone = bidding_zone
        self._charges_kwh = charges_kwh
        self._vat_rate = vat_rate

    @property
    def provider_id(self) -> str:
        return "EnergyCharts"

    # ── API request ───────────────────────────────────────────────────

    async def _request_prices(self, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
        """Call ``https://api.energy-charts.info/price``."""
        url = "https://api.energy-charts.info/price"
        params = {
            "bzn": self._bidding_zone,
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "end": end.strftime("%Y-%m-%dT%H:%M"),
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

        timestamps_s: list[int] = data.get("unix_seconds", [])
        prices_mwh: list[float | None] = data.get("price", [])
        charges_wh = self._charges_kwh / 1000.0

        result: list[tuple[datetime, float]] = []
        for ts, price_mwh in zip(timestamps_s, prices_mwh):
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
        """Extend *history* by *steps* 15-minute intervals using ETS or median."""
        capped = self._cap_outliers(history)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            sp_hours = 168 if len(capped) > 800 else 24
            sp = sp_hours * 4  # convert hours -> number of 15-minute periods
            if len(capped) > sp * 2:
                model = ExponentialSmoothing(capped, seasonal="add", seasonal_periods=sp).fit(
                    optimized=True
                )
                return [float(v) for v in model.forecast(steps)]
        except Exception:
            pass
        from statistics import median
        med = median(capped) if capped else 0.0
        return [med] * steps

    # ── public API ────────────────────────────────────────────────────

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        ts_list: list[datetime] = timestamps.to_list()
        start = ts_list[0]
        end = ts_list[-1]

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)

        # Pull 5 weeks of history for ETS
        hist_start = start_utc - timedelta(days=35)
        raw = await self._request_prices(hist_start, end_utc + timedelta(hours=1))
        if not raw:
            return pl.Series([0.0] * len(ts_list), dtype=pl.Float32)

        # Build 15-minute-indexed lookup (key = unix 15-min bucket since epoch)
        history: list[float] = []
        lookup: dict[int, float] = {}
        start_bucket = int(start_utc.timestamp()) // 900

        for dt, price in raw:
            b = int(dt.timestamp()) // 900
            history.append(price)
            if b >= start_bucket:
                lookup[b] = price

        # Map each target timestamp to its 15-minute bucket
        result: list[float | None] = []
        for ts in ts_list:
            b = int(_to_utc(ts).timestamp()) // 900
            result.append(lookup.get(b))

        # Fill gaps with ETS / median forecast
        missing = sum(1 for v in result if v is None)
        if missing and history:
            forecast = self._forecast(history, missing)
            fi = 0
            for i in range(len(result)):
                if result[i] is None:
                    result[i] = forecast[fi]
                    fi += 1

        filled = [v if v is not None else 0.0 for v in result]
        return pl.Series(filled, dtype=pl.Float32)
