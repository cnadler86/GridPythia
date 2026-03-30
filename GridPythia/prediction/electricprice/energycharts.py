"""Energy-Charts electricity price provider.

Caching strategy
----------------
Day-ahead prices are published by ENTSO-E around 12:00-13:00 CET each day.
The provider fetches complete calendar days (from midnight of the first
requested date to midnight of the last) and stores the result in a single
in-memory price map.

Cache invalidation rules (checked on every ``fetch()`` call):

1. Cache is empty.
2. The requested date range is not fully covered by the cached date range.
3. **Poll window** - all three conditions are true simultaneously:

   - The real current time (``datetime.now(UTC)``) is *beyond* the last
     real API timestamp in the cache, meaning cached real prices no longer
     cover "now".
   - The real current time falls *within* the requested series
     (``requested_start ≤ now ≤ requested_end``).
   - The real current time is at or after **12:30 UTC** - the earliest
     realistic day-ahead publication time.

Outside that narrow publishing window the cache is served as-is, avoiding
repeated API calls.  Once new day-ahead data extends ``_last_real_ts``
beyond ``now``, the poll condition becomes false and no further fetches
happen until the next publication cycle.

Price map construction
----------------------
Real API prices cover the available portion of the requested window.
Any remaining future slots (up to ``last_real_ts + horizon_buffer``) are
filled by ETS (``statsmodels``, optional) or a median fallback.

Concurrency
-----------
An ``asyncio.Lock`` serialises rebuilds so that concurrent ``fetch()``
callers do not each trigger a separate API round-trip.  A double-checked
pattern avoids unnecessary lock contention on the hot path.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

import aiohttp
import polars as pl
from pydantic import BaseModel, Field, field_validator

from GridPythia.prediction.electricprice.provider import ElecPriceProvider

logger = logging.getLogger(__name__)

_HISTORY_WINDOW = timedelta(days=15)
_HORIZON_BUFFER_DEFAULT = timedelta(hours=25)


class EnergyChartsConfig(BaseModel):
    """Pydantic config model for ElecPriceEnergyCharts."""

    bidding_zone: str = Field("DE-LU", min_length=1)
    charges_kwh: float = Field(0.0, ge=0.0)
    vat_rate: float = Field(0.19, ge=0.0, lt=1.0)
    horizon_buffer: timedelta = Field(default=_HORIZON_BUFFER_DEFAULT)

    @field_validator("horizon_buffer", mode="before")
    def _ensure_timedelta(cls, v):
        if isinstance(v, (int, float)):
            return timedelta(hours=float(v))
        return v


class ElecPriceEnergyCharts(ElecPriceProvider):
    """Fetch day-ahead electricity prices from the Energy-Charts API.

    Prices beyond the last available API timestamp are extended using
    Exponential Smoothing (requires ``statsmodels``, optional) or a simple
    median fallback.

    Parameters
    ----------
    bidding_zone:
        ENTSO-E bidding zone string (default ``"DE-LU"``).
    charges_kwh:
        Additional grid/levy charges in EUR/kWh added on top of the
        day-ahead price.
    vat_rate:
        VAT rate applied *after* adding charges (default ``0.19`` for 19% VAT).
    horizon_buffer:
        Extra time added *beyond the last real API data point* when
        pre-computing the price map.  Defaults to 25 h so that forecast
        coverage extends well into the next day after publication.
    """

    def __init__(self, config: EnergyChartsConfig) -> None:
        """Initialize with an EnergyChartsConfig instance or a mapping.

        The constructor accepts either an EnergyChartsConfig instance or a
        mapping/dict that can be validated into one.
        """
        if not isinstance(config, EnergyChartsConfig):
            config = EnergyChartsConfig(**config)

        self._bidding_zone = config.bidding_zone
        self._charges_kwh = config.charges_kwh
        self._vat_rate = config.vat_rate
        self._horizon_buffer = config.horizon_buffer

        # ── cache state ───────────────────────────────────────────────
        # bucket = unix_timestamp // 900  (15-min granularity)
        self._price_map: dict[int, float] = {}
        # Last real (API-provided) timestamp in the price map; everything
        # beyond this was filled by ETS/median forecast.
        self._last_real_ts: datetime | None = None
        # Calendar-day bounds of the currently cached window.
        self._cache_start_day: date | None = None
        self._cache_end_day: date | None = None
        self._lock = asyncio.Lock()

    @property
    def provider_id(self) -> str:
        return "EnergyCharts"

    # ── API request ───────────────────────────────────────────────────

    async def _request_prices(self, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
        """Call ``https://api.energy-charts.info/price`` and return parsed results."""
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

        # Return raw EPEX/Energy-Charts prices in EUR/Wh. Charges and VAT
        # are applied later when serving values so that forecasts (ETS)
        # operate on the underlying market prices.
        result: list[tuple[datetime, float]] = []
        for ts, price_mwh in zip(timestamps_s, prices_mwh, strict=False):
            if price_mwh is None:
                continue
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            price_wh = price_mwh / 1_000_000.0  # EUR/MWh → EUR/Wh (raw)
            result.append((dt, price_wh))

        return result

    # ── ETS / fallback ────────────────────────────────────────────────

    def _forecast(self, history: list[float], steps: int) -> list[float]:
        """Extend *history* by *steps* 15-minute intervals using ETS or median."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            # For 15-minute data there are 4 samples per hour
            periods_per_hour = 4
            weekly_sp = 168 * periods_per_hour  # 168 h -> 672 periods
            daily_sp = 24 * periods_per_hour  # 24 h -> 96 periods

            # Prefer weekly seasonality when we have at least two full weekly
            # cycles in the history; otherwise fall back to daily if possible.
            sp = None
            if len(history) >= 2 * weekly_sp:
                sp = weekly_sp
            elif len(history) >= 2 * daily_sp:
                sp = daily_sp

            if sp is not None and len(history) >= 2 * sp:
                logger.debug("Using ETS with seasonal_periods=%d (history=%d)", sp, len(history))
                model = ExponentialSmoothing(history, seasonal="add", seasonal_periods=sp).fit(
                    optimized=True
                )
                # Ensure forecasts are non-negative (floor at 0.0)
                return [max(0.0, float(v)) for v in model.forecast(steps)]
        except Exception:  # noqa: S110
            pass
        from statistics import median

        med = median(history) if history else 0.0
        # Floor median fallback at 0.0 as well
        return [max(0.0, med)] * steps

    # ── cache management ──────────────────────────────────────────────

    def _needs_refresh(
        self, now_utc: datetime, requested_start: datetime, requested_end: datetime
    ) -> bool:
        """Return True when the price map must be rebuilt before serving.

        Rebuild if:

        - the cache is empty,
        - the requested date range is not fully covered by the cache, or
        - the poll window is open: ``now_utc`` is past the last real API
          data point, falls within the requested series, and is at or
          after 12:30 UTC (earliest realistic day-ahead publication time).
        """
        if not self._price_map:
            return True
        if (
            self._cache_start_day is None
            or self._cache_end_day is None
            or requested_start.date() < self._cache_start_day
            or requested_end.date() > self._cache_end_day
        ):
            return True
        if (
            self._last_real_ts is not None
            and (now_utc + timedelta(days=1)) > self._last_real_ts
            and requested_start <= now_utc <= requested_end
            and (now_utc.hour > 12 or (now_utc.hour == 12 and now_utc.minute >= 30))
        ):
            logger.debug(
                "Poll window open: now=%s > last_real_ts=%s, within series, after 12:30 UTC",
                now_utc.strftime("%H:%M"),
                self._last_real_ts.strftime("%Y-%m-%dT%H:%M"),
            )
            return True
        return False

    async def _refresh(
        self, now_utc: datetime, requested_start: datetime, requested_end: datetime
    ) -> None:
        """Fetch full calendar days and rebuild the price map.

        Fetches from midnight of *requested_start*'s day through 23:59 of
        *requested_end*'s day.  ``_last_real_ts`` is set to the maximum
        timestamp returned by the API; forecast slots are appended up to
        ``_last_real_ts + horizon_buffer``.
        """
        map_start = requested_start.replace(hour=0, minute=0, second=0, microsecond=0)
        hist_start = now_utc - _HISTORY_WINDOW
        fetch_end = requested_end.replace(hour=23, minute=59, second=0, microsecond=0)

        logger.debug(
            "Polling Energy-Charts (bzn=%s, history=[%s, map=[%s, %s UTC])",
            self._bidding_zone,
            hist_start.strftime("%Y-%m-%dT%H:%M"),
            map_start.strftime("%Y-%m-%dT%H:%M"),
            fetch_end.strftime("%Y-%m-%dT%H:%M"),
        )
        raw = await self._request_prices(hist_start, fetch_end)

        if not raw:
            logger.warning("Energy-Charts returned no data for bzn=%s", self._bidding_zone)
            return

        last_real_ts = max(dt for dt, _ in raw)

        if self._last_real_ts is not None and last_real_ts <= self._last_real_ts:
            # Only skip rebuilding the cache if the requested range is
            # already fully covered by the existing cached window. If the
            # requested range extends beyond the cached days we must rebuild
            # (even if no newer API data was found) so that the price map
            # covers the requested timestamps.
            if (
                self._cache_start_day is not None
                and self._cache_end_day is not None
                and requested_start.date() >= self._cache_start_day
                and requested_end.date() <= self._cache_end_day
            ):
                logger.debug(
                    "No new data beyond last_real_ts=%s (fetched last_real_ts=%s) – skipping refresh (requested range already cached)",
                    self._last_real_ts.strftime("%Y-%m-%dT%H:%M"),
                    last_real_ts.strftime("%Y-%m-%dT%H:%M"),
                )
                return
        horizon_end = max(last_real_ts + self._horizon_buffer, fetch_end)

        logger.info(
            "Energy-Charts: %d real data points, last_real_ts=%s, forecast until %s (+%.0f h)",
            len(raw),
            last_real_ts.strftime("%Y-%m-%dT%H:%M"),
            horizon_end.strftime("%Y-%m-%dT%H:%M"),
            self._horizon_buffer.total_seconds() / 3600,
        )
        self._build_price_map(raw, map_start, horizon_end)
        self._last_real_ts = last_real_ts
        self._cache_start_day = requested_start.date()
        self._cache_end_day = requested_end.date()

    def _build_price_map(
        self, raw: list[tuple[datetime, float]], map_start: datetime, horizon_end: datetime
    ) -> None:
        """Populate ``_price_map`` from *raw* API data plus ETS/median forecast.

        The map covers ``[map_start, horizon_end]`` at 15-minute granularity.
        Slots present in *raw* use the real API price; remaining slots are
        filled by :meth:`_forecast`.
        """
        start_bucket = int(map_start.timestamp()) // 900
        end_bucket = int(horizon_end.timestamp()) // 900
        total_slots = end_bucket - start_bucket + 1
        target_buckets = [start_bucket + i for i in range(total_slots)]

        # Separate history (for ETS input) from API-covered future buckets
        history: list[float] = []
        lookup: dict[int, float] = {}
        for dt, price in raw:
            b = int(dt.timestamp()) // 900
            history.append(price)
            if b >= start_bucket:
                lookup[b] = price

        new_map: dict[int, float] = {}
        missing_pos: list[int] = []
        for i, b in enumerate(target_buckets):
            if b in lookup:
                new_map[b] = lookup[b]
            else:
                missing_pos.append(i)

        api_hits = len(new_map)

        if missing_pos and history:
            logger.debug(
                "Forecasting %d/%d future slots via ETS/median", len(missing_pos), total_slots
            )
            forecast = self._forecast(history, len(missing_pos))
            for pos, fc_val in zip(missing_pos, forecast, strict=False):
                new_map[target_buckets[pos]] = fc_val
        elif missing_pos:
            logger.warning(
                "No history for ETS forecast – %d slots will default to 0.0", len(missing_pos)
            )

        # Apply configured charges and VAT to every slot once here so the
        # in-memory price map contains final end-prices.
        charges_wh = self._charges_kwh / 1000.0
        if charges_wh != 0.0 or self._vat_rate != 0.0:
            for k in list(new_map.keys()):
                new_map[k] = (new_map[k] + charges_wh) * (1 + self._vat_rate)

        self._price_map = new_map
        logger.info(
            "Price map ready: %d API slots + %d forecast slots = %.1f h coverage",
            api_hits,
            len(missing_pos),
            total_slots * 0.25,
        )

    # ── public API ────────────────────────────────────────────────────

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        """Return prices in EUR/Wh for each timestamp in *timestamps*.

        Contacts Energy-Charts only when the cache is stale (see module
        docstring); all other calls are served from the in-memory price map
        without any I/O.
        """
        ts_list: list[datetime] = timestamps.to_list()

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        requested_start = _to_utc(ts_list[0])
        requested_end = _to_utc(ts_list[-1])
        now_utc = datetime.now(timezone.utc)

        # Fast path: avoid lock acquisition when no refresh is required
        if self._needs_refresh(now_utc, requested_start, requested_end):
            async with self._lock:
                # Re-check inside the lock – another coroutine may have just
                # finished a refresh while we were waiting
                if self._needs_refresh(now_utc, requested_start, requested_end):
                    await self._refresh(now_utc, requested_start, requested_end)

        # Serve from the price map (pure in-memory, no I/O)
        result: list[float] = []
        cache_misses = 0
        for ts in ts_list:
            b = int(_to_utc(ts).timestamp()) // 900
            val = self._price_map.get(b)
            if val is None:
                cache_misses += 1
                val = 0.0
            result.append(val)

        if cache_misses:
            logger.warning(
                "%d/%d requested timestamps not covered by price map (returning 0.0)",
                cache_misses,
                len(ts_list),
            )

        return pl.Series(result, dtype=pl.Float32)
