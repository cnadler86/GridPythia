"""Energy-Charts electricity price provider.

This provider uses a generic time-bucket cache with source-validity metadata.
Real API values carry a validity timestamp (the newest fetched real timestamp).
Any synthesized extension (forecast) inherits that same source validity timestamp.

Refresh is transactional:

1. Fetch raw API values.
2. Build a candidate map that covers the requested window plus horizon buffer.
3. Commit cache only when step 2 succeeded.

If a refresh fails:
  - If the previous cache still covers the requested range, it is served.
  - Otherwise, an error is raised for the decorator chain to handle fallback.

Errors are propagated to allow decorator-based fallback chains (e.g.,
FallbackProvider) to switch to alternative providers.
"""

import asyncio
from datetime import date, datetime, timedelta, timezone

import aiohttp
import numpy as np
from pydantic import BaseModel, Field, field_validator
from structlog import get_logger

from GridPythia.prediction.cache import TimeBucketCache, to_utc
from GridPythia.prediction.electricprice.provider import ElecPriceProvider

logger = get_logger(__name__)

_HISTORY_WINDOW = timedelta(days=15)
_HORIZON_BUFFER_DEFAULT = timedelta(hours=25)

# Day-ahead prices are published once per day around 12:30–12:45 UTC.
_DAY_AHEAD_PUB_HOUR = 12
_DAY_AHEAD_PUB_MINUTE = 30  # check from 12:30; publication usually at ~12:45
_RETRY_AFTER_FAILED_REFRESH = timedelta(minutes=15)


class EnergyChartsConfig(BaseModel):
    """Pydantic config model for ElecPriceEnergyCharts."""

    model_config = {"frozen": True}

    bidding_zone: str = Field("DE-LU", min_length=1)
    charges_kwh: float = Field(0.0, ge=0.0)
    vat_rate: float = Field(0.19, ge=0.0)
    horizon_buffer: timedelta = Field(default=_HORIZON_BUFFER_DEFAULT)

    @field_validator("horizon_buffer", mode="before")
    def _ensure_timedelta(cls, v):
        if isinstance(v, (int, float)):
            return timedelta(hours=float(v))
        return v


class ElecPriceEnergyCharts(ElecPriceProvider):
    """Fetch day-ahead electricity prices from the Energy-Charts API.

    Prices beyond the last available API timestamp are extended using
    Exponential Smoothing (requires ``statsmodels``, optional).

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

        self._cache = TimeBucketCache(bucket_seconds=900)
        # Compatibility aliases used by existing tests.
        self._price_map: dict[int, float] = self._cache.values
        self._last_real_ts: datetime | None = None
        self._cache_start_day: date | None = None
        self._cache_end_day: date | None = None
        self._min_history_points = 8
        self._lock = asyncio.Lock()

    @property
    def provider_id(self) -> str:
        return "EnergyCharts"

    @property
    def last_real_ts(self) -> "datetime | None":
        """Timestamp of the last real API data point.

        Values *after* this timestamp were synthesised by the statistical
        model (ETS / median fallback).  Use this to shade the forecast
        region in plots.  ``None`` before the first successful fetch.
        """
        return self._last_real_ts

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
                logger.debug("ets_forecast_selected", seasonal_periods=sp, history_len=len(history))
                model = ExponentialSmoothing(history, seasonal="add", seasonal_periods=sp).fit(
                    optimized=True
                )
                # Ensure forecasts are non-negative (floor at 0.0)
                return [max(0.0, float(v)) for v in model.forecast(steps)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("ets_forecast_failed_using_median", error=str(exc))
        from statistics import median

        med = median(history) if history else 0.0
        # Floor median fallback at 0.0 as well
        return [max(0.0, med)] * steps

    # ── cache management ──────────────────────────────────────────────

    def _next_pub_time(self, now_utc: datetime) -> datetime:
        """Return the next expected day-ahead publication time (rolling 12:30 UTC)."""
        today_pub = now_utc.replace(
            hour=_DAY_AHEAD_PUB_HOUR, minute=_DAY_AHEAD_PUB_MINUTE, second=0, microsecond=0
        )
        if now_utc < today_pub:
            return today_pub
        return today_pub + timedelta(days=1)

    def _compute_source_valid_until(self, now_utc: datetime, last_real_ts: datetime) -> datetime:
        """Return until-when the newly fetched data is valid.

        Day-ahead prices for the next delivery day are published around 12:30–12:45 UTC.

        * If *last_real_ts* is **beyond tomorrow's** publication time we already have
          next-day prices.  The cache is valid until that deadline (tomorrow 12:30 UTC)
          because that is when fresher prices will supersede ours.
        * If we only have today's prices and we are **before** today's publication time,
          the cache is valid until today's publication time (we re-check then).
        * If we only have today's prices but we are already **past** today's publication
          time, the publication is delayed; we set a short retry window so we check
          again soon instead of waiting until tomorrow.
        """
        tomorrow = (now_utc + timedelta(days=1)).date()
        tomorrow_pub = datetime(
            tomorrow.year,
            tomorrow.month,
            tomorrow.day,
            _DAY_AHEAD_PUB_HOUR,
            _DAY_AHEAD_PUB_MINUTE,
            tzinfo=timezone.utc,
        )
        if last_real_ts > tomorrow_pub:
            # We have next-day prices → valid until tomorrow's publication.
            return tomorrow_pub

        # Only today's (or earlier) prices.
        today_pub = now_utc.replace(
            hour=_DAY_AHEAD_PUB_HOUR, minute=_DAY_AHEAD_PUB_MINUTE, second=0, microsecond=0
        )
        if now_utc >= today_pub:
            # Past today's publication window but still no next-day prices →
            # publication is delayed; schedule a short retry.
            return now_utc + _RETRY_AFTER_FAILED_REFRESH

        # Before today's publication time → valid until then.
        return today_pub

    def _needs_refresh(
        self, now_utc: datetime, requested_start: datetime, requested_end: datetime
    ) -> bool:
        """Return True when cache should be refreshed before serving."""
        if not self._cache.has_data():
            return True
        if not self._cache.covers(requested_start, requested_end):
            return True
        if self._cache.source_valid_until is not None and now_utc >= self._cache.source_valid_until:
            logger.debug(
                "cache_stale",
                now_utc=now_utc.isoformat(),
                source_valid_until=self._cache.source_valid_until.isoformat(),
            )
            return True
        return False

    def _build_price_map(
        self,
        raw: list[tuple[datetime, float]],
        map_start: datetime,
        horizon_end: datetime,
        fallback_map: dict[int, float],
    ) -> tuple[dict[int, float], int, int, int]:
        """Build a complete 15-minute map for ``[map_start, horizon_end]``.

        Missing slots are filled in this order:

        1. Existing cache values (same bucket).
        2. Forecast from fresh history values.

        Raises:
            ValueError: If resulting map cannot fully cover the target range.
        """
        start_bucket = int(map_start.timestamp()) // 900
        end_bucket = int(horizon_end.timestamp()) // 900
        target_buckets = list(range(start_bucket, end_bucket + 1))

        lookup: dict[int, float] = {}
        history: list[float] = []
        for dt, price in sorted(raw, key=lambda item: item[0]):
            b = int(to_utc(dt).timestamp()) // 900
            history.append(price)
            if b >= start_bucket:
                lookup[b] = price

        new_map: dict[int, float] = {}
        missing_buckets: list[int] = []
        fallback_hits = 0

        for b in target_buckets:
            if b in lookup:
                new_map[b] = lookup[b]
                continue
            fallback_val = fallback_map.get(b)
            if fallback_val is not None:
                new_map[b] = fallback_val
                fallback_hits += 1
                continue
            missing_buckets.append(b)

        if missing_buckets:
            if len(history) < self._min_history_points:
                raise ValueError(
                    f"Insufficient fresh history for forecast: {len(history)} points "
                    f"(< {self._min_history_points})"
                )
            logger.debug(
                "energy_charts_forecasting_slots",
                missing_slots=len(missing_buckets),
                total_slots=len(target_buckets),
            )
            forecast = self._forecast(history, len(missing_buckets))
            for b, fc_val in zip(missing_buckets, forecast, strict=False):
                new_map[b] = fc_val

        if len(new_map) != len(target_buckets):
            raise ValueError("Could not build a fully covered price map.")

        charges_wh = self._charges_kwh / 1000.0
        if charges_wh != 0.0 or self._vat_rate != 0.0:
            for k in list(new_map.keys()):
                new_map[k] = (new_map[k] + charges_wh) * (1 + self._vat_rate)

        api_hits = sum(1 for b in target_buckets if b in lookup)
        forecast_hits = len(missing_buckets)
        return new_map, api_hits, forecast_hits, fallback_hits

    async def _refresh(
        self, now_utc: datetime, requested_start: datetime, requested_end: datetime
    ) -> None:
        """Fetch fresh data and transactionally rebuild cache.

        Fetch window logic:
        - Expand requested window with history for forecast context and current-day context.
        - Replace cache entirely with new data (no fallback to old cache values).
        - If fetch does not cover the requested range, raise to let fallbacks handle it.
        """
        # Expand fetch window: history for forecast context, horizon for forecasted slots
        fetch_start = requested_start - _HISTORY_WINDOW
        fetch_end = requested_end + self._horizon_buffer

        logger.debug(
            "energy_charts_polling",
            bidding_zone=self._bidding_zone,
            fetch_start=fetch_start.strftime("%Y-%m-%dT%H:%M"),
            fetch_end=fetch_end.strftime("%Y-%m-%dT%H:%M"),
            requested_start=requested_start.strftime("%Y-%m-%dT%H:%M"),
            requested_end=requested_end.strftime("%Y-%m-%dT%H:%M"),
        )
        raw = await self._request_prices(fetch_start, fetch_end)

        if not raw:
            raise ValueError("Energy-Charts returned no usable data.")

        last_real_ts = max(to_utc(dt) for dt, _ in raw)

        # Determine coverage bounds from fetched data
        map_start = min(to_utc(dt) for dt, _ in raw)
        map_end = max(to_utc(dt) for dt, _ in raw)

        # Horizon extends to ensure we have buffer for forecasts
        horizon_end = max(map_end, requested_end) + self._horizon_buffer

        # Build price map without fallback to old cache (complete replacement)
        new_map, api_hits, forecast_hits, fallback_hits = self._build_price_map(
            raw=raw,
            map_start=map_start,
            horizon_end=horizon_end,
            fallback_map={},  # No fallback to old cache; replace entirely
        )

        # Verify that the new cache covers at least the start of the requested range
        # (remaining gaps into the future will be filled by forecast)
        if map_start > requested_start:
            raise ValueError(
                f"Fetched data starts at {map_start} but requested range starts at "
                f"{requested_start}; cannot cover historical gap."
            )

        source_valid_until = self._compute_source_valid_until(now_utc, last_real_ts)
        self._cache.update(
            values=new_map,
            coverage_start=map_start,
            coverage_end=horizon_end,
            source_valid_until=source_valid_until,
        )

        # Compatibility aliases used by existing tests.
        self._price_map = self._cache.values
        self._last_real_ts = last_real_ts
        self._cache_start_day = (
            self._cache.coverage_start.date() if self._cache.coverage_start else None
        )
        self._cache_end_day = self._cache.coverage_end.date() if self._cache.coverage_end else None

        logger.info(
            "energy_charts_refresh_complete",
            real_data_points=len(raw),
            source_valid_until=source_valid_until.strftime("%Y-%m-%dT%H:%M"),
            last_real_api_ts=last_real_ts.strftime("%Y-%m-%dT%H:%M"),
            coverage_start=self._cache.coverage_start.strftime("%Y-%m-%dT%H:%M")
            if self._cache.coverage_start
            else None,
            coverage_end=self._cache.coverage_end.strftime("%Y-%m-%dT%H:%M")
            if self._cache.coverage_end
            else None,
            api_slots=api_hits,
            forecast_slots=forecast_hits,
            fallback_slots=fallback_hits,
            horizon_buffer_h=self._horizon_buffer.total_seconds() / 3600,
        )

    # ── public API ────────────────────────────────────────────────────

    async def fetch(self, timestamps: list) -> np.ndarray:
        """Return prices in EUR/Wh for each timestamp in *timestamps*.

        Contacts Energy-Charts only when the cache is stale (see module
        docstring); all other calls are served from the in-memory price map
        without any I/O.
        """
        ts_list = timestamps

        requested_start = to_utc(ts_list[0])
        requested_end = to_utc(ts_list[-1])
        now_utc = datetime.now(timezone.utc)

        if self._needs_refresh(now_utc, requested_start, requested_end):
            async with self._lock:
                if self._needs_refresh(now_utc, requested_start, requested_end):
                    try:
                        await self._refresh(now_utc, requested_start, requested_end)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "energy_charts_refresh_failed_cache_fallback",
                            error=str(exc),
                            cache_covers_request=self._cache.covers(requested_start, requested_end),
                        )
                        if not self._cache.covers(requested_start, requested_end):
                            raise RuntimeError(
                                "Energy-Charts refresh failed and cache does not cover request."
                            ) from exc
                        # Cache still covers; schedule a short retry to avoid
                        # hammering the API while the publication is delayed.
                        retry_at = now_utc + _RETRY_AFTER_FAILED_REFRESH
                        if (
                            self._cache.source_valid_until is None
                            or retry_at < self._cache.source_valid_until
                        ):
                            self._cache.source_valid_until = retry_at

        if not self._cache.covers(requested_start, requested_end):
            raise RuntimeError("Energy-Charts cache does not cover requested timestamps.")

        result: list[float] = []
        cache_misses = 0
        for ts in ts_list:
            val = self._cache.value_at(ts)
            if val is None:
                cache_misses += 1
                val = 0.0
            result.append(val)

        if cache_misses:
            logger.warning(
                "energy_charts_cache_misses",
                cache_misses=cache_misses,
                total_timestamps=len(ts_list),
            )

        return np.array(result, dtype=np.float32)

    def plot(self, values: np.ndarray, timestamps: list) -> "go.Figure":
        """Return a Plotly figure for *values*, highlighting the forecast region.

        The forecast region (timestamps after :attr:`last_real_ts`) is shaded
        with a light pastel background.

        Args:
            values:     EUR/Wh array returned by :meth:`fetch`.
            timestamps: The same timestamp list passed to :meth:`fetch`.
        """
        import plotly.graph_objects as go  # noqa: F401 – local import keeps plotly optional

        from GridPythia.prediction.plots.electricprice import ElecPricePlotter

        return ElecPricePlotter().plot(
            values,
            list(timestamps),
            forecast_from=self._last_real_ts,
            title=f"Electricity Price – {self._bidding_zone}",
        )
