"""EPEX day-ahead predictor electricity price provider.

Uses the EPEXPredictor ML API (https://epexpredictor.batzill.com) which
provides day-ahead price predictions powered by machine learning.

Unlike EnergyCharts, this API already extends prices beyond the last real
data point using an ML model, so no local ETS forecasting is needed.

Cache validity logic:
  Day-ahead prices are published around 12:00 UTC for the following day.

  * If ``known_until`` (last real API data point) covers tomorrow, the cache
    is valid until tomorrow 12:00 UTC (then fresher data supersedes ours).
  * If we only have today's real data and it is already past 12:00 UTC,
    publication was delayed → set a short retry window (15 min) so we
    re-check soon without hammering the API.
  * Before 12:00 UTC without tomorrow's real data → valid until 12:00 UTC.

Error handling is identical to EnergyCharts: errors propagate so that
:class:`ElecPriceFallbackChain` can fall through to an alternative provider.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import aiohttp
import numpy as np
from pydantic import BaseModel, Field, field_validator
from structlog import get_logger

from GridPythia.prediction.cache import TimeBucketCache, to_utc
from GridPythia.prediction.electricprice.provider import ElecPriceProvider

logger = get_logger(__name__)

_HORIZON_BUFFER_DEFAULT = timedelta(hours=25)
# Small lookback window so the cache covers recent optimization slots.
_LOOKBACK = timedelta(hours=2)

# Day-ahead prices are published around 12:00 UTC for the next delivery day.
_DAY_AHEAD_PUB_HOUR = 12
_DAY_AHEAD_PUB_MINUTE = 0
_RETRY_AFTER_FAILED_REFRESH = timedelta(minutes=15)

_BASE_URL = "https://epexpredictor.batzill.com"

# The EPEXPredictor API returns 15-minute prices by default (hourly=false).
_BUCKET_SECONDS = 900


class EpexPredictorConfig(BaseModel):
    """Pydantic config for :class:`ElecPriceEpexPredictor`."""

    model_config = {"frozen": True}

    region: str = Field("DE", min_length=1)
    charges_kwh: float = Field(0.0, ge=0.0)
    vat_rate: float = Field(0.19, ge=0.0)
    horizon_buffer: timedelta = Field(default=_HORIZON_BUFFER_DEFAULT)
    base_url: str = Field(default=_BASE_URL)

    @field_validator("horizon_buffer", mode="before")
    @classmethod
    def _ensure_timedelta(cls, v: object) -> timedelta:
        if isinstance(v, (int, float)):
            return timedelta(hours=float(v))
        if isinstance(v, timedelta):
            return v
        raise TypeError(f"horizon_buffer must be timedelta or hours number, got {type(v).__name__}")


class ElecPriceEpexPredictor(ElecPriceProvider):
    """Fetch electricity price predictions from the EPEXPredictor ML API.

    The API always provides a full forecast (real prices up to ``knownUntil``,
    ML-predicted prices beyond).  Charges and VAT are applied locally so the
    raw API values are stored in the cache for internal consistency.

    Parameters
    ----------
    config:
        :class:`EpexPredictorConfig` instance or mapping.
    """

    def __init__(self, config: EpexPredictorConfig | dict) -> None:
        if not isinstance(config, EpexPredictorConfig):
            config = EpexPredictorConfig(**config)  # type: ignore[arg-type]

        self._region = config.region
        self._charges_kwh = config.charges_kwh
        self._vat_rate = config.vat_rate
        self._horizon_buffer = config.horizon_buffer
        self._base_url = config.base_url.rstrip("/")

        # 15-minute buckets matching the API's native resolution.
        self._cache = TimeBucketCache(bucket_seconds=_BUCKET_SECONDS)
        self._known_until: datetime | None = None
        self._lock = asyncio.Lock()

    # ── ElecPriceProvider interface ───────────────────────────────────

    @property
    def provider_id(self) -> str:
        return "EpexPredictor"

    @property
    def last_real_ts(self) -> datetime | None:
        """Boundary between real EPEX data and ML-predicted values.

        Returns ``None`` before the first successful fetch.
        Equivalent to ``last_real_ts`` in :class:`ElecPriceEnergyCharts`.
        """
        return self._known_until

    # ── API request ───────────────────────────────────────────────────

    async def _request_prices(
        self,
        start_ts: datetime,
        hours: int,
    ) -> tuple[list[tuple[datetime, float]], datetime]:
        """Call ``GET /prices`` and return ``(prices, known_until)``.

        Prices are returned as raw EUR/Wh values (before charges/VAT).

        Args:
            start_ts: Window start (UTC-aware).
            hours:    Number of hours to fetch.

        Returns:
            ``([(datetime, EUR/Wh)], known_until_utc)``

        Raises:
            :exc:`aiohttp.ClientResponseError`: on HTTP errors.
            :exc:`ValueError`: if the response is missing required fields.
        """
        url = f"{self._base_url}/prices"
        params: dict[str, object] = {
            "region": self._region,
            # ISO 8601 UTC; API accepts date-time strings.
            "startTs": start_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": hours,
            "unit": "EUR_PER_MWH",  # we convert to EUR/Wh locally
            # hourly omitted → default false → native 15-minute resolution
            "timezone": "UTC",  # always use UTC for consistent bucket arithmetic
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

        raw_prices: list[dict] = data.get("prices", [])
        known_until_str: str | None = data.get("knownUntil")

        if not known_until_str:
            raise ValueError("EPEXPredictor response missing 'knownUntil' field.")

        known_until = datetime.fromisoformat(known_until_str.replace("Z", "+00:00"))
        if known_until.tzinfo is None:
            known_until = known_until.replace(tzinfo=timezone.utc)
        known_until = to_utc(known_until)

        result: list[tuple[datetime, float]] = []
        for entry in raw_prices:
            starts_at_str: str | None = entry.get("startsAt")
            total: float | None = entry.get("total")
            if starts_at_str is None or total is None:
                continue
            dt = datetime.fromisoformat(starts_at_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            # EUR/MWh → EUR/Wh (raw, before charges/VAT)
            result.append((to_utc(dt), float(total) / 1_000_000.0))

        if not result:
            raise ValueError("EPEXPredictor returned empty prices array.")

        return result, known_until

    # ── Cache validity ────────────────────────────────────────────────

    def _compute_source_valid_until(self, now_utc: datetime, known_until: datetime) -> datetime:
        """Compute how long the newly fetched cache is valid.

        Mirrors the EnergyCharts publication-window logic but with the
        EPEX publication time of ~12:00 UTC.
        """
        tomorrow = (now_utc + timedelta(days=1)).date()
        tomorrow_start = datetime(
            tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, tzinfo=timezone.utc
        )
        tomorrow_pub = datetime(
            tomorrow.year,
            tomorrow.month,
            tomorrow.day,
            _DAY_AHEAD_PUB_HOUR,
            _DAY_AHEAD_PUB_MINUTE,
            tzinfo=timezone.utc,
        )

        if known_until >= tomorrow_start:
            # We have next-day real prices → valid until tomorrow's publication.
            return tomorrow_pub

        # Only today's (or earlier) real prices available.
        today_pub = now_utc.replace(
            hour=_DAY_AHEAD_PUB_HOUR,
            minute=_DAY_AHEAD_PUB_MINUTE,
            second=0,
            microsecond=0,
        )
        if now_utc >= today_pub:
            # Past today's publication window but still no next-day real prices →
            # delayed publication; schedule a short retry.
            return now_utc + _RETRY_AFTER_FAILED_REFRESH

        # Before today's publication → valid until then.
        return today_pub

    def _needs_refresh(
        self,
        now_utc: datetime,
        requested_start: datetime,
        requested_end: datetime,
    ) -> bool:
        if not self._cache.has_data():
            return True
        if not self._cache.covers(requested_start, requested_end):
            return True
        if self._cache.source_valid_until is not None and now_utc >= self._cache.source_valid_until:
            logger.debug(
                "epex_cache_stale",
                now_utc=now_utc.isoformat(),
                source_valid_until=self._cache.source_valid_until.isoformat(),
            )
            return True
        return False

    async def _refresh(
        self,
        now_utc: datetime,
        requested_start: datetime,
        requested_end: datetime,
    ) -> None:
        """Fetch fresh data and transactionally rebuild the cache."""
        # Fetch from a small lookback so recent timestamps are always covered.
        fetch_start = min(requested_start, now_utc) - _LOOKBACK
        # Request enough hours to cover requested_end + horizon_buffer.
        fetch_end = requested_end + self._horizon_buffer
        total_hours = max(1, int((fetch_end - fetch_start).total_seconds() / 3600) + 2)

        logger.debug(
            "epex_predictor_polling",
            region=self._region,
            fetch_start=fetch_start.strftime("%Y-%m-%dT%H:%M"),
            fetch_end=fetch_end.strftime("%Y-%m-%dT%H:%M"),
            total_hours=total_hours,
        )

        raw, known_until = await self._request_prices(fetch_start, total_hours)

        map_start = min(to_utc(dt) for dt, _ in raw)
        last_api_ts = max(to_utc(dt) for dt, _ in raw)

        if map_start > requested_start:
            raise ValueError(
                f"API data starts at {map_start.isoformat()} but "
                f"requested range starts at {requested_start.isoformat()}."
            )

        charges_wh = self._charges_kwh / 1000.0
        new_map: dict[int, float] = {}
        for dt, raw_price in raw:
            b = int(dt.timestamp()) // _BUCKET_SECONDS
            adjusted = (raw_price + charges_wh) * (1.0 + self._vat_rate)
            new_map[b] = adjusted

        # Coverage end: last 15-min slot covered
        # (last_api_ts starts the last slot, which covers [last_api_ts, last_api_ts + 15min)).
        coverage_end = last_api_ts + timedelta(minutes=15) - timedelta(seconds=1)

        source_valid_until = self._compute_source_valid_until(now_utc, known_until)
        self._cache.update(
            values=new_map,
            coverage_start=map_start,
            coverage_end=coverage_end,
            source_valid_until=source_valid_until,
        )
        self._known_until = known_until

        logger.info(
            "epex_predictor_refresh_complete",
            region=self._region,
            data_points=len(raw),
            known_until=known_until.strftime("%Y-%m-%dT%H:%M"),
            source_valid_until=source_valid_until.strftime("%Y-%m-%dT%H:%M"),
            coverage_start=self._cache.coverage_start.strftime("%Y-%m-%dT%H:%M")
            if self._cache.coverage_start
            else None,
            coverage_end=self._cache.coverage_end.strftime("%Y-%m-%dT%H:%M")
            if self._cache.coverage_end
            else None,
        )

    # ── Public API ────────────────────────────────────────────────────

    async def fetch(self, timestamps: list) -> np.ndarray:
        """Return prices in EUR/Wh for each timestamp in *timestamps*.

        Contacts the EPEXPredictor API only when the cache is stale; all
        other calls are served from the in-memory price map without I/O.
        """
        requested_start = to_utc(timestamps[0])
        requested_end = to_utc(timestamps[-1])
        now_utc = datetime.now(timezone.utc)

        if self._needs_refresh(now_utc, requested_start, requested_end):
            async with self._lock:
                # Double-checked locking: another coroutine may have refreshed while
                # we were waiting for the lock.
                if self._needs_refresh(now_utc, requested_start, requested_end):
                    try:
                        await self._refresh(now_utc, requested_start, requested_end)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "epex_predictor_refresh_failed_cache_fallback",
                            error=str(exc),
                            cache_covers_request=self._cache.covers(requested_start, requested_end),
                        )
                        if not self._cache.covers(requested_start, requested_end):
                            raise RuntimeError(
                                "EPEXPredictor refresh failed and cache does not cover request."
                            ) from exc
                        # Cache still covers; short retry to avoid hammering the API.
                        retry_at = now_utc + _RETRY_AFTER_FAILED_REFRESH
                        if (
                            self._cache.source_valid_until is None
                            or retry_at < self._cache.source_valid_until
                        ):
                            self._cache.source_valid_until = retry_at

        if not self._cache.covers(requested_start, requested_end):
            raise RuntimeError("EPEXPredictor cache does not cover requested timestamps.")

        result: list[float] = []
        cache_misses = 0
        for ts in timestamps:
            val = self._cache.value_at(ts)
            if val is None:
                cache_misses += 1
                val = 0.0
            result.append(val)

        if cache_misses:
            logger.warning("epex_predictor_cache_misses", count=cache_misses)

        return np.asarray(result, dtype=np.float32)
