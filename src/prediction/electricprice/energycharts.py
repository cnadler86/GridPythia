"""Energy-Charts electricity price provider.

Caching strategy
----------------
Day-ahead prices change at most once per day (Energy-Charts publishes the
next day's prices around 14:00 CET, though the exact time can vary).
Calling ``fetch()`` every 5–15 minutes without caching would therefore redo
expensive ETS fitting and hit the external API far more than necessary.

The provider maintains an internal *price map* – a ``dict[int, float]``
mapping 15-minute unix buckets to EUR/Wh prices.  The map always covers
``[ts_first, ts_last + horizon_buffer]`` (default buffer 25 h) where
``ts_first``/``ts_last`` are the first and last timestamps of the current
``fetch()`` call.  A 25 h buffer beyond the last requested timestamp means
an hour-by-hour consumer can run for ~25 h without any re-anchor.
Only after the buffer is exhausted (or ``poll_interval`` expires) does the
provider contact the API again.

Map invalidation rules (checked on every ``fetch()``):

1. Map is empty (first call or after reset).
2. ``poll_interval`` has elapsed since the last Energy-Charts API check –
   we re-poll to detect a new day-ahead publication.  If the API's maximum
   timestamp has *not* advanced the existing map is kept as-is.
3. The latest requested timestamp falls outside the current map coverage.

When new Energy-Charts data is detected the map is fully rebuilt:
real API prices are used for all available buckets; remaining future
buckets are filled by ETS (``statsmodels``, optional) or a median fallback.

Concurrency
-----------
An ``asyncio.Lock`` serialises rebuilds so that concurrent ``fetch()``
callers do not each trigger a separate API round-trip.  A double-checked
pattern avoids unnecessary lock contention on the hot path.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import aiohttp
import polars as pl

from src.prediction.electricprice.provider import ElecPriceProvider

logger = logging.getLogger(__name__)

_HISTORY_WINDOW = timedelta(days=35)
_POLL_INTERVAL_DEFAULT = timedelta(minutes=30)
_HORIZON_BUFFER_DEFAULT = timedelta(hours=25)


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
        VAT multiplier applied *after* adding charges (default ``1.19``).
    poll_interval:
        How often to check Energy-Charts for newly published prices.
    horizon_buffer:
        Extra time added *beyond the last requested timestamp* when
        pre-computing the price map.  Defaults to 25 h so that a 48 h
        fetch window can slide forward ~25 h before a re-anchor is needed.
    """

    def __init__(
        self,
        bidding_zone: str = "DE-LU",
        charges_kwh: float = 0.0,
        vat_rate: float = 1.19,
        poll_interval: timedelta = _POLL_INTERVAL_DEFAULT,
        horizon_buffer: timedelta = _HORIZON_BUFFER_DEFAULT,
    ) -> None:
        self._bidding_zone = bidding_zone
        self._charges_kwh = charges_kwh
        self._vat_rate = vat_rate
        self._poll_interval = poll_interval
        self._horizon_buffer = horizon_buffer

        # ── cache state ───────────────────────────────────────────────
        # bucket = unix_timestamp // 900  (15-min granularity)
        self._price_map: dict[int, float] = {}
        # Highest Energy-Charts bucket seen; advances only when EC publishes
        # new day-ahead data – used to detect stale-vs-fresh API responses.
        self._ec_max_bucket: int | None = None
        self._last_api_check: datetime | None = None
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
        except Exception:  # noqa: S110
            pass
        from statistics import median

        med = median(capped) if capped else 0.0
        return [med] * steps

    # ── cache management ──────────────────────────────────────────────

    def _needs_refresh(self, now_utc: datetime, max_bucket_needed: int) -> bool:
        """Return True when the price map must be rebuilt before serving."""
        if not self._price_map:
            return True
        if self._last_api_check is None or (now_utc - self._last_api_check) >= self._poll_interval:
            return True
        if max_bucket_needed not in self._price_map:
            logger.debug(
                "Cache miss: bucket %d not in price map – forcing refresh", max_bucket_needed
            )
            return True
        return False

    async def _refresh(self, now_utc: datetime, max_bucket_needed: int, end_utc: datetime) -> None:
        """Poll Energy-Charts and rebuild the price map when new data is found."""
        hist_start = now_utc - _HISTORY_WINDOW
        horizon_end = end_utc + self._horizon_buffer

        logger.debug(
            "Polling Energy-Charts (bzn=%s, window=[%s, %s UTC])",
            self._bidding_zone,
            hist_start.strftime("%Y-%m-%dT%H:%M"),
            horizon_end.strftime("%Y-%m-%dT%H:%M"),
        )
        raw = await self._request_prices(hist_start, horizon_end)
        self._last_api_check = now_utc

        if not raw:
            logger.warning("Energy-Charts returned no data for bzn=%s", self._bidding_zone)
            return

        new_max_bucket = max(int(dt.timestamp()) // 900 for dt, _ in raw)

        if (
            new_max_bucket == self._ec_max_bucket
            and self._price_map
            and max_bucket_needed in self._price_map
        ):
            logger.debug(
                "Energy-Charts data unchanged (ec_max_bucket=%d) – price map kept",
                new_max_bucket,
            )
            return

        if new_max_bucket == self._ec_max_bucket and self._price_map:
            logger.debug(
                "Energy-Charts data unchanged (ec_max_bucket=%d) but coverage gap "
                "(bucket %d missing) – re-anchoring price map to new now_utc",
                new_max_bucket,
                max_bucket_needed,
            )

        logger.info(
            "New Energy-Charts data detected (ec_max_bucket %s → %d, %d raw points) "
            "– rebuilding price map [now, last_ts + %.0f h]",
            self._ec_max_bucket,
            new_max_bucket,
            len(raw),
            self._horizon_buffer.total_seconds() / 3600,
        )
        self._ec_max_bucket = new_max_bucket
        self._build_price_map(raw, now_utc, horizon_end)

    def _build_price_map(
        self, raw: list[tuple[datetime, float]], now_utc: datetime, horizon_end: datetime
    ) -> None:
        """Populate ``_price_map`` from *raw* API data plus ETS/median forecast.

        The map covers ``[now_utc, horizon_end]`` at 15-minute granularity where
        ``horizon_end = ts_last + horizon_buffer``.  Slots present in *raw* are
        taken directly; remaining future slots are filled by :meth:`_forecast`.
        """
        start_bucket = int(now_utc.timestamp()) // 900
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
            for pos, fc_val in zip(missing_pos, forecast):
                new_map[target_buckets[pos]] = fc_val
        elif missing_pos:
            logger.warning(
                "No history for ETS forecast – %d slots will default to 0.0", len(missing_pos)
            )

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

        The first call (or after the poll interval expires) contacts
        Energy-Charts; subsequent calls within the poll window are served
        entirely from the in-memory price map without any I/O.
        """
        ts_list: list[datetime] = timestamps.to_list()

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        # First timestamp drives the cache anchor; last timestamp determines
        # how far forward the map must reach (+ horizon_buffer).
        now_utc = _to_utc(ts_list[0])
        end_utc = _to_utc(ts_list[-1])
        max_bucket_needed = int(end_utc.timestamp()) // 900

        # Fast path: avoid lock acquisition when no refresh is required
        if self._needs_refresh(now_utc, max_bucket_needed):
            async with self._lock:
                # Re-check inside the lock – another coroutine may have just
                # finished a refresh while we were waiting
                if self._needs_refresh(now_utc, max_bucket_needed):
                    await self._refresh(now_utc, max_bucket_needed, end_utc)

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
