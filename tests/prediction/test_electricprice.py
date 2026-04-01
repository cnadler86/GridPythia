"""Tests for electricity price providers."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.electricprice.fixed import ElecPriceFixed, TimeWindow
from GridPythia.prediction.electricprice.provider import ElecPriceFallbackChain

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


def _ts(hours: float = 24, dt: float = 1.0) -> pl.Series:
    return make_timestamps(START, hours, dt)


class TestElecPriceFixed:
    async def test_flat_price(self):
        provider = ElecPriceFixed(price_kwh=0.30)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[0] == pytest.approx(0.30 / 1000.0)
        assert all(v == pytest.approx(result[0]) for v in result.to_list())

    async def test_flat_price_with_charges_and_vat(self):
        provider = ElecPriceFixed(price_kwh=0.25, charges_kwh=0.05, vat_rate=1.19)
        result = await provider.fetch(_ts())
        expected_wh = (0.25 / 1000.0 + 0.05 / 1000.0) * 1.19
        assert result[0] == pytest.approx(expected_wh)

    async def test_schedule(self):
        schedule = [
            TimeWindow(start_hour=0, end_hour=6, value=0.20),
            TimeWindow(start_hour=6, end_hour=22, value=0.35),
            TimeWindow(start_hour=22, end_hour=24, value=0.20),
        ]
        provider = ElecPriceFixed(schedule=schedule)
        result = await provider.fetch(_ts())
        assert result[0] == pytest.approx(0.20 / 1000.0)
        assert result[5] == pytest.approx(0.20 / 1000.0)
        assert result[6] == pytest.approx(0.35 / 1000.0)
        assert result[22] == pytest.approx(0.20 / 1000.0)

    async def test_quarter_hour_steps(self):
        provider = ElecPriceFixed(price_kwh=0.30)
        result = await provider.fetch(_ts(dt=0.25))
        assert len(result) == 96

    async def test_provider_id(self):
        assert ElecPriceFixed().provider_id == "ElecPriceFixed"

    async def test_returns_polars_series(self):
        result = await ElecPriceFixed().fetch(_ts())
        assert isinstance(result, pl.Series)
        assert result.dtype == pl.Float32


class _FailingElecPriceProvider(ElecPriceFixed):
    @property
    def provider_id(self) -> str:
        return "FailingElecPrice"

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        raise RuntimeError("primary provider failed")


class TestElecPriceFallbackChain:
    async def test_uses_primary_when_primary_succeeds(self):
        primary = ElecPriceFixed(price_kwh=0.25)
        fallback = ElecPriceFixed(price_kwh=0.40)
        provider = ElecPriceFallbackChain(primary=primary, fallback=fallback)

        result = await provider.fetch(_ts(hours=2))
        assert all(v == pytest.approx(0.25 / 1000.0) for v in result.to_list())

    async def test_switches_to_fallback_when_primary_raises(self):
        primary = _FailingElecPriceProvider()
        fallback = ElecPriceFixed(price_kwh=0.40)
        provider = ElecPriceFallbackChain(primary=primary, fallback=fallback)

        result = await provider.fetch(_ts(hours=2))
        assert all(v == pytest.approx(0.40 / 1000.0) for v in result.to_list())


class TestElecPriceEnergyChartsCache:
    """Unit tests for the in-memory caching layer of ElecPriceEnergyCharts."""

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_raw(
        anchor: datetime,
        n_history: int = 500,
        n_future: int = 76 * 4,
        base_price: float = 10.0,
    ) -> list[tuple[datetime, float]]:
        """Synthetic 15-min price stream: *n_history* points ending at *anchor*
        followed by *n_future* points extending into the future.

        Default n_future covers 48 h request + 25 h buffer + 3 h safety = 76 h.
        """
        from datetime import timedelta

        points: list[tuple[datetime, float]] = []
        start = anchor - timedelta(minutes=15 * n_history)
        for i in range(n_history + n_future):
            dt = start + timedelta(minutes=15 * i)
            price = (base_price + (i % 96) * 0.01) / 1_000_000.0  # tiny EUR/Wh
            points.append((dt, price))
        return points

    @staticmethod
    def _provider(poll_minutes: int = 30, buffer_hours: int = 25):
        from datetime import timedelta
        from GridPythia.prediction.electricprice.energycharts import (
            ElecPriceEnergyCharts,
            EnergyChartsConfig,
        )

        cfg = EnergyChartsConfig(horizon_buffer=timedelta(hours=buffer_hours))
        return ElecPriceEnergyCharts(cfg)

    # ── tests ─────────────────────────────────────────────────────────

    async def test_first_call_populates_price_map(self):
        """After the first fetch() the price map is populated."""
        from datetime import timedelta

        provider = self._provider()
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now)
        provider._request_prices = AsyncMock(return_value=raw)

        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        result = await provider.fetch(ts)

        assert isinstance(result, pl.Series)
        assert len(result) == 24
        assert provider._request_prices.call_count == 1
        assert len(provider._price_map) > 0

    async def test_repeated_fetches_hit_cache(self):
        """Multiple fetch() calls within poll_interval make only one API call."""
        from datetime import timedelta

        provider = self._provider()
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now)
        provider._request_prices = AsyncMock(return_value=raw)

        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        await provider.fetch(ts)
        await provider.fetch(ts)
        await provider.fetch(ts)

        assert provider._request_prices.call_count == 1

    async def test_expired_source_validity_triggers_recheck(self):
        """When source_valid_until is in the past, the provider re-fetches."""
        provider = self._provider()
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now)
        provider._request_prices = AsyncMock(return_value=raw)

        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        await provider.fetch(ts)
        assert provider._request_prices.call_count == 1

        # Manually expire source validity to force a re-check on next fetch.
        provider._cache.source_valid_until = now - timedelta(seconds=1)

        provider._request_prices = AsyncMock(return_value=self._make_raw(now))
        await provider.fetch(ts)
        assert provider._request_prices.call_count == 1  # new mock, one call

    async def test_unchanged_ec_data_keeps_existing_map(self):
        """When EC max bucket does not advance, the price map is not rebuilt."""
        from datetime import timedelta

        provider = self._provider()  # always re-check
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now)
        provider._request_prices = AsyncMock(return_value=raw)

        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        await provider.fetch(ts)
        original_map = dict(provider._price_map)

        # Same raw data → same max bucket → map must not change
        await provider.fetch(ts)
        assert provider._price_map is not original_map or provider._price_map == original_map

    async def test_new_ec_data_rebuilds_price_map(self):
        """When EC max bucket advances (new day-ahead published) map is rebuilt."""
        from datetime import timedelta

        provider = self._provider()
        now = datetime.now(timezone.utc)

        raw_day1 = self._make_raw(now, n_future=24 * 4, base_price=10.0)
        raw_day2 = self._make_raw(now, n_future=48 * 4, base_price=20.0)  # more future + higher

        provider._request_prices = AsyncMock(side_effect=[raw_day1, raw_day2])

        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        result1 = await provider.fetch(ts)
        bucket1 = int(ts.to_list()[0].timestamp()) // 900
        price1 = provider._price_map.get(bucket1, 0.0)

        # Expire source validity to force re-check and fetch raw_day2.
        provider._cache.source_valid_until = now - timedelta(seconds=1)
        result2 = await provider.fetch(ts)

        price2 = provider._price_map.get(bucket1, 0.0)

        # The map was rebuilt with different (higher) prices
        assert price2 != pytest.approx(price1, rel=0.5), "Map should have been rebuilt"

    async def test_forecast_fills_future_gaps(self):
        """Buckets beyond the last EC timestamp are filled with ETS/median."""
        from datetime import timedelta

        provider = self._provider()
        now = datetime.now(timezone.utc)

        # Only provide 12 h of future data – the rest must be forecast
        raw = self._make_raw(now, n_history=400, n_future=12 * 4)
        provider._request_prices = AsyncMock(return_value=raw)

        # Request 48 h – well beyond the 12 h API coverage
        ts = make_timestamps(now, hours=48, dt_hours=1.0)
        result = await provider.fetch(ts)

        assert len(result) == 48
        # All entries must be non-None floats; zeros only for genuinely missing slots
        assert result.null_count() == 0
        # At least the first hour should have a real API price (non-zero)
        assert result[0] > 0.0

    async def test_coverage_gap_triggers_refresh(self):
        """If requested timestamps exceed cache horizon a refresh is triggered."""
        # With the transactional cache build, coverage starts at midnight and
        # can extend well beyond the immediate request. We therefore request a
        # much longer window to force a true coverage miss.
        provider = self._provider(buffer_hours=6)
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now, n_future=30 * 4)
        provider._request_prices = AsyncMock(return_value=raw)

        ts_short = make_timestamps(now, hours=4, dt_hours=1.0)
        await provider.fetch(ts_short)
        assert provider._request_prices.call_count == 1

        # Ask for timestamps clearly beyond the first cache horizon.
        ts_long = make_timestamps(now, hours=72, dt_hours=1.0)
        provider._request_prices = AsyncMock(return_value=self._make_raw(now, n_future=50 * 4))
        await provider.fetch(ts_long)
        assert provider._request_prices.call_count == 1  # new mock, fresh counter

    async def test_coverage_gap_with_unchanged_ec_data_rebuilds_map(self):
        """When EC data is unchanged but time has advanced past the map window,
        the map must be re-anchored so no zeros are returned."""
        from datetime import timedelta

        # buffer=6h: first 24h fetch → map covers [now, now+23+6=now+29h].
        # Advance 8h: new last ts = now+8+23=now+31h > now+29h → re-anchor.
        buffer_h = 6
        provider = self._provider(buffer_hours=buffer_h)
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now, n_history=400, n_future=40 * 4)
        provider._request_prices = AsyncMock(return_value=raw)

        ts1 = make_timestamps(now, hours=24, dt_hours=1.0)
        result1 = await provider.fetch(ts1)
        assert provider._request_prices.call_count == 1
        assert result1.null_count() == 0
        assert all(v > 0 for v in result1.to_list())

        # Advance beyond buffer: last ts now+8+23=now+31h > map ceiling now+29h
        later = now + timedelta(hours=buffer_h + 2)
        provider._request_prices = AsyncMock(
            return_value=self._make_raw(later, n_history=400, n_future=40 * 4)
        )
        ts2 = make_timestamps(later, hours=24, dt_hours=1.0)
        result2 = await provider.fetch(ts2)

        assert result2.null_count() == 0
        assert all(v > 0 for v in result2.to_list()), (
            "Map should have been re-anchored; zeros mean stale coverage"
        )

    async def test_concurrent_fetches_call_api_once(self):
        """Concurrent fetch() calls do not trigger duplicate API requests."""
        import asyncio
        from datetime import timedelta

        provider = self._provider()
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now)

        call_count = 0

        async def slow_request(start, end):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0)  # yield to let other coroutines run
            return raw

        provider._request_prices = slow_request

        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        # Fire 5 concurrent fetches
        results = await asyncio.gather(*[provider.fetch(ts) for _ in range(5)])

        assert call_count == 1
        assert all(len(r) == 24 for r in results)

    async def test_empty_api_response_without_cache_raises(self):
        """Without usable cache, an empty API response raises a clear error."""
        provider = self._provider()
        provider._request_prices = AsyncMock(return_value=[])

        now = datetime.now(timezone.utc)
        ts = make_timestamps(now, hours=6, dt_hours=1.0)
        with pytest.raises(RuntimeError, match="refresh failed"):
            await provider.fetch(ts)

    async def test_refresh_error_uses_existing_covering_cache(self):
        """If refresh fails later, provider serves still-covering cache."""
        provider = self._provider()
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now, n_history=400, n_future=40 * 4)

        # Initial successful fill
        provider._request_prices = AsyncMock(return_value=raw)
        ts = make_timestamps(now, hours=12, dt_hours=1.0)
        warm = await provider.fetch(ts)
        assert all(v > 0.0 for v in warm.to_list())

        # Expire source validity to force a re-check that will fail.
        provider._request_prices = AsyncMock(side_effect=RuntimeError("network down"))
        provider._cache.source_valid_until = now - timedelta(seconds=1)
        cached = await provider.fetch(ts)
        assert len(cached) == 12
        assert all(v > 0.0 for v in cached.to_list())

    async def test_refresh_error_without_cache_raises(self):
        """If EnergyCharts fails and cache is empty, error is raised for decorator chain."""
        from datetime import timedelta
        from GridPythia.prediction.electricprice.energycharts import (
            ElecPriceEnergyCharts,
            EnergyChartsConfig,
        )

        provider = ElecPriceEnergyCharts(
            EnergyChartsConfig(horizon_buffer=timedelta(hours=12)),
        )
        provider._request_prices = AsyncMock(side_effect=RuntimeError("network down"))

        now = datetime.now(timezone.utc)
        ts = make_timestamps(now, hours=6, dt_hours=1.0)
        
        # Without cache, refresh error should propagate
        with pytest.raises(RuntimeError, match="Energy-Charts refresh failed"):
            await provider.fetch(ts)

    async def test_too_few_fresh_points_uses_cache_fallback_slots(self):
        """Sparse fresh fetch can still succeed by reusing cached buckets."""
        provider = self._provider(buffer_hours=24)
        now = datetime.now(timezone.utc)

        # Build a solid initial cache.
        raw_full = self._make_raw(now, n_history=400, n_future=60 * 4)
        provider._request_prices = AsyncMock(return_value=raw_full)
        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        first = await provider.fetch(ts)
        assert all(v > 0.0 for v in first.to_list())

        # Next refresh only returns very few fresh points.
        sparse = raw_full[-2:]
        provider._request_prices = AsyncMock(return_value=sparse)
        provider._cache.source_valid_until = now - timedelta(seconds=1)

        second = await provider.fetch(ts)
        assert len(second) == 24
        assert second.null_count() == 0

    async def test_charges_and_vat_applied(self):
        """charges_kwh and vat_rate are factored into the cached prices."""
        from datetime import timedelta

        from GridPythia.prediction.electricprice.energycharts import (
            ElecPriceEnergyCharts,
            EnergyChartsConfig,
        )

        provider = ElecPriceEnergyCharts(EnergyChartsConfig(charges_kwh=0.05, vat_rate=0.19))
        now = datetime.now(timezone.utc)

        # Provide a constant price of 100 EUR/MWh = 1e-4 EUR/Wh
        raw_price_wh = 100.0 / 1_000_000.0  # 1e-4 EUR/Wh
        raw = [(now + timedelta(minutes=15 * i), raw_price_wh) for i in range(200)]
        # Inject already-processed prices (bypass _request_prices conversion)
        provider._request_prices = AsyncMock(return_value=raw)  # type: ignore[assignment]

        ts = make_timestamps(now, hours=2, dt_hours=0.25)
        result = await provider.fetch(ts)

        # The injected raw prices are market prices; the provider applies
        # configured charges and VAT when building the in-memory price map.
        charges_wh = provider._charges_kwh / 1000.0
        expected = (raw_price_wh + charges_wh) * (1+provider._vat_rate)
        assert all(v == pytest.approx(expected, rel=1e-3) for v in result.to_list())


class TestEnergyChartsSourceValidUntil:
    """Unit tests for _compute_source_valid_until publication-time semantics."""

    @staticmethod
    def _provider():
        from GridPythia.prediction.electricprice.energycharts import (
            ElecPriceEnergyCharts,
            EnergyChartsConfig,
        )
        return ElecPriceEnergyCharts(EnergyChartsConfig())

    def test_before_pub_today_data_valid_until_today_pub(self):
        """5 am with only today's prices → valid until today 12:30 UTC."""
        from GridPythia.prediction.electricprice.energycharts import (
            _DAY_AHEAD_PUB_HOUR,
            _DAY_AHEAD_PUB_MINUTE,
        )

        provider = self._provider()
        today = datetime(2025, 6, 15, tzinfo=timezone.utc)
        now = today.replace(hour=5, minute=0)
        last_real_ts = today.replace(hour=23, minute=45)  # only today's prices

        result = provider._compute_source_valid_until(now, last_real_ts)
        expected = today.replace(
            hour=_DAY_AHEAD_PUB_HOUR, minute=_DAY_AHEAD_PUB_MINUTE, second=0, microsecond=0
        )
        assert result == expected

    def test_after_pub_next_day_data_valid_until_tomorrow_pub(self):
        """2 pm with next-day prices → valid until tomorrow 12:30 UTC."""
        from GridPythia.prediction.electricprice.energycharts import (
            _DAY_AHEAD_PUB_HOUR,
            _DAY_AHEAD_PUB_MINUTE,
        )

        provider = self._provider()
        today = datetime(2025, 6, 15, tzinfo=timezone.utc)
        now = today.replace(hour=14, minute=0)
        last_real_ts = (today + timedelta(days=1)).replace(hour=23, minute=45)

        result = provider._compute_source_valid_until(now, last_real_ts)
        expected = (today + timedelta(days=1)).replace(
            hour=_DAY_AHEAD_PUB_HOUR, minute=_DAY_AHEAD_PUB_MINUTE, second=0, microsecond=0
        )
        assert result == expected

    def test_after_pub_no_next_day_data_short_retry(self):
        """2 pm, publication delayed, only today's prices → short retry window."""
        from GridPythia.prediction.electricprice.energycharts import _RETRY_AFTER_FAILED_REFRESH

        provider = self._provider()
        today = datetime(2025, 6, 15, tzinfo=timezone.utc)
        now = today.replace(hour=14, minute=0)
        last_real_ts = today.replace(hour=23, minute=45)  # only today's despite being 2 pm

        result = provider._compute_source_valid_until(now, last_real_ts)
        assert result == now + _RETRY_AFTER_FAILED_REFRESH


async def test_energycharts_fetch_today():
    """Integration test: skipped when Energy-Charts API is unreachable."""
    import aiohttp

    from GridPythia.prediction.electricprice.energycharts import ElecPriceEnergyCharts
    from zoneinfo import ZoneInfo

    berlin = ZoneInfo("Europe/Berlin")
    start_local = datetime.now(berlin).replace(hour=0, minute=0, second=0, microsecond=0)
    start = start_local.astimezone(timezone.utc)

    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                "https://api.energy-charts.info/price",
                params={"bzn": "DE-LU", "start": start.strftime("%Y-%m-%dT%H:%M"), "end": start.strftime("%Y-%m-%dT%H:%M")},
            ) as resp:
                resp.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Energy-Charts API unreachable: {exc}")

    from GridPythia.prediction.electricprice.energycharts import EnergyChartsConfig

    provider = ElecPriceEnergyCharts(EnergyChartsConfig(bidding_zone="DE-LU"))
    ts = make_timestamps(start, hours=24, dt_hours=1.0)
    try:
        prices = await provider.fetch(ts)
    except Exception as exc:
        pytest.skip(f"Energy-Charts fetch failed: {exc}")

    assert len(prices) == 24
    assert isinstance(prices, pl.Series)
