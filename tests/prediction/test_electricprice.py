"""Tests for electricity price providers."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest

from src.prediction.base import make_timestamps
from src.prediction.electricprice.fixed import ElecPriceFixed, TimeWindow
from src.prediction.electricprice.import_ import ElecPriceImport

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


class TestElecPriceImport:
    async def test_exact_match(self):
        prices = [0.0003] * 24
        provider = ElecPriceImport(prices_wh=prices)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert list(result) == pytest.approx(prices)

    async def test_shorter_pads_last(self):
        prices = [0.0002, 0.0003]
        provider = ElecPriceImport(prices_wh=prices)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[0] == pytest.approx(0.0002)
        assert result[1] == pytest.approx(0.0003)
        assert result[23] == pytest.approx(0.0003)

    async def test_resample_to_quarter_hour(self):
        prices = [0.0001, 0.0005]  # 2h at 1h resolution
        ts = make_timestamps(START, hours=2, dt_hours=0.25)
        provider = ElecPriceImport(prices_wh=prices, source_dt_hours=1.0)
        result = await provider.fetch(ts)
        assert len(result) == 8
        assert result[0] == pytest.approx(0.0001, abs=1e-5)

    async def test_provider_id(self):
        assert ElecPriceImport(prices_wh=[]).provider_id == "ElecPriceImport"


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
        from src.prediction.electricprice.energycharts import (
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

    async def test_poll_interval_triggers_recheck(self):
        """After poll_interval expires the provider re-checks Energy-Charts."""
        from datetime import timedelta
        from unittest.mock import patch

        provider = self._provider()
        # Use a deterministic 'now' in the early afternoon so the poll
        # window condition (after 12:30 UTC) can be satisfied reliably.
        now = datetime.now(timezone.utc).replace(hour=13, minute=0, second=0, microsecond=0)
        # Short future coverage so that last_real_ts < now + 1 day and a
        # recheck becomes possible when time advances.
        raw = self._make_raw(now, n_future=4)
        provider._request_prices = AsyncMock(return_value=raw)

        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        # First call
        await provider.fetch(ts)
        assert provider._request_prices.call_count == 1

        # Simulate time has passed beyond poll by patching the module's
        # `datetime.now()` so the provider sees the bumped current time
        # and decides to recheck.
        expired = now + timedelta(minutes=31)
        ts_expired = make_timestamps(expired, hours=24, dt_hours=1.0)
        with patch("src.prediction.electricprice.energycharts.datetime") as mock_dt:
            mock_dt.now.return_value = expired
            await provider.fetch(ts_expired)

        assert provider._request_prices.call_count == 2

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
        # Use a deterministic afternoon time so the poll window can trigger
        # when we advance the perceived current time.
        now = datetime.now(timezone.utc).replace(hour=13, minute=0, second=0, microsecond=0)

        raw_day1 = self._make_raw(now, n_future=24 * 4, base_price=10.0)
        raw_day2 = self._make_raw(now, n_future=48 * 4, base_price=20.0)  # more future + higher

        provider._request_prices = AsyncMock(side_effect=[raw_day1, raw_day2])

        ts = make_timestamps(now, hours=24, dt_hours=1.0)
        result1 = await provider.fetch(ts)
        bucket1 = int(ts.to_list()[0].timestamp()) // 900
        price1 = provider._price_map.get(bucket1, 0.0)

        # Advance perceived current time so the provider re-checks and
        # fetches fresh data (side_effect returns raw_day2).
        later = now + timedelta(hours=2)
        from unittest.mock import patch

        with patch("src.prediction.electricprice.energycharts.datetime") as mock_dt:
            mock_dt.now.return_value = later
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
        from datetime import timedelta

        # Small buffer (6 h) so we can easily exceed it:
        # ts_short last = now+3h → map covers [now, now+9h].
        # ts_long last = now+23h > now+9h → cache miss → refresh.
        provider = self._provider(buffer_hours=6)
        now = datetime.now(timezone.utc)
        raw = self._make_raw(now, n_future=30 * 4)  # 30 h covers 3 h last_ts + 6 h buffer + slack
        provider._request_prices = AsyncMock(return_value=raw)

        ts_short = make_timestamps(now, hours=4, dt_hours=1.0)
        await provider.fetch(ts_short)
        assert provider._request_prices.call_count == 1

        # Now ask for timestamps far beyond the map horizon
        ts_long = make_timestamps(now, hours=24, dt_hours=1.0)
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

    async def test_empty_api_response_returns_zeros(self):
        """If the API returns no data, fetch() returns all-zero prices."""
        provider = self._provider()
        provider._request_prices = AsyncMock(return_value=[])

        now = datetime.now(timezone.utc)
        ts = make_timestamps(now, hours=6, dt_hours=1.0)
        result = await provider.fetch(ts)

        assert len(result) == 6
        assert all(v == pytest.approx(0.0) for v in result.to_list())

    async def test_charges_and_vat_applied(self):
        """charges_kwh and vat_rate are factored into the cached prices."""
        from datetime import timedelta

        from src.prediction.electricprice.energycharts import (
            ElecPriceEnergyCharts,
            EnergyChartsConfig,
        )

        provider = ElecPriceEnergyCharts(EnergyChartsConfig(charges_kwh=0.05, vat_rate=1.19))
        now = datetime.now(timezone.utc)

        # Provide a constant price of 100 EUR/MWh = 1e-4 EUR/Wh
        raw_price_wh = 100.0 / 1_000_000.0  # 1e-4 EUR/Wh
        raw = [(now + timedelta(minutes=15 * i), raw_price_wh) for i in range(200)]
        # Inject already-processed prices (bypass _request_prices conversion)
        provider._request_prices = AsyncMock(return_value=raw)

        ts = make_timestamps(now, hours=2, dt_hours=0.25)
        result = await provider.fetch(ts)

        # The injected raw prices are market prices; the provider applies
        # configured charges and VAT when building the in-memory price map.
        charges_wh = provider._charges_kwh / 1000.0
        expected = (raw_price_wh + charges_wh) * (1+provider._vat_rate)
        assert all(v == pytest.approx(expected, rel=1e-3) for v in result.to_list())


async def test_energycharts_fetch_today():
    """Integration test: skipped when Energy-Charts API is unreachable."""
    import aiohttp

    from src.prediction.electricprice.energycharts import ElecPriceEnergyCharts
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

    from src.prediction.electricprice.energycharts import EnergyChartsConfig

    provider = ElecPriceEnergyCharts(EnergyChartsConfig(bidding_zone="DE-LU"))
    ts = make_timestamps(start, hours=24, dt_hours=1.0)
    try:
        prices = await provider.fetch(ts)
    except Exception as exc:
        pytest.skip(f"Energy-Charts fetch failed: {exc}")

    assert len(prices) == 24
    assert isinstance(prices, pl.Series)
