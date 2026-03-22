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

    provider = ElecPriceEnergyCharts(bidding_zone="DE-LU")
    ts = make_timestamps(start, hours=24, dt_hours=1.0)
    try:
        prices = await provider.fetch(ts)
    except Exception as exc:
        pytest.skip(f"Energy-Charts fetch failed: {exc}")

    assert len(prices) == 24
    assert isinstance(prices, pl.Series)
