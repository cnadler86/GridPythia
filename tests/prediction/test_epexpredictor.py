"""Tests for the ElecPriceEpexPredictor provider."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.electricprice.epexpredictor import (
    ElecPriceEpexPredictor,
    EpexPredictorConfig,
    _BUCKET_SECONDS,
    _DAY_AHEAD_PUB_HOUR,
    _DAY_AHEAD_PUB_MINUTE,
    _RETRY_AFTER_FAILED_REFRESH,
)

# ── Helpers ───────────────────────────────────────────────────────────────

NOW_UTC = datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc)  # before 13:00 pub

_EUR_PER_WH = 50.0 / 1_000_000.0  # 50 EUR/MWh → EUR/Wh raw


def _make_provider(
    charges_kwh: float = 0.0,
    vat_rate: float = 0.0,
    region: str = "DE",
) -> ElecPriceEpexPredictor:
    cfg = EpexPredictorConfig(charges_kwh=charges_kwh, vat_rate=vat_rate, region=region)
    return ElecPriceEpexPredictor(cfg)


def _make_api_response(
    start: datetime,
    hours: int,
    base_eur_mwh: float = 50.0,
    known_until_offset_h: int = 36,
) -> tuple[list[tuple[datetime, float]], datetime]:
    """Build a synthetic API response: *hours* hours of 15-min prices + known_until."""
    prices: list[tuple[datetime, float]] = []
    slots = hours * 4  # 4 quarter-hours per hour
    for i in range(slots):
        dt = start + timedelta(minutes=15 * i)
        prices.append((dt, base_eur_mwh / 1_000_000.0))
    known_until = start + timedelta(hours=known_until_offset_h)
    return prices, known_until


def _ts(start: datetime = NOW_UTC, hours: float = 24.0, dt: float = 1.0) -> list[datetime]:
    return make_timestamps(start, hours, dt)


# ── Config tests ──────────────────────────────────────────────────────────


class TestEpexPredictorConfig:
    def test_defaults(self):
        cfg = EpexPredictorConfig()
        assert cfg.region == "DE"
        assert cfg.charges_kwh == 0.0
        assert cfg.vat_rate == 0.19
        assert cfg.horizon_buffer == timedelta(hours=25)
        assert "epexpredictor.batzill.com" in cfg.base_url

    def test_horizon_buffer_from_int(self):
        cfg = EpexPredictorConfig(horizon_buffer=cast(Any, 10))
        assert cfg.horizon_buffer == timedelta(hours=10)

    def test_horizon_buffer_from_float(self):
        cfg = EpexPredictorConfig(horizon_buffer=cast(Any, 6.5))
        assert cfg.horizon_buffer == timedelta(hours=6, minutes=30)

    def test_immutable(self):
        cfg = EpexPredictorConfig()
        with pytest.raises(Exception):
            cfg.region = "AT"  # type: ignore[misc]


# ── provider_id ───────────────────────────────────────────────────────────


class TestProviderIdentity:
    def test_provider_id(self):
        assert _make_provider().provider_id == "EpexPredictor"

    def test_last_real_ts_initially_none(self):
        assert _make_provider().last_real_ts is None


# ── Cache-validity computation ────────────────────────────────────────────


class TestComputeSourceValidUntil:
    def _provider(self) -> ElecPriceEpexPredictor:
        return _make_provider()

    def test_before_publication_no_next_day(self):
        """Before 12:00 UTC without tomorrow's data → valid until today 12:00 UTC."""
        provider = self._provider()
        now = datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc)
        # known_until = today 23:00 UTC (today's real data only)
        known_until = datetime(2025, 6, 15, 23, 0, tzinfo=timezone.utc)
        result = provider._compute_source_valid_until(now, known_until)
        expected = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_after_publication_with_next_day(self):
        """After 12:00 UTC with tomorrow's real data → valid until tomorrow 12:00 UTC."""
        provider = self._provider()
        now = datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc)
        # known_until = tomorrow (next-day real prices available)
        known_until = datetime(2025, 6, 16, 11, 0, tzinfo=timezone.utc)
        result = provider._compute_source_valid_until(now, known_until)
        expected = datetime(2025, 6, 16, 12, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_after_publication_no_next_day_short_retry(self):
        """After 12:00 UTC without tomorrow's data → short retry (~15 min)."""
        provider = self._provider()
        now = datetime(2025, 6, 15, 13, 30, tzinfo=timezone.utc)
        known_until = datetime(2025, 6, 15, 23, 0, tzinfo=timezone.utc)
        result = provider._compute_source_valid_until(now, known_until)
        assert result == now + _RETRY_AFTER_FAILED_REFRESH

    def test_known_until_covers_tomorrow_midnight(self):
        """known_until at tomorrow midnight → next-day data available."""
        provider = self._provider()
        now = datetime(2025, 6, 15, 13, 30, tzinfo=timezone.utc)
        known_until = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)  # exactly tomorrow
        result = provider._compute_source_valid_until(now, known_until)
        expected = datetime(2025, 6, 16, 12, 0, tzinfo=timezone.utc)
        assert result == expected


# ── Fetch / cache flow ────────────────────────────────────────────────────


class TestEpexPredictorFetch:
    def _provider(self) -> ElecPriceEpexPredictor:
        return _make_provider()

    def _mock_request(
        self,
        provider: ElecPriceEpexPredictor,
        start: datetime,
        hours: int = 80,
        base_eur_mwh: float = 50.0,
        known_until_offset_h: int = 36,
    ) -> AsyncMock:
        raw, known_until = _make_api_response(
            start, hours, base_eur_mwh, known_until_offset_h
        )
        mock = AsyncMock(return_value=(raw, known_until))
        cast(Any, provider)._request_prices = mock
        return mock

    async def test_first_call_populates_cache(self):
        provider = self._provider()
        ts = _ts(NOW_UTC, hours=24)
        self._mock_request(provider, NOW_UTC - timedelta(hours=2))
        result = await provider.fetch(ts)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 24
        assert cast(Any, provider)._request_prices.call_count == 1
        assert provider.last_real_ts is not None

    async def test_repeated_calls_use_cache(self):
        provider = self._provider()
        ts = _ts(NOW_UTC, hours=24)
        self._mock_request(provider, NOW_UTC - timedelta(hours=2))
        await provider.fetch(ts)
        await provider.fetch(ts)
        # Cache is valid; API should be called exactly once.
        assert cast(Any, provider)._request_prices.call_count == 1

    async def test_price_values_with_charges_vat(self):
        """Charges + VAT are applied correctly to raw EUR/MWh values."""
        charges_kwh = 0.10
        vat_rate = 0.19
        base_eur_mwh = 50.0  # → 50e-6 EUR/Wh raw
        provider = ElecPriceEpexPredictor(
            EpexPredictorConfig(charges_kwh=charges_kwh, vat_rate=vat_rate)
        )
        ts = _ts(NOW_UTC, hours=2, dt=1.0)
        raw, known_until = _make_api_response(NOW_UTC - timedelta(hours=2), 80, base_eur_mwh)
        cast(Any, provider)._request_prices = AsyncMock(return_value=(raw, known_until))
        result = await provider.fetch(ts)
        charges_wh = charges_kwh / 1000.0
        raw_wh = base_eur_mwh / 1_000_000.0
        expected = (raw_wh + charges_wh) * (1.0 + vat_rate)
        assert result[0] == pytest.approx(expected, rel=1e-4)

    async def test_returns_float32_array(self):
        provider = self._provider()
        ts = _ts(NOW_UTC, hours=6)
        self._mock_request(provider, NOW_UTC - timedelta(hours=2))
        result = await provider.fetch(ts)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    async def test_quarter_hour_timestamps_each_get_own_bucket(self):
        """Each 15-min slot maps to its own 900-second bucket."""
        provider = self._provider()
        ts = make_timestamps(NOW_UTC, hours=1, dt_hours=0.25)  # 4 slots
        self._mock_request(provider, NOW_UTC - timedelta(hours=2))
        result = await provider.fetch(ts)
        assert len(result) == 4
        # All from same mock price, so all equal
        assert all(v == pytest.approx(result[0], rel=1e-4) for v in result)

    async def test_cache_stale_after_valid_until(self):
        """Cache is refreshed once source_valid_until is passed."""
        provider = self._provider()
        ts = _ts(NOW_UTC, hours=24)
        mock = self._mock_request(provider, NOW_UTC - timedelta(hours=2))
        await provider.fetch(ts)
        assert mock.call_count == 1

        # Expire the cache
        provider._cache.source_valid_until = NOW_UTC - timedelta(seconds=1)
        mock2 = self._mock_request(provider, NOW_UTC - timedelta(hours=2))
        await provider.fetch(ts)
        assert mock2.call_count == 1  # new mock was called

    async def test_api_failure_raises_when_cache_empty(self):
        provider = self._provider()
        ts = _ts(NOW_UTC, hours=2)
        cast(Any, provider)._request_prices = AsyncMock(side_effect=RuntimeError("network error"))
        with pytest.raises(RuntimeError, match="EPEXPredictor refresh failed"):
            await provider.fetch(ts)

    async def test_api_failure_falls_back_to_cache_when_covered(self):
        """When the cache already covers the range, a failed refresh is tolerated."""
        provider = self._provider()
        ts = _ts(NOW_UTC, hours=2)
        # Populate cache first
        self._mock_request(provider, NOW_UTC - timedelta(hours=2))
        await provider.fetch(ts)

        # Expire the cache and make the API fail
        provider._cache.source_valid_until = NOW_UTC - timedelta(seconds=1)
        cast(Any, provider)._request_prices = AsyncMock(side_effect=RuntimeError("network error"))
        # Should NOT raise — stale cache still covers the range
        result = await provider.fetch(ts)
        assert len(result) == len(ts)

    async def test_known_until_stored_after_fetch(self):
        provider = self._provider()
        ts = _ts(NOW_UTC, hours=24)
        cast(Any, provider)._request_prices = AsyncMock(
            return_value=_make_api_response(NOW_UTC - timedelta(hours=2), 80)
        )
        await provider.fetch(ts)
        assert provider.last_real_ts is not None
        # known_until should be approximately 36h from start (NOW - 2h + 36h = NOW + 34h)
        assert provider.last_real_ts > NOW_UTC


# ── Fallback chain integration ────────────────────────────────────────────


class TestEpexPredictorFallbackChain:
    async def test_fallback_chain_uses_fallback_on_failure(self):
        """When EPEXPredictor fails, EnergyCharts fallback is used."""
        from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
        from GridPythia.prediction.electricprice.provider import ElecPriceFallbackChain

        primary = _make_provider()
        cast(Any, primary)._request_prices = AsyncMock(side_effect=RuntimeError("EPEX down"))

        fallback = ElecPriceFixed(price_kwh=0.30)
        chain = ElecPriceFallbackChain(primary=primary, fallback=fallback)

        ts = _ts(NOW_UTC, hours=2)
        result = await chain.fetch(ts)
        expected = 0.30 / 1000.0
        assert all(v == pytest.approx(expected) for v in result)

    async def test_fallback_chain_provider_id(self):
        from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
        from GridPythia.prediction.electricprice.provider import ElecPriceFallbackChain

        primary = _make_provider()
        fallback = ElecPriceFixed(price_kwh=0.30)
        chain = ElecPriceFallbackChain(primary=primary, fallback=fallback)
        assert "EpexPredictor" in chain.provider_id
        assert "fallback" in chain.provider_id

    async def test_fallback_chain_last_real_ts_from_primary(self):
        """When primary succeeds, last_real_ts is from primary."""
        from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
        from GridPythia.prediction.electricprice.provider import ElecPriceFallbackChain

        primary = _make_provider()
        raw, known_until = _make_api_response(NOW_UTC - timedelta(hours=2), 80)
        cast(Any, primary)._request_prices = AsyncMock(return_value=(raw, known_until))

        fallback = ElecPriceFixed(price_kwh=0.30)
        chain = ElecPriceFallbackChain(primary=primary, fallback=fallback)

        ts = _ts(NOW_UTC, hours=2)
        await chain.fetch(ts)
        assert chain.last_real_ts == primary.last_real_ts
        assert chain.last_real_ts is not None
