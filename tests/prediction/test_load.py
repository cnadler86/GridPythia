"""Tests for load forecast provider base class (fetch / get_profile_series)."""

from datetime import datetime, timezone

import numpy as np
import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.load.provider import DayType, LoadProvider, day_type_for_date

START = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)  # Monday


def _ts(hours: float = 24, dt: float = 1.0) -> list:
    return make_timestamps(START, hours, dt)


class _ConstantLoadProvider(LoadProvider):
    """Concrete test stub: returns constant power per day type."""

    def __init__(self, power_w: float = 500.0, src_dt: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._power_w = power_w
        self._src_dt = src_dt

    @property
    def provider_id(self) -> str:
        return "ConstantLoad"

    def _get_day_profile_w(self, day_type: DayType) -> tuple[list[float], float]:
        slots = int(24 / self._src_dt)
        # weekday = base, weekend = half, vacation = quarter
        if day_type == DayType.VACATIONS:
            pw = self._power_w / 4
        elif day_type in (DayType.SATURDAY, DayType.SUNDAY, DayType.WEEKEND):
            pw = self._power_w / 2
        else:
            pw = self._power_w
        return [pw] * slots, self._src_dt


class TestDayTypeForDate:
    def test_weekday(self):
        # 2025-06-16 is a Monday
        assert day_type_for_date(datetime(2025, 6, 16).date()) == DayType.WEEKDAY

    def test_saturday(self):
        assert day_type_for_date(datetime(2025, 6, 21).date()) == DayType.SATURDAY

    def test_sunday(self):
        assert day_type_for_date(datetime(2025, 6, 22).date()) == DayType.SUNDAY

    def test_public_holiday_is_sunday(self):
        # 2025-01-01 = Neujahr
        assert day_type_for_date(datetime(2025, 1, 1).date(), country="DE") == DayType.SUNDAY

    def test_non_holiday_weekday(self):
        # A normal Tuesday
        assert (
            day_type_for_date(datetime(2025, 6, 17).date(), country="DE") == DayType.WEEKDAY
        )


class TestLoadProviderFetch:
    @pytest.mark.asyncio
    async def test_fetch_hourly_24h(self):
        provider = _ConstantLoadProvider(power_w=1000.0)
        ts = _ts(hours=24, dt=1.0)
        result = await provider.fetch(ts)

        assert len(result) == 24
        assert result.dtype == np.float32
        # 1000 W * 1 h = 1000 Wh per slot
        np.testing.assert_allclose(result, 1000.0, atol=1.0)

    @pytest.mark.asyncio
    async def test_fetch_quarter_hour(self):
        provider = _ConstantLoadProvider(power_w=1000.0)
        ts = _ts(hours=24, dt=0.25)
        result = await provider.fetch(ts)

        assert len(result) == 96
        assert result.dtype == np.float32
        # 1000 W * 0.25 h = 250 Wh per slot
        np.testing.assert_allclose(result, 250.0, atol=1.0)

    @pytest.mark.asyncio
    async def test_fetch_energy_conservation(self):
        """Total energy over 24h should be preserved regardless of step width."""
        provider = _ConstantLoadProvider(power_w=500.0)

        r1h = await provider.fetch(_ts(hours=24, dt=1.0))
        r15m = await provider.fetch(_ts(hours=24, dt=0.25))

        total_1h = float(np.sum(r1h))
        total_15m = float(np.sum(r15m))
        # 500 W * 24 h = 12000 Wh
        assert total_1h == pytest.approx(12000.0, rel=0.01)
        assert total_15m == pytest.approx(12000.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_fetch_empty_timestamps(self):
        provider = _ConstantLoadProvider()
        result = await provider.fetch([])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_vacation_mode(self):
        provider = _ConstantLoadProvider(power_w=1000.0)
        ts = _ts(hours=24, dt=1.0)

        normal = await provider.fetch(ts, use_vacation_profile=False)
        vacation = await provider.fetch(ts, use_vacation_profile=True)

        # Vacation load (1/4) should be less than normal weekday load
        assert float(np.sum(vacation)) < float(np.sum(normal))

    @pytest.mark.asyncio
    async def test_fetch_weekend_profile(self):
        """Saturday timestamps should use the weekend profile."""
        provider = _ConstantLoadProvider(power_w=1000.0)
        sat_start = datetime(2025, 6, 21, 0, 0, tzinfo=timezone.utc)
        ts = make_timestamps(sat_start, hours=24, dt_hours=1.0)
        result = await provider.fetch(ts)

        # Weekend = half power → 500 W * 1 h = 500 Wh
        np.testing.assert_allclose(result, 500.0, atol=1.0)


class TestLoadProviderProfileSeries:
    @pytest.mark.asyncio
    async def test_single_day_type(self):
        provider = _ConstantLoadProvider(power_w=600.0)
        ts = _ts(hours=24, dt=1.0)
        result = await provider.get_profile_series(ts, [DayType.WEEKDAY])

        assert len(result) == 24
        assert result.dtype == np.float32
        # 600 W * 1 h = 600 Wh
        np.testing.assert_allclose(result, 600.0, atol=1.0)

    @pytest.mark.asyncio
    async def test_multi_day_types(self):
        provider = _ConstantLoadProvider(power_w=600.0)
        ts = _ts(hours=24, dt=1.0)
        result = await provider.get_profile_series(
            ts, [DayType.WEEKDAY, DayType.WEEKEND]
        )

        # Two days concatenated = 48 slots
        assert len(result) == 48

    @pytest.mark.asyncio
    async def test_empty_day_types(self):
        provider = _ConstantLoadProvider()
        ts = _ts(hours=24, dt=1.0)
        result = await provider.get_profile_series(ts, [])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_vacation_profile_series(self):
        provider = _ConstantLoadProvider(power_w=800.0)
        ts = _ts(hours=24, dt=1.0)
        result = await provider.get_profile_series(ts, [DayType.VACATIONS])

        assert len(result) == 24
        # Vacation = 1/4 of 800 W = 200 W * 1 h = 200 Wh
        np.testing.assert_allclose(result, 200.0, atol=1.0)

