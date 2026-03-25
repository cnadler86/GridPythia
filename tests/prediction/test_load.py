"""Tests for load forecast providers."""

from datetime import datetime, timezone

import polars as pl
import pytest

from src.prediction.base import make_timestamps
from src.prediction.load.fixed import LoadFixed, LoadTimeWindow
from src.prediction.load.import_ import LoadImport

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


def _ts(hours: float = 24, dt: float = 1.0) -> pl.Series:
    return make_timestamps(START, hours, dt)


class TestLoadFixed:
    async def test_constant_load(self):
        provider = LoadFixed(power_w=750.0)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert all(v == pytest.approx(750.0) for v in result.to_list())

    async def test_schedule(self):
        schedule = [
            LoadTimeWindow(start_hour=0, end_hour=7, power_w=200),
            LoadTimeWindow(start_hour=7, end_hour=22, power_w=800),
            LoadTimeWindow(start_hour=22, end_hour=24, power_w=300),
        ]
        provider = LoadFixed(schedule=schedule)
        result = await provider.fetch(_ts())
        assert result[0] == pytest.approx(200.0)
        assert result[6] == pytest.approx(200.0)
        assert result[7] == pytest.approx(800.0)
        assert result[22] == pytest.approx(300.0)

    async def test_quarter_hour(self):
        provider = LoadFixed(power_w=500.0)
        result = await provider.fetch(_ts(dt=0.25))
        assert len(result) == 96

    async def test_provider_id(self):
        assert LoadFixed().provider_id == "LoadFixed"

    async def test_returns_polars_float32(self):
        result = await LoadFixed().fetch(_ts())
        assert isinstance(result, pl.Series)
        assert result.dtype == pl.Float32


class TestLoadImport:
    async def test_exact_length(self):
        load = [500.0] * 24
        provider = LoadImport(load_w=load)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert list(result) == pytest.approx(load)

    async def test_padding(self):
        load = [100.0, 200.0]
        provider = LoadImport(load_w=load)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[0] == pytest.approx(100.0)
        assert result[1] == pytest.approx(200.0)
        assert result[23] == pytest.approx(200.0)

    async def test_resample(self):
        load = [0.0, 100.0]  # 2h hourly
        ts = make_timestamps(START, hours=2, dt_hours=0.25)
        provider = LoadImport(load_w=load, source_dt_hours=1.0)
        result = await provider.fetch(ts)
        assert len(result) == 8
        assert result[0] == pytest.approx(0.0, abs=1.0)

    async def test_provider_id(self):
        assert LoadImport(load_w=[]).provider_id == "LoadImport"
