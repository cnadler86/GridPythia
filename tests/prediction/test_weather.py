"""Tests for weather providers."""

from datetime import datetime, timezone

import polars as pl
import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.weather.import_ import WeatherImport

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


def _ts(hours: float = 24, dt: float = 1.0) -> pl.Series:
    return make_timestamps(START, hours, dt)


class TestWeatherImport:
    async def test_basic_channels(self):
        data = {
            "temperature_c": [20.0 + i * 0.5 for i in range(24)],
            "cloud_cover_pct": [30.0] * 24,
        }
        provider = WeatherImport(data=data)
        result = await provider.fetch(_ts())
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 24
        assert result["temperature_c"][0] == pytest.approx(20.0)
        assert result["temperature_c"][23] == pytest.approx(20.0 + 23 * 0.5)
        assert result["cloud_cover_pct"][0] == pytest.approx(30.0)

    async def test_optional_channels_absent(self):
        data = {
            "temperature_c": [15.0] * 24,
            "cloud_cover_pct": [50.0] * 24,
        }
        provider = WeatherImport(data=data)
        result = await provider.fetch(_ts())
        assert "wind_speed_kmh" not in result.columns
        assert "ghi_wm2" not in result.columns

    async def test_all_channels(self):
        full = {
            "temperature_c": [20.0] * 24,
            "cloud_cover_pct": [50.0] * 24,
            "wind_speed_kmh": [10.0] * 24,
            "humidity_pct": [60.0] * 24,
            "precipitation_mm": [0.0] * 24,
            "pressure_hpa": [1013.0] * 24,
            "ghi_wm2": [300.0] * 24,
            "dni_wm2": [200.0] * 24,
            "dhi_wm2": [100.0] * 24,
        }
        provider = WeatherImport(data=full)
        result = await provider.fetch(_ts())
        assert "ghi_wm2" in result.columns
        assert result["ghi_wm2"][0] == pytest.approx(300.0)

    async def test_quarter_hour_resample(self):
        data = {
            "temperature_c": [10.0, 20.0],
            "cloud_cover_pct": [0.0, 100.0],
        }
        ts = make_timestamps(START, hours=2, dt_hours=0.25)
        provider = WeatherImport(data=data, source_dt_hours=1.0)
        result = await provider.fetch(ts)
        assert len(result) == 8
        assert result["temperature_c"][0] == pytest.approx(10.0, abs=0.5)

    async def test_shorter_data_zero_padded(self):
        data = {
            "temperature_c": [15.0] * 12,
            "cloud_cover_pct": [50.0] * 12,
        }
        provider = WeatherImport(data=data)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result["temperature_c"][11] == pytest.approx(15.0)
        assert result["temperature_c"][12] == pytest.approx(0.0)

    async def test_provider_id(self):
        data = {"temperature_c": [], "cloud_cover_pct": []}
        assert WeatherImport(data=data).provider_id == "WeatherImport"

    async def test_returns_dataframe_with_float32(self):
        data = {"temperature_c": [1.0] * 3, "cloud_cover_pct": [0.0] * 3}
        ts = make_timestamps(START, hours=3, dt_hours=1.0)
        result = await WeatherImport(data=data).fetch(ts)
        assert result["temperature_c"].dtype == pl.Float32
