"""Tests for weather providers."""

from datetime import datetime, timezone

import pytest

from src.prediction.weather.import_ import WeatherImport

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)
END_24H = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)


class TestWeatherImport:
    def test_basic_channels(self):
        data = {
            "temperature_c": [20.0 + i * 0.5 for i in range(24)],
            "cloud_cover_pct": [30.0] * 24,
        }
        provider = WeatherImport(data=data)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result.temperature_c) == 24
        assert len(result.cloud_cover_pct) == 24
        assert result.temperature_c[0] == pytest.approx(20.0)
        assert result.temperature_c[23] == pytest.approx(20.0 + 23 * 0.5)
        assert result.cloud_cover_pct[0] == pytest.approx(30.0)

    def test_optional_channels_none(self):
        data = {
            "temperature_c": [15.0] * 24,
            "cloud_cover_pct": [50.0] * 24,
        }
        provider = WeatherImport(data=data)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert result.wind_speed_kmh is None
        assert result.ghi_wm2 is None

    def test_all_channels(self):
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
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert result.ghi_wm2 is not None
        assert result.ghi_wm2[0] == pytest.approx(300.0)

    def test_quarter_hour_resample(self):
        data = {
            "temperature_c": [10.0, 20.0],
            "cloud_cover_pct": [0.0, 100.0],
        }
        end_2h = datetime(2025, 6, 15, 2, 0, tzinfo=timezone.utc)
        provider = WeatherImport(data=data, source_dt_hours=1.0)
        result = provider.fetch(START, end_2h, dt_hours=0.25)
        assert len(result.temperature_c) == 8
        assert result.temperature_c[0] == pytest.approx(10.0, abs=0.5)

    def test_shorter_data_zero_padded(self):
        data = {
            "temperature_c": [15.0] * 12,
            "cloud_cover_pct": [50.0] * 12,
        }
        provider = WeatherImport(data=data)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result.temperature_c) == 24
        assert result.temperature_c[11] == pytest.approx(15.0)
        assert result.temperature_c[12] == pytest.approx(0.0)  # zero-filled

    def test_provider_id(self):
        data = {"temperature_c": [], "cloud_cover_pct": []}
        assert WeatherImport(data=data).provider_id == "WeatherImport"
