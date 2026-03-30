"""Tests for JSON-backed load profile provider."""

from datetime import datetime, timezone
import json

import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.load.config import LoadProfileConfig
from GridPythia.prediction.load.profilejson import LoadProfileJSON


def _write_profiles(path, weekday: list[float], weekend: list[float]) -> None:
    payload = {
        "estimated_annual_kwh": 1400,
        "profile_dt_hours": 1.0,
        "profiles": {
            "overall": {"mean_wh": weekday, "std_wh": [0.0] * 24},
            "weekday": {"mean_wh": weekday, "std_wh": [0.0] * 24},
            "weekend": {"mean_wh": weekend, "std_wh": [0.0] * 24},
            "vacation": {"mean_wh": [50.0] * 24, "std_wh": [0.0] * 24},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestLoadProfileJSON:
    async def test_provider_id(self, tmp_path):
        p = tmp_path / "profiles.json"
        _write_profiles(p, [100.0] * 24, [200.0] * 24)
        assert LoadProfileJSON(LoadProfileConfig(path=p)).provider_id == "LoadProfileJSON"

    async def test_weekday_and_weekend_profile_selection(self, tmp_path):
        p = tmp_path / "profiles.json"
        _write_profiles(p, [100.0] * 24, [200.0] * 24)
        provider = LoadProfileJSON(LoadProfileConfig(path=p))

        ts_weekday = make_timestamps(datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc), 24, 0.25)
        weekday_values = (await provider.fetch(ts_weekday)).to_list()
        # Source profile is hourly mean_wh=100, so each 15-minute slot is 25 Wh.
        assert all(v == pytest.approx(25.0) for v in weekday_values)

        ts_weekend = make_timestamps(datetime(2025, 6, 21, 0, 0, tzinfo=timezone.utc), 24, 0.25)
        weekend_values = (await provider.fetch(ts_weekend)).to_list()
        # Source profile is hourly mean_wh=200, so each 15-minute slot is 50 Wh.
        assert all(v == pytest.approx(50.0) for v in weekend_values)

    async def test_hourly_profile_interpolates_to_15min(self, tmp_path):
        p = tmp_path / "profiles.json"
        weekday = [0.0, 100.0] + [100.0] * 22
        _write_profiles(p, weekday=weekday, weekend=[100.0] * 24)
        provider = LoadProfileJSON(LoadProfileConfig(path=p))

        ts = make_timestamps(datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc), 1.25, 0.25)
        values = (await provider.fetch(ts)).to_list()
        # Interpolation runs in power-space, then converts to Wh for 15-minute slots.
        assert values == pytest.approx([0.0, 6.25, 12.5, 18.75, 25.0], abs=1e-6)

    async def test_data_loaded_once_into_memory(self, tmp_path):
        p = tmp_path / "profiles.json"
        _write_profiles(p, [100.0] * 24, [200.0] * 24)
        provider = LoadProfileJSON(LoadProfileConfig(path=p))

        ts = make_timestamps(datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc), 2.0, 0.25)
        assert provider._loaded_data is None
        await provider.fetch(ts)
        loaded = provider._loaded_data
        assert loaded is not None
        # Second fetch reuses the same object
        await provider.fetch(ts)
        assert provider._loaded_data is loaded
