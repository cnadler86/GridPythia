"""Tests for JSON-backed load profile provider with weekly cache."""

from datetime import datetime, timezone
import json

import pytest

from src.prediction.base import make_timestamps
from src.prediction.load.profilejson import LoadProfileJSON


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
        assert LoadProfileJSON(data_path=p).provider_id == "LoadProfileJSON"

    async def test_weekday_and_weekend_profile_selection(self, tmp_path):
        p = tmp_path / "profiles.json"
        _write_profiles(p, [100.0] * 24, [200.0] * 24)
        provider = LoadProfileJSON(data_path=p)

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
        provider = LoadProfileJSON(data_path=p)

        ts = make_timestamps(datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc), 1.25, 0.25)
        values = (await provider.fetch(ts)).to_list()
        # Interpolation runs in power-space, then converts to Wh for 15-minute slots.
        assert values == pytest.approx([0.0, 6.25, 12.5, 18.75, 25.0], abs=1e-6)

    async def test_cache_file_created_and_reused(self, tmp_path):
        p = tmp_path / "profiles.json"
        _write_profiles(p, [100.0] * 24, [200.0] * 24)
        provider = LoadProfileJSON(data_path=p)

        ts = make_timestamps(datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc), 2.0, 0.25)
        await provider.fetch(ts)
        cache_file = provider.cache_file
        assert cache_file.exists()
        mtime_before = cache_file.stat().st_mtime

        await provider.fetch(ts)
        mtime_after = cache_file.stat().st_mtime
        assert mtime_after == pytest.approx(mtime_before)

    async def test_cache_hash_changes_when_input_changes(self, tmp_path):
        p = tmp_path / "profiles.json"
        _write_profiles(p, [100.0] * 24, [200.0] * 24)

        provider_before = LoadProfileJSON(data_path=p)
        ts = make_timestamps(datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc), 1.0, 0.25)
        await provider_before.fetch(ts)
        old_cache = provider_before.cache_file

        _write_profiles(p, [150.0] * 24, [200.0] * 24)
        provider_after = LoadProfileJSON(data_path=p)
        await provider_after.fetch(ts)
        new_cache = provider_after.cache_file

        assert old_cache != new_cache
        assert new_cache.exists()
