"""Tests for the CSV-backed load profile provider."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.load.config import LoadProfileConfig
from GridPythia.prediction.load.provider import DayType, day_type_for_date
from GridPythia.prediction.load.provider import load_provider_from_config
from GridPythia.prediction.load.profilecsv import LoadProfileCSV

# Monday 2025-06-16 00:00 UTC
_START_MON = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)
# Saturday 2025-06-21 00:00 UTC
_START_SAT = datetime(2025, 6, 21, 0, 0, tzinfo=timezone.utc)
# Sunday 2025-06-22 00:00 UTC
_START_SUN = datetime(2025, 6, 22, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# CSV file helpers
# ---------------------------------------------------------------------------

def _write_csv_weekend(path: Path, weekday_wh: float, weekend_wh: float) -> None:
    """Write a 1-h resolution file with weekday/weekend columns."""
    rows = ["time,weekday,weekend"]
    for h in range(24):
        rows.append(f"{h:02d}:00,{weekday_wh},{weekend_wh}")
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_csv_sat_sun(
    path: Path, weekday_wh: float, sat_wh: float, sun_wh: float
) -> None:
    """Write a 1-h resolution file with weekday/saturday/sunday columns."""
    rows = ["time,weekday,saturday,sunday"]
    for h in range(24):
        rows.append(f"{h:02d}:00,{weekday_wh},{sat_wh},{sun_wh}")
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_csv_15min(path: Path, weekday_wh: float, weekend_wh: float) -> None:
    """Write a 15-min resolution file."""
    rows = ["time,weekday,weekend"]
    for slot in range(96):
        h, m = divmod(slot * 15, 60)
        rows.append(f"{h:02d}:{m:02d},{weekday_wh},{weekend_wh}")
    path.write_text("\n".join(rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# provider_id
# ---------------------------------------------------------------------------


class TestLoadProfileCSVProviderID:
    async def test_provider_id(self, tmp_path):
        p = tmp_path / "myprofile.csv"
        _write_csv_weekend(p, 100.0, 200.0)
        assert LoadProfileCSV(LoadProfileConfig(path=p)).provider_id == "LoadProfileCSV(myprofile.csv)"


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


class TestLoadProfileCSVFetch:
    async def test_weekday_returns_weekday_values(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        # Mon 00:00 – 24h at 1-h steps
        ts = make_timestamps(_START_MON, 24, 1.0)
        result = await provider.fetch(ts)
        assert len(result) == 24
        # Source is 1-h, target is 1-h → each slot must be exactly 100 Wh
        assert all(v == pytest.approx(100.0) for v in result)

    async def test_saturday_returns_saturday_values(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_sat_sun(p, weekday_wh=100.0, sat_wh=200.0, sun_wh=300.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        ts = make_timestamps(_START_SAT, 24, 1.0)
        result = await provider.fetch(ts)
        assert all(v == pytest.approx(200.0) for v in result)

    async def test_sunday_returns_sunday_values(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_sat_sun(p, weekday_wh=100.0, sat_wh=200.0, sun_wh=300.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        ts = make_timestamps(_START_SUN, 24, 1.0)
        result = await provider.fetch(ts)
        assert all(v == pytest.approx(300.0) for v in result)

    async def test_downsample_1h_source_to_1h_target_gives_exact_energy(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        ts = make_timestamps(_START_MON, 24, 1.0)
        result = await provider.fetch(ts)
        assert pytest.approx(24 * 100.0, rel=1e-4) == result.sum()

    async def test_upsample_1h_source_to_15min_conserves_energy(self, tmp_path):
        """When upsampling 1-h → 15-min, daily energy should be conserved."""
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        ts_15 = make_timestamps(_START_MON, 24, 0.25)
        result = await provider.fetch(ts_15)
        assert len(result) == 96
        # Total energy in 24 h is 24 × 100 Wh = 2400 Wh
        assert pytest.approx(2400.0, rel=0.01) == result.sum()

    async def test_weekend_fallback_when_only_weekend_column(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=50.0, weekend_wh=150.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        # Saturday → should use weekend column = 150 Wh
        ts = make_timestamps(_START_SAT, 24, 1.0)
        result = await provider.fetch(ts)
        assert all(v == pytest.approx(150.0) for v in result)

    async def test_15min_source_file(self, tmp_path):
        p = tmp_path / "p15.csv"
        _write_csv_15min(p, weekday_wh=25.0, weekend_wh=50.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        ts = make_timestamps(_START_MON, 24, 0.25)
        result = await provider.fetch(ts)
        assert len(result) == 96
        assert all(v == pytest.approx(25.0) for v in result)


# ---------------------------------------------------------------------------
# get_profile_series
# ---------------------------------------------------------------------------


class TestLoadProfileCSVGetProfileSeries:
    async def test_two_day_types_returns_double_length(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        day_ts = make_timestamps(_START_MON, 24, 1.0)
        result = await provider.get_profile_series(day_ts, [DayType.WEEKDAY, DayType.WEEKEND])
        assert len(result) == 48

    async def test_single_day_type_correct_values(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        day_ts = make_timestamps(_START_MON, 24, 1.0)
        result = await provider.get_profile_series(day_ts, [DayType.WEEKDAY])
        assert pytest.approx(24 * 100.0, rel=1e-4) == result.sum()

    async def test_vacation_is_constant_minimum(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=60.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        provider._ensure_loaded()
        vac = provider._profiles["vacations"]  # type: ignore[index]
        assert all(v == pytest.approx(60.0) for v in vac)

    async def test_upsample_15min_target_smooth_transition(self, tmp_path):
        """Day boundary should not be a hard step after smoothing."""
        p = tmp_path / "p.csv"
        # Two distinct levels to force a hard boundary if no smoothing is used
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        day_ts = make_timestamps(_START_MON, 24, 0.25)
        result = await provider.get_profile_series(day_ts, [DayType.WEEKDAY, DayType.WEEKEND])
        values = result
        # With smoothing the boundary transition shouldn't be a step > source wh/4
        boundary_idx = 96  # last slot of day 1 / first slot of day 2
        jump = abs(values[boundary_idx] - values[boundary_idx - 1])
        # jump should be smaller than a single source step (100 Wh / 4 = 25 Wh)
        assert jump < 25.0

    async def test_saturday_sunday_separation(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_sat_sun(p, weekday_wh=100.0, sat_wh=200.0, sun_wh=300.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        day_ts = make_timestamps(_START_MON, 24, 1.0)
        result = await provider.get_profile_series(
            day_ts, [DayType.SATURDAY, DayType.SUNDAY]
        )
        # First 24 = sat (200 Wh each), next 24 = sun (300 Wh each)
        first_half = result[:24].sum()
        second_half = result[24:].sum()
        assert pytest.approx(24 * 200.0, rel=1e-3) == first_half
        assert pytest.approx(24 * 300.0, rel=1e-3) == second_half

    async def test_empty_day_types_returns_empty(self, tmp_path):
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p))
        day_ts = make_timestamps(_START_MON, 24, 1.0)
        result = await provider.get_profile_series(day_ts, [])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# day_type_for_date helper
# ---------------------------------------------------------------------------


class TestDayTypeForDate:
    from datetime import date

    def test_weekday_no_country(self) -> None:
        from datetime import date
        # 2025-06-16 is a Monday
        assert day_type_for_date(date(2025, 6, 16)) == DayType.WEEKDAY

    def test_saturday_no_country(self) -> None:
        from datetime import date
        assert day_type_for_date(date(2025, 6, 21)) == DayType.SATURDAY

    def test_sunday_no_country(self) -> None:
        from datetime import date
        assert day_type_for_date(date(2025, 6, 22)) == DayType.SUNDAY

    def test_german_holiday_maps_to_sunday(self) -> None:
        from datetime import date
        # 2025-01-01 is New Year's Day (public holiday in Germany)
        assert day_type_for_date(date(2025, 1, 1), country="DE") == DayType.SUNDAY

    def test_german_bw_holiday_maps_to_sunday(self) -> None:
        from datetime import date
        # 2025-01-06 is Epiphany, public holiday in BW but not in all German states
        assert day_type_for_date(date(2025, 1, 6), country="DE", subdivision="BW") == DayType.SUNDAY

    def test_bw_epiphany_is_weekday_in_hessen(self) -> None:
        from datetime import date
        # Epiphany is NOT a holiday in Hessen (HE)
        # 2025-01-06 is a Monday
        result = day_type_for_date(date(2025, 1, 6), country="DE", subdivision="HE")
        assert result == DayType.WEEKDAY

    def test_normal_weekday_with_country_stays_weekday(self) -> None:
        from datetime import date
        # 2025-06-16 is a regular Monday in Germany
        assert day_type_for_date(date(2025, 6, 16), country="DE", subdivision="BW") == DayType.WEEKDAY


# ---------------------------------------------------------------------------
# Holiday-aware fetch (CSV provider)
# ---------------------------------------------------------------------------


class TestLoadProfileCSVHolidays:
    async def test_holiday_uses_sunday_profile(self, tmp_path: Path) -> None:
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p, country="DE", subdivision="BW"))
        # 2025-01-01 is New Year's Day (holiday) → treated as Sunday → weekend = 200 Wh
        ts = make_timestamps(datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc), 24, 1.0)
        result = await provider.fetch(ts)
        assert all(v == pytest.approx(200.0) for v in result)

    async def test_non_holiday_weekday_unaffected(self, tmp_path: Path) -> None:
        p = tmp_path / "p.csv"
        _write_csv_weekend(p, weekday_wh=100.0, weekend_wh=200.0)
        provider = LoadProfileCSV(LoadProfileConfig(path=p, country="DE", subdivision="BW"))
        # 2025-06-16 is a regular Monday
        ts = make_timestamps(datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc), 24, 1.0)
        result = await provider.fetch(ts)
        assert all(v == pytest.approx(100.0) for v in result)


# ---------------------------------------------------------------------------
# LoadProfileConfig + load_provider_from_config
# ---------------------------------------------------------------------------


class TestLoadProfileConfig:
    def test_csv_suffix_accepted(self, tmp_path: Path) -> None:
        p = tmp_path / "prof.csv"
        p.write_text("time,weekday\n", encoding="utf-8")
        cfg = LoadProfileConfig(path=p)
        assert cfg.path.suffix == ".csv"

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "prof.xlsx"
        p.write_text("", encoding="utf-8")
        with pytest.raises(Exception, match="Unsupported"):
            LoadProfileConfig(path=p)

    def test_factory_returns_csv_provider(self, tmp_path: Path) -> None:
        p = tmp_path / "prof.csv"
        rows = ["time,weekday,weekend"] + [f"{h:02d}:00,100,200" for h in range(24)]
        p.write_text("\n".join(rows), encoding="utf-8")
        cfg = LoadProfileConfig(path=p, country="DE")
        provider = load_provider_from_config(cfg)
        assert isinstance(provider, LoadProfileCSV)
        assert provider._country == "DE"
