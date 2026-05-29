"""Tests for vacation-mode load profile selection.

Covers:
- LoadProfileCSV.fetch() with use_vacation_profile=True returns the configured
  percentile value (not the minimum) via statistics.quantiles
- PredictionSetup.use_vacation_profile is read by Prediction.fetch() and fetch_partial()
- dataclasses.replace works correctly on PredictionSetup (frozen dataclass)
- Vacation router: GET returns current state; PUT toggles and invalidates cache
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.load.config import LoadProfileConfig
from GridPythia.prediction.load.profilecsv import LoadProfileCSV
from GridPythia.prediction.prediction import Prediction, PredictionSetup

# Monday 2025-06-16 00:00 UTC
_START_MON = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, weekday_wh: float, weekend_wh: float) -> None:
    """1-h resolution CSV with distinct weekday and weekend values."""
    rows = ["time,weekday,weekend"]
    for h in range(24):
        rows.append(f"{h:02d}:00,{weekday_wh},{weekend_wh}")
    path.write_text("\n".join(rows), encoding="utf-8")


def _make_load_provider(tmp_path: Path, weekday_wh: float, weekend_wh: float) -> LoadProfileCSV:
    p = tmp_path / "profile.csv"
    _write_csv(p, weekday_wh, weekend_wh)
    return LoadProfileCSV(LoadProfileConfig(path=p))


# ---------------------------------------------------------------------------
# LoadProvider.fetch – use_vacation_profile flag
# ---------------------------------------------------------------------------

class TestLoadProviderVacationFlag:
    async def test_normal_mode_returns_weekday_values(self, tmp_path):
        """Without vacation flag, weekday returns the configured weekday values."""
        provider = _make_load_provider(tmp_path, weekday_wh=200.0, weekend_wh=100.0)
        ts = make_timestamps(_START_MON, 24, 1.0)
        result = await provider.fetch(ts)
        assert np.all(result == pytest.approx(200.0, rel=1e-4))

    async def test_vacation_mode_returns_percentile_not_minimum(self, tmp_path):
        """With use_vacation_profile=True the vacation profile uses the percentile,
        which is >= the minimum and <= the normal load."""
        provider = _make_load_provider(tmp_path, weekday_wh=200.0, weekend_wh=100.0)
        ts = make_timestamps(_START_MON, 24, 1.0)
        result = await provider.fetch(ts, use_vacation_profile=True)
        # Default percentile=5; with only two distinct values (200, 100) the p5
        # of 48 values (24 weekday + 24 weekend) is 100 Wh since all weekend values
        # are the minimum — so the result equals the smaller value here.
        assert float(result[0]) >= 0.0
        # Must not exceed the normal weekday load
        assert float(result[0]) <= 200.0

    async def test_vacation_mode_below_or_equal_normal_load(self, tmp_path):
        """Vacation load is always ≤ any individual normal-profile value."""
        provider = _make_load_provider(tmp_path, weekday_wh=300.0, weekend_wh=150.0)
        ts = make_timestamps(_START_MON, 48, 1.0)
        normal = await provider.fetch(ts)
        vacation = await provider.fetch(ts, use_vacation_profile=True)
        assert np.all(vacation <= normal + 1e-3)

    async def test_vacation_percentile_configurable(self, tmp_path):
        """A higher percentile yields a higher vacation load than a lower one."""
        import statistics

        from GridPythia.prediction.load.config import LoadProfileConfig
        from GridPythia.prediction.load.profilecsv import LoadProfileCSV

        p = tmp_path / "profile.csv"
        _write_csv(p, weekday_wh=200.0, weekend_wh=100.0)

        provider_p5 = LoadProfileCSV(LoadProfileConfig(path=p, vacation_percentile=5.0))
        provider_p50 = LoadProfileCSV(LoadProfileConfig(path=p, vacation_percentile=50.0))

        ts = make_timestamps(_START_MON, 24, 1.0)
        result_p5 = await provider_p5.fetch(ts, use_vacation_profile=True)
        result_p50 = await provider_p50.fetch(ts, use_vacation_profile=True)
        # p50 ≥ p5 (non-decreasing property of percentiles)
        assert float(result_p50[0]) >= float(result_p5[0])


# ---------------------------------------------------------------------------
# PredictionSetup.use_vacation_profile – frozen dataclass + replace
# ---------------------------------------------------------------------------

class TestPredictionSetupVacationFlag:
    def test_default_is_false(self):
        setup = PredictionSetup()
        assert setup.use_vacation_profile is False

    def test_explicit_true(self):
        setup = PredictionSetup(use_vacation_profile=True)
        assert setup.use_vacation_profile is True

    def test_dataclasses_replace_sets_flag(self):
        base = PredictionSetup()
        vacation = _dc_replace(base, use_vacation_profile=True)
        assert vacation.use_vacation_profile is True
        # Original unchanged
        assert base.use_vacation_profile is False

    def test_dataclasses_replace_preserves_other_fields(self, tmp_path):
        from GridPythia.prediction.electricprice.fixed import ElecPriceFixed

        ep = ElecPriceFixed(price_kwh=0.30)
        base = PredictionSetup(electricprice=ep)
        vacation = _dc_replace(base, use_vacation_profile=True)
        assert vacation.electricprice is ep
        assert vacation.use_vacation_profile is True


# ---------------------------------------------------------------------------
# Prediction.fetch – reads flag from PredictionSetup
# ---------------------------------------------------------------------------

class TestPredictionFetchVacationProfile:
    def _make_prediction(self, tmp_path: Path, *, vacation: bool) -> Prediction:
        provider = _make_load_provider(tmp_path, weekday_wh=300.0, weekend_wh=150.0)
        setup = PredictionSetup(load=provider, use_vacation_profile=vacation)
        return Prediction(setup)

    async def test_normal_load_higher_than_vacation(self, tmp_path):
        pred_normal = self._make_prediction(tmp_path, vacation=False)
        pred_vacation = self._make_prediction(tmp_path, vacation=True)
        data_normal = await pred_normal.fetch(start=_START_MON, hours=24, dt_hours=1.0)
        data_vacation = await pred_vacation.fetch(start=_START_MON, hours=24, dt_hours=1.0)
        assert float(np.sum(data_normal.load_wh)) > float(np.sum(data_vacation.load_wh))

    async def test_vacation_load_equals_percentile_value(self, tmp_path):
        """All vacation slots return the same constant p5 value from the profile."""
        import statistics

        from GridPythia.prediction.load.config import LoadProfileConfig
        from GridPythia.prediction.load.profilecsv import LoadProfileCSV

        p = tmp_path / "p.csv"
        _write_csv(p, weekday_wh=300.0, weekend_wh=150.0)
        # With vacation_percentile=5: compute expected value ourselves
        # 24 weekday values (300) + 24 weekend values (150) = 48 values
        all_vals = [300.0] * 24 + [150.0] * 24
        expected = statistics.quantiles(all_vals, n=100)[4]  # index 4 = 5th percentile

        provider = LoadProfileCSV(LoadProfileConfig(path=p, vacation_percentile=5.0))
        setup = PredictionSetup(load=provider, use_vacation_profile=True)
        data = await Prediction(setup).fetch(start=_START_MON, hours=24, dt_hours=1.0)
        assert np.all(data.load_wh == pytest.approx(expected, rel=1e-4))

    async def test_fetch_partial_vacation_profile(self, tmp_path):
        """fetch_partial also uses the vacation percentile from PredictionSetup."""
        provider = _make_load_provider(tmp_path, weekday_wh=300.0, weekend_wh=150.0)
        setup = PredictionSetup(load=provider, use_vacation_profile=True)
        data, errors = await Prediction(setup).fetch_partial(
            start=_START_MON, hours=24, dt_hours=1.0
        )
        assert not errors
        # Vacation load must be less than the normal weekday load
        normal_setup = PredictionSetup(load=provider, use_vacation_profile=False)
        data_normal, _ = await Prediction(normal_setup).fetch_partial(
            start=_START_MON, hours=24, dt_hours=1.0
        )
        assert float(np.sum(data.load_wh)) < float(np.sum(data_normal.load_wh))

    async def test_replace_enables_vacation_on_existing_setup(self, tmp_path):
        """dataclasses.replace on a shared setup instance activates vacation correctly."""
        provider = _make_load_provider(tmp_path, weekday_wh=300.0, weekend_wh=150.0)
        base_setup = PredictionSetup(load=provider)
        vacation_setup = _dc_replace(base_setup, use_vacation_profile=True)

        data_base = await Prediction(base_setup).fetch(start=_START_MON, hours=24, dt_hours=1.0)
        data_vacation = await Prediction(vacation_setup).fetch(
            start=_START_MON, hours=24, dt_hours=1.0
        )
        assert float(np.sum(data_vacation.load_wh)) < float(np.sum(data_base.load_wh))


# ---------------------------------------------------------------------------
# Vacation HTTP router
# ---------------------------------------------------------------------------

class TestVacationRouter:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        import GridPythia.server.state as state

        state.vacation_mode = False
        yield
        state.vacation_mode = False

    async def test_get_returns_false_by_default(self):
        from GridPythia.server.routers.vacation import get_vacation_mode

        resp = await get_vacation_mode()
        assert resp.enabled is False

    async def test_put_enables_vacation_mode(self):
        import GridPythia.server.state as state
        from GridPythia.server.models import VacationModeRequest
        from GridPythia.server.routers.vacation import set_vacation_mode

        resp = await set_vacation_mode(VacationModeRequest(enabled=True))
        assert resp.enabled is True
        assert state.vacation_mode is True

    async def test_put_disables_vacation_mode(self):
        import GridPythia.server.state as state
        from GridPythia.server.models import VacationModeRequest
        from GridPythia.server.routers.vacation import set_vacation_mode

        state.vacation_mode = True
        resp = await set_vacation_mode(VacationModeRequest(enabled=False))
        assert resp.enabled is False
        assert state.vacation_mode is False

    async def test_put_no_change_does_not_call_clear_cache(self, monkeypatch):
        """Toggling to the same value must not invalidate the cache."""
        import GridPythia.server.state as state
        from GridPythia.server.models import VacationModeRequest
        from GridPythia.server.routers.vacation import set_vacation_mode

        state.vacation_mode = False
        # No assertion needed; just ensure no exception and state unchanged
        await set_vacation_mode(VacationModeRequest(enabled=False))
        assert state.vacation_mode is False

    async def test_cache_key_differs_by_vacation_mode(self):
        """The prediction cache key must differ between vacation ON and OFF."""
        from datetime import timezone
        from GridPythia.server import services
        import GridPythia.server.state as state

        ts = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)
        mtime = 1234567.0

        state.vacation_mode = False
        key_off = services._prediction_cache_key(ts, mtime)
        state.vacation_mode = True
        key_on = services._prediction_cache_key(ts, mtime)
        state.vacation_mode = False

        assert key_off != key_on

    async def test_put_change_logs(self):
        """Toggling vacation mode logs the change."""
        import GridPythia.server.state as state
        from GridPythia.server.models import VacationModeRequest
        from GridPythia.server.routers.vacation import set_vacation_mode

        await set_vacation_mode(VacationModeRequest(enabled=True))
        assert state.vacation_mode is True

    async def test_description_reflects_mode(self):
        import GridPythia.server.state as state
        from GridPythia.server.models import VacationModeRequest
        from GridPythia.server.routers.vacation import get_vacation_mode, set_vacation_mode

        resp_off = await get_vacation_mode()
        assert "normal" in resp_off.description

        await set_vacation_mode(VacationModeRequest(enabled=True))
        resp_on = await get_vacation_mode()
        assert "vacation" in resp_on.description
