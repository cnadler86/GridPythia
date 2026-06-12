"""Tests for services helper functions (SoC mapping, appliance load snapping, config)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from GridPythia.config.optimization import BatteryParameters, InverterParameters
from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.prediction.base import make_timestamps
from GridPythia.server.services import (
    cap_runtime_soc_wh_for_solver,
    snap_appliance_forecasts_to_grid,
    soc_overrides_wh_for_solver,
)
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


def _make_inverter_with_battery(
    device_id: str = "inv1",
    battery_id: str = "bat1",
    capacity_wh: float = 5000.0,
    min_soc_pct: float = 10.0,
    max_soc_pct: float = 90.0,
) -> InverterBase:
    bat_params = BatteryParameters(
        device_id=battery_id,
        capacity_wh=capacity_wh,
        initial_soc_percentage=50.0,
        min_soc_percentage=min_soc_pct,
        max_soc_percentage=max_soc_pct,
    )
    bat = Battery(bat_params)
    inv_params = InverterParameters(
        device_id=device_id,
        battery_id=battery_id,
        has_pv=True,
    )
    return InverterBase(inv_params, battery=bat)


class TestSocOverridesWhForSolver:
    def test_empty_overrides_returns_empty(self):
        inv = _make_inverter_with_battery()
        optimizer = MagicMock()
        optimizer.inverters = [inv]
        result = soc_overrides_wh_for_solver(optimizer, {})
        assert result == {}

    def test_maps_battery_pct_to_inverter_wh(self):
        inv = _make_inverter_with_battery(capacity_wh=10000.0)
        optimizer = MagicMock()
        optimizer.inverters = [inv]
        result = soc_overrides_wh_for_solver(optimizer, {"bat1": 75.0})
        assert "inv1" in result
        assert result["inv1"] == pytest.approx(7500.0)

    def test_clamps_to_min_soc(self):
        inv = _make_inverter_with_battery(
            capacity_wh=10000.0, min_soc_pct=20.0
        )
        optimizer = MagicMock()
        optimizer.inverters = [inv]
        result = soc_overrides_wh_for_solver(optimizer, {"bat1": 5.0})
        # Should clamp to min_soc (20%) → 2000 Wh
        assert result["inv1"] == pytest.approx(2000.0)

    def test_clamps_to_max_soc(self):
        inv = _make_inverter_with_battery(
            capacity_wh=10000.0, max_soc_pct=80.0
        )
        optimizer = MagicMock()
        optimizer.inverters = [inv]
        result = soc_overrides_wh_for_solver(optimizer, {"bat1": 95.0})
        # Should clamp to max_soc (80%) → 8000 Wh
        assert result["inv1"] == pytest.approx(8000.0)

    def test_unknown_battery_id_ignored(self):
        inv = _make_inverter_with_battery()
        optimizer = MagicMock()
        optimizer.inverters = [inv]
        result = soc_overrides_wh_for_solver(optimizer, {"unknown_bat": 50.0})
        assert result == {}


class TestCapRuntimeSocWhForSolver:
    def test_empty_input(self):
        optimizer = MagicMock()
        optimizer.inverters = []
        assert cap_runtime_soc_wh_for_solver(optimizer, {}) == {}

    def test_clamps_within_battery_limits(self):
        inv = _make_inverter_with_battery(
            capacity_wh=10000.0, min_soc_pct=10.0, max_soc_pct=90.0
        )
        optimizer = MagicMock()
        optimizer.inverters = [inv]
        # min_soc_wh = 1000, max_soc_wh = 9000
        result = cap_runtime_soc_wh_for_solver(optimizer, {"inv1": 500.0})
        assert result["inv1"] == pytest.approx(1000.0)

        result = cap_runtime_soc_wh_for_solver(optimizer, {"inv1": 9500.0})
        assert result["inv1"] == pytest.approx(9000.0)

    def test_within_bounds_unchanged(self):
        inv = _make_inverter_with_battery(capacity_wh=10000.0)
        optimizer = MagicMock()
        optimizer.inverters = [inv]
        result = cap_runtime_soc_wh_for_solver(optimizer, {"inv1": 5000.0})
        assert result["inv1"] == pytest.approx(5000.0)


class TestSnapApplianceForecastsToGrid:
    def test_empty_forecasts(self):
        ts = make_timestamps(datetime(2026, 5, 30, tzinfo=timezone.utc), 24, 0.25)
        result = snap_appliance_forecasts_to_grid({}, ts, 0.25)
        assert result == {}

    def test_empty_timestamps(self):
        forecasts = {"dev1": [{"time": "2026-05-30T10:00:00+00:00", "load_wh": 100}]}
        result = snap_appliance_forecasts_to_grid(forecasts, [], 0.25)
        assert result == {}

    def test_valid_slots_snapped_to_grid(self):
        # Future start so the slots are not dropped as "in the past".
        start = (datetime.now(tz=timezone.utc) + timedelta(days=1)).replace(
            minute=0, second=0, microsecond=0
        )
        ts = make_timestamps(start, 2, 0.25)
        forecasts = {
            "washer": [
                {"time": ts[0].isoformat(), "load_wh": 100.0},
                {"time": ts[1].isoformat(), "load_wh": 200.0},
            ]
        }
        result = snap_appliance_forecasts_to_grid(forecasts, ts, 0.25)
        assert "washer" in result
        assert result["washer"][0] == pytest.approx(100.0)
        assert result["washer"][1] == pytest.approx(200.0)

    def test_malformed_slot_skipped(self):
        start = (datetime.now(tz=timezone.utc) + timedelta(days=1)).replace(
            minute=0, second=0, microsecond=0
        )
        ts = make_timestamps(start, 1, 0.25)
        forecasts = {
            "dev": [
                {"time": ts[0].isoformat(), "load_wh": 50.0},
                {"load_wh": 100.0},  # missing time
                {"time": "bad-iso", "load_wh": 100.0},  # bad time
            ]
        }
        result = snap_appliance_forecasts_to_grid(forecasts, ts, 0.25)
        assert "dev" in result
        assert result["dev"][0] == pytest.approx(50.0)

    def test_past_slots_dropped(self):
        """Slots before 'now' should be silently dropped."""
        start = datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)
        ts = make_timestamps(start, 1, 0.25)
        past_time = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
        forecasts = {
            "dev": [
                {"time": past_time, "load_wh": 100.0},
            ]
        }
        result = snap_appliance_forecasts_to_grid(forecasts, ts, 0.25)
        # Either empty dict or array of zeros
        if "dev" in result:
            assert float(np.sum(result["dev"])) == 0.0
