"""Tests for the coordination layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from GridPythia.coordination.inverter_coordinator import (
    InverterCoordinator,
    InverterState,
    next_optimization_slot,
)
from GridPythia.simulation.devices import InverterMode


# ── InverterState ─────────────────────────────────────────────────────────


class TestInverterState:
    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def test_is_fresh_when_new(self):
        state = InverterState(
            device_id="inv1", soc=60.0, mode=InverterMode.IDLE, reported_at=self._now()
        )
        assert state.is_fresh(max_age_s=300) is True

    def test_is_stale_when_old(self):
        old = self._now() - timedelta(seconds=400)
        state = InverterState(device_id="inv1", soc=60.0, mode=InverterMode.IDLE, reported_at=old)
        assert state.is_fresh(max_age_s=300) is False

    def test_age_seconds(self):
        t0 = self._now()
        state = InverterState(device_id="inv1", soc=50.0, mode=InverterMode.IDLE, reported_at=t0)
        now = t0 + timedelta(seconds=42)
        assert abs(state.age_s(now) - 42.0) < 0.01


# ── InverterCoordinator ───────────────────────────────────────────────────


class TestInverterCoordinator:
    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def test_unknown_device_not_fresh(self):
        coord = InverterCoordinator(max_age_s=300)
        assert coord.is_fresh("unknown") is False

    def test_update_and_read(self):
        coord = InverterCoordinator(max_age_s=300)
        coord.update_status("inv1", soc=75.0, mode=InverterMode.DISCHARGE)
        state = coord.get_state("inv1")
        assert state is not None
        assert state.soc == 75.0
        assert state.mode == InverterMode.DISCHARGE

    def test_fresh_after_update(self):
        coord = InverterCoordinator(max_age_s=300)
        coord.update_status("inv1", soc=50.0)
        assert coord.is_fresh("inv1") is True

    def test_stale_after_max_age(self):
        coord = InverterCoordinator(max_age_s=60)
        old_ts = self._now() - timedelta(seconds=120)
        coord.update_status("inv1", soc=50.0, reported_at=old_ts)
        assert coord.is_fresh("inv1") is False

    def test_invalid_soc_raises(self):
        coord = InverterCoordinator()
        with pytest.raises(ValueError, match="soc"):
            coord.update_status("inv1", soc=150.0)

    def test_naive_timestamp_raises(self):
        coord = InverterCoordinator()
        with pytest.raises(ValueError, match="timezone-aware"):
            coord.update_status("inv1", soc=50.0, reported_at=datetime.now())

    def test_mode_as_int(self):
        coord = InverterCoordinator()
        coord.update_status("inv1", soc=50.0, mode=2)  # DISCHARGE_ZFI
        state = coord.get_state("inv1")
        assert state is not None
        assert state.mode == InverterMode.DISCHARGE_ZERO_FEED_IN

    def test_snapshot_is_copy(self):
        coord = InverterCoordinator()
        coord.update_status("inv1", soc=50.0)
        snap = coord.snapshot()
        snap["inv1"] = None  # type: ignore
        assert coord.get_state("inv1") is not None


class TestInverterCoordinatorReadiness:
    """Tests for all_optimizable_ready and optimizer input helpers."""

    def _make_inverter(self, device_id: str, *, is_optimizable: bool = True, capacity_wh: float = 1920.0):
        from unittest.mock import MagicMock
        inv = MagicMock()
        inv.device_id = device_id
        inv.is_optimizable = is_optimizable
        if is_optimizable:
            inv.battery = MagicMock()
            inv.battery.capacity_wh = capacity_wh
        else:
            inv.battery = None
        return inv

    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def test_all_ready_when_all_fresh(self):
        coord = InverterCoordinator(max_age_s=300)
        inv = self._make_inverter("inv1")
        coord.update_status("inv1", soc=60.0)
        ready, missing = coord.all_optimizable_ready([inv])
        assert ready is True
        assert missing == []

    def test_not_ready_when_no_status(self):
        coord = InverterCoordinator(max_age_s=300)
        inv = self._make_inverter("inv1")
        ready, missing = coord.all_optimizable_ready([inv])
        assert ready is False
        assert "inv1" in missing

    def test_not_ready_when_stale(self):
        coord = InverterCoordinator(max_age_s=60)
        inv = self._make_inverter("inv1")
        old = self._now() - timedelta(seconds=120)
        coord.update_status("inv1", soc=50.0, reported_at=old)
        ready, missing = coord.all_optimizable_ready([inv])
        assert ready is False
        assert "inv1" in missing

    def test_non_optimizable_skipped(self):
        coord = InverterCoordinator(max_age_s=300)
        inv = self._make_inverter("pv_only", is_optimizable=False)
        # No status update – should still be "ready" because it's not optimizable
        ready, missing = coord.all_optimizable_ready([inv])
        assert ready is True
        assert missing == []

    def test_soc_overrides_wh(self):
        coord = InverterCoordinator(max_age_s=300)
        inv = self._make_inverter("inv1", capacity_wh=2000.0)
        coord.update_status("inv1", soc=50.0)
        overrides = coord.get_soc_overrides_wh([inv])
        assert "inv1" in overrides
        assert abs(overrides["inv1"] - 1000.0) < 0.01  # 50% of 2000 Wh

    def test_soc_overrides_excludes_no_battery(self):
        coord = InverterCoordinator(max_age_s=300)
        inv = self._make_inverter("pv_only", is_optimizable=False)
        coord.update_status("pv_only", soc=0.0)
        overrides = coord.get_soc_overrides_wh([inv])
        assert overrides == {}

    def test_get_initial_modes(self):
        coord = InverterCoordinator(max_age_s=300)
        inv = self._make_inverter("inv1")
        coord.update_status("inv1", soc=50.0, mode=InverterMode.AC_CHARGE_ZERO_FEED_IN)
        modes = coord.get_initial_modes([inv])
        assert modes["inv1"] == InverterMode.AC_CHARGE_ZERO_FEED_IN


# ── next_optimization_slot ────────────────────────────────────────────────


class TestNextOptimizationSlot:
    UTC = timezone.utc

    def _dt(self, h: int, m: int, s: int = 0) -> datetime:
        return datetime(2026, 4, 23, h, m, s, tzinfo=self.UTC)

    def test_fires_at_next_quarter(self):
        # 14:07 → next is 14:15
        assert next_optimization_slot(self._dt(14, 7), 15) == self._dt(14, 15)

    def test_fires_at_top_of_hour(self):
        # 14:47 → next is 15:00
        assert next_optimization_slot(self._dt(14, 47), 15) == self._dt(15, 0)

    def test_on_boundary_stays_on_boundary(self):
        # Exactly on slot within 1 s grace
        assert next_optimization_slot(self._dt(14, 0, 0), 15) == self._dt(14, 0)

    def test_30_min_interval(self):
        assert next_optimization_slot(self._dt(14, 7), 30) == self._dt(14, 30)

    def test_60_min_interval(self):
        assert next_optimization_slot(self._dt(14, 20), 60) == self._dt(15, 0)

    def test_invalid_interval_not_divisor(self):
        with pytest.raises(ValueError, match="divisor"):
            next_optimization_slot(self._dt(14, 0), 7)

    def test_invalid_interval_zero(self):
        with pytest.raises(ValueError):
            next_optimization_slot(self._dt(14, 0), 0)

    @pytest.mark.parametrize("interval", [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60])
    def test_valid_divisors_of_60(self, interval: int):
        slot = next_optimization_slot(self._dt(14, 7), interval)
        # Result must be >= now (on-boundary grace) and aligned to the grid
        assert slot >= self._dt(14, 7)
        total_minutes = slot.hour * 60 + slot.minute
        assert total_minutes % interval == 0, f"slot {slot} not aligned to {interval} min grid"
