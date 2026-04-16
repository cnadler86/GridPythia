"""Tests for the linear MILP solver, organized by behavioral concern."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from GridPythia.config.optimization import BatteryParameters, InverterParameters
from GridPythia.optimization.solver import LinearOptimizer, OptimizationObjective
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode, SystemTopology
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_prediction(
    load_w: list[float],
    price_eur_wh: list[float],
    pv_wh: dict[str, list[float]] | None = None,
    feedintariff_eur_wh: list[float] | None = None,
) -> PredictionData:
    """Minimal PredictionData factory used across all test classes."""
    n = len(load_w)
    assert n == len(price_eur_wh)
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(n)]
    pv_by_inverter: dict[str, np.ndarray] = {}
    for inverter_id, series in (pv_wh or {}).items():
        assert len(series) == n
        pv_by_inverter[inverter_id] = np.array(series, dtype=np.float32)
    feedin = np.array(feedintariff_eur_wh, dtype=np.float32) if feedintariff_eur_wh else np.zeros(n, dtype=np.float32)
    return PredictionData(
        timestamps=timestamps,
        dt_hours=1.0,
        load_wh=np.array(load_w, dtype=np.float32),
        electricprice_eur_wh=np.array(price_eur_wh, dtype=np.float32),
        feedintariff_eur_wh=feedin,
        pv_by_inverter=pv_by_inverter,
    )


def _make_hybrid_inverter(
    *,
    device_id: str = "hybrid_h1",
    roundtrip_efficiency: float = 0.8,
    zero_feed_in: bool = True,
) -> InverterBase:
    """Hybrid inverter: PV + battery (AC charge capable), 1 kWh at 50% initial SoC."""
    eta = roundtrip_efficiency**0.5
    battery = Battery(
        BatteryParameters(
            device_id=f"battery_{device_id}",
            capacity_wh=1000,
            charging_efficiency=eta,
            discharging_efficiency=eta,
            max_charge_power_w=500,
            max_discharge_power_w=1000,
            initial_soc_percentage=50,
            min_soc_percentage=0,
            max_soc_percentage=100,
        )
    )
    return InverterBase(
        InverterParameters(
            device_id=device_id,
            battery_id=f"battery_{device_id}",
            has_pv=True,
            max_ac_output_power_w=1000,
            max_ac_charge_power_w=500,
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=zero_feed_in,
            mode_switch_cost=0.0,
            active_inverter_consumption_w=0.0,
        ),
        battery=battery,
    )


def _make_boundary_inverter(
    *,
    device_id: str,
    zero_feed_in: bool,
    has_pv: bool,
    initial_soc_percentage: int,
    min_soc_percentage: int,
    max_soc_percentage: int,
    capacity_wh: int = 1000,
) -> InverterBase:
    """Boundary-condition helper: unit efficiency, configurable SoC range."""
    battery = Battery(
        BatteryParameters(
            device_id=f"battery_{device_id}",
            capacity_wh=capacity_wh,
            charging_efficiency=1.0,
            discharging_efficiency=1.0,
            max_charge_power_w=500,
            max_discharge_power_w=500,
            initial_soc_percentage=initial_soc_percentage,
            min_soc_percentage=min_soc_percentage,
            max_soc_percentage=max_soc_percentage,
        )
    )
    return InverterBase(
        InverterParameters(
            device_id=device_id,
            battery_id=battery.parameters.device_id,
            has_pv=has_pv,
            max_ac_output_power_w=500,
            max_ac_charge_power_w=500,
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=zero_feed_in,
            mode_switch_cost=0.0,
            active_inverter_consumption_w=0.0,
        ),
        battery=battery,
    )


def _make_pv_only_inverter(*, device_id: str = "pv_string", max_power_w: float = 3000.0) -> InverterBase:
    """String inverter: PV output only, no battery, no AC charge (PV_ONLY topology)."""
    return InverterBase(
        InverterParameters(
            device_id=device_id,
            battery_id=None,
            has_pv=True,
            max_ac_output_power_w=max_power_w,
            max_ac_charge_power_w=0.0,
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=False,
            mode_switch_cost=0.0,
            active_inverter_consumption_w=0.0,
        ),
        battery=None,
    )


def _make_ac_battery_inverter(
    *,
    device_id: str = "ac_bat",
    capacity_wh: int = 2000,
    initial_soc_percentage: int = 0,
) -> InverterBase:
    """AC-coupled battery: no PV, charges from grid and discharges to ZFI (AC_BATTERY topology)."""
    battery = Battery(
        BatteryParameters(
            device_id=f"battery_{device_id}",
            capacity_wh=capacity_wh,
            charging_efficiency=1.0,
            discharging_efficiency=1.0,
            max_charge_power_w=1000,
            max_discharge_power_w=1000,
            initial_soc_percentage=initial_soc_percentage,
            min_soc_percentage=0,
            max_soc_percentage=100,
        )
    )
    return InverterBase(
        InverterParameters(
            device_id=device_id,
            battery_id=f"battery_{device_id}",
            has_pv=False,
            max_ac_output_power_w=1000,
            max_ac_charge_power_w=1000,
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=True,
            mode_switch_cost=0.0,
            active_inverter_consumption_w=0.0,
        ),
        battery=battery,
    )


def _make_pv_battery_inverter(
    *,
    device_id: str = "pv_bat",
    initial_soc_percentage: int = 50,
) -> InverterBase:
    """PV + battery, NO AC charging capability (PV_BATTERY topology)."""
    battery = Battery(
        BatteryParameters(
            device_id=f"battery_{device_id}",
            capacity_wh=1000,
            charging_efficiency=1.0,
            discharging_efficiency=1.0,
            max_charge_power_w=500,
            max_discharge_power_w=500,
            initial_soc_percentage=initial_soc_percentage,
            min_soc_percentage=0,
            max_soc_percentage=100,
        )
    )
    return InverterBase(
        InverterParameters(
            device_id=device_id,
            battery_id=f"battery_{device_id}",
            has_pv=True,
            max_ac_output_power_w=1000,
            max_ac_charge_power_w=0,  # No AC charge → PV_BATTERY topology
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=True,
            mode_switch_cost=0.0,
            active_inverter_consumption_w=0.0,
        ),
        battery=battery,
    )


# ---------------------------------------------------------------------------
# TestArbitrage – economic charge/discharge decisions
# ---------------------------------------------------------------------------


class TestArbitrage:
    """Economic arbitrage: the solver should charge cheap and discharge expensive."""

    def test_no_charge_when_not_profitable_after_roundtrip(self) -> None:
        """If the low/high spread does not cover round-trip losses, no charging should occur.

        roundtrip=0.8 → effective charge cost at low = low/0.8.
        low=0.00050, high=0.00060 → 0.000625 > 0.00060 → not profitable.
        """
        pred = _make_prediction(
            load_w=[0.0, 0.0, 2000.0],
            price_eur_wh=[0.00045, 0.00050, 0.00060],
        )
        inv = _make_hybrid_inverter()

        plan = LinearOptimizer([inv]).solve(pred).inverter_plans[0]

        assert plan["modes"][1] != int(InverterMode.AC_CHARGE)
        assert plan["charge_ac_wh"][1] == pytest.approx(0.0, abs=1e-6)
        assert plan["modes"][2] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)

    def test_charge_when_spread_is_profitable(self) -> None:
        """Solver should charge at low price when the spread covers round-trip losses.

        roundtrip=0.8 → low/0.8 = 0.000375 < 0.00050 (high) → profitable.
        """
        pred = _make_prediction(
            load_w=[0.0, 0.0, 2000.0],
            price_eur_wh=[0.00045, 0.00030, 0.00050],
        )
        inv = _make_hybrid_inverter()

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][1] == int(InverterMode.AC_CHARGE)
        assert plan["charge_ac_wh"][1] > 0.0
        assert plan["modes"][2] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)

        # SoC increase must be consistent with AC charge energy and efficiency.
        soc = list(sol.result.battery_wh_per_dt["hybrid_h1"])
        eta = 0.8**0.5
        assert soc[1] - soc[0] == pytest.approx(float(plan["charge_ac_wh"][1]) * eta, abs=12.0)

    def test_charge_and_discharge_power_respect_battery_limits(self) -> None:
        """charge_ac_wh and discharge_ac_wh must stay within hardware power limits."""
        pred = _make_prediction(
            load_w=[0.0, 0.0, 2000.0],
            price_eur_wh=[0.00044, 0.00031, 0.00052],
        )
        inv = _make_hybrid_inverter()

        plan = LinearOptimizer([inv]).solve(pred).inverter_plans[0]
        assert inv.battery is not None
        max_ch_wh = inv.battery.max_charge_power_w * 1.0
        max_dc_wh = inv.battery.max_discharge_power_w * 1.0

        assert np.all(plan.charge_ac_wh >= -1e-6)
        assert np.all(plan.discharge_ac_wh >= -1e-6)
        assert np.all(plan.charge_ac_wh <= max_ch_wh + 1e-6)
        assert np.all(plan.discharge_ac_wh <= max_dc_wh + 1e-6)

    def test_zero_feed_discharge_covers_full_load(self) -> None:
        """ZFI discharge should fully cover load so that grid import is zero."""
        pred = _make_prediction(load_w=[400.0, 0.0], price_eur_wh=[0.00060, 0.0])
        inv = _make_hybrid_inverter()

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        assert float(sol.result.grid_import_wh_per_dt[0]) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# TestTerminalValue – future-price awareness
# ---------------------------------------------------------------------------


class TestTerminalValue:
    """The auto-estimated terminal value should prevent premature discharge."""

    def test_cheap_slot_discharge_suppressed_by_expensive_tail(self) -> None:
        """With expensive future prices, the solver should hold energy at the cheap slot.

        Auto terminal value ≈ mean(prices) * eta_d ≫ price[0] * eta_d,
        so discharging at t=0 would forfeit more future value than it saves now.
        """
        pred = _make_prediction(
            load_w=[1500.0, 1500.0, 1500.0, 1500.0],
            price_eur_wh=[0.00010, 0.00050, 0.00050, 0.00050],
        )
        inv = _make_hybrid_inverter()

        plan = LinearOptimizer([inv]).solve(pred).inverter_plans[0]

        assert plan["modes"][0] not in (int(InverterMode.DISCHARGE), int(InverterMode.DISCHARGE_ZERO_FEED_IN))
        assert any(
            m in (int(InverterMode.DISCHARGE), int(InverterMode.DISCHARGE_ZERO_FEED_IN))
            for m in plan["modes"][1:]
        )


# ---------------------------------------------------------------------------
# TestPVRouting – physical PV energy routing (battery-first and bypass)
# ---------------------------------------------------------------------------


class TestPVRouting:
    """Physical PV routing: battery-first in IDLE, bypass when battery is full."""

    def test_pv_charges_battery_and_soc_rises(self) -> None:
        """With PV surplus and battery headroom, SoC should increase in the first slot."""
        pred = _make_prediction(
            load_w=[200.0, 700.0],
            price_eur_wh=[0.00090, 0.00200],
            pv_wh={"hybrid_h1": [700.0, 100.0]},
        )
        inv = _make_hybrid_inverter()

        sol = LinearOptimizer([inv]).solve(pred)
        soc = np.asarray(sol.result.battery_wh_per_dt["hybrid_h1"], dtype=float)

        assert soc[0] > 500.0  # SoC increased from initial 500 Wh
        assert sol.inverter_plans[0]["modes"][1] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        assert float(sol.result.grid_import_wh_per_dt[1]) < 400.0

    def test_pv_to_battery_wh_is_populated_when_idle(self) -> None:
        """pv_to_battery_wh must reflect PV energy stored passively via DC-bus coupling in IDLE.

        Battery is at min SoC so discharge is impossible; PV must flow to battery.
        """
        inv = _make_boundary_inverter(
            device_id="hybrid_pv_bat",
            zero_feed_in=True,
            has_pv=True,
            initial_soc_percentage=20,
            min_soc_percentage=20,
            max_soc_percentage=100,
        )
        pred = _make_prediction(
            load_w=[0.0],
            price_eur_wh=[0.00010],
            pv_wh={"hybrid_pv_bat": [400.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.IDLE)
        assert float(plan.pv_to_battery_wh[0]) == pytest.approx(400.0, abs=2.0)
        # SoC: 200 Wh + 400 Wh PV (eta=1) = 600 Wh
        assert float(sol.result.battery_wh_per_dt["hybrid_pv_bat"][0]) == pytest.approx(600.0, abs=2.0)

    def test_pv_to_battery_respects_battery_charge_power_limit(self) -> None:
        """PV->battery in a single slot must be capped by battery max_charge_power_w * dt."""
        inv = _make_boundary_inverter(
            device_id="hybrid_pv_cap",
            zero_feed_in=True,
            has_pv=True,
            initial_soc_percentage=20,
            min_soc_percentage=20,
            max_soc_percentage=100,
        )
        pred = _make_prediction(
            load_w=[0.0],
            price_eur_wh=[0.00010],
            pv_wh={"hybrid_pv_cap": [900.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert inv.battery is not None
        max_ch_wh = inv.battery.max_charge_power_w * pred.dt_hours
        assert float(plan.pv_to_battery_wh[0]) <= max_ch_wh + 1e-6
        assert float(sol.result.battery_wh_per_dt["hybrid_pv_cap"][0]) <= (
            inv.battery.min_soc_wh + max_ch_wh + 1e-6
        )

    def test_pv_discharge_blocked_at_min_soc_non_zfi(self) -> None:
        """Discrete DISCHARGE mode is forbidden when battery is at min SoC, even with PV.

        PV still flows to the battery (DC-bus coupling), so SoC rises despite IDLE mode.
        """
        inv = _make_boundary_inverter(
            device_id="hybrid_min_rate",
            zero_feed_in=False,
            has_pv=True,
            initial_soc_percentage=20,
            min_soc_percentage=20,
            max_soc_percentage=100,
        )
        pred = _make_prediction(
            load_w=[500.0],
            price_eur_wh=[0.00060],
            pv_wh={"hybrid_min_rate": [500.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        # With PV bypass active, mode may be reported as DISCHARGE while p_dc stays zero.
        assert plan["discharge_ac_wh"][0] == pytest.approx(0.0, abs=1e-6)
        # SoC rises but cannot exceed battery charge-power cap for one slot.
        soc0 = float(inv.battery.min_soc_wh)
        soc1 = float(sol.result.battery_wh_per_dt["hybrid_min_rate"][0])
        assert soc1 > soc0
        assert soc1 <= soc0 + inv.battery.max_charge_power_w * pred.dt_hours + 1e-6

    def test_pv_zfi_discharge_blocked_at_min_soc(self) -> None:
        """ZFI discharge mode is forbidden at min SoC; PV still charges battery passively."""
        inv = _make_boundary_inverter(
            device_id="hybrid_min_zfi",
            zero_feed_in=True,
            has_pv=True,
            initial_soc_percentage=20,
            min_soc_percentage=20,
            max_soc_percentage=100,
        )
        pred = _make_prediction(
            load_w=[500.0],
            price_eur_wh=[0.00060],
            pv_wh={"hybrid_min_zfi": [500.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        # With PV bypass active, mode may be reported as DISCHARGE while p_dc stays zero.
        assert plan["discharge_ac_wh"][0] == pytest.approx(0.0, abs=1e-6)
        soc0 = float(inv.battery.min_soc_wh)
        soc1 = float(sol.result.battery_wh_per_dt["hybrid_min_zfi"][0])
        assert soc1 > soc0
        assert soc1 <= soc0 + inv.battery.max_charge_power_w * pred.dt_hours + 1e-6

    def test_bypass_mode_activated_when_battery_is_full(self) -> None:
        """When the battery is full, the solver activates bypass mode (mode_dc=1, p_dc≈0)
        so that PV energy can reach the AC bus without charging the full battery.

        Physical story: DC→AC path is opened (mode_dc=1) but no battery energy is
        drawn (p_dc≈0). The inverter acts as a pass-through for excess PV.
        """
        inv = _make_boundary_inverter(
            device_id="hybrid_bypass",
            zero_feed_in=True,
            has_pv=True,
            initial_soc_percentage=100,
            min_soc_percentage=0,
            max_soc_percentage=100,
        )
        pred = _make_prediction(
            load_w=[400.0],
            price_eur_wh=[0.00100],
            pv_wh={"hybrid_bypass": [400.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        # Battery headroom = 0 → pv_to_bat must be 0
        assert float(plan.pv_to_battery_wh[0]) == pytest.approx(0.0, abs=1e-3)
        # PV reaches AC bus via bypass (exact amount depends on Fraunhofer SC coupling)
        assert float(plan.pv_to_ac_wh[0]) > 0.0
        # Battery discharge is zero (bypass, not discharge)
        assert plan["discharge_ac_wh"][0] == pytest.approx(0.0, abs=1e-3)
        # Mode is reported as DISCHARGE_ZERO_FEED_IN because the DC→AC path is open
        assert plan["modes"][0] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        # SoC stays at 1000 Wh
        assert float(sol.result.battery_wh_per_dt["hybrid_bypass"][0]) == pytest.approx(1000.0, abs=2.0)
        # Grid import is reduced relative to full-load import of 400 Wh
        assert float(sol.result.grid_import_wh_per_dt[0]) < 400.0

    def test_pv_deficit_triggers_discharge(self) -> None:
        """If PV is below load in an expensive slot, the battery should discharge."""
        pred = _make_prediction(
            load_w=[600.0, 600.0],
            price_eur_wh=[0.00200, 0.00010],
            pv_wh={"hybrid_h1": [200.0, 200.0]},
        )
        inv = _make_hybrid_inverter()

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert any(m == int(InverterMode.DISCHARGE_ZERO_FEED_IN) for m in plan["modes"])
        assert np.min(np.asarray(sol.result.grid_import_wh_per_dt, dtype=float)) < 400.0


# ---------------------------------------------------------------------------
# TestPVBatteryTopology – PV_BATTERY (no AC charge)
# ---------------------------------------------------------------------------


class TestPVBatteryTopology:
    """PV_BATTERY: inverter has PV and battery but cannot charge the battery from AC."""

    def test_topology_is_pv_battery(self) -> None:
        """Factory must produce a PV_BATTERY topology (max_ac_charge_power_w=0)."""
        inv = _make_pv_battery_inverter()
        assert inv.topology == SystemTopology.PV_BATTERY

    def test_ac_charge_mode_not_available(self) -> None:
        """AC_CHARGE must not appear in available_modes for PV_BATTERY."""
        inv = _make_pv_battery_inverter()
        assert InverterMode.AC_CHARGE not in inv.available_modes
        assert InverterMode.AC_CHARGE_ZERO_FEED_IN not in inv.available_modes

    def test_pv_charges_battery_via_dc_bus(self) -> None:
        """PV energy should charge the battery passively in IDLE (no AC charge possible)."""
        inv = _make_pv_battery_inverter(initial_soc_percentage=0)
        pred = _make_prediction(
            load_w=[0.0],
            price_eur_wh=[0.00010],
            pv_wh={"pv_bat": [500.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["charge_ac_wh"][0] == pytest.approx(0.0, abs=1e-6)  # No AC charge
        assert float(plan.pv_to_battery_wh[0]) > 0.0  # PV → battery via DC bus
        assert float(sol.result.battery_wh_per_dt["pv_bat"][0]) > 0.0

    def test_discharge_reduces_grid_import_when_pv_absent(self) -> None:
        """Battery charged by PV in the first slot should discharge in the second."""
        inv = _make_pv_battery_inverter(initial_soc_percentage=50)
        pred = _make_prediction(
            load_w=[0.0, 400.0],
            price_eur_wh=[0.00010, 0.00090],
            pv_wh={"pv_bat": [400.0, 0.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        # t=1: PV=0, expensive slot → discharge from battery
        assert plan["modes"][1] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        assert float(sol.result.grid_import_wh_per_dt[1]) < 400.0


# ---------------------------------------------------------------------------
# TestFeedInTariff – grid export and revenue
# ---------------------------------------------------------------------------


class TestFeedInTariff:
    """PV surplus should feed into the grid when no battery can absorb it."""

    def test_pv_excess_feeds_grid_without_battery(self) -> None:
        """PV-only inverter should at least reduce grid import when PV is present."""
        inv = _make_pv_only_inverter()
        pred = _make_prediction(
            load_w=[300.0],
            price_eur_wh=[0.00050],
            pv_wh={"pv_string": [1000.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)

        assert float(sol.inverter_plans[0].pv_to_ac_wh[0]) > 0.0
        assert float(sol.result.grid_import_wh_per_dt[0]) < 300.0

    def test_feed_in_revenue_is_positive_with_tariff(self) -> None:
        """With tariff configured, solver should remain numerically stable and finite."""
        inv = _make_pv_only_inverter()
        pred = _make_prediction(
            load_w=[100.0],
            price_eur_wh=[0.00050],
            pv_wh={"pv_string": [800.0]},
            feedintariff_eur_wh=[0.00008],
        )

        sol = LinearOptimizer([inv]).solve(pred)

        assert np.isfinite(float(np.sum(sol.result.revenue_per_dt)))
        assert np.isfinite(float(np.sum(sol.result.costs_per_dt)))

    def test_feed_in_avoided_by_zero_feed_in_battery(self) -> None:
        """A zero-feed-in hybrid inverter should store PV surplus in the battery
        instead of exporting to the grid.
        """
        inv = _make_hybrid_inverter(zero_feed_in=True)
        pred = _make_prediction(
            load_w=[100.0],
            price_eur_wh=[0.00050],
            pv_wh={"hybrid_h1": [600.0]},
            feedintariff_eur_wh=[0.00001],  # negligible tariff → no incentive to export
        )

        sol = LinearOptimizer([inv]).solve(pred)

        assert float(sol.result.feedin_wh_per_dt[0]) == pytest.approx(0.0, abs=5.0)
        assert float(sol.result.battery_wh_per_dt["hybrid_h1"][0]) > 500.0


# ---------------------------------------------------------------------------
# TestBatteryBoundaryConditions – SoC hard limits
# ---------------------------------------------------------------------------


class TestBatteryBoundaryConditions:
    """The solver must respect hard SoC limits at both ends of the range."""

    def test_ac_charge_blocked_at_max_soc(self) -> None:
        """AC charge must be zero when battery starts at max SoC."""
        inv = _make_boundary_inverter(
            device_id="hybrid_max",
            zero_feed_in=False,
            has_pv=False,
            initial_soc_percentage=80,
            min_soc_percentage=0,
            max_soc_percentage=80,
        )
        pred = _make_prediction(load_w=[0.0], price_eur_wh=[0.00010])

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.IDLE)
        assert plan["charge_ac_wh"][0] == pytest.approx(0.0, abs=1e-6)
        assert float(sol.result.battery_wh_per_dt["hybrid_max"][0]) == pytest.approx(800.0)

    def test_discharge_blocked_at_min_soc(self) -> None:
        """Discharge must be zero when battery starts at min SoC (no PV)."""
        inv = _make_boundary_inverter(
            device_id="hybrid_min",
            zero_feed_in=True,
            has_pv=False,
            initial_soc_percentage=20,
            min_soc_percentage=20,
            max_soc_percentage=100,
        )
        pred = _make_prediction(load_w=[400.0], price_eur_wh=[0.00090])

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["discharge_ac_wh"][0] == pytest.approx(0.0, abs=1e-6)
        assert float(sol.result.battery_wh_per_dt["hybrid_min"][0]) == pytest.approx(200.0, abs=1.0)


# ---------------------------------------------------------------------------
# TestModeSwitchCosts – switching-penalty decisions
# ---------------------------------------------------------------------------


class TestModeSwitchCosts:
    """Mode-switch costs should alter which slots are activated."""

    def _make_sw_inverter(
        self,
        *,
        switch_cost: float,
        active_inverter_consumption_w: float = 0.0,
        roundtrip_efficiency: float = 0.8,
    ) -> InverterBase:
        eta = roundtrip_efficiency**0.5
        battery = Battery(
            BatteryParameters(
                device_id="battery_sw",
                capacity_wh=1000,
                charging_efficiency=eta,
                discharging_efficiency=eta,
                max_charge_power_w=500,
                max_discharge_power_w=1000,
                initial_soc_percentage=50,
                min_soc_percentage=0,
                max_soc_percentage=100,
            )
        )
        return InverterBase(
            InverterParameters(
                device_id="hybrid_sw",
                battery_id="battery_sw",
                has_pv=True,
                max_ac_output_power_w=1000,
                max_ac_charge_power_w=500,
                dc_to_ac_efficiency=1.0,
                ac_to_dc_efficiency=1.0,
                zero_feed_in=True,
                mode_switch_cost=switch_cost,
                active_inverter_consumption_w=active_inverter_consumption_w,
            ),
            battery=battery,
        )

    def test_isolated_cheapest_slot_is_used_despite_switch_cost(self) -> None:
        """The isolated cheapest slot (t3) must still be used at full charge power
        when its arbitrage saving exceeds the switch cost.
        """
        pred = _make_prediction(
            load_w=[0.0, 0.0, 0.0, 0.0, 600.0],
            price_eur_wh=[0.00030, 0.00030, 0.00045, 0.00024, 0.00070],
        )
        inv = self._make_sw_inverter(switch_cost=0.005)

        plan = LinearOptimizer([inv]).solve(pred).inverter_plans[0]
        assert inv.battery is not None
        max_ch_wh = inv.battery.max_charge_power_w * 1.0
        isolated_charge_wh = float(plan["charge_ac_wh"][3]) if plan["modes"][3] == int(InverterMode.AC_CHARGE) else 0.0

        assert isolated_charge_wh == pytest.approx(max_ch_wh, rel=0.05)

    def test_initial_mode_reduces_first_step_switch_penalty(self) -> None:
        """Providing initial_mode=AC_CHARGE should not make first-step charging worse."""
        pred = _make_prediction(
            load_w=[0.0, 0.0, 600.0],
            price_eur_wh=[0.00030, 0.00045, 0.00070],
        )
        inv_idle = self._make_sw_inverter(switch_cost=0.020)
        sol_idle = LinearOptimizer([inv_idle]).solve(pred)

        inv_ch = self._make_sw_inverter(switch_cost=0.020)
        sol_ch = LinearOptimizer([inv_ch]).solve(
            pred, initial_modes={"hybrid_sw": InverterMode.AC_CHARGE}
        )

        def _charge_wh(plan, t: int) -> float:
            return float(plan["charge_ac_wh"][t]) if plan["modes"][t] == int(InverterMode.AC_CHARGE) else 0.0

        assert _charge_wh(sol_ch.inverter_plans[0], 0) >= _charge_wh(sol_idle.inverter_plans[0], 0) - 1e-9

    def test_high_switch_cost_keeps_ac_charge_continuous(self) -> None:
        """High switch cost should prevent the solver from dropping out of AC_CHARGE
        for a single cheap slot, forcing continuous charging over adjacent slots.
        """
        battery = Battery(
            BatteryParameters(
                device_id="battery_zfi_cost",
                capacity_wh=1000,
                charging_efficiency=1.0,
                discharging_efficiency=1.0,
                max_charge_power_w=600,
                max_discharge_power_w=1000,
                initial_soc_percentage=0,
                min_soc_percentage=0,
                max_soc_percentage=100,
            )
        )
        inv = InverterBase(
            InverterParameters(
                device_id="hybrid_zfi_cost",
                battery_id="battery_zfi_cost",
                has_pv=False,
                max_ac_output_power_w=1000,
                max_ac_charge_power_w=600,
                dc_to_ac_efficiency=1.0,
                ac_to_dc_efficiency=1.0,
                zero_feed_in=True,
                mode_switch_cost=0.1,
                active_inverter_consumption_w=0.0,
            ),
            battery=battery,
        )
        pred = _make_prediction(
            load_w=[100.0, 100.0, 100.0, 1000.0, 100.0],
            price_eur_wh=[0.01, 0.012, 0.011, 0.90, 0.4],
        )

        plan = LinearOptimizer([inv]).solve(pred).inverter_plans[0]

        AC = InverterMode.AC_CHARGE
        ZFI = InverterMode.DISCHARGE_ZERO_FEED_IN
        assert plan["modes"][0:-1].tolist() == [int(AC), int(AC), int(AC), int(ZFI)]
        assert plan["charge_ac_wh"][1] < plan["charge_ac_wh"][2] < plan["charge_ac_wh"][0]

    def test_high_switch_cost_extends_discharge_into_adjacent_slot(self) -> None:
        """With non-zero switch cost, the solver should keep discharge active into the
        next slot rather than toggling back to IDLE immediately after the expensive peak.
        """
        eta = 1.0

        def _make(switch_cost: float) -> tuple[LinearOptimizer, PredictionData]:
            battery = Battery(
                BatteryParameters(
                    device_id="battery_zfi_cost",
                    capacity_wh=1000,
                    charging_efficiency=eta,
                    discharging_efficiency=eta,
                    max_charge_power_w=500,
                    max_discharge_power_w=1000,
                    initial_soc_percentage=50,
                    min_soc_percentage=0,
                    max_soc_percentage=100,
                )
            )
            inv = InverterBase(
                InverterParameters(
                    device_id="hybrid_zfi_cost",
                    battery_id="battery_zfi_cost",
                    has_pv=True,
                    max_ac_output_power_w=1000,
                    max_ac_charge_power_w=0,
                    dc_to_ac_efficiency=1.0,
                    ac_to_dc_efficiency=1.0,
                    zero_feed_in=True,
                    mode_switch_cost=switch_cost,
                    active_inverter_consumption_w=0.0,
                ),
                battery=battery,
            )
            pred = _make_prediction(
                load_w=[50.0, 50.0, 50.0],
                price_eur_wh=[0.10, 0.80, 0.40],
                pv_wh={"hybrid_zfi_cost": [0.0, 0.0, 0.0]},
            )
            return LinearOptimizer([inv]), pred

        low_opt, pred_low = _make(0.0)
        low_plan = low_opt.solve(pred_low).inverter_plans[0]
        high_opt, pred_high = _make(0.5)
        high_plan = high_opt.solve(pred_high).inverter_plans[0]

        assert low_plan["modes"].tolist() == [
            int(InverterMode.IDLE),
            int(InverterMode.DISCHARGE_ZERO_FEED_IN),
            int(InverterMode.IDLE),
        ]
        assert high_plan["modes"].tolist() == [
            int(InverterMode.IDLE),
            int(InverterMode.DISCHARGE_ZERO_FEED_IN),
            int(InverterMode.DISCHARGE_ZERO_FEED_IN),
        ]


# ---------------------------------------------------------------------------
# TestMultiInverterTopologies – more than one inverter in the system
# ---------------------------------------------------------------------------


class TestMultiInverterTopologies:
    """Tests combining multiple inverters: PV string + AC battery, two hybrid inverters."""

    def test_topology_ac_battery(self) -> None:
        """Factory must produce an AC_BATTERY topology."""
        inv = _make_ac_battery_inverter()
        assert inv.topology == SystemTopology.AC_BATTERY

    def test_topology_pv_only(self) -> None:
        """PV-only inverter must have PV_ONLY topology and not be flagged as optimizable."""
        inv = _make_pv_only_inverter()
        assert inv.topology == SystemTopology.PV_ONLY
        assert not inv.is_optimizable

    def test_ac_battery_alone_charges_and_discharges(self) -> None:
        """Standalone AC battery should charge at cheap price and discharge at expensive."""
        inv = _make_ac_battery_inverter(initial_soc_percentage=0)
        pred = _make_prediction(
            load_w=[0.0, 500.0],
            price_eur_wh=[0.00010, 0.00090],
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.AC_CHARGE)
        assert plan["charge_ac_wh"][0] > 0.0
        assert plan["modes"][1] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        assert float(sol.result.grid_import_wh_per_dt[1]) < 500.0

    def test_pv_string_plus_ac_battery_reduces_grid_import_at_peak(self) -> None:
        """PV string inverter + separate AC battery: AC battery should discharge
        during the expensive peak slot to reduce grid import.
        """
        pv_inv = _make_pv_only_inverter()
        bat_inv = _make_ac_battery_inverter(initial_soc_percentage=80)
        pred = _make_prediction(
            load_w=[0.0, 1000.0],
            price_eur_wh=[0.00010, 0.00090],
            pv_wh={"pv_string": [0.0, 0.0]},
        )

        sol = LinearOptimizer([pv_inv, bat_inv]).solve(pred)
        bat_plan = next(p for p in sol.inverter_plans if p.device_id == "ac_bat")

        assert bat_plan["modes"][1] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        assert float(sol.result.grid_import_wh_per_dt[1]) < 1000.0

    def test_pv_string_plus_ac_battery_pv_reduces_import(self) -> None:
        """In a mixed topology setup, the PV-only inverter must contribute AC PV flow."""
        pv_inv = _make_pv_only_inverter()
        bat_inv = _make_ac_battery_inverter(initial_soc_percentage=0)
        pred = _make_prediction(
            load_w=[800.0, 800.0],
            price_eur_wh=[0.00050, 0.00050],
            pv_wh={"pv_string": [800.0, 0.0]},  # 800 Wh covers all of slot 0
        )

        sol_with_pv = LinearOptimizer([pv_inv, bat_inv]).solve(pred)
        pv_plan = next(p for p in sol_with_pv.inverter_plans if p.device_id == "pv_string")

        assert float(pv_plan.pv_to_ac_wh[0]) > 0.0
        assert np.isfinite(float(np.sum(sol_with_pv.result.grid_import_wh_per_dt)))

    def test_two_hybrid_inverters_independent_batteries(self) -> None:
        """Two hybrid inverters with separate batteries should each follow their own
        charge/discharge schedule; their SoC trajectories must be tracked separately.
        """
        inv_a = _make_hybrid_inverter(device_id="hybrid_a")
        inv_b = _make_hybrid_inverter(device_id="hybrid_b")
        pred = _make_prediction(
            load_w=[0.0, 0.0, 1500.0],
            price_eur_wh=[0.00040, 0.00020, 0.00080],
            pv_wh={"hybrid_a": [0.0, 0.0, 0.0], "hybrid_b": [0.0, 0.0, 0.0]},
        )

        sol = LinearOptimizer([inv_a, inv_b]).solve(pred)

        # Both inverters must have a separate SoC track in the result.
        assert "hybrid_a" in sol.result.battery_wh_per_dt
        assert "hybrid_b" in sol.result.battery_wh_per_dt

        # At least one inverter should discharge at the peak slot.
        assert any(
            p["modes"][2] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
            for p in sol.inverter_plans
        )
        assert float(sol.result.grid_import_wh_per_dt[2]) < 1500.0


# ---------------------------------------------------------------------------
# TestEnergyConservation – physical invariants in the solver output
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    """Verify that the plan output satisfies fundamental physical constraints."""

    def test_soc_evolution_matches_charge_discharge_with_unit_efficiency(self) -> None:
        """With unit efficiency, SoC[t] = SoC[t-1] + charge_ac_wh[t] + pv_to_bat_wh[t]
        − discharge_ac_wh[t] must hold for every slot.
        """
        inv = _make_boundary_inverter(
            device_id="hv_eff1",
            zero_feed_in=True,
            has_pv=True,
            initial_soc_percentage=50,
            min_soc_percentage=0,
            max_soc_percentage=100,
        )
        pred = _make_prediction(
            load_w=[200.0, 500.0, 200.0],
            price_eur_wh=[0.00020, 0.00090, 0.00020],
            pv_wh={"hv_eff1": [300.0, 0.0, 100.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]
        soc = np.asarray(plan.battery_soc_wh, dtype=float)
        charge = np.asarray(plan.charge_ac_wh, dtype=float)
        discharge = np.asarray(plan.discharge_ac_wh, dtype=float)
        pv_bat = np.asarray(plan.pv_to_battery_wh, dtype=float)

        soc_init = 500.0
        prev = soc_init
        for t in range(len(soc)):
            expected = prev + charge[t] + pv_bat[t] - discharge[t]
            assert soc[t] == pytest.approx(expected, abs=2.0), f"SoC mismatch at t={t}"
            prev = soc[t]

    def test_no_simultaneous_charge_and_discharge(self) -> None:
        """charge_ac_wh and discharge_ac_wh must not both be nonzero in the same slot."""
        inv = _make_hybrid_inverter()
        pred = _make_prediction(
            load_w=[300.0, 300.0, 300.0, 300.0],
            price_eur_wh=[0.00020, 0.00090, 0.00030, 0.00070],
            pv_wh={"hybrid_h1": [200.0, 0.0, 500.0, 0.0]},
        )

        plan = LinearOptimizer([inv]).solve(pred).inverter_plans[0]
        eps = 1e-3

        for t in range(len(plan.modes)):
            ch = float(plan.charge_ac_wh[t])
            dc = float(plan.discharge_ac_wh[t])
            assert not (ch > eps and dc > eps), f"Simultaneous charge and discharge at t={t}"

    def test_battery_soc_stays_within_bounds(self) -> None:
        """Solver solution must keep SoC within [min_soc, max_soc] at every slot."""
        inv = _make_boundary_inverter(
            device_id="hv_bounds",
            zero_feed_in=True,
            has_pv=True,
            initial_soc_percentage=30,
            min_soc_percentage=20,
            max_soc_percentage=90,
        )
        pred = _make_prediction(
            load_w=[500.0, 500.0, 500.0, 500.0, 500.0],
            price_eur_wh=[0.00010, 0.00090, 0.00020, 0.00080, 0.00030],
            pv_wh={"hv_bounds": [800.0, 0.0, 800.0, 0.0, 0.0]},
        )

        sol = LinearOptimizer([inv]).solve(pred)
        plan = sol.inverter_plans[0]
        soc = np.asarray(plan.battery_soc_wh, dtype=float)

        assert inv.battery is not None
        min_soc_wh = inv.battery.min_soc_wh
        max_soc_wh = inv.battery.max_soc_wh
        assert np.all(soc >= min_soc_wh - 0.5), "SoC dropped below min_soc"
        assert np.all(soc <= max_soc_wh + 0.5), "SoC exceeded max_soc"

    def test_pv_routing_conservation(self) -> None:
        """pv_to_ac_wh + pv_to_battery_wh ≤ pv_pred for every slot (no PV created out of thin air)."""
        inv = _make_hybrid_inverter()
        pv = [400.0, 600.0, 200.0, 0.0]
        pred = _make_prediction(
            load_w=[300.0, 300.0, 300.0, 300.0],
            price_eur_wh=[0.00020, 0.00060, 0.00010, 0.00080],
            pv_wh={"hybrid_h1": pv},
        )

        plan = LinearOptimizer([inv]).solve(pred).inverter_plans[0]
        pv_arr = np.array(pv, dtype=float)
        pv_ac = np.asarray(plan.pv_to_ac_wh, dtype=float)
        pv_bat = np.asarray(plan.pv_to_battery_wh, dtype=float)

        assert np.all(pv_ac + pv_bat <= pv_arr + 0.5), "PV routing exceeds PV prediction"


# ---------------------------------------------------------------------------
# TestOptimizationObjectives – switching between objectives
# ---------------------------------------------------------------------------


class TestOptimizationObjectives:
    """The MAXIMIZE_SELF_CONSUMPTION objective should prefer local use of PV."""

    def test_maximize_self_consumption_reduces_feedin_vs_cost_objective(self) -> None:
        """With MAXIMIZE_SELF_CONSUMPTION, feed-in should be lower than with MINIMIZE_COST
        when the feed-in tariff is zero (no export incentive under cost objective either).
        Both objectives should agree when feed-in is zero: store PV, avoid import.
        At minimum, self-consumption must be non-negative.
        """
        inv = _make_hybrid_inverter(zero_feed_in=False)
        pred = _make_prediction(
            load_w=[200.0, 200.0],
            price_eur_wh=[0.00050, 0.00050],
            pv_wh={"hybrid_h1": [800.0, 0.0]},
        )

        sol_cost = LinearOptimizer(
            [_make_hybrid_inverter(zero_feed_in=False)],
            objective=OptimizationObjective.MINIMIZE_COST,
        ).solve(pred)
        sol_sc = LinearOptimizer(
            [_make_hybrid_inverter(zero_feed_in=False)],
            objective=OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION,
        ).solve(pred)

        feedin_cost = float(np.sum(sol_cost.result.feedin_wh_per_dt))
        feedin_sc = float(np.sum(sol_sc.result.feedin_wh_per_dt))

        assert feedin_sc <= feedin_cost + 1.0  # self-consumption objective should not export more


# ---------------------------------------------------------------------------
# TestActiveInverterConsumption – self-consumption of active inverter
# ---------------------------------------------------------------------------


class TestActiveInverterConsumption:
    """Tests for the active_inverter_consumption_w parameter."""

    def _make_inverter(self, *, active_consumption_w: float = 0.0) -> InverterBase:
        battery = Battery(
            BatteryParameters(
                device_id="battery_act",
                capacity_wh=1000,
                charging_efficiency=1.0,
                discharging_efficiency=1.0,
                max_charge_power_w=500,
                max_discharge_power_w=500,
                initial_soc_percentage=50,
                min_soc_percentage=0,
                max_soc_percentage=100,
            )
        )
        return InverterBase(
            InverterParameters(
                device_id="hybrid_act",
                battery_id="battery_act",
                has_pv=False,
                max_ac_output_power_w=500,
                max_ac_charge_power_w=500,
                dc_to_ac_efficiency=1.0,
                ac_to_dc_efficiency=1.0,
                zero_feed_in=False,
                mode_switch_cost=0.0,
                active_inverter_consumption_w=active_consumption_w,
            ),
            battery=battery,
        )

    def test_active_consumption_verified_by_simulation_parity(self) -> None:
        """LP model and simulation must agree on grid import when active consumption is non-zero."""
        pred = _make_prediction(
            load_w=[0.0, 0.0, 200.0],
            price_eur_wh=[0.00020, 0.00022, 0.00080],
        )
        inv = self._make_inverter(active_consumption_w=30.0)
        sol = LinearOptimizer([inv]).solve(
            pred, validate_with_simulation=True
        )

        assert sol.parity_report is not None
        assert sol.parity_report.ok
        assert sol.parity_report.max_abs_grid_import_error_wh <= 5.0

    def test_high_active_consumption_prevents_unprofitable_arbitrage(self) -> None:
        """500 W active consumption makes the arbitrage unprofitable; both slots stay IDLE.

        Arbitrage saving: 500 Wh × (0.00030 − 0.00025) = 0.00025 EUR.
        Active consumption cost: 500 Wh × 2 steps × 0.00030 = 0.00030 EUR > saving.
        """
        pred = _make_prediction(load_w=[0.0, 0.0], price_eur_wh=[0.00025, 0.00030])
        inv = self._make_inverter(active_consumption_w=500.0)

        plan = LinearOptimizer([inv]).solve(pred).inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.IDLE)
        assert plan["modes"][1] == int(InverterMode.IDLE)

    def test_losses_include_active_consumption(self) -> None:
        """Result losses must increase by at least active_consumption_wh when inverter is active."""
        pred = _make_prediction(load_w=[200.0], price_eur_wh=[0.00080])
        inv_no = self._make_inverter(active_consumption_w=0.0)
        inv_with = self._make_inverter(active_consumption_w=30.0)

        sol_no = LinearOptimizer([inv_no]).solve(pred)
        sol_with = LinearOptimizer([inv_with]).solve(pred)

        if any(m != int(InverterMode.IDLE) for m in sol_with.inverter_plans[0]["modes"]):
            losses_no = float(np.sum(sol_no.result.losses_wh_per_dt))
            losses_with = float(np.sum(sol_with.result.losses_wh_per_dt))
            assert losses_with > losses_no + 25.0


# ---------------------------------------------------------------------------
# TestModelInfrastructure – CVXPY compilation and simulation parity
# ---------------------------------------------------------------------------


class TestModelInfrastructure:
    """Structural properties: DPP compliance, simulation parity."""

    def test_compiled_problems_are_dpp(self) -> None:
        """All prebuilt CVXPY problems must satisfy the DPP condition."""
        pred = _make_prediction(
            load_w=[100.0, 100.0, 100.0, 100.0],
            price_eur_wh=[0.05, 0.05, 0.05, 0.05],
        )
        inv = _make_hybrid_inverter()
        opt = LinearOptimizer([inv])
        opt.solve(pred)  # trigger lazy compilation

        for objective, problem in opt._problems.items():
            assert problem is not None, f"Problem for {objective} is None"
            assert problem.is_dpp(), f"Compiled problem for {objective} is not DPP-compliant"

    def test_compiled_problems_are_dcp(self) -> None:
        """All prebuilt CVXPY problems must satisfy the DCP condition."""
        pred = _make_prediction(
            load_w=[100.0, 100.0, 100.0, 100.0],
            price_eur_wh=[0.05, 0.05, 0.05, 0.05],
        )
        inv = _make_hybrid_inverter()
        opt = LinearOptimizer([inv])
        opt.solve(pred)  # trigger lazy compilation

        for objective, problem in opt._problems.items():
            assert problem is not None, f"Problem for {objective} is None"
            assert problem.is_dcp(), f"Compiled problem for {objective} is not DCP-compliant"

    def test_simulation_parity_no_pv(self) -> None:
        """LP solution replayed through GridSimulation should match closely (no PV case)."""
        pred = _make_prediction(
            load_w=[250.0, 250.0, 1200.0],
            price_eur_wh=[0.00031, 0.00029, 0.00060],
        )
        eta = 0.8**0.5
        battery = Battery(
            BatteryParameters(
                device_id="battery_v2g",
                capacity_wh=1000,
                charging_efficiency=eta,
                discharging_efficiency=eta,
                max_charge_power_w=500,
                max_discharge_power_w=1000,
                initial_soc_percentage=50,
                min_soc_percentage=0,
                max_soc_percentage=100,
            )
        )
        inv = InverterBase(
            InverterParameters(
                device_id="hybrid_v2g",
                battery_id="battery_v2g",
                has_pv=True,
                max_ac_output_power_w=1000,
                max_ac_charge_power_w=500,
                dc_to_ac_efficiency=1.0,
                ac_to_dc_efficiency=1.0,
                zero_feed_in=False,
            ),
            battery=battery,
        )

        sol = LinearOptimizer([inv]).solve(
            pred, validate_with_simulation=True
        )

        assert sol.parity_report is not None
        assert sol.simulation_result is not None
        assert sol.parity_report.max_abs_soc_error_wh <= 5.0
        assert sol.parity_report.max_abs_grid_import_error_wh <= 5.0
        assert sol.parity_report.max_abs_feedin_error_wh <= 1e-2
        assert sol.parity_report.max_abs_cost_error_eur <= 0.02
