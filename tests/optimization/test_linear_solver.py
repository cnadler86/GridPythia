"""Basic economic behavior tests for the linear MILP solver."""

from __future__ import annotations
from pygments.unistring import No

from datetime import datetime, timedelta

import numpy as np
import pytest

from GridPythia.config.optimization import BatteryParameters, InverterParameters
from GridPythia.optimization.solver import LinearOptimizer, OptimizationObjective
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


def _make_prediction(
    load_w: list[float],
    price_eur_wh: list[float],
    pv_wh: dict[str, list[float]] | None = None,
) -> PredictionData:
    """Create a minimal PredictionData with aligned timestamp/load/price arrays.
    
    Args:
        load_w: Load values in Wh (energy, not power)
        price_eur_wh: Prices in EUR/Wh
    """
    n = len(load_w)
    assert n == len(price_eur_wh)

    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(n)]
    pv_by_inverter: dict[str, np.ndarray] = {}
    for inverter_id, series in (pv_wh or {}).items():
        assert len(series) == n
        pv_by_inverter[inverter_id] = np.array(series, dtype=np.float32)

    return PredictionData(
        timestamps=timestamps,
        dt_hours=1.0,
        load_wh=np.array(load_w, dtype=np.float32),
        electricprice_eur_wh=np.array(price_eur_wh, dtype=np.float32),
        feedintariff_eur_wh=np.zeros(n, dtype=np.float32),
        pv_by_inverter=pv_by_inverter,
    )


def _make_hybrid_inverter(roundtrip_efficiency: float = 0.8) -> InverterBase:
    """Create one hybrid inverter with 0.5 kWh headroom from initial SoC.

    Capacity is 1 kWh with initial SoC at 50%, so charging headroom is 500 Wh.
    """
    eta = roundtrip_efficiency**0.5

    battery = Battery(
        BatteryParameters(
            device_id="battery_h1",
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
            device_id="hybrid_h1",
            battery_id="battery_h1",
                has_pv=True,
            max_ac_output_power_w=1000,
            max_ac_charge_power_w=500,
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=True,
            mode_switch_cost=0.0,
        ),
        battery=battery,
    )
    return inv


def _make_switching_case_inverter(
    *,
    roundtrip_efficiency: float = 0.8,
    switch_cost: float,
    active_inverter_consumption_w: float = 0.0,
) -> InverterBase:
    """Hybrid inverter for switching-cost edge cases.

    Starts at 0% SoC so profitable charging windows become the deciding factor.
    """
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
            active_inverter_consumption_w = active_inverter_consumption_w,
        ),
        battery=battery,
    )


def _make_boundary_inverter(
    *,
    device_id: str,
    zero_feed_in: bool,
    has_pv: bool | None,
    initial_soc_percentage: int,
    min_soc_percentage: int,
    max_soc_percentage: int,
) -> InverterBase:
    battery = Battery(
        BatteryParameters(
            device_id=f"battery_{device_id}",
            capacity_wh=1000,
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
                has_pv=bool(has_pv),
            max_ac_output_power_w=500,
            max_ac_charge_power_w=500,
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=zero_feed_in,
        ),
        battery=battery,
    )


class TestLinearSolverHybridEconomics:
    def test_no_charge_when_low_period_not_profitable_after_roundtrip(self) -> None:
        """If low/high arbitrage is not profitable after losses, solver should not charge."""
        # Periods: medium, low, high
        # roundtrip = 0.8  -> effective charge cost at low is low/0.8
        # low=0.00050, high=0.00060 => low/0.8=0.000625 > high (not profitable)
        pred = _make_prediction(
            load_w=[0.0, 0.0, 2000.0],
            price_eur_wh=[0.00045, 0.00050, 0.00060],
        )
        inv = _make_hybrid_inverter(roundtrip_efficiency=0.8)

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        # Low period should not charge.
        assert plan["modes"][1] != int(InverterMode.AC_CHARGE)
        assert plan["rates"][1] == pytest.approx(0.0, abs=1e-6)

        # High-price period should discharge in zero-feed-in mode.
        assert plan["modes"][2] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        assert plan["rates"][2] == pytest.approx(0.0, abs=1e-6)

    def test_charge_when_low_period_is_profitable_after_roundtrip(self) -> None:
        """If high > low/roundtrip, solver should charge in low period before discharging at high."""
        # roundtrip = 0.8, low=0.00030, high=0.00050 => low/0.8=0.000375 < high (profitable)
        pred = _make_prediction(
            load_w=[0.0, 0.0, 2000.0],
            price_eur_wh=[0.00045, 0.00030, 0.00050],
        )
        inv = _make_hybrid_inverter(roundtrip_efficiency=0.8)

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        # Low period should charge.
        assert plan["modes"][1] == int(InverterMode.AC_CHARGE)
        assert plan["rates"][1] > 0.0

        # SoC increase must match returned integer rate approximately.
        soc = list(sol.result.battery_wh_per_dt["hybrid_h1"])
        eta = 0.8**0.5
        charged_wh = 500.0 * (int(plan["rates"][1]) / 100.0)
        assert soc[1] - soc[0] == pytest.approx(charged_wh * eta, abs=12.0)

        # High-price period should discharge in zero-feed-in mode.
        assert plan["modes"][2] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)

    def test_rates_are_integer_percent_for_rate_modes(self) -> None:
        """Returned rates should be integer percentages in [0, 100]."""
        pred = _make_prediction(
            load_w=[0.0, 0.0, 2000.0],
            price_eur_wh=[0.00044, 0.00031, 0.00052],
        )
        inv = _make_hybrid_inverter(roundtrip_efficiency=0.8)

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]
        rates = [int(r) for r in plan["rates"]]

        assert all(0 <= r <= 100 for r in rates)
        assert all(float(r).is_integer() for r in rates)

    def test_auto_terminal_value_suppresses_cheap_slot_discharge(self) -> None:
        """Auto terminal value based on high tail prices should prevent discharge in cheap slots.

        Prices: cheap at t=0, expensive at t=1..3.
        Auto estimate = mean(all 4 prices) * eta_d ≈ 0.00040 * 0.894 = 0.000357 EUR/Wh.
        Discharge threshold at t=0 = price[0] * eta_d ≈ 0.00010 * 0.894 = 0.0000894 EUR/Wh.
        estimate >> threshold -> optimizer should stay IDLE at t=0 and discharge later.
        """
        pred = _make_prediction(
            load_w=[1500.0, 1500.0, 1500.0, 1500.0],
            price_eur_wh=[0.00010, 0.00050, 0.00050, 0.00050],
        )
        inv = _make_hybrid_inverter(roundtrip_efficiency=0.8)

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)

        plan = sol.inverter_plans[0]
        # Cheap slot: retaining energy for expensive future slots is more valuable.
        assert plan["modes"][0] not in (
            int(InverterMode.DISCHARGE),
            int(InverterMode.DISCHARGE_ZERO_FEED_IN),
        )
        # At least one later expensive slot should use the stored energy.
        assert any(
            mode in (int(InverterMode.DISCHARGE), int(InverterMode.DISCHARGE_ZERO_FEED_IN))
            for mode in plan["modes"][1:]
        )

    def test_zero_feed_discharge_can_compensate_full_load_without_rate(self) -> None:
        """Zero-feed discharge should be energy-target driven, not rate driven."""
        pred = _make_prediction(
            load_w=[400.0, 0.0],
            price_eur_wh=[0.00060, 0.0],
        )
        inv = _make_hybrid_inverter(roundtrip_efficiency=0.8)

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        assert plan["rates"][0] == pytest.approx(0.0, abs=1e-6)
        assert float(sol.result.grid_import_wh_per_dt[0]) == pytest.approx(0.0, abs=1e-3)

    def test_simulation_parity_report_for_no_pv_case(self) -> None:
        """With no PV signal, LP replay should match simulation very closely."""
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

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(
            pred,
            validate_with_simulation=True,
        )

        assert sol.parity_report is not None
        assert sol.simulation_result is not None
        # LP -> simulation replay includes mode/rate projection and can differ slightly.
        assert sol.parity_report.max_abs_soc_error_wh <= 5.0
        assert sol.parity_report.max_abs_grid_import_error_wh <= 5.0
        assert sol.parity_report.max_abs_feedin_error_wh <= 1e-2
        assert sol.parity_report.max_abs_cost_error_eur <= 0.02
    
    def test_switch_cost_below_threshold_prefers_isolated_cheapest_slot(self) -> None:
        """With lower switching cost, solver should use the isolated cheapest slot.

        Auto terminal value means the optimizer may charge fully (not just the minimum
        for discharge).  t3 (price=0.00024) MUST be used because its price saving
        exceeds the switch cost.  t2 (price=0.00045) may also be used because the
        terminal reward for stored energy makes it profitable.
        """
        pred = _make_prediction(
            load_w=[0.0, 0.0, 0.0, 0.0, 600.0],
            price_eur_wh=[0.00030, 0.00030, 0.00045, 0.00024, 0.00070],
        )
        inv = _make_switching_case_inverter(switch_cost=0.005)

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        isolated_charge = (
            float(plan["rates"][3]) if plan["modes"][3] == int(InverterMode.AC_CHARGE) else 0.0
        )

        # The isolated cheapest slot (t3) must be used at full rate.
        assert isolated_charge == pytest.approx(100.0, abs=1e-6)

    def test_initial_mode_reduces_first_step_switch_penalty(self) -> None:
        """If initial mode is AC_CHARGE, charging at t=0 should avoid first-step switch cost."""
        pred = _make_prediction(
            load_w=[0.0, 0.0, 600.0],
            price_eur_wh=[0.00030, 0.00045, 0.00070],
        )

        inv_idle_start = _make_switching_case_inverter(switch_cost=0.020)
        sol_idle_start = LinearOptimizer([inv_idle_start], pred.steps, pred.dt_hours).solve(pred)
        plan_idle_start = sol_idle_start.inverter_plans[0]

        inv_charge_start = _make_switching_case_inverter(switch_cost=0.020)
        sol_charge_start = LinearOptimizer([inv_charge_start], pred.steps, pred.dt_hours).solve(
            pred,
            initial_modes={"hybrid_sw": InverterMode.AC_CHARGE},
        )
        plan_charge_start = sol_charge_start.inverter_plans[0]

        # Starting in AC_CHARGE should never make t=0 charging less attractive than idle start.
        charge_rate_idle_start = (
            float(plan_idle_start["rates"][0])
            if plan_idle_start["modes"][0] == int(InverterMode.AC_CHARGE)
            else 0.0
        )
        charge_rate_charge_start = (
            float(plan_charge_start["rates"][0])
            if plan_charge_start["modes"][0] == int(InverterMode.AC_CHARGE)
            else 0.0
        )
        assert charge_rate_charge_start >= charge_rate_idle_start - 1e-9

    def test_rate_discharge_mode_is_blocked_at_min_soc_even_with_pv(self) -> None:
        """Discrete discharge must stay idle when the battery starts at min SoC."""
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

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.IDLE)
        assert plan["rates"][0] == pytest.approx(0.0, abs=1e-6)
        assert float(sol.result.battery_wh_per_dt["hybrid_min_rate"][0]) == pytest.approx(700.0)

    def test_zero_feed_discharge_mode_is_blocked_at_min_soc_even_with_pv(self) -> None:
        """Zero-feed discharge must stay idle when the battery starts at min SoC."""
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

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.IDLE)
        assert plan["rates"][0] == pytest.approx(0.0, abs=1e-6)
        assert float(sol.result.battery_wh_per_dt["hybrid_min_zfi"][0]) == pytest.approx(700.0)

    def test_ac_charge_mode_is_blocked_at_max_soc(self) -> None:
        """AC charge must stay idle when the battery starts at max SoC."""
        inv = _make_boundary_inverter(
            device_id="hybrid_max_charge",
            zero_feed_in=False,
            has_pv=False,
            initial_soc_percentage=80,
            min_soc_percentage=0,
            max_soc_percentage=80,
        )
        pred = _make_prediction(
            load_w=[0.0],
            price_eur_wh=[0.00010],
        )

        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        assert plan["modes"][0] == int(InverterMode.IDLE)
        assert plan["rates"][0] == pytest.approx(0.0, abs=1e-6)
        assert float(sol.result.battery_wh_per_dt["hybrid_max_charge"][0]) == pytest.approx(800.0)

    def test_ac_charge_respects_mode_switch_cost(self) -> None:
        """Mode-switch costs should apply to ac charge.
        """
        # Create a zero-feed-in inverter with ONLY DISCHARGE_ZERO_FEED_IN (no DISCHARGE with rates)
        eta = 1.0

        battery = Battery(
            BatteryParameters(
                device_id="battery_zfi_cost",
                capacity_wh=1000,
                charging_efficiency=eta,
                discharging_efficiency=eta,
                max_charge_power_w=600,
                max_discharge_power_w=1000,
                initial_soc_percentage=0,  # Start at 0% SoC
                min_soc_percentage=0,
                max_soc_percentage=100,
            )
        )

        # Inverter with only DISCHARGE_ZERO_FEED_IN mode (no AC_CHARGE, no discrete DISCHARGE rates)
        inv = InverterBase(
            InverterParameters(
                device_id="hybrid_zfi_cost",
                battery_id="battery_zfi_cost",
                has_pv=False,
                max_ac_output_power_w=1000,
                max_ac_charge_power_w=600,
                dc_to_ac_efficiency=1.0,
                ac_to_dc_efficiency=1.0,
                zero_feed_in=True,  # Only DISCHARGE_ZERO_FEED_IN
                mode_switch_cost=0.1,  # High switch cost: 0.1 EUR per mode change
                active_inverter_consumption_w=0.0,
            ),
            battery=battery,
        )

        pred = _make_prediction(
            load_w=[100.0, 100.0, 100.0, 1000.0, 100.0],
            price_eur_wh=[0.01, 0.012, 0.011, 0.90, 0.4],
        )

        # With high switch cost, the optimizer should avoid mode switches
        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        # The test passes, if ac charge is continuous for the first 3 slots, becasue the switch costs
        # dominate and the solver sould maintain ac charging mode with a low rate at slot 2 instead
        # of switching to idle and back to ac charge.
        ID = InverterMode.IDLE
        AC = InverterMode.AC_CHARGE
        ZFI = InverterMode.DISCHARGE_ZERO_FEED_IN
        assert plan["modes"][0:-1].tolist() == [int(AC), int(AC), int(AC), int(ZFI)]
        assert plan["rates"][1] < plan["rates"][2] < plan["rates"][0]


    def test_zero_feed_discharge_continuous_respects_mode_switch_cost(self) -> None:
        """Switch costs should affect continuous zero-feed discharge mode decisions.

        This case uses a PV-battery topology with no PV energy and no AC charging,
        so the only active battery mode is continuous DISCHARGE_ZERO_FEED_IN.
        Without switch cost, only the most expensive slot is discharged. With a
        non-zero switch cost, the optimizer prefers keeping discharge active into
        the following slot instead of toggling back to idle immediately.
        """
        eta = 1.0

        def make_solver(switch_cost: float) -> tuple[LinearOptimizer, PredictionData]:
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
            return LinearOptimizer([inv], pred.steps, pred.dt_hours), pred

        low_opt, pred_low = make_solver(0.0)
        low_cost_plan = low_opt.solve(pred_low).inverter_plans[0]
        high_opt, pred_high = make_solver(0.5)
        high_cost_plan = high_opt.solve(pred_high).inverter_plans[0]

        assert low_cost_plan["modes"].tolist() == [
            int(InverterMode.IDLE),
            int(InverterMode.DISCHARGE_ZERO_FEED_IN),
            int(InverterMode.IDLE),
        ]
        assert high_cost_plan["modes"].tolist() == [
            int(InverterMode.IDLE),
            int(InverterMode.DISCHARGE_ZERO_FEED_IN),
            int(InverterMode.DISCHARGE_ZERO_FEED_IN),
        ]

    def test_optimizer_compiled_problems_are_dpp(self) -> None:
        """The optimizer compiles CVXPY problems that must be DPP-compliant.

        This asserts that all prebuilt problems on a freshly-constructed
        LinearOptimizer pass CVXPY's dedicated DPP check.
        """
        pred = _make_prediction(load_w=[100.0, 100.0, 100.0, 100.0], price_eur_wh=[0.05, 0.05, 0.05, 0.05])
        inv = _make_hybrid_inverter()

        opt = LinearOptimizer([inv], pred.steps, pred.dt_hours)

        for objective, problem in opt._problems.items():
            assert problem is not None, f"Problem for {objective} is None"
            assert problem.is_dpp(), f"Compiled problem for {objective} is not DPP-compliant"


class TestActiveInverterConsumption:
    """Tests for the active_inverter_consumption_w parameter."""

    def _make_inverter(self, *, active_consumption_w: float = 0.0) -> InverterBase:
        """Hybrid inverter with configurable active self-consumption."""
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

    def test_solver_active_consumption_increases_grid_import(self) -> None:
        """Active consumption is accounted for in the energy balance.

        When the inverter is active, the solver must supply load + active_consumption.
        We verify this via simulation parity: both the LP model and the simulation
        include the same active consumption via the energy balance, so the parity
        report should confirm the two are consistent.
        """
        pred = _make_prediction(
            load_w=[0.0, 0.0, 200.0],
            price_eur_wh=[0.00020, 0.00022, 0.00080],
        )
        inv = self._make_inverter(active_consumption_w=30.0)
        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(
            pred,
            validate_with_simulation=True,
        )

        assert sol.parity_report is not None
        assert sol.parity_report.ok
        assert sol.parity_report.max_abs_grid_import_error_wh <= 5.0

    def test_solver_high_active_consumption_prevents_unprofitable_charging(self) -> None:
        """With a large active consumption, activating the inverter for marginal arbitrage
        becomes unprofitable, so the optimizer should keep the inverter IDLE.

        Arbitrage opportunity: low=0.00025 EUR/Wh, high=0.00030 EUR/Wh.
        Savings per charge/discharge cycle: 500 Wh * (0.00030 - 0.00025) = 0.25 EUR.
        Active consumption cost: 200 W * 1 h = 200 Wh per active step,
        at the high price: 200 * 0.00030 * 2 steps active = 0.12 EUR → still profitable.
        But at 500 W active consumption: 500 * 2 * 0.00030 = 0.30 EUR > 0.25 EUR → not profitable.
        """
        pred = _make_prediction(
            load_w=[0.0, 0.0],
            price_eur_wh=[0.00025, 0.00030],
        )

        # Very high active consumption makes staying IDLE cheaper
        inv = self._make_inverter(active_consumption_w=500.0)
        sol = LinearOptimizer([inv], pred.steps, pred.dt_hours).solve(pred)
        plan = sol.inverter_plans[0]

        # Both slots should be IDLE (arbitrage not worth the inverter consumption cost)
        assert plan["modes"][0] == int(InverterMode.IDLE)
        assert plan["modes"][1] == int(InverterMode.IDLE)

    def test_solver_losses_include_active_consumption(self) -> None:
        """Losses in the solver result should include the inverter's active consumption."""
        pred = _make_prediction(
            load_w=[200.0],
            price_eur_wh=[0.00080],
        )
        inv_no_loss = self._make_inverter(active_consumption_w=0.0)
        inv_with_loss = self._make_inverter(active_consumption_w=30.0)

        sol_no = LinearOptimizer([inv_no_loss], pred.steps, pred.dt_hours).solve(pred)
        sol_with = LinearOptimizer([inv_with_loss], pred.steps, pred.dt_hours).solve(pred)

        losses_no = float(np.sum(sol_no.result.losses_wh_per_dt))
        losses_with = float(np.sum(sol_with.result.losses_wh_per_dt))

        # If inverter is active in either solution, losses_with must exceed losses_no.
        plan_with = sol_with.inverter_plans[0]
        if any(m != int(InverterMode.IDLE) for m in plan_with["modes"]):
            assert losses_with > losses_no + 25.0
