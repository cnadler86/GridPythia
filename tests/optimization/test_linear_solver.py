"""Basic economic behavior tests for the linear MILP solver."""

from __future__ import annotations

import polars as pl
import pytest

from src.config.models import BatteryParameters, InverterParameters
from src.optimization.solver import LinearOptimizer, OptimizationObjective
from src.prediction.prediction import PredictionData
from src.simulation.devices import InverterMode
from src.simulation.devices.battery import Battery
from src.simulation.devices.inverterbase import InverterBase


def _make_prediction(load_w: list[float], price_eur_wh: list[float]) -> PredictionData:
    """Create a minimal PredictionData with aligned timestamp/load/price arrays.
    
    Args:
        load_w: Load values in Wh (energy, not power)
        price_eur_wh: Prices in EUR/Wh
    """
    n = len(load_w)
    assert n == len(price_eur_wh)

    # The linear solver only requires aligned columns; a simple integer
    # timestamp keeps tests robust across Polars versions on Windows.
    df = pl.DataFrame(
        {
            "timestamp": pl.Series(range(n), dtype=pl.Int64),
            "electricprice_eur_wh": pl.Series(price_eur_wh, dtype=pl.Float32),
            "feedintariff_eur_wh": pl.Series([0.0] * n, dtype=pl.Float32),
            "load_wh": pl.Series(load_w, dtype=pl.Float32),
        }
    )
    return PredictionData(_df=df, dt_hours=1.0)


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
        ),
        prediction_hours=3,
    )

    inv = InverterBase(
        InverterParameters(
            device_id="hybrid_h1",
            battery_id="battery_h1",
            pv_source="hybrid_h1",
            max_ac_output_power_w=1000,
            max_ac_charge_power_w=500,
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=True,
            ac_rates=(0.5, 1.0),
            mode_switch_cost=0.0,
        ),
        battery=battery,
    )
    return inv


def _make_switching_case_inverter(
    *,
    prediction_hours: int,
    roundtrip_efficiency: float = 0.8,
    switch_cost: float,
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
            initial_soc_percentage=0,
            min_soc_percentage=0,
            max_soc_percentage=100,
        ),
        prediction_hours=prediction_hours,
    )

    return InverterBase(
        InverterParameters(
            device_id="hybrid_sw",
            battery_id="battery_sw",
            pv_source="hybrid_sw",
            max_ac_output_power_w=1000,
            max_ac_charge_power_w=500,
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=1.0,
            zero_feed_in=True,
            ac_rates=(0.5, 1.0),
            mode_switch_cost=switch_cost,
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

        sol = LinearOptimizer([inv], pred).solve(OptimizationObjective.MINIMIZE_COST)
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

        sol = LinearOptimizer([inv], pred).solve(OptimizationObjective.MINIMIZE_COST)
        plan = sol.inverter_plans[0]

        # Low period should charge.
        assert plan["modes"][1] == int(InverterMode.AC_CHARGE)
        assert plan["rates"][1] > 0.0

        # SoC increase must match selected discrete rate.
        soc = list(sol.result.battery_wh_per_dt["hybrid_h1"])
        eta = 0.8**0.5
        charged_wh = 500.0 * float(plan["rates"][1])
        assert soc[1] - soc[0] == pytest.approx(charged_wh * eta, abs=1e-2)

        # High-price period should discharge in zero-feed-in mode.
        assert plan["modes"][2] == int(InverterMode.DISCHARGE_ZERO_FEED_IN)

    def test_uses_only_configured_discrete_rates(self) -> None:
        """Returned rates must be one of the inverter's configured discrete levels."""
        pred = _make_prediction(
            load_w=[0.0, 0.0, 2000.0],
            price_eur_wh=[0.00044, 0.00031, 0.00052],
        )
        inv = _make_hybrid_inverter(roundtrip_efficiency=0.8)

        sol = LinearOptimizer([inv], pred).solve(OptimizationObjective.MINIMIZE_COST)
        rates = [float(r) for r in sol.result.inverter_ac_rate_per_dt["hybrid_h1"] if float(r) > 1e-8]

        allowed = {0.5, 1.0}
        assert rates
        assert set(round(r, 6) for r in rates).issubset(allowed)

    def test_terminal_energy_value_can_prevent_economically_bad_discharge(self) -> None:
        """A high terminal value should keep energy in battery instead of discharging."""
        pred = _make_prediction(
            load_w=[1500.0],
            price_eur_wh=[0.00040],
        )
        inv = _make_hybrid_inverter(roundtrip_efficiency=0.8)

        # Terminal value dominates immediate import savings -> do not discharge.
        sol = LinearOptimizer(
            [inv], pred, battery_end_value_eur_wh=0.001
        ).solve(OptimizationObjective.MINIMIZE_COST)

        plan = sol.inverter_plans[0]
        assert plan["modes"][0] != int(InverterMode.DISCHARGE_ZERO_FEED_IN)
        assert plan["modes"][0] != int(InverterMode.DISCHARGE)

    def test_zero_feed_discharge_can_compensate_full_load_without_rate(self) -> None:
        """Zero-feed discharge should be energy-target driven, not rate driven."""
        pred = _make_prediction(
            load_w=[400.0],
            price_eur_wh=[0.00060],
        )
        inv = _make_hybrid_inverter(roundtrip_efficiency=0.8)

        sol = LinearOptimizer([inv], pred).solve(OptimizationObjective.MINIMIZE_COST)
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
            ),
            prediction_hours=3,
        )
        inv = InverterBase(
            InverterParameters(
                device_id="hybrid_v2g",
                battery_id="battery_v2g",
                pv_source="hybrid_v2g",
                max_ac_output_power_w=1000,
                max_ac_charge_power_w=500,
                dc_to_ac_efficiency=1.0,
                ac_to_dc_efficiency=1.0,
                zero_feed_in=False,
                ac_rates=(0.5, 1.0),
            ),
            battery=battery,
        )

        sol = LinearOptimizer([inv], pred).solve(
            OptimizationObjective.MINIMIZE_COST,
            validate_with_simulation=True,
        )

        assert sol.parity_report is not None
        assert sol.simulation_result is not None
        assert sol.parity_report.ok
        assert sol.parity_report.max_abs_soc_error_wh <= 1e-2
        assert sol.parity_report.max_abs_grid_import_error_wh <= 1e-2
        assert sol.parity_report.max_abs_feedin_error_wh <= 1e-2

    def test_switch_cost_above_threshold_keeps_ac_charge_contiguous_block(self) -> None:
        """With sufficiently high switching cost, charging stays in adjacent slots."""
        # Three profitable cheap windows exist: t0/t1 (adjacent, same price) and t3 (isolated, cheaper).
        # Required charge is 750 Wh (for 600 Wh discharge at 80% roundtrip):
        # either split (0.5 at adjacent + 1.0 at isolated) or contiguous block (1.0 + 0.5 adjacent).
        pred = _make_prediction(
            load_w=[0.0, 0.0, 0.0, 0.0, 600.0],
            price_eur_wh=[0.00030, 0.00030, 0.00045, 0.00024, 0.00070],
        )
        inv = _make_switching_case_inverter(prediction_hours=5, switch_cost=0.020)

        sol = LinearOptimizer([inv], pred).solve(OptimizationObjective.MINIMIZE_COST)
        plan = sol.inverter_plans[0]

        # High switching penalty should avoid split charging and keep charging contiguous.
        assert plan["modes"][0] == int(InverterMode.AC_CHARGE)
        assert plan["modes"][1] == int(InverterMode.AC_CHARGE)
        assert plan["modes"][3] != int(InverterMode.AC_CHARGE)
        assert float(plan["rates"][0]) + float(plan["rates"][1]) == pytest.approx(1.5, abs=1e-6)

    def test_switch_cost_below_threshold_prefers_split_with_isolated_cheapest_slot(self) -> None:
        """With lower switching cost, solver should use the isolated cheapest slot plus one adjacent half-slot."""
        pred = _make_prediction(
            load_w=[0.0, 0.0, 0.0, 0.0, 600.0],
            price_eur_wh=[0.00030, 0.00030, 0.00045, 0.00024, 0.00070],
        )
        inv = _make_switching_case_inverter(prediction_hours=5, switch_cost=0.005)

        sol = LinearOptimizer([inv], pred).solve(OptimizationObjective.MINIMIZE_COST)
        plan = sol.inverter_plans[0]

        # Low switching penalty allows split charging: full on isolated cheapest slot,
        # remaining half on one of the adjacent equal-price slots.
        adjacent_charge = (
            (float(plan["rates"][0]) if plan["modes"][0] == int(InverterMode.AC_CHARGE) else 0.0)
            + (float(plan["rates"][1]) if plan["modes"][1] == int(InverterMode.AC_CHARGE) else 0.0)
        )
        isolated_charge = (
            float(plan["rates"][3]) if plan["modes"][3] == int(InverterMode.AC_CHARGE) else 0.0
        )

        assert isolated_charge == pytest.approx(1.0, abs=1e-6)
        assert adjacent_charge == pytest.approx(0.5, abs=1e-6)

    def test_initial_mode_reduces_first_step_switch_penalty(self) -> None:
        """If initial mode is AC_CHARGE, charging at t=0 should avoid first-step switch cost."""
        pred = _make_prediction(
            load_w=[0.0, 0.0, 600.0],
            price_eur_wh=[0.00030, 0.00045, 0.00070],
        )

        inv_idle_start = _make_switching_case_inverter(prediction_hours=3, switch_cost=0.020)
        sol_idle_start = LinearOptimizer([inv_idle_start], pred).solve(
            OptimizationObjective.MINIMIZE_COST
        )
        plan_idle_start = sol_idle_start.inverter_plans[0]

        inv_charge_start = _make_switching_case_inverter(prediction_hours=3, switch_cost=0.020)
        sol_charge_start = LinearOptimizer(
            [inv_charge_start],
            pred,
            initial_modes={"hybrid_sw": InverterMode.AC_CHARGE},
        ).solve(OptimizationObjective.MINIMIZE_COST)
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
