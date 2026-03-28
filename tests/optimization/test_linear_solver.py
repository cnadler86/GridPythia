"""Basic economic behavior tests for the linear MILP solver."""

from __future__ import annotations

import polars as pl
import pytest

from src.optimization.linear.solver import LinearOptimizer, OptimizationObjective
from src.prediction.prediction import PredictionData
from src.simulation.devices import InverterMode
from src.simulation.devices.battery import Battery, BatteryParameters
from src.simulation.devices.inverterbase import InverterBase, InverterParameters


def _make_prediction(load_w: list[float], price_eur_wh: list[float]) -> PredictionData:
    """Create a minimal PredictionData with aligned timestamp/load/price arrays."""
    n = len(load_w)
    assert n == len(price_eur_wh)

    # The linear solver only requires aligned columns; a simple integer
    # timestamp keeps tests robust across Polars versions on Windows.
    df = pl.DataFrame(
        {
            "timestamp": pl.Series(range(n), dtype=pl.Int64),
            "electricprice_eur_wh": pl.Series(price_eur_wh, dtype=pl.Float32),
            "feedintariff_eur_wh": pl.Series([0.0] * n, dtype=pl.Float32),
            "load_w": pl.Series(load_w, dtype=pl.Float32),
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
        ),
        battery=battery,
    )
    return inv


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
        assert plan["rates"][2] > 0.0

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
