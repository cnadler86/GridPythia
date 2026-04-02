"""Tests for the GridSimulation engine."""

from array import array
from datetime import datetime, timedelta

import numpy as np
import pytest

from GridPythia.config.models import BatteryParameters, InverterParameters
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.homeappliance import HomeAppliance, HomeApplianceParameters
from GridPythia.simulation.devices.inverterbase import InverterBase
from GridPythia.simulation.grid_simulation import GridSimulation, SimulationResult

START_IDX = 1

PV_WH = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    8.05,
    352.91,
    728.51,
    930.28,
    1043.25,
    1106.74,
    1161.69,
    6018.82,
    5519.07,
    3969.88,
    3017.96,
    1943.07,
    1007.17,
    319.67,
    7.88,
    0,
    0,
    0,
    0,
    0,
    0,
    5.04,
    335.59,
    705.32,
    1121.12,
    1604.79,
    2157.38,
    1433.25,
    5718.49,
    4553.96,
    3027.55,
    2574.46,
    1720.4,
    963.4,
    383.3,
    0,
    0,
    0,
    0,
    0,
    0,
]

PRICES = [
    0.0003384,
    0.0003318,
    0.0003284,
    0.0003283,
    0.0003289,
    0.0003334,
    0.0003290,
    0.0003302,
    0.0003042,
    0.0002430,
    0.0002280,
    0.0002212,
    0.0002093,
    0.0001879,
    0.0001838,
    0.0002004,
    0.0002198,
    0.0002270,
    0.0002997,
    0.0003195,
    0.0003081,
    0.0002969,
    0.0002921,
    0.0002780,
    0.0003384,
    0.0003318,
    0.0003284,
    0.0003283,
    0.0003289,
    0.0003334,
    0.0003290,
    0.0003302,
    0.0003042,
    0.0002430,
    0.0002280,
    0.0002212,
    0.0002093,
    0.0001879,
    0.0001838,
    0.0002004,
    0.0002198,
    0.0002270,
    0.0002997,
    0.0003195,
    0.0003081,
    0.0002969,
    0.0002921,
    0.0002780,
]

LOAD = [
    676.71,
    876.19,
    527.13,
    468.88,
    531.38,
    517.95,
    483.15,
    472.28,
    1011.68,
    995.00,
    1053.07,
    1063.91,
    1320.56,
    1132.03,
    1163.67,
    1176.82,
    1216.22,
    1103.78,
    1129.12,
    1178.71,
    1050.98,
    988.56,
    912.38,
    704.61,
    516.37,
    868.05,
    694.34,
    608.79,
    556.31,
    488.89,
    506.91,
    804.89,
    1141.98,
    1056.97,
    992.46,
    1155.99,
    827.01,
    1257.98,
    1232.67,
    871.26,
    860.88,
    1158.03,
    1222.72,
    1221.04,
    949.99,
    987.01,
    733.99,
    592.97,
]

PREDICTION_HOURS = 48
OPTIMIZATION_HOURS = 24


def _make_prediction(steps: int) -> PredictionData:
    """Build PredictionData from the test constants (first *steps* entries)."""
    n = steps
    arrays = {
        "electricprice_eur_wh": np.array(PRICES[:n], dtype=np.float32),
        "feedintariff_eur_wh": np.full(n, 0.00007, dtype=np.float32),
        "load_wh": np.array(LOAD[:n], dtype=np.float32),
        "pv_inverter1_wh": np.array(PV_WH[:n], dtype=np.float32),
    }
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(n)]
    return PredictionData(_timestamps=timestamps, _arrays=arrays, dt_hours=1.0)


@pytest.fixture
def grid_simulation() -> GridSimulation:
    """GridSimulation fixture with a PV_BATTERY inverter."""
    akku = Battery(
        BatteryParameters(
            device_id="battery1",
            capacity_wh=5000,
            initial_soc_percentage=80,
            min_soc_percentage=10,
        ),
        prediction_hours=PREDICTION_HOURS,
    )

    inverter = InverterBase(
        InverterParameters(
            device_id="inverter1",
            max_ac_output_power_w=10000,
            battery_id="battery1",
            pv_source="inverter1",
            dc_to_ac_efficiency=1.0,
            ac_to_dc_efficiency=0.0,
            max_ac_charge_power_w=0.0,
        ),
        battery=akku,
    )

    home_appliance = HomeAppliance(
        HomeApplianceParameters(
            device_id="dishwasher1",
            consumption_wh=2000,
            duration_h=2,
        ),
        optimization_hours=OPTIMIZATION_HOURS,
        prediction_hours=PREDICTION_HOURS,
    )

    return GridSimulation(
        prediction=_make_prediction(OPTIMIZATION_HOURS),
        inverters=[inverter],
        home_appliances=[home_appliance],
    )


def test_simulation(grid_simulation: GridSimulation) -> None:
    """Simulate from START_IDX and validate the result structure."""
    sim = grid_simulation
    n_hours = sim.simulation_steps

    inverter_modes = {
        inv.device_id: array("i", [InverterMode.IDLE] * n_hours)
        for inv in sim._inv_list
    }
    inverter_ac_rates = {
        inv.device_id: array("i", [0] * n_hours)
        for inv in sim._inv_list
    }
    appliance_load = array("f", [0.0] * n_hours)

    result = sim.simulate(
        inverter_modes=inverter_modes,
        inverter_ac_rates=inverter_ac_rates,
        appliance_load=appliance_load,
        start_idx=START_IDX,
    )

    assert result is not None
    assert isinstance(result, SimulationResult)

    expected_len = n_hours - START_IDX
    assert len(result.costs_per_dt) == expected_len
    assert len(result.feedin_wh_per_dt) == expected_len
    assert len(result.self_consumption_wh_per_dt) == expected_len

    assert result.net_balance == pytest.approx(
        result.total_revenue - result.total_cost, abs=1e-4
    )
    assert result.total_losses >= 0.0



def test_simulation_discharge_reduces_grid_draw(
    grid_simulation: GridSimulation,
) -> None:
    """Battery discharge should reduce grid import compared to IDLE mode."""
    sim = grid_simulation
    n_hours = sim.simulation_steps

    idle_modes = {
        inv.device_id: array("i", [InverterMode.IDLE] * n_hours)
        for inv in sim._inv_list
    }
    discharge_modes = {
        inv.device_id: array("i", [InverterMode.DISCHARGE_ZERO_FEED_IN] * n_hours)
        for inv in sim._inv_list
    }
    rates = {
        inv.device_id: array("i", [0] * n_hours)
        for inv in sim._inv_list
    }
    appliance_load = array("f", [0.0] * n_hours)

    r_idle = sim.simulate(idle_modes, rates, appliance_load, start_idx=START_IDX)
    r_discharge = sim.simulate(
        discharge_modes, rates, appliance_load, start_idx=START_IDX
    )

    assert r_idle is not None and r_discharge is not None
    assert r_discharge.total_cost <= r_idle.total_cost + 1e-4


def test_simulation_reset(grid_simulation: GridSimulation) -> None:
    """simulate() calls reset() so two identical runs produce identical results."""
    sim = grid_simulation
    n_hours = sim.simulation_steps

    modes = {
        inv.device_id: array("i", [InverterMode.DISCHARGE_ZERO_FEED_IN] * n_hours)
        for inv in sim._inv_list
    }
    rates = {
        inv.device_id: array("i", [0] * n_hours)
        for inv in sim._inv_list
    }
    appliance_load = array("f", [0.0] * n_hours)

    r1 = sim.simulate(modes, rates, appliance_load, start_idx=START_IDX)
    r2 = sim.simulate(modes, rates, appliance_load, start_idx=START_IDX)

    assert r1 is not None and r2 is not None
    assert r1.net_balance == pytest.approx(r2.net_balance, abs=1e-5)
    assert r1.total_cost == pytest.approx(r2.total_cost, abs=1e-5)
