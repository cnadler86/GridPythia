from __future__ import annotations

import numpy as np
import pytest

from GridPythia.optimization.solver import InverterPlan, LinearOptimizer, OptimizationObjective
from GridPythia.simulation.devices import InverterMode
from tests.optimization.solver_fixture_support import load_solver_fixture_scenario


_FAST_FIXTURE_SOLVER_OPTS = {
    "mip_lp_solver": "ipm",
    "mip_rel_gap": 0.02,
}


def test_fixture_solution_matches_regression_signature() -> None:
    scenario = load_solver_fixture_scenario()

    solution = LinearOptimizer(scenario.inverters, scenario.prediction).solve(
        OptimizationObjective.MINIMIZE_COST,
        validate_with_simulation=False,
        solver_opts=_FAST_FIXTURE_SOLVER_OPTS,
    )
    plan = solution.inverter_plans[0]

    assert solution.solver_status == "optimal"
    assert isinstance(plan, InverterPlan)
    assert plan.device_id == "SF800Pro"
    assert plan.battery_soc_wh is not None
    assert np.max(np.minimum(plan.charge_ac_wh, plan.discharge_ac_wh)) == pytest.approx(0.0, abs=1e-6)
    assert np.all(plan.pv_to_ac_wh + plan.pv_to_battery_wh <= scenario.prediction.pv_by_inverter[plan.device_id] + 1e-5)

    assert solution.result.total_cost == pytest.approx(0.9403953, abs=1e-3)
    assert solution.result.total_grid_import == pytest.approx(5266.249, abs=1.0)
    assert solution.result.total_losses == pytest.approx(665.415, abs=2.0)
    assert float(plan.battery_soc_wh[-1]) == pytest.approx(763.492, abs=2.0)

    active_idx = np.flatnonzero((plan.charge_ac_wh > 1e-6) | (plan.discharge_ac_wh > 1e-6))
    assert active_idx.size == 85
    assert active_idx[0] == 7
    assert active_idx[-1] == 182
    np.testing.assert_allclose(
        plan.discharge_ac_wh[active_idx[:8]],
        np.array([13.858, 22.668, 30.842, 31.028, 23.226, 14.601, 12.5, 12.5], dtype=np.float32),
        atol=0.05,
    )
    assert plan.modes[active_idx[:8]].tolist() == [2, 2, 2, 2, 2, 2, 2, 2]
    assert plan.rates[active_idx[:8]].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]


def test_linear_shift_solution_for_next_horizon_aligns_modes_and_rates() -> None:
    scenario = load_solver_fixture_scenario()
    solution = LinearOptimizer(scenario.inverters, scenario.prediction).solve(
        OptimizationObjective.MINIMIZE_COST,
        validate_with_simulation=False,
        solver_opts=_FAST_FIXTURE_SOLVER_OPTS,
    )

    horizon = scenario.prediction.steps
    warm = LinearOptimizer.shift_solution_for_next_horizon(
        solution,
        horizon_steps=horizon,
        shift_steps=1,
    )
    inv_id = solution.inverter_plans[0].device_id
    shifted_modes, shifted_rates = warm[inv_id]

    assert shifted_modes.shape[0] == horizon
    assert shifted_rates.shape[0] == horizon
    assert int(shifted_modes[0]) == int(solution.inverter_plans[0].modes[1])

    tail_mode = int(solution.inverter_plans[0].modes[-1])
    tail_rate = int(solution.inverter_plans[0].rates[-1])
    assert int(shifted_modes[-1]) == tail_mode
    if tail_mode in (int(InverterMode.DISCHARGE), int(InverterMode.AC_CHARGE)):
        assert int(shifted_rates[-1]) == tail_rate
    else:
        assert int(shifted_rates[-1]) == 0