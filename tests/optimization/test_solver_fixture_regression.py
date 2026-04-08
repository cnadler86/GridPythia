from __future__ import annotations

import numpy as np
import pytest

from GridPythia.optimization.solver import InverterPlan, LinearOptimizer, OptimizationObjective
from tests.optimization.solver_fixture_support import load_solver_fixture_scenario


def test_fixture_solution_matches_regression_signature() -> None:
    scenario = load_solver_fixture_scenario()

    solution = LinearOptimizer(scenario.inverters, scenario.prediction).solve(
        OptimizationObjective.MINIMIZE_COST,
        validate_with_simulation=True,
    )
    plan = solution.inverter_plans[0]

    assert solution.solver_status == "optimal"
    assert isinstance(plan, InverterPlan)
    assert plan.device_id == "SF800Pro"
    assert plan.battery_soc_wh is not None
    assert np.max(np.minimum(plan.charge_ac_wh, plan.discharge_ac_wh)) == pytest.approx(0.0, abs=1e-6)
    assert np.all(plan.pv_to_ac_wh + plan.pv_to_battery_wh <= scenario.prediction.pv_by_inverter[plan.device_id] + 1e-5)

    assert solution.result.total_cost == pytest.approx(0.9261264, abs=1e-3)
    assert solution.result.total_grid_import == pytest.approx(5091.8125, abs=1.0)
    assert solution.result.total_losses == pytest.approx(627.5436, abs=2.0)
    assert float(plan.battery_soc_wh[-1]) == pytest.approx(724.1880, abs=2.0)

    active_idx = np.flatnonzero((plan.charge_ac_wh > 1e-6) | (plan.discharge_ac_wh > 1e-6))
    assert active_idx.size == 74
    assert active_idx[0] == 21
    assert active_idx[-1] == 182
    np.testing.assert_allclose(
        plan.discharge_ac_wh[active_idx[:8]],
        np.array([18.632, 19.166, 19.493, 18.615, 18.260, 20.290, 24.705, 33.521], dtype=np.float32),
        atol=0.05,
    )
    assert plan.modes[active_idx[:8]].tolist() == [2, 2, 2, 2, 2, 2, 2, 2]
    assert plan.rates[active_idx[:8]].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]

    assert solution.parity_report is not None
    assert solution.parity_report.max_abs_soc_error_wh < 300.0
    assert solution.parity_report.max_abs_grid_import_error_wh < 40.0
    assert solution.parity_report.max_abs_feedin_error_wh < 40.0
    assert solution.parity_report.max_abs_cost_error_eur < 0.01