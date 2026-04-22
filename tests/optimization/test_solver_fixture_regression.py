from __future__ import annotations

import numpy as np
import pytest

from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.optimization.solution import InverterPlan, OptimizationObjective
from tests.optimization.solver_fixture_support import load_solver_fixture_scenario


_FAST_FIXTURE_SOLVER_OPTS = {
    "mip_lp_solver": "ipm",
    "mip_rel_gap": 0.02,
    "random_seed": 0,
}


def test_fixture_solution_matches_regression_signature() -> None:
    scenario = load_solver_fixture_scenario()
    pred = scenario.prediction

    solution = LinearOptimizer(
        scenario.inverters,
        solver_opts=_FAST_FIXTURE_SOLVER_OPTS,
    ).solve(pred)
    plan = solution.inverter_plans[0]

    assert solution.solver_status == "optimal"
    assert isinstance(plan, InverterPlan)
    assert plan.device_id == "SF800Pro"
    assert plan.battery_soc_wh is not None
    assert np.max(np.minimum(plan.charge_ac_wh, plan.discharge_ac_wh)) == pytest.approx(0.0, abs=1e-6)
    assert np.all(plan.pv_to_ac_wh + plan.pv_to_battery_wh <= scenario.prediction.pv_by_inverter[plan.device_id] + 1e-5)

    assert solution.result.total_cost == pytest.approx(0.814117, abs=1e-3)
    assert solution.result.total_grid_import == pytest.approx(4738.7, abs=1.0)
    assert solution.result.total_losses == pytest.approx(772.6, abs=2.0)
    assert float(plan.battery_soc_wh[-1]) == pytest.approx(672.8, abs=2.0)

    active_idx = np.flatnonzero((plan.charge_ac_wh > 1e-6) | (plan.discharge_ac_wh > 1e-6))
    assert active_idx.size == 98
    assert active_idx[0] == 21
    assert active_idx[-1] == 182
    np.testing.assert_allclose(
        plan.discharge_ac_wh[active_idx[:8]],
        np.array([14.882, 15.416, 15.743, 14.865, 14.510, 16.540, 20.955, 29.771], dtype=np.float32),
        atol=0.05,
    )
    assert plan.modes[active_idx[:8]].tolist() == [2, 2, 2, 2, 2, 2, 2, 2]


def test_rolling_horizon_auto_roll_aligns_modes() -> None:
    """Two consecutive solves with shifted timestamps should both succeed and
    the optimizer should auto-roll the warm-start plan by the correct number of steps."""
    from profile_rolling_horizon import slice_prediction

    scenario = load_solver_fixture_scenario()
    pred = scenario.prediction
    window = pred.steps - 1  # one step shorter to allow slicing

    optimizer = LinearOptimizer(
        scenario.inverters,
        solver_opts=_FAST_FIXTURE_SOLVER_OPTS,
    )

    pred0 = slice_prediction(pred, start_idx=0, length=window)
    sol0 = optimizer.solve(pred0)
    assert sol0.solver_status in {"optimal", "optimal_inaccurate", "user_limit"}

    pred1 = slice_prediction(pred, start_idx=1, length=window)
    sol1 = optimizer.solve(pred1)
    assert sol1.solver_status in {"optimal", "optimal_inaccurate", "user_limit"}

    # Verify the optimizer updated its cache to pred1 and detects no further shift from pred0.
    assert optimizer._cached_prediction is pred1
    assert optimizer._compute_roll_steps(pred0) == 0  # pred0 is behind pred1 (negative shift)
    assert optimizer._compute_roll_steps(pred1) == 0  # same timestamp = no roll