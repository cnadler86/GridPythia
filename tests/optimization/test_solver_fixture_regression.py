from __future__ import annotations

import time

import numpy as np
import pytest

from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.optimization.solution import InverterPlan, OptimizationObjective
from GridPythia.simulation.devices import InverterMode
from tests.optimization.solver_fixture_support import FIXTURE_PATHS, load_solver_fixture_scenario


_FAST_FIXTURE_SOLVER_OPTS = {
    "mip_lp_solver": "ipm",
    "mip_rel_gap": 0.02,
    "random_seed": 0,
}

_HARD_FOLLOWUP_MAX_S = 0.350
_TARGET_FOLLOWUP_AVG_S = 0.300
_ROLLING_SHIFT_HOURS = 4.0
_PERF_BENCHMARK_STEPS = 48  # 12 h at 15 min resolution


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

    assert solution.result.total_cost == pytest.approx(0.816, abs=2e-3)
    assert solution.result.total_grid_import == pytest.approx(4778.8, abs=5.0)
    assert solution.result.total_losses == pytest.approx(734.0, abs=5.0)
    assert float(plan.battery_soc_wh[-1]) == pytest.approx(680.2, abs=5.0)

    active_idx = np.flatnonzero((plan.charge_ac_wh > 1e-6) | (plan.discharge_ac_wh > 1e-6))
    assert 73 <= active_idx.size <= 77
    assert active_idx[0] == 19
    assert active_idx[-1] == 182
    np.testing.assert_allclose(
        plan.discharge_ac_wh[active_idx[:8]],
        np.array([13.257, 14.140, 14.882, 15.416, 15.743, 14.865, 14.510, 16.540], dtype=np.float32),
        atol=0.05,
    )
    assert plan.modes[active_idx[:8]].tolist() == [2, 2, 2, 2, 2, 2, 2, 2]


@pytest.mark.parametrize("fixture_key", sorted(FIXTURE_PATHS))
def test_fixture_smoke_solves_for_all_registered_fixtures(fixture_key: str) -> None:
    scenario = load_solver_fixture_scenario(fixture_key=fixture_key)
    pred = scenario.prediction

    solution = LinearOptimizer(
        scenario.inverters,
        solver_opts=_FAST_FIXTURE_SOLVER_OPTS,
    ).solve(pred)

    assert solution.solver_status in {"optimal", "optimal_inaccurate", "user_limit"}
    assert len(solution.inverter_plans) == len(scenario.inverters)
    for plan in solution.inverter_plans:
        assert plan.steps == pred.steps


@pytest.mark.parametrize("fixture_key", sorted(FIXTURE_PATHS))
def test_rolling_horizon_auto_roll_aligns_modes(fixture_key: str) -> None:
    """Two consecutive solves with shifted timestamps should both succeed and
    the optimizer should auto-roll the warm-start plan by the correct number of steps."""
    from utils.profile_rolling_horizon import slice_prediction

    scenario = load_solver_fixture_scenario(fixture_key=fixture_key)
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


def test_rolling_horizon_followup_wall_time_guard() -> None:
    """Protect rolling-horizon runtime against refactors.

    Expectations:
    - with suitable warm-start handover (mode + SoC from t+1), follow-up solves stay under
      a hard limit of 350 ms;
    - the preferred operating range is <= 300 ms on average.

    Important: do not tune solver parameters in this test. It intentionally uses the
    optimizer defaults because those parameters are already optimized.
    """
    from utils.profile_rolling_horizon import slice_prediction

    scenario = load_solver_fixture_scenario(fixture_key="today_2026_04_25")

    # Use a fixed, representative 12h slice to keep this guard deterministic across runs.
    pred = slice_prediction(scenario.prediction, start_idx=0, length=_PERF_BENCHMARK_STEPS)
    roll_shift_steps = max(1, int(round(_ROLLING_SHIFT_HOURS / float(pred.dt_hours))))
    window_steps = pred.steps - roll_shift_steps

    # Keep solver options at optimizer defaults; no per-test retuning here.
    optimizer = LinearOptimizer(scenario.inverters)

    current_modes: dict[str, InverterMode] | None = None
    current_soc: dict[str, float] | None = None
    wall_times: list[float] = []

    for roll in range(roll_shift_steps):
        pred_window = slice_prediction(pred, start_idx=roll, length=window_steps)
        t0 = time.perf_counter()
        sol = optimizer.solve(
            pred_window,
            soc=current_soc,
            initial_modes=current_modes,
        )
        wall_times.append(time.perf_counter() - t0)

        assert sol.solver_status in {"optimal", "optimal_inaccurate", "user_limit"}

        # Match profiler handover: next roll starts one timestep later, so seed from t+1.
        current_modes = {
            plan.device_id: InverterMode(
                int(plan.modes[1] if plan.modes.size > 1 else plan.modes[0])
            )
            for plan in sol.inverter_plans
            if plan.modes.size > 0
        } or None
        current_soc = {
            plan.device_id: float(soc_trace[1] if len(soc_trace) > 1 else soc_trace[0])
            for plan in sol.inverter_plans
            if (soc_trace := sol.result.battery_wh_per_dt.get(plan.device_id)) is not None
            and len(soc_trace) > 0
        } or None

    followup = wall_times[1:]
    assert followup, "Need at least one follow-up solve for rolling-horizon timing checks"

    followup_max_s = max(followup)
    followup_avg_s = float(np.mean(followup))

    assert followup_max_s <= _HARD_FOLLOWUP_MAX_S, (
        f"Follow-up solve exceeded hard limit: {followup_max_s:.3f}s > {_HARD_FOLLOWUP_MAX_S:.3f}s. "
        "Warm-start quality likely regressed."
    )
    assert followup_avg_s <= _TARGET_FOLLOWUP_AVG_S, (
        f"Follow-up average above target: {followup_avg_s:.3f}s > {_TARGET_FOLLOWUP_AVG_S:.3f}s"
    )