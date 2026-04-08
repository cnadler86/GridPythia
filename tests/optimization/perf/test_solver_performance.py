from __future__ import annotations

import time

from GridPythia.optimization.solver import LinearOptimizer, OptimizationObjective
from tests.optimization.solver_fixture_support import load_solver_fixture_scenario


def test_compiled_solver_fast_path_on_fixture() -> None:
    scenario = load_solver_fixture_scenario()
    optimizer = LinearOptimizer(scenario.inverters, scenario.prediction)

    elapsed: list[float] = []
    statuses: list[str] = []
    for _ in range(2):
        start = time.perf_counter()
        solution = optimizer.solve(
            OptimizationObjective.MINIMIZE_COST,
            solver_opts={"time_limit": 1, "mip_rel_gap": 0.05},
            validate_with_simulation=False,
        )
        elapsed.append(time.perf_counter() - start)
        statuses.append(solution.solver_status)

    assert all(status in {"user_limit", "optimal", "optimal_inaccurate"} for status in statuses)
    assert sum(elapsed) / len(elapsed) < 1.5