"""Rolling-horizon warm-start performance tests.

Each fixture is solved as a rolling sequence (1-step advances).  After the
first (cold-start) solve we assert that the *average* follow-up step time is
within STEP_BUDGET_S seconds.  This catches regressions in warm-start quality
and guards against unexpected recompilations without failing on individual
spikes that are within normal solver variance.
"""
from __future__ import annotations

import math

import pytest

from tests.optimization.solver_fixture_support import FIXTURE_PATHS, load_solver_fixture_scenario
from utils.profile_rolling_horizon import run_rolling_horizon

# Maximum *average* wall time (seconds) allowed for warm-start follow-up solves.
STEP_BUDGET_S = 0.350

# Number of rolling hours to execute per fixture (first step is cold-start, not asserted).
ROLL_HOURS = 1.0  # 1 h → 4 steps at dt=15 min (1 cold + 3 warm)


@pytest.mark.parametrize("fixture_key", list(FIXTURE_PATHS))
def test_rolling_horizon_warm_start_speed(fixture_key: str) -> None:
    """Average follow-up rolling-horizon solve time must stay below STEP_BUDGET_S."""
    scenario = load_solver_fixture_scenario(fixture_key=fixture_key)

    # Silent solver opts (no log files, minimal verbosity) to keep test fast.
    solver_opts = {
        "warm_start": True,
        "time_limit": 30,
        "mip_rel_gap": 0.03,
        "verbose": False,
        "output_flag": False,
        "log_to_console": False,
    }

    rows = run_rolling_horizon(
        inverters=scenario.inverters,
        pred=scenario.prediction,
        roll_shift_hours=ROLL_HOURS,
        base_opts=solver_opts,
        output_dir=None,  # no log files
    )

    assert rows, "run_rolling_horizon returned no rows"

    # First row is the cold-start solve – skip the budget assertion for it.
    follow_up = rows[1:]
    if not follow_up:
        pytest.skip(f"Fixture {fixture_key!r} too short to produce follow-up rolls")

    avg_s = sum(r.solve_wall_s for r in follow_up) / len(follow_up)
    summary_lines = [
        f"  Roll {r.roll} (idx={r.start_idx}): {r.solve_wall_s*1000:.0f} ms  status={r.status}"
        for r in rows
    ]

    assert avg_s <= STEP_BUDGET_S, (
        f"[{fixture_key}] Average follow-up solve time {avg_s*1000:.0f} ms "
        f"exceeds {STEP_BUDGET_S*1000:.0f} ms budget "
        f"(n={len(follow_up)} follow-up rolls)\n\nAll rolls:\n"
        + "\n".join(summary_lines)
    )

    # Sanity: all statuses must be accepted.
    bad_statuses = [r for r in rows if r.status not in {"optimal", "optimal_inaccurate", "user_limit"}]
    assert not bad_statuses, (
        f"[{fixture_key}] Unexpected solver statuses: "
        + ", ".join(f"roll {r.roll}={r.status!r}" for r in bad_statuses)
    )

    # Confirm warm-start speedup: follow-up avg must be faster than first solve.
    first_s = rows[0].solve_wall_s
    follow_avg = sum(r.solve_wall_s for r in follow_up) / len(follow_up)
    speedup = first_s / follow_avg if follow_avg > 0 else math.nan
    # At least 1.5× speedup on a warm-start solve vs cold start is expected.
    assert speedup >= 1.5 or math.isnan(speedup), (
        f"[{fixture_key}] Warm-start speedup too low: {speedup:.2f}× "
        f"(first={first_s*1000:.0f} ms, follow_avg={follow_avg*1000:.0f} ms)"
    )
