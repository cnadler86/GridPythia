"""Core optimization orchestration layer.

Connects prediction fetching with MILP solving, enforcing TZ awareness and
proper 15-minute slot alignment.  This layer is intentionally free of server
concerns (HTTP, MQTT, WebSocket, state singletons).

Usage example::

    result = await run_optimization(
        start=datetime.now(tz=ZoneInfo("Europe/Berlin")),
        end=datetime.now(tz=ZoneInfo("Europe/Berlin")) + timedelta(hours=48),
        prediction=prediction,
        optimizer=optimizer,
        dt_hours=0.25,
        soc={"SF800Pro": 960.0},
        initial_modes={"SF800Pro": InverterMode.IDLE},
    )
    solution = result.solution
    print(result.solver_start.isoformat())
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from structlog import get_logger

from GridPythia.optimization.solution import LinearSolution, OptimizationObjective
from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.prediction.base import ceil_to_slot, floor_to_slot
from GridPythia.prediction.prediction import Prediction, PredictionData
from GridPythia.simulation.devices import InverterMode

logger = get_logger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a single optimization run.

    Attributes:
        solution:     MILP solver output.
        fetch_pdata:  Full prediction data fetched from providers,
                      spanning ``[floor(start), last_slot_before(end)]``.
        solver_pdata: Prediction slice actually passed to the solver,
                      starting at ``ceil(start)`` (first complete slot).
        solver_start: First timestamp of *solver_pdata*  (== ``ceil_to_slot(start)``).
    """

    solution: LinearSolution
    fetch_pdata: PredictionData
    solver_pdata: PredictionData
    solver_start: datetime


async def run_optimization(
    start: datetime,
    end: datetime,
    prediction: Prediction,
    optimizer: LinearOptimizer,
    dt_hours: float,
    *,
    soc: Mapping[str, float] | None = None,
    initial_modes: Mapping[str, InverterMode | int] | None = None,
    solver_opts: Mapping[str, Any] | None = None,
    objective: OptimizationObjective | None = None,
    validate_with_simulation: bool = False,
    pdata_transform: Callable[[PredictionData], PredictionData] | None = None,
) -> OptimizationResult:
    """Fetch predictions and run the MILP energy optimizer.

    Slot alignment rules:

    * ``fetch_start = floor_to_slot(start)`` – providers are always queried from
      a clean slot boundary so caches hit consistently.
    * ``solver_start = ceil_to_slot(start)`` – the solver only sees *complete*
      slots that have not yet started.  When *start* is already on a boundary,
      ``solver_start == start`` and no data is dropped.
    * ``fetch_end   = last slot before (start + hours)`` – the fetch window covers
      the full requested range but never adds spurious slots when the end already
      falls on a boundary.

    The optional *pdata_transform* callback (e.g. ``services.apply_appliance_loads``)
    is applied to the full *fetch_pdata* **before** slicing so that appliance
    forecasts are correctly distributed across all returned slots.

    Args:
        start:      Horizon start (timezone-aware).
        end:        Horizon end (timezone-aware).
        prediction: Configured :class:`~GridPythia.prediction.prediction.Prediction`.
        optimizer:  :class:`~GridPythia.optimization.solver.LinearOptimizer`.
        dt_hours:   Slot duration in hours (must match prediction and optimizer).
        soc:        Per-inverter battery SoC in Wh.
        initial_modes: Per-inverter active mode at the start of the horizon.
        solver_opts: HiGHS option overrides for this call.
        objective:   Optimization objective override.
        validate_with_simulation: Attach a simulation parity report.
        pdata_transform: Optional callable applied to *fetch_pdata* before slicing.

    Raises:
        ValueError: If *start* or *end* are naive (timezone-unaware) datetimes.

    Returns:
        :class:`OptimizationResult` with solution and both prediction data objects.
    """
    if start.tzinfo is None:
        raise ValueError("run_optimization: start must be timezone-aware")
    if end.tzinfo is None:
        raise ValueError("run_optimization: end must be timezone-aware")

    fetch_start = floor_to_slot(start, dt_hours)
    solver_start = ceil_to_slot(start, dt_hours)
    fetch_hours = (end - fetch_start).total_seconds() / 3600.0

    logger.info(
        "run_optimization_start",
        requested_start=start.isoformat(),
        requested_end=end.isoformat(),
        fetch_start=fetch_start.isoformat(),
        solver_start=solver_start.isoformat(),
        fetch_hours=round(fetch_hours, 4),
        dt_hours=dt_hours,
    )

    fetch_pdata = await prediction.fetch(
        start=fetch_start,
        hours=fetch_hours,
        dt_hours=dt_hours,
    )

    if pdata_transform is not None:
        fetch_pdata = pdata_transform(fetch_pdata)

    solver_pdata = fetch_pdata.slice_from(solver_start)

    logger.debug(
        "run_optimization_windows",
        fetch_steps=fetch_pdata.steps,
        solver_steps=solver_pdata.steps,
        fetch_first=fetch_pdata.timestamps[0].isoformat(),
        fetch_last=fetch_pdata.timestamps[-1].isoformat(),
        solver_first=solver_pdata.timestamps[0].isoformat(),
        solver_last=solver_pdata.timestamps[-1].isoformat(),
    )

    solution = await asyncio.to_thread(
        lambda: optimizer.solve(
            solver_pdata,
            soc=dict(soc) if soc else None,
            initial_modes=dict(initial_modes) if initial_modes else None,
            objective=objective,
            solver_opts=dict(solver_opts) if solver_opts else None,
            validate_with_simulation=validate_with_simulation,
        )
    )

    logger.info(
        "run_optimization_done",
        solver_status=solution.solver_status,
        solve_time_s=round(solution.solve_time_s, 2),
        solver_start=solver_start.isoformat(),
        solver_steps=solver_pdata.steps,
    )

    return OptimizationResult(
        solution=solution,
        fetch_pdata=fetch_pdata,
        solver_pdata=solver_pdata,
        solver_start=solver_start,
    )
