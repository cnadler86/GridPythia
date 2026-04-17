"""Consumer-facing solution contract for the energy optimizer.

:class:`EnergySolution` is the typed, immutable base that every optimizer
must produce.  It contains everything a downstream consumer needs.

:class:`LinearSolution` extends the base with diagnostic fields that are
specific to the MILP solver (simulation parity report, GridSimulation
replay result).  These are always optional from the consumer's perspective.

``OptimizationObjective`` and ``SimulationParityReport`` live here because
they are part of the solution contract, not of the solver internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from GridPythia.optimization.plan import InverterPlan
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.grid_simulation import SimulationResult


class OptimizationObjective(str, Enum):
    """Supported objective functions for :class:`~GridPythia.optimization.solver.LinearOptimizer`."""

    MINIMIZE_COST = "cost"
    MAXIMIZE_SELF_CONSUMPTION = "self_consumption"


@dataclass(frozen=True)
class SimulationParityReport:
    """Difference report between LP result and GridSimulation replay.

    Attached to :class:`LinearSolution` when ``validate_with_simulation=True``
    is passed to :meth:`~GridPythia.optimization.solver.LinearOptimizer.solve`.
    """

    ok: bool
    max_abs_soc_error_wh: float
    max_abs_grid_import_error_wh: float
    max_abs_feedin_error_wh: float
    max_abs_cost_error_eur: float


@dataclass(frozen=True)
class EnergySolution:
    """Immutable consumer contract for any energy optimization result.

    This is the primary interface for downstream consumers (GUI, scheduler,
    reporting).  All fields are always populated — there are no ``None``
    sentinel values in the base contract.

    Attributes:
        prediction:     The :class:`~GridPythia.prediction.prediction.PredictionData`
                        the optimizer was solved against.  Consumers can call
                        ``prediction.to_solver_view()`` or read
                        ``prediction.timestamps`` to align result arrays with
                        wall-clock time.
        inverter_plans: Immutable tuple of per-inverter schedules.  Use
                        ``plan.windows()`` for a time-window view.
        objective:      Which objective function was optimized.
        solver_status:  Raw status string returned by the solver backend.
        solve_time_s:   Wall-clock seconds spent inside the solver.
        result:         Aggregated grid-level energy flows (costs, imports,
                        feed-in, losses) computed from the LP solution.
    """

    prediction: PredictionData
    inverter_plans: tuple[InverterPlan, ...]
    objective: OptimizationObjective
    solver_status: str
    solve_time_s: float
    result: SimulationResult


@dataclass(frozen=True)
class LinearSolution(EnergySolution):
    """Solution produced by :class:`~GridPythia.optimization.solver.LinearOptimizer`.

    Extends :class:`EnergySolution` with optional diagnostic information that
    is meaningful only for MILP-based solvers.

    Attributes:
        parity_report:     Comparison between LP solution and a GridSimulation
                           replay.  Populated only when
                           ``validate_with_simulation=True``.
        simulation_result: Full :class:`~GridPythia.simulation.grid_simulation.SimulationResult`
                           from the GridSimulation replay used for parity
                           checking.  ``None`` when parity checking was
                           skipped.
    """

    parity_report: SimulationParityReport | None = None
    simulation_result: SimulationResult | None = None
