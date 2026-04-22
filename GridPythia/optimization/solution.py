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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.grid_simulation import SimulationResult


class OptimizationObjective(str, Enum):
    """Supported objective functions for :class:`~GridPythia.optimization.solver.LinearOptimizer`."""

    MINIMIZE_COST = "cost"
    MAXIMIZE_SELF_CONSUMPTION = "self_consumption"


@dataclass(frozen=True, slots=True)
class PlanWindow:
    """Contiguous block of timesteps sharing the same inverter mode.

    Energy arrays inside a window are *views* into the parent :class:`InverterPlan`
    arrays — no copying occurs when iterating windows.
    """

    start_idx: int  # inclusive index into the full plan
    end_idx: int  # exclusive index into the full plan
    mode: InverterMode
    charge_ac_wh: np.ndarray  # AC energy drawn from grid for charging [Wh/slot]
    discharge_ac_wh: np.ndarray  # AC energy delivered to home from battery [Wh/slot]
    pv_to_ac_wh: np.ndarray  # PV energy routed directly to AC bus [Wh/slot]
    pv_to_battery_wh: np.ndarray  # PV energy routed into battery [Wh/slot]
    battery_soc_wh: np.ndarray | None  # SoC at end of each slot [Wh], or None

    @property
    def steps(self) -> int:
        """Number of timestep slots in this window."""
        return self.end_idx - self.start_idx

    @property
    def total_charge_ac_wh(self) -> float:
        """Total AC charge energy across the window [Wh]."""
        return float(self.charge_ac_wh.sum())

    @property
    def total_discharge_ac_wh(self) -> float:
        """Total AC discharge energy across the window [Wh]."""
        return float(self.discharge_ac_wh.sum())

    @property
    def total_pv_to_ac_wh(self) -> float:
        """Total PV-to-AC energy across the window [Wh]."""
        return float(self.pv_to_ac_wh.sum())

    @property
    def total_pv_to_battery_wh(self) -> float:
        """Total PV-to-battery energy across the window [Wh]."""
        return float(self.pv_to_battery_wh.sum())


@dataclass(frozen=True, slots=True)
class InverterPlan:
    """Raw per-timestep schedule produced by the optimizer for one inverter.

    Per-slot arrays are indexed ``[0 … steps-1]`` and correspond 1-to-1 with
    the :class:`~GridPythia.prediction.prediction.PredictionData` timestamps of
    the solution they belong to.

    Use :meth:`windows` to obtain a time-window view that is easier to
    display or act on (consecutive same-mode slots are merged into a single
    :class:`PlanWindow`).
    """

    device_id: str
    modes: np.ndarray  # InverterMode int per timestep (np.int8)
    charge_ac_wh: np.ndarray  # AC energy drawn from grid for charging [Wh/dt]
    discharge_ac_wh: np.ndarray  # AC energy delivered to home from battery [Wh/dt]
    pv_to_ac_wh: np.ndarray  # PV energy routed directly to AC [Wh/dt]
    pv_to_battery_wh: np.ndarray  # PV energy routed into battery [Wh/dt]
    battery_soc_wh: np.ndarray | None = None  # SoC at end of each slot [Wh]

    @property
    def steps(self) -> int:
        """Number of timestep slots in the plan."""
        return int(self.modes.shape[0])

    def windows(self) -> list[PlanWindow]:
        """Group consecutive same-mode slots into contiguous :class:`PlanWindow` objects.

        Returns an empty list for a zero-length plan.  The returned windows
        cover the full plan without gaps and in chronological order.
        Array slices inside each window are views — no data is copied.
        """
        T = self.steps
        if T == 0:
            return []

        result: list[PlanWindow] = []
        start = 0
        current_mode_int = int(self.modes[0])

        for i in range(1, T + 1):
            if i == T or int(self.modes[i]) != current_mode_int:
                sl = slice(start, i)
                result.append(
                    PlanWindow(
                        start_idx=start,
                        end_idx=i,
                        mode=InverterMode(current_mode_int),
                        charge_ac_wh=self.charge_ac_wh[sl],
                        discharge_ac_wh=self.discharge_ac_wh[sl],
                        pv_to_ac_wh=self.pv_to_ac_wh[sl],
                        pv_to_battery_wh=self.pv_to_battery_wh[sl],
                        battery_soc_wh=(
                            self.battery_soc_wh[sl] if self.battery_soc_wh is not None else None
                        ),
                    )
                )
                if i < T:
                    start = i
                    current_mode_int = int(self.modes[i])

        return result

    # ------------------------------------------------------------------
    # Dict-style access kept for backward compatibility with consumers
    # that index plans as plan["charge_ac_wh"] etc.
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return getattr(self, key, default)


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

    def plot(self, *, title: str = "Optimization Result") -> "go.Figure":
        """Return a Plotly figure visualising the solution's energy flows.

        Requires *plotly* to be installed (``pip install plotly``).
        """
        import plotly.graph_objects as go  # noqa: F401 – local import

        from GridPythia.optimization.plots import SolutionPlotter  # local import

        return SolutionPlotter().plot(self, title=title)


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
