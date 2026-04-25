"""Linear (MILP) energy-management optimizer using CVXPY + HiGHS.

This module builds a topology-aware mathematical model from the same
core signals that drive GridSimulation (load, PV, battery, prices) and
solves it with HiGHS. Home appliances are intentionally excluded from
this LP model for now.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from platform import machine
from typing import Any, cast

# ARM libatomic preloading is handled in main.py (before any CVXPY import).
# The message below is kept for visibility when the solver is loaded.
if machine() in ("armv7l", "armv6l"):
    pass  # libatomic already loaded by main.py via ctypes.CDLL(RTLD_GLOBAL)

import cvxpy as cp
import numpy as np
from structlog import get_logger

from GridPythia.optimization.solution import (
    InverterPlan,
    LinearSolution,
    OptimizationObjective,
    SimulationParityReport,
)
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.inverterbase import InverterBase
from GridPythia.simulation.grid_interpolator import FraunhoferSCModel
from GridPythia.simulation.grid_simulation import GridSimulation, SimulationResult

logger = get_logger(__name__)


_DEFAULT_HIGHS_OPTS = {
    "verbose": False,
    "warm_start": True,
    "time_limit": 30,
    "mip_rel_gap": 0.03,
    "presolve": "on",
    "mip_lp_solver": "ipm",
    "mip_heuristic_effort": 0.05,
    "mip_heuristic_run_rens": False,
    "mip_heuristic_run_rins": True,
    "mip_heuristic_run_feasibility_jump": False,
    "mip_heuristic_run_root_reduced_cost": False,
    "small_matrix_value": 1e-4,
}


@dataclass
class _PreparedInputs:
    """Numerical input series for one optimization horizon."""

    T: int
    dt: float
    load_wh: np.ndarray
    price: np.ndarray
    feedin_tariff: np.ndarray
    pv_by_source: Mapping[str, np.ndarray]


@dataclass
class _InverterModelBlock:
    """Decision and helper expressions for one inverter block."""

    inverter: InverterBase
    # AC-side charging energy per timestep [Wh] (variable when inverter+bat present)
    p_ch: cp.Expression
    # AC-side discharging energy per timestep [Wh] (variable when inverter+bat present)
    p_dc: cp.Expression
    # AC power from PV routed to the AC bus per timestep [Wh]
    pv_ac: cp.Expression
    # PV energy routed into the battery (DC side) per timestep [Wh]
    pv_to_bat: cp.Expression
    # Buffered portion of pv_ac: backed by battery headroom, 100% controllable [Wh]
    pv_ac_buffered: cp.Expression
    # State-of-charge time series for the connected battery [Wh] or None
    soc: cp.Variable | None
    # Net battery energy flow [Wh/dt]: positive=charge, negative=discharge
    battery_net_flow_wh: cp.Expression | None
    # Binary/indicator for AC-charge mode activity (1 when AC charging active)
    mode_ch_activity: cp.Expression | None
    # Binary/indicator for DC-discharge mode activity (1 when discharge/bypass active)
    mode_dc_activity: cp.Expression | None


class LinearOptimizer:
    """MILP optimizer with a compiled reusable CVXPY model.

    The problem structure (variables + constraints) is compiled once per optimizer
    instance and subsequent solves only update runtime Parameters (prediction arrays,
    battery start SoC, and initial inverter mode).
    """

    _MONETARY_OBJECTIVE_SCALE = 100.0

    def __init__(
        self,
        inverters: list[InverterBase],
        *,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST,
        solver_opts: Mapping[str, Any] | None = None,
    ) -> None:
        self.inverters = inverters
        self._objective = objective
        self._solver_opts: dict[str, Any] = dict(solver_opts) if solver_opts else {}
        self._cached_prediction: PredictionData | None = None
        self._cached_solution: LinearSolution | None = None
        self._T = 0
        self._dt = 0.0

        self._log = logger.bind(
            component="optimizer",
            inverter_ids=[inv.device_id for inv in inverters],
        )

        self._compiled = False

    def solve(
        self,
        prediction: PredictionData,
        *,
        soc: Mapping[str, float] | None = None,
        initial_modes: Mapping[str, InverterMode | int] | None = None,
        objective: OptimizationObjective | None = None,
        solver_opts: Mapping[str, Any] | None = None,
        validate_with_simulation: bool = False,
    ) -> LinearSolution:
        """Solve the compiled MILP for the given prediction horizon.

        Args:
            prediction: Current forecast data including timestamps.
            soc:         Per-inverter battery state-of-charge in Wh.  Falls back to
                         ``inv.battery.soc_wh`` when omitted.
            initial_modes: Per-inverter active mode at the start of the horizon.
            objective:   Override the instance-level objective for this call only.
            solver_opts: Override or extend the instance-level HiGHS options.
            validate_with_simulation: Run a GridSimulation replay and attach a
                         parity report to the returned solution.
        """
        effective_objective = objective if objective is not None else self._objective
        prep = self._prepare_inputs(prediction)
        self._ensure_compiled_layout(prep)

        normalized_initial_modes: dict[str, InverterMode] = {}
        for inv in self.inverters:
            raw_mode = (
                initial_modes.get(inv.device_id, InverterMode.IDLE)
                if initial_modes is not None
                else InverterMode.IDLE
            )
            normalized_initial_modes[inv.device_id] = (
                raw_mode if isinstance(raw_mode, InverterMode) else InverterMode(int(raw_mode))
            )

        warm_start_plan = self._build_warm_start_plan(self._compute_roll_steps(prediction))
        self._update_runtime_parameters(prep, normalized_initial_modes, soc=soc)
        self._apply_warm_start_values(prep, warm_start_plan)

        problem = self._problems[effective_objective]
        size = problem.size_metrics
        self._log.info(
            "optimizer_solve_start",
            objective=effective_objective.value,
            num_variables=size.num_scalar_variables,
            num_constraints=size.num_scalar_eq_constr + size.num_scalar_leq_constr,
        )

        # Merge: defaults → instance opts → per-call overrides.
        opts = dict(_DEFAULT_HIGHS_OPTS)
        opts.update(self._solver_opts)
        if solver_opts:
            opts.update(dict(solver_opts))

        t0 = time.perf_counter()
        try:
            problem.solve(solver=cp.HIGHS, canon_backend=cp.SCIPY_CANON_BACKEND, **opts)
        except cp.SolverError as exc:
            raise RuntimeError(f"CVXPY/HiGHS solver error: {exc}") from exc
        solve_time = time.perf_counter() - t0

        status = problem.status
        accepted_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if (
            status == "user_limit"
            and self._g_import.value is not None
            and self._g_feedin.value is not None
        ):
            accepted_statuses.add("user_limit")

        if status not in accepted_statuses:
            self._log.error(
                "optimizer_solve_failed",
                solver_status=status,
                solve_time_s=round(solve_time, 3),
            )
            raise RuntimeError(
                f"Optimisation did not converge: solver status='{status}'. "
                "Check feasibility (battery bounds, rates, and capacities)."
            )

        if self._g_import.value is None or self._g_feedin.value is None:
            self._log.error("optimizer_no_values", solver_status=status)
            raise RuntimeError("Solver returned no values for grid variables")

        self._log.info(
            "optimizer_solve_complete",
            solver_status=status,
            solve_time_s=round(solve_time, 3),
            objective=effective_objective.value,
            objective_value=round(
                self._objective_value_for_logging(effective_objective, problem.value), 4
            )
            if problem.value is not None
            else None,
        )

        solution = self._build_solution(
            prep=prep,
            blocks=self._blocks,
            g_import_val=np.asarray(self._g_import.value, dtype=float),
            g_feedin_val=np.asarray(self._g_feedin.value, dtype=float),
            objective=effective_objective,
            solver_status=status,
            solve_time=solve_time,
            prediction=prediction,
        )

        self._cached_prediction = prediction
        self._cached_solution = solution

        if validate_with_simulation:
            parity, sim_res = self._validate_with_simulation(solution, prep, prediction)
            solution = replace(solution, parity_report=parity, simulation_result=sim_res)
            self._cached_solution = solution
            if parity.ok:
                self._log.debug("optimizer_parity_ok")
            else:
                self._log.warning(
                    "optimizer_parity_mismatch",
                    max_soc_error_wh=round(parity.max_abs_soc_error_wh, 4),
                    max_grid_import_error_wh=round(parity.max_abs_grid_import_error_wh, 4),
                    max_feedin_error_wh=round(parity.max_abs_feedin_error_wh, 4),
                    max_cost_error_eur=round(parity.max_abs_cost_error_eur, 6),
                )

        return solution

    def _compute_roll_steps(self, new_prediction: PredictionData) -> int:
        """Return how many horizon steps the new prediction has advanced vs. the cached one.

        Compares the first timestamp of *new_prediction* against the first timestamp
        of the most-recently cached prediction and converts the elapsed time to a
        whole number of ``dt_hours`` steps (rounded to nearest).  Returns 0 when
        there is no cached prediction or when the first timestamp has not changed.
        """
        if self._cached_prediction is None:
            return 0
        cached_ts = self._cached_prediction.timestamps
        new_ts = new_prediction.timestamps
        if not cached_ts or not new_ts:
            return 0
        delta_s = (new_ts[0] - cached_ts[0]).total_seconds()
        if delta_s <= 0:
            return 0
        dt_s = self._dt * 3600.0
        return min(max(0, round(delta_s / dt_s)), self._T)

    def _build_warm_start_plan(
        self, shift_steps: int
    ) -> dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
    ]:
        """Build a warm-start seed from the cached solution, shifted by *shift_steps*.

        Returns a dict mapping device_id → (modes, charge_ac_wh, discharge_ac_wh,
        pv_to_bat_wh, pv_ac_wh, soc_wh | None).
        """
        if self._cached_solution is None or shift_steps <= 0:
            return {}

        T = self._T
        out: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
        ] = {}
        for inv_plan in self._cached_solution.inverter_plans:
            src_modes = np.asarray(inv_plan.modes, dtype=np.int8)
            src_ch = np.asarray(inv_plan.charge_ac_wh, dtype=np.float32)
            src_dc = np.asarray(inv_plan.discharge_ac_wh, dtype=np.float32)
            src_pvb = np.asarray(inv_plan.pv_to_battery_wh, dtype=np.float32)
            src_pva = np.asarray(inv_plan.pv_to_ac_wh, dtype=np.float32)

            if src_modes.size == 0:
                out[inv_plan.device_id] = (
                    np.full(T, int(InverterMode.IDLE), dtype=np.int8),
                    np.zeros(T, dtype=np.float32),
                    np.zeros(T, dtype=np.float32),
                    np.zeros(T, dtype=np.float32),
                    np.zeros(T, dtype=np.float32),
                    None,
                )
                continue

            src_start = min(shift_steps, int(src_modes.size))
            copied = min(max(int(src_modes.size) - src_start, 0), T)

            shifted_modes = np.full(T, int(InverterMode.IDLE), dtype=np.int8)
            shifted_ch = np.zeros(T, dtype=np.float32)
            shifted_dc = np.zeros(T, dtype=np.float32)
            shifted_pvb = np.zeros(T, dtype=np.float32)
            shifted_pva = np.zeros(T, dtype=np.float32)

            if copied > 0:
                sl = slice(src_start, src_start + copied)
                shifted_modes[:copied] = src_modes[sl]
                shifted_ch[:copied] = src_ch[sl]
                shifted_dc[:copied] = src_dc[sl]
                shifted_pvb[:copied] = src_pvb[sl]
                shifted_pva[:copied] = src_pva[sl]

            if copied < T:
                shifted_modes[copied:] = src_modes[-1]
                shifted_ch[copied:] = src_ch[-1]
                shifted_dc[copied:] = src_dc[-1]
                shifted_pvb[copied:] = src_pvb[-1]
                shifted_pva[copied:] = src_pva[-1]

            # ZFI/IDLE modes carry no explicit AC power target
            zfi_mask = np.isin(
                shifted_modes,
                [
                    int(InverterMode.IDLE),
                    int(InverterMode.DISCHARGE_ZERO_FEED_IN),
                    int(InverterMode.AC_CHARGE_ZERO_FEED_IN),
                ],
            )
            shifted_ch[zfi_mask] = 0.0
            shifted_dc[zfi_mask] = 0.0
            # pv_to_bat may still be nonzero in IDLE (PV → battery passively)
            # pv_ac is 0 in IDLE (gated by mode_dc)
            shifted_pva[shifted_modes == int(InverterMode.IDLE)] = 0.0

            # Shift the SoC trace: new_soc[t] = old_soc[t + shift_steps]
            shifted_soc: np.ndarray | None = None
            if inv_plan.battery_soc_wh is not None and len(inv_plan.battery_soc_wh) > 0:
                src_soc = np.asarray(inv_plan.battery_soc_wh, dtype=np.float32)
                shifted_soc = np.empty(T, dtype=np.float32)
                soc_src_start = min(shift_steps, int(src_soc.size))
                soc_copied = min(max(int(src_soc.size) - soc_src_start, 0), T)
                if soc_copied > 0:
                    shifted_soc[:soc_copied] = src_soc[soc_src_start : soc_src_start + soc_copied]
                if soc_copied < T:
                    shifted_soc[soc_copied:] = src_soc[-1]

            out[inv_plan.device_id] = (
                shifted_modes,
                shifted_ch,
                shifted_dc,
                shifted_pvb,
                shifted_pva,
                shifted_soc,
            )
        return out

    def _prepare_inputs(self, prediction: PredictionData) -> _PreparedInputs:
        solver_view = prediction.to_solver_view(dtype=np.float64)
        return _PreparedInputs(
            T=solver_view.steps,
            dt=solver_view.dt_hours,
            load_wh=solver_view.load_wh,
            price=solver_view.electricprice_eur_wh,
            feedin_tariff=solver_view.feedintariff_eur_wh,
            pv_by_source=solver_view.pv_by_inverter,
        )

    def _compile_problem(self, *, horizon: int, dt: float) -> None:
        self._T = int(horizon)
        self._dt = float(dt)

        self._constraints: list[cp.Constraint] = []
        self._mode_switch_costs: list[cp.Expression] = []

        self._load_param = cp.Parameter(self._T, name="load_wh")
        self._price_param = cp.Parameter(self._T, name="price")
        self._feedin_param = cp.Parameter(self._T, name="feedin_tariff")
        self._terminal_value_param = cp.Parameter(nonneg=True, name="terminal_value")

        self._pv_params: dict[str, cp.Parameter] = {
            inv.device_id: cp.Parameter(self._T, nonneg=True, name=f"pv_pred_{inv.device_id}")
            for inv in self.inverters
            if inv.parameters.has_pv
        }
        self._pv_external_param = cp.Parameter(self._T, nonneg=True, name="pv_pred_external")

        self._soc_init_params: dict[str, cp.Parameter] = {}
        self._init_mode_ch_params: dict[str, cp.Parameter] = {}
        self._init_mode_dc_params: dict[str, cp.Parameter] = {}

        self._g_import = cp.Variable(self._T, nonneg=True, name="g_import")
        self._g_feedin = cp.Variable(self._T, nonneg=True, name="g_feedin")
        self._pv_unbuffered_self = cp.Variable(self._T, nonneg=True, name="pv_unbuf_self")

        # Fraunhofer SC model parameters (updated per solve with linearization coefficients).
        # Applied only to unbuffered PV (PV on AC not backed by battery headroom).
        # Keep DPP compliance by avoiding parameter*parameter terms in constraints.
        self._sc_rhs_base = cp.Parameter(self._T, name="sc_rhs_base")
        self._sc_c_pv = cp.Parameter(self._T, name="sc_c_pv")
        self._sc_c_buf = cp.Parameter(self._T, name="sc_c_buf")

        total_p_ch_terms: list[object] = []
        total_p_dc_terms: list[object] = []
        total_pv_ac_var_terms: list[object] = []
        total_pv_buffered_terms: list[object] = []
        blocks = [self._build_inverter_block(inv) for inv in self.inverters]
        for _inv, block in zip(self.inverters, blocks, strict=False):
            total_p_ch_terms.append(block.p_ch)
            total_p_dc_terms.append(block.p_dc)
            total_pv_ac_var_terms.append(block.pv_ac)
            total_pv_buffered_terms.append(block.pv_ac_buffered)

        total_p_ch = self._sum_terms(total_p_ch_terms, self._T)
        total_p_dc = self._sum_terms(total_p_dc_terms, self._T)
        total_pv_ac_var = self._sum_terms(total_pv_ac_var_terms, self._T)
        total_pv_ac = total_pv_ac_var + self._pv_external_param
        total_pv_buffered = self._sum_terms(total_pv_buffered_terms, self._T)

        # Store expressions for use in other methods
        self._total_pv_ac_expr = total_pv_ac
        self._total_pv_buffered_expr = total_pv_buffered

        # --- Fraunhofer self-consumption model (on UNBUFFERED PV only) ---
        # Unbuffered PV = total_pv_ac - total_pv_buffered.
        # Buffered PV is backed by battery headroom → 100% controllable → directly
        # covers load.  Only the unbuffered remainder goes through Fraunhofer.
        #
        # Linearized SC constraint:
        #   pv_unbuf_self <= sc_rhs_base + c_pv * total_pv_ac_var + c_buf * total_pv_buffered
        # where sc_rhs_base folds c_const + c_pv * pv_external + c_load * load, and
        # c_buf = -(c_pv + c_load) (more buffered → less remaining load → less SC).
        self._constraints.append(
            self._pv_unbuffered_self
            <= self._sc_rhs_base
            + cp.multiply(self._sc_c_pv, total_pv_ac_var)
            + cp.multiply(self._sc_c_buf, total_pv_buffered)
        )
        # Physical bounds on unbuffered self-consumption
        self._constraints.extend(
            [
                # Can't self-consume more unbuffered PV than exists
                self._pv_unbuffered_self <= total_pv_ac - total_pv_buffered,
                # Buffered PV + unbuffered self-consumption can't exceed load
                self._pv_unbuffered_self <= self._load_param - total_pv_buffered,
            ]
        )

        # --- Corrected AC-bus energy balance ---
        # IN:  g_import + p_dc + total_pv_ac
        # OUT: load + p_ch + g_feedin
        self._constraints.append(
            self._g_import - self._g_feedin
            == self._load_param + total_p_ch - total_p_dc - total_pv_ac
        )

        # --- Minimum feedin from uncontrollable PV ---
        # Unbuffered PV that can't be self-consumed must be exported.
        self._constraints.append(
            self._g_feedin >= total_pv_ac - total_pv_buffered - self._pv_unbuffered_self
        )

        terminal_terms = [block.soc[-1] for block in blocks if block.soc is not None]
        self._terminal_soc_sum: cp.Expression | float
        if terminal_terms:
            self._terminal_soc_sum = cp.sum(cp.hstack(terminal_terms))
        else:
            self._terminal_soc_sum = 0.0

        mode_switch_costs_term = cp.sum(self._mode_switch_costs) if self._mode_switch_costs else 0.0
        terminal_reward = self._terminal_reward_expr()

        feedin_revenue_term = cp.multiply(self._g_feedin, self._feedin_param)

        objective_cost_eur = (
            cp.sum(cp.multiply(self._g_import, self._price_param) - feedin_revenue_term)
            + mode_switch_costs_term
            - terminal_reward
        )
        objective_cost = self._MONETARY_OBJECTIVE_SCALE * objective_cost_eur

        objective_self = cp.sum(self._g_feedin) - terminal_reward

        self._blocks = blocks
        self._problems = {
            OptimizationObjective.MINIMIZE_COST: cp.Problem(
                cp.Minimize(objective_cost),
                self._constraints,
            ),
            OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION: cp.Problem(
                cp.Minimize(objective_self),
                self._constraints,
            ),
        }

        self._compiled = True

    def _ensure_compiled_layout(self, prep: _PreparedInputs) -> None:
        if not self._compiled:
            self._compile_problem(horizon=prep.T, dt=prep.dt)
            return
        if prep.T != self._T or prep.dt != self._dt:
            self._log.warning(
                "optimizer_recompile",
                old_T=self._T,
                new_T=prep.T,
                old_dt=self._dt,
                new_dt=prep.dt,
            )
            self._cached_prediction = None
            self._cached_solution = None
            self._compile_problem(horizon=prep.T, dt=prep.dt)

    def _update_runtime_parameters(
        self,
        prep: _PreparedInputs,
        initial_modes: Mapping[str, InverterMode],
        soc: Mapping[str, float] | None = None,
    ) -> None:
        self._load_param.value = prep.load_wh
        self._price_param.value = prep.price
        self._feedin_param.value = prep.feedin_tariff

        for inv in self.inverters:
            inv_id = inv.device_id
            if inv_id in self._pv_params:
                pv_arr = prep.pv_by_source.get(inv_id, np.zeros(prep.T, dtype=float))
                self._pv_params[inv_id].value = pv_arr

            if inv_id in self._soc_init_params:
                if inv.battery is None:
                    self._soc_init_params[inv_id].value = 0.0
                elif soc is not None and inv_id in soc:
                    self._soc_init_params[inv_id].value = float(
                        np.clip(soc[inv_id], inv.battery.min_soc_wh, inv.battery.max_soc_wh)
                    )
                else:
                    self._soc_init_params[inv_id].value = float(inv.battery.soc_wh)

            init_mode = initial_modes.get(inv_id, InverterMode.IDLE)
            init_ch, init_dc = self._mode_flags(init_mode)
            if inv_id in self._init_mode_ch_params:
                self._init_mode_ch_params[inv_id].value = float(init_ch)
            if inv_id in self._init_mode_dc_params:
                self._init_mode_dc_params[inv_id].value = float(init_dc)

        mapped = set(self._pv_params)
        external = np.zeros(prep.T, dtype=float)
        for src_id, arr in prep.pv_by_source.items():
            if src_id not in mapped:
                external += np.asarray(arr, dtype=float)
        self._pv_external_param.value = external

        # Fraunhofer SC model linearization (vectorized)
        # Compute total PV AC estimate as operating point for linearization
        total_pv_ac_estimate = external.copy()
        for inv_id in self._pv_params:
            pv_arr = prep.pv_by_source.get(inv_id, np.zeros(prep.T, dtype=float))
            total_pv_ac_estimate += pv_arr

        # Estimate buffered PV per inverter: min(pv, headroom_in_ac_terms)
        total_buffered_estimate = np.zeros(prep.T, dtype=float)
        for block in self._blocks:
            inv = block.inverter
            if inv.battery is not None and inv.device_id in self._pv_params:
                bat = inv.battery
                soc_init = float(self._soc_init_params[inv.device_id].value or bat.soc_wh)
                headroom_wh = max(bat.max_soc_wh - soc_init, 0.0)
                eta_buf = bat.charging_efficiency / inv.parameters.dc_to_ac_efficiency
                headroom_ac = headroom_wh / max(eta_buf, 1e-9)
                pv_arr = prep.pv_by_source.get(inv.device_id, np.zeros(prep.T, dtype=float))
                buf_est = np.minimum(pv_arr, headroom_ac)
                buf_est = np.minimum(buf_est, prep.load_wh)
                total_buffered_estimate += buf_est

        # Unbuffered PV estimate (operating point for Fraunhofer linearization)
        total_unbuffered_estimate = np.maximum(total_pv_ac_estimate - total_buffered_estimate, 0.0)
        remaining_load_estimate = np.maximum(prep.load_wh - total_buffered_estimate, 0.0)

        # Use a strictly positive baseload to satisfy FraunhoferSCModel input contract.
        min_load = (
            float(np.min(remaining_load_estimate)) if remaining_load_estimate.size > 0 else 0.0
        )
        sc_model = FraunhoferSCModel(baseload_wh=max(min_load, 10.0), dt=prep.dt)

        # Linearize around (unbuffered_estimate, remaining_load_estimate).
        c_const, c_pv, c_load = sc_model.linearize_batch(
            pv_0=total_unbuffered_estimate, load_0=remaining_load_estimate
        )
        # Fold all parameter-only parts into sc_rhs_base to keep the CVXPY model DPP.
        # The constraint is: pv_unbuf_self <= c_const + c_pv * unbuf + c_load * remaining_load
        #   unbuf = (total_pv_ac_var - total_pv_buffered) + pv_external
        #   remaining_load = load - total_pv_buffered
        # Expanding: c_const + c_pv * pv_external + c_load * load  [→ sc_rhs_base]
        #   + c_pv * total_pv_ac_var                                [→ sc_c_pv]
        #   + (-(c_pv + c_load)) * total_pv_buffered                [→ sc_c_buf]
        self._sc_rhs_base.value = c_const + c_pv * external + c_load * prep.load_wh
        self._sc_c_pv.value = c_pv
        self._sc_c_buf.value = -(c_pv + c_load)

        terminal_value = self._estimate_terminal_value(prep, self._blocks)
        self._terminal_value_param.value = terminal_value

    def _apply_warm_start_values(
        self,
        prep: _PreparedInputs,
        warm_start_plan: Mapping[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
        ]
        | None,
    ) -> None:
        if not warm_start_plan or prep.T <= 0:
            return

        def _pad(arr: np.ndarray, fill: float | int) -> np.ndarray:
            if arr.size < prep.T:
                return np.pad(arr, (0, prep.T - arr.size), constant_values=fill)
            return arr

        for block in self._blocks:
            inv = block.inverter
            seed = warm_start_plan.get(inv.device_id)
            if seed is None:
                continue

            seed_modes, seed_ch_wh, seed_dc_wh, seed_pvb_wh, seed_pva_wh, seed_soc_wh = seed
            if seed_modes.size == 0:
                continue

            seed_modes = _pad(seed_modes, int(InverterMode.IDLE))
            seed_ch_wh = _pad(seed_ch_wh, 0.0)
            seed_dc_wh = _pad(seed_dc_wh, 0.0)
            seed_pvb_wh = _pad(seed_pvb_wh, 0.0)
            seed_pva_wh = _pad(seed_pva_wh, 0.0)

            if isinstance(block.mode_ch_activity, cp.Variable):
                ch = np.isin(
                    seed_modes,
                    [int(InverterMode.AC_CHARGE), int(InverterMode.AC_CHARGE_ZERO_FEED_IN)],
                ).astype(float)
                block.mode_ch_activity.value = ch

            if isinstance(block.mode_dc_activity, cp.Variable):
                dc = np.isin(
                    seed_modes,
                    [int(InverterMode.DISCHARGE), int(InverterMode.DISCHARGE_ZERO_FEED_IN)],
                ).astype(float)
                block.mode_dc_activity.value = dc

            if isinstance(block.p_ch, cp.Variable):
                p_ch_val = seed_ch_wh[: prep.T].astype(float)
                # Project to semi-continuous domain: when mode_ch=1, p_ch >= min.
                # This ensures the warm start satisfies the min-power constraint so
                # HiGHS can accept it directly without an LP repair step.
                _min_p_w = inv.parameters.min_ac_output_power_w
                if _min_p_w > 0.0 and isinstance(block.mode_ch_activity, cp.Variable):
                    _ch_on = (
                        block.mode_ch_activity.value > 0.5
                        if block.mode_ch_activity.value is not None
                        else np.zeros(prep.T, dtype=bool)
                    )
                    _min_ch_wh = _min_p_w * prep.dt
                    p_ch_val = p_ch_val.copy()
                    p_ch_val[~_ch_on] = 0.0
                    p_ch_val[_ch_on] = np.maximum(p_ch_val[_ch_on], _min_ch_wh)
                block.p_ch.value = p_ch_val

            if isinstance(block.p_dc, cp.Variable):
                p_dc_val = seed_dc_wh[: prep.T].astype(float)
                # Project to semi-continuous domain: when mode_dc=1, p_dc >= min.
                _min_p_w = inv.parameters.min_ac_output_power_w
                if _min_p_w > 0.0 and isinstance(block.mode_dc_activity, cp.Variable):
                    _dc_on = (
                        block.mode_dc_activity.value > 0.5
                        if block.mode_dc_activity.value is not None
                        else np.zeros(prep.T, dtype=bool)
                    )
                    _min_dc_wh = _min_p_w * prep.dt
                    p_dc_val = p_dc_val.copy()
                    p_dc_val[~_dc_on] = 0.0
                    p_dc_val[_dc_on] = np.maximum(p_dc_val[_dc_on], _min_dc_wh)
                block.p_dc.value = p_dc_val

            if isinstance(block.pv_to_bat, cp.Variable):
                block.pv_to_bat.value = seed_pvb_wh[: prep.T].astype(float)

            if isinstance(block.pv_ac, cp.Variable):
                block.pv_ac.value = seed_pva_wh[: prep.T].astype(float)

            # Warm-start pv_ac_buffered: estimate from battery headroom
            if isinstance(block.pv_ac_buffered, cp.Variable):
                pv_ac_val = seed_pva_wh[: prep.T].astype(float)
                if inv.battery is not None and inv.device_id in self._soc_init_params:
                    bat = inv.battery
                    soc0 = float(self._soc_init_params[inv.device_id].value or bat.soc_wh)
                    headroom_wh = max(bat.max_soc_wh - soc0, 0.0)
                    eta_buf = bat.charging_efficiency / inv.parameters.dc_to_ac_efficiency
                    headroom_ac = headroom_wh / max(eta_buf, 1e-9)
                    buf_val = np.minimum(pv_ac_val, headroom_ac)
                    buf_val = np.minimum(buf_val, np.maximum(prep.load_wh, 0.0))
                    block.pv_ac_buffered.value = buf_val
                else:
                    block.pv_ac_buffered.value = np.zeros(prep.T, dtype=float)

            if isinstance(block.soc, cp.Variable) and inv.device_id in self._soc_init_params:
                if seed_soc_wh is not None and seed_soc_wh.size >= prep.T:
                    block.soc.value = seed_soc_wh[: prep.T].astype(float)
                else:
                    soc0 = float(self._soc_init_params[inv.device_id].value or 0.0)
                    block.soc.value = np.full(prep.T, soc0, dtype=float)

        # Warm-start pv_unbuffered_self with Fraunhofer model estimate
        if isinstance(self._pv_unbuffered_self, cp.Variable):
            # Compute total PV AC and buffered estimates
            total_pv_ac_est = np.zeros(prep.T, dtype=float)
            total_buffered_est = np.zeros(prep.T, dtype=float)
            for block in self._blocks:
                inv_id = block.inverter.device_id
                if inv_id in prep.pv_by_source:
                    total_pv_ac_est += prep.pv_by_source[inv_id]
                if (
                    isinstance(block.pv_ac_buffered, cp.Variable)
                    and block.pv_ac_buffered.value is not None
                ):
                    total_buffered_est += np.asarray(block.pv_ac_buffered.value, dtype=float)

            total_unbuffered_est = np.maximum(total_pv_ac_est - total_buffered_est, 0.0)
            remaining_load_est = np.maximum(prep.load_wh - total_buffered_est, 0.0)

            # Use Fraunhofer model for warm-start (vectorized).
            mean_load = float(np.mean(remaining_load_est)) if remaining_load_est.size > 0 else 0.0
            sc_model = FraunhoferSCModel(baseload_wh=max(mean_load, 1.0), dt=prep.dt)
            unbuf_self_est = sc_model.self_consumed_wh(total_unbuffered_est, remaining_load_est)

            self._pv_unbuffered_self.value = np.asarray(unbuf_self_est, dtype=float)

    def _build_inverter_block(self, inv: InverterBase) -> _InverterModelBlock:
        T = self._T
        dt = self._dt
        inv_id = inv.device_id

        has_charge_mode = InverterMode.AC_CHARGE in inv.available_modes
        has_discharge_mode = any(
            mode in inv.available_modes
            for mode in (InverterMode.DISCHARGE, InverterMode.DISCHARGE_ZERO_FEED_IN)
        )

        mode_ch_activity: cp.Expression | None = None
        mode_dc_activity: cp.Expression | None = None
        battery_net_flow_wh: cp.Expression | None = None

        if inv.battery is not None and inv.is_optimizable and has_charge_mode:
            max_ch_power_w = float(inv.battery.max_charge_power_w)
            if inv.parameters.max_ac_charge_power_w > 0:
                max_ch_power_w = min(max_ch_power_w, float(inv.parameters.max_ac_charge_power_w))
            max_ch_wh = max_ch_power_w * dt
            mode_ch_activity = cp.Variable(T, boolean=True, name=f"mode_ch_{inv_id}")
            if max_ch_wh > 0:
                p_ch = cp.Variable(T, nonneg=True, name=f"p_ch_{inv_id}")
                self._constraints.append(p_ch <= max_ch_wh * mode_ch_activity)
            else:
                p_ch = cp.Constant(np.zeros(T, dtype=float))
            if inv.parameters.min_ac_output_power_w > 0 and max_ch_wh > 0:
                min_ch_wh = inv.parameters.min_ac_output_power_w * dt
                self._constraints.append(p_ch >= min_ch_wh * mode_ch_activity)
        else:
            p_ch = cp.Constant(np.zeros(T, dtype=float))

        if inv.battery is not None and inv.is_optimizable and has_discharge_mode:
            max_dc_power_w = float(inv.battery.max_discharge_power_w)
            if inv.parameters.max_ac_output_power_w > 0:
                max_dc_power_w = min(max_dc_power_w, float(inv.parameters.max_ac_output_power_w))
            max_dc_wh = max_dc_power_w * dt
            mode_dc_activity = cp.Variable(T, boolean=True, name=f"mode_dc_{inv_id}")
            if max_dc_wh > 0:
                p_dc = cp.Variable(T, nonneg=True, name=f"p_dc_{inv_id}")
                self._constraints.append(p_dc <= max_dc_wh * mode_dc_activity)
            else:
                p_dc = cp.Constant(np.zeros(T, dtype=float))
            if inv.parameters.min_ac_output_power_w > 0 and max_dc_wh > 0:
                min_dc_wh = inv.parameters.min_ac_output_power_w * dt
                self._constraints.append(p_dc >= min_dc_wh * mode_dc_activity)
        else:
            p_dc = cp.Constant(np.zeros(T, dtype=float))

        if mode_ch_activity is not None and mode_dc_activity is not None:
            self._constraints.append(mode_ch_activity + mode_dc_activity <= 1)

        if inv_id in self._pv_params:
            pv_pred = self._pv_params[inv_id]
            pv_ac = cp.Variable(T, nonneg=True, name=f"pv_ac_{inv_id}")
            if inv.battery is not None:
                pv_to_bat = cp.Variable(T, nonneg=True, name=f"pv_to_bat_{inv_id}")
                pv_ac_buffered = cp.Variable(T, nonneg=True, name=f"pv_ac_buf_{inv_id}")
                self._constraints.append(
                    pv_ac + pv_to_bat <= pv_pred
                )  # Curtailment allowed when AC and battery are saturated
                # Buffered PV cannot exceed total PV on AC.
                self._constraints.append(pv_ac_buffered <= pv_ac)
                if mode_dc_activity is not None:
                    # Gate PV→AC on the discharge binary ("bypass mode"):
                    # the DC→AC path is only open when the inverter is active (mode_dc=1).
                    # In IDLE (mode_dc=0) all PV flows to the battery via passive DC-bus
                    # coupling (pv_to_bat).  When the battery is full the headroom constraint
                    # forces pv_to_bat=0; the solver then activates mode_dc=1 with p_dc=0
                    # (zero-discharge bypass) so that excess PV can reach the AC bus.
                    # No new binary variable is needed.
                    self._constraints.append(pv_ac <= cp.multiply(pv_pred, mode_dc_activity))
                # No mode gate on pv_to_bat: battery absorbs PV passively in any mode;
                # the combined headroom constraint prevents overfilling.
            else:
                pv_to_bat = cp.Constant(np.zeros(T, dtype=float))
                pv_ac_buffered = cp.Constant(np.zeros(T, dtype=float))
                self._constraints.append(pv_ac <= pv_pred)
        else:
            pv_ac = cp.Constant(np.zeros(T, dtype=float))
            pv_to_bat = cp.Constant(np.zeros(T, dtype=float))
            pv_ac_buffered = cp.Constant(np.zeros(T, dtype=float))

        max_ac_out = inv.parameters.max_ac_output_power_w * dt
        self._constraints.append(pv_ac + p_dc <= max_ac_out)

        soc_init_param: cp.Parameter | None = None
        if inv.battery is not None:
            bat = inv.battery
            soc = cp.Variable(T, nonneg=True, name=f"soc_{inv_id}")
            soc_init_param = cp.Parameter(nonneg=True, name=f"soc_init_{inv_id}")
            self._soc_init_params[inv_id] = soc_init_param

            eta_c_ac = inv.parameters.ac_to_dc_efficiency * bat.charging_efficiency
            eta_c_pv = bat.charging_efficiency
            eta_d = bat.discharging_efficiency * inv.parameters.dc_to_ac_efficiency

            # Net battery flow in stored-energy units [Wh/dt].
            battery_net_flow_wh = p_ch * eta_c_ac + pv_to_bat * eta_c_pv - p_dc / eta_d

            _active_w_self = inv.parameters.active_inverter_consumption_w
            _is_active_self = None
            if _active_w_self > 0.0:
                _act_wh_self = _active_w_self * dt
                if mode_ch_activity is not None and mode_dc_activity is not None:
                    _is_active_self = mode_ch_activity + mode_dc_activity
                elif mode_ch_activity is not None:
                    _is_active_self = mode_ch_activity
                elif mode_dc_activity is not None:
                    _is_active_self = mode_dc_activity
                if _is_active_self is not None:
                    battery_net_flow_wh = battery_net_flow_wh - _act_wh_self * _is_active_self

            start_soc = (
                cp.hstack([soc_init_param, soc[:-1]]) if T > 1 else cp.hstack([soc_init_param])
            )

            if _is_active_self is not None:
                self._constraints.append(
                    p_dc / eta_d + _act_wh_self * _is_active_self <= start_soc - bat.min_soc_wh
                )
            else:
                self._constraints.append(p_dc / eta_d <= start_soc - bat.min_soc_wh)

            # RLT strengthening cuts for semi-continuous min-power constraints.
            # Derived by substituting p_dc >= min_dc * mode_dc into the SoC discharge
            # constraint above.  This creates a direct binary→SoC link that tightens
            # the LP relaxation at each B&B node, reducing tree size.
            _min_p_w = inv.parameters.min_ac_output_power_w
            if _min_p_w > 0.0 and mode_dc_activity is not None:
                _min_dc_wh_rlt = _min_p_w * dt
                if _is_active_self is not None:
                    # Strongest valid cut: combine min discharge + active consumption.
                    # mode_dc*(min_dc/eta_d + act) + act*mode_ch (if present) <= SoC headroom
                    if mode_ch_activity is not None:
                        self._constraints.append(
                            mode_dc_activity * (_min_dc_wh_rlt / eta_d + _act_wh_self)
                            + _act_wh_self * mode_ch_activity
                            <= start_soc - bat.min_soc_wh
                        )
                    else:
                        self._constraints.append(
                            mode_dc_activity * (_min_dc_wh_rlt / eta_d + _act_wh_self)
                            <= start_soc - bat.min_soc_wh
                        )
                else:
                    self._constraints.append(
                        mode_dc_activity * _min_dc_wh_rlt / eta_d <= start_soc - bat.min_soc_wh
                    )
            # RLT cut for min charge: mode_ch * min_ch * eta_c_ac <= SoC headroom
            # (Commented out: in practice this cut interacts poorly with presolve
            #  on specific prediction windows and can be slower overall.)
            # if _min_p_w > 0.0 and mode_ch_activity is not None:
            #     _min_ch_wh_rlt = _min_p_w * dt
            #     self._constraints.append(
            #         mode_ch_activity * _min_ch_wh_rlt * eta_c_ac <= bat.max_soc_wh - start_soc
            #     )

            # Combined battery charge-rate limit in raw battery input energy [Wh/dt]:
            # AC charging contributes after AC->DC conversion, PV->battery is already
            # on the DC side before battery charging losses.
            self._constraints.append(
                p_ch * inv.parameters.ac_to_dc_efficiency + pv_to_bat <= bat.max_charge_power_w * dt
            )

            # Headroom constraint: actual charging + buffered PV headroom <= available SoC capacity.
            # pv_ac_buffered is PV on AC that is backed by battery headroom (the inverter
            # COULD route it to battery instead of AC).  eta_buf converts AC-side Wh to
            # stored-energy Wh:  PV_AC → (÷ dc_to_ac) → DC → (× charging_eff) → stored.
            eta_buf = bat.charging_efficiency / inv.parameters.dc_to_ac_efficiency
            self._constraints.append(
                p_ch * eta_c_ac + pv_to_bat * eta_c_pv + pv_ac_buffered * eta_buf
                <= bat.max_soc_wh - start_soc
            )

            # Physical model: PV is passively coupled to the battery DC bus.
            # pv_to_bat is ungated — the battery absorbs PV automatically in any mode.
            # pv_ac is gated by mode_dc (bypass gate above): PV only reaches the AC bus
            # when the inverter DC→AC path is active.  The headroom constraint ensures
            # AC + PV charging never overfills the battery.

            self._constraints.append(soc == start_soc + battery_net_flow_wh)
            self._constraints.extend([soc >= bat.min_soc_wh, soc <= bat.max_soc_wh])
        else:
            soc = None

        self._add_mode_switch_costs(inv, inv_id, mode_ch_activity, mode_dc_activity)

        return _InverterModelBlock(
            inverter=inv,
            p_ch=p_ch,
            p_dc=p_dc,
            pv_ac=pv_ac,
            pv_to_bat=pv_to_bat,
            pv_ac_buffered=pv_ac_buffered,
            soc=soc,
            battery_net_flow_wh=battery_net_flow_wh,
            mode_ch_activity=mode_ch_activity,
            mode_dc_activity=mode_dc_activity,
        )

    def _add_mode_switch_costs(
        self,
        inv: InverterBase,
        inv_id: str,
        mode_ch_activity: cp.Expression | None,
        mode_dc_activity: cp.Expression | None,
    ) -> None:
        if not inv.battery:
            return

        mode_switch_cost = inv.parameters.mode_switch_cost
        if mode_switch_cost <= 0.0:
            return

        init_ch_param = cp.Parameter(nonneg=True, name=f"init_ch_{inv_id}")
        init_dc_param = cp.Parameter(nonneg=True, name=f"init_dc_{inv_id}")
        self._init_mode_ch_params[inv_id] = init_ch_param
        self._init_mode_dc_params[inv_id] = init_dc_param

        is_ch = mode_ch_activity if mode_ch_activity is not None else np.zeros(self._T, dtype=float)
        is_dc = mode_dc_activity if mode_dc_activity is not None else np.zeros(self._T, dtype=float)

        # Use separate delta variables per binary so that a simultaneous
        # AC_CHARGE→DISCHARGE transition incurs 2×switch_cost (one for each
        # binary change), identical to going through IDLE in two steps.
        # The old single delta_mode variable used max(Δch, Δdc) semantics,
        # which created a spurious incentive to "park" at 1 % charge just
        # before discharge to save one switch.
        delta_ch = cp.Variable(self._T, nonneg=True, name=f"delta_ch_{inv_id}")
        delta_dc = cp.Variable(self._T, nonneg=True, name=f"delta_dc_{inv_id}")
        self._constraints.extend(
            [
                delta_ch[0] >= is_ch[0] - init_ch_param,
                delta_ch[0] >= init_ch_param - is_ch[0],
                delta_dc[0] >= is_dc[0] - init_dc_param,
                delta_dc[0] >= init_dc_param - is_dc[0],
            ]
        )

        if self._T > 1:
            self._constraints.extend(
                [
                    delta_ch[1:] >= is_ch[1:] - is_ch[:-1],
                    delta_ch[1:] >= is_ch[:-1] - is_ch[1:],
                    delta_dc[1:] >= is_dc[1:] - is_dc[:-1],
                    delta_dc[1:] >= is_dc[:-1] - is_dc[1:],
                ]
            )

        self._mode_switch_costs.append((cp.sum(delta_ch) + cp.sum(delta_dc)) * mode_switch_cost)

    @staticmethod
    def _mode_flags(mode: InverterMode) -> tuple[int, int]:
        if mode in (InverterMode.AC_CHARGE, InverterMode.AC_CHARGE_ZERO_FEED_IN):
            return 1, 0
        if mode in (InverterMode.DISCHARGE, InverterMode.DISCHARGE_ZERO_FEED_IN):
            return 0, 1
        return 0, 0

    def _estimate_terminal_value(
        self,
        prep: _PreparedInputs,
        blocks: list[_InverterModelBlock],
    ) -> float:
        if prep.T == 0:
            return 0.0

        eta_d_values = [
            block.inverter.battery.discharging_efficiency
            * block.inverter.parameters.dc_to_ac_efficiency
            for block in blocks
            if block.inverter.battery is not None
        ]
        if not eta_d_values:
            return 0.0

        mean_eta_d = float(np.mean(eta_d_values))
        N = min(prep.T, 6)
        mean_price = float(np.mean(prep.price[-N:]))
        return mean_price * mean_eta_d

    def _terminal_reward_expr(self) -> cp.Expression | float:
        if isinstance(self._terminal_soc_sum, (int, float)):
            return 0.0
        return self._terminal_value_param * self._terminal_soc_sum

    def _objective_value_for_logging(
        self,
        objective: OptimizationObjective,
        value: float | None,
    ) -> float:
        if value is None:
            return 0.0
        if objective == OptimizationObjective.MINIMIZE_COST:
            return float(value) / self._MONETARY_OBJECTIVE_SCALE
        return float(value)

    @staticmethod
    def _sum_terms(terms: list[object], T: int) -> cp.Expression:
        if not terms:
            return cp.Constant(np.zeros(T, dtype=float))
        if len(terms) == 1:
            return cast(cp.Expression, terms[0])
        return cp.sum(cp.vstack([cast(cp.Expression, term) for term in terms]), axis=0)

    @staticmethod
    def _expr_to_vec(expr: cp.Expression, T: int) -> np.ndarray:
        val = expr.value
        if val is None:
            return np.zeros(T, dtype=float)
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return np.full(T, float(arr), dtype=float)
        return np.maximum(arr, 0.0)

    def _extract_modes(self, block: _InverterModelBlock, T: int) -> np.ndarray:
        """Map solver solution to InverterMode per timestep.

        Binary-based extraction is used as the primary signal when ``mode_switch_cost > 0``.
        In that case the solver controls the binaries deliberately:
        - binary=1, energy>0  → mode is actively running
        - binary=1, energy=0  → "parked" mode: held at 1 to avoid a future switch penalty

        When ``mode_switch_cost = 0`` the binaries are degenerate.  The semi-continuous
        constraint only provides an upper bound (p ≤ max × binary), so binary=1 with p=0
        is feasible and HiGHS may choose it as an arbitrary tie-break.  Energy-flow based
        extraction is used instead so that IDLE is reported when no energy actually flows.

        PV→AC bypass (pv_ac > 0) always implies the DC→AC path is open regardless of
        switch cost, so it is always captured via the energy-flow check.
        """
        inv = block.inverter
        eps = 1e-3
        modes = np.full(T, int(InverterMode.IDLE), dtype=np.int8)

        ch_mode = (
            InverterMode.AC_CHARGE
            if InverterMode.AC_CHARGE in inv.available_modes
            else InverterMode.AC_CHARGE_ZERO_FEED_IN
        )
        dc_mode = (
            InverterMode.DISCHARGE
            if InverterMode.DISCHARGE in inv.available_modes
            else InverterMode.DISCHARGE_ZERO_FEED_IN
        )

        if inv.parameters.mode_switch_cost > 0:
            # --- Binary-based (primary) ---
            # With switch cost the solver only sets binary=1 deliberately.
            # Parked slots (binary=1, energy=0) are a valid mode state and must be reported.
            if block.mode_ch_activity is not None:
                ch_binary = self._expr_to_vec(block.mode_ch_activity, T) > 0.5
                modes[ch_binary] = int(ch_mode)
            if block.mode_dc_activity is not None:
                dc_binary = self._expr_to_vec(block.mode_dc_activity, T) > 0.5
                # mode_ch + mode_dc <= 1 ensures mutual exclusion; dc takes precedence
                # for bypass/discharge reporting (overwriting ch where both would apply).
                modes[dc_binary] = int(dc_mode)
        else:
            # --- Energy-flow based (primary) ---
            # With switch_cost=0 the binary is a free variable when p=0 and may be set to
            # 1 by HiGHS as a degenerate tie-break. Use actual energy flow to infer mode.
            ch_flow = self._expr_to_vec(block.p_ch, T) > eps
            dc_flow = self._expr_to_vec(block.p_dc, T) > eps
            modes[ch_flow] = int(ch_mode)
            modes[dc_flow] = int(dc_mode)

        # PV→AC bypass: the bypass-gate constraint (pv_ac <= pv_pred * mode_dc) forces
        # mode_dc=1 whenever pv_ac > 0.  For switch_cost>0 this is already captured by
        # the binary above; the energy-flow check below covers switch_cost=0 explicitly.
        if inv.battery is not None:
            pv_ac_flow = self._expr_to_vec(block.pv_ac, T) > eps
            modes[pv_ac_flow] = int(dc_mode)

        return modes

    def _build_solution(
        self,
        *,
        prep: _PreparedInputs,
        blocks: list[_InverterModelBlock],
        g_import_val: np.ndarray,
        g_feedin_val: np.ndarray,
        objective: OptimizationObjective,
        solver_status: str,
        solve_time: float,
        prediction: PredictionData,
    ) -> LinearSolution:
        T = prep.T
        dt = prep.dt

        gi = np.maximum(g_import_val, 0.0)
        gf = np.maximum(g_feedin_val, 0.0)
        self_consumption = np.maximum(prep.load_wh - gi, 0.0)

        costs = gi * prep.price
        revenue = gf * prep.feedin_tariff

        losses_arr = np.zeros(T, dtype=float)
        battery_wh_per_dt: dict[str, np.ndarray] = {}
        battery_soc_pct: dict[str, np.ndarray] = {}
        battery_initial_soc_pct: dict[str, float] = {}
        inv_modes_per_dt: dict[str, np.ndarray] = {}
        inverter_plans: list[InverterPlan] = []

        for block in blocks:
            inv = block.inverter
            inv_id = inv.device_id

            p_ch = self._expr_to_vec(block.p_ch, T)
            p_dc = self._expr_to_vec(block.p_dc, T)
            pv_to_bat = self._expr_to_vec(block.pv_to_bat, T)
            battery_soc_wh: np.ndarray | None = None

            if inv.battery is not None and block.soc is not None:
                bat = inv.battery
                soc0_wh = float(self._soc_init_params[inv_id].value or bat.soc_wh)
                battery_initial_soc_pct[inv_id] = float(soc0_wh * (100.0 / bat.capacity_wh))
                soc_vals = np.maximum(np.asarray(block.soc.value, dtype=float), 0.0)
                battery_soc_wh = np.asarray(soc_vals, dtype=np.float32)
                battery_wh_per_dt[inv_id] = np.asarray(soc_vals, dtype=np.float32)
                battery_soc_pct[inv_id] = np.asarray(
                    soc_vals * (100.0 / bat.capacity_wh), dtype=np.float32
                )

                eta_c_ac = inv.parameters.ac_to_dc_efficiency * bat.charging_efficiency
                eta_c_pv = bat.charging_efficiency
                eta_d = bat.discharging_efficiency * inv.parameters.dc_to_ac_efficiency
                losses_arr += p_ch * (1.0 - eta_c_ac)
                losses_arr += pv_to_bat * (1.0 - eta_c_pv)
                losses_arr += p_dc * (1.0 / eta_d - 1.0)

            modes = self._extract_modes(block, T)
            inv_modes_per_dt[inv_id] = modes
            inverter_plans.append(
                InverterPlan(
                    device_id=inv_id,
                    modes=modes,
                    charge_ac_wh=np.asarray(p_ch, dtype=np.float32),
                    discharge_ac_wh=np.asarray(p_dc, dtype=np.float32),
                    pv_to_ac_wh=np.asarray(self._expr_to_vec(block.pv_ac, T), dtype=np.float32),
                    pv_to_battery_wh=np.asarray(pv_to_bat, dtype=np.float32),
                    battery_soc_wh=battery_soc_wh,
                )
            )

            active_w = inv.parameters.active_inverter_consumption_w
            if active_w > 0.0:
                active_wh = active_w * dt
                is_active = (modes != int(InverterMode.IDLE)).astype(float)
                losses_arr += active_wh * is_active

        result = SimulationResult(
            costs_per_dt=np.asarray(costs, dtype=np.float32),
            revenue_per_dt=np.asarray(revenue, dtype=np.float32),
            grid_import_wh_per_dt=np.asarray(gi, dtype=np.float32),
            self_consumption_wh_per_dt=np.asarray(self_consumption, dtype=np.float32),
            feedin_wh_per_dt=np.asarray(gf, dtype=np.float32),
            losses_wh_per_dt=np.asarray(losses_arr, dtype=np.float32),
            inverter_modes_per_dt=inv_modes_per_dt,
            battery_wh_per_dt=battery_wh_per_dt,
            battery_soc_percentage_per_dt=battery_soc_pct,
            battery_initial_soc_percentage=battery_initial_soc_pct,
        )

        return LinearSolution(
            prediction=prediction,
            inverter_plans=tuple(inverter_plans),
            objective=objective,
            solver_status=solver_status,
            solve_time_s=solve_time,
            result=result,
        )

    def _validate_with_simulation(
        self,
        solution: LinearSolution,
        prep: _PreparedInputs,
        prediction: PredictionData,
    ) -> tuple[SimulationParityReport, SimulationResult | None]:
        sim = GridSimulation(prediction=prediction, inverters=self.inverters, home_appliances=None)

        modes: dict[str, np.ndarray] = {}
        rates: dict[str, np.ndarray] = {}
        energy_wh: dict[str, np.ndarray] = {}

        for inv in self.inverters:
            inv_id = inv.device_id
            plan = next((p for p in solution.inverter_plans if p.device_id == inv_id), None)
            if plan is None:
                modes[inv_id] = np.full(prep.T, int(InverterMode.IDLE), dtype=np.int32)
                rates[inv_id] = np.zeros(prep.T, dtype=np.int32)
                energy_wh[inv_id] = np.zeros(prep.T, dtype=np.float32)
            else:
                modes[inv_id] = np.asarray(plan.modes, dtype=np.int32)
                rates[inv_id] = np.zeros(prep.T, dtype=np.int32)  # rates no longer in plan
                energy_wh[inv_id] = np.asarray(
                    np.maximum(plan.charge_ac_wh, plan.discharge_ac_wh),
                    dtype=np.float32,
                )

        sim_result = sim.simulate(
            inverter_modes=modes,
            inverter_ac_rates=rates,
            inverter_ac_energy_wh=energy_wh,
            appliance_load=None,
            start_idx=0,
            dt=prep.dt,
        )
        if sim_result is None:
            return (
                SimulationParityReport(
                    ok=False,
                    max_abs_soc_error_wh=float("inf"),
                    max_abs_grid_import_error_wh=float("inf"),
                    max_abs_feedin_error_wh=float("inf"),
                    max_abs_cost_error_eur=float("inf"),
                ),
                None,
            )

        lp = solution.result
        gi_err = float(
            np.max(
                np.abs(
                    np.asarray(lp.grid_import_wh_per_dt)
                    - np.asarray(sim_result.grid_import_wh_per_dt)
                )
            )
        )
        gf_err = float(
            np.max(
                np.abs(np.asarray(lp.feedin_wh_per_dt) - np.asarray(sim_result.feedin_wh_per_dt))
            )
        )
        cost_err = float(
            np.max(np.abs(np.asarray(lp.costs_per_dt) - np.asarray(sim_result.costs_per_dt)))
        )

        soc_err = 0.0
        shared_ids = set(lp.battery_wh_per_dt).intersection(sim_result.battery_wh_per_dt)
        for inv_id in shared_ids:
            cur = float(
                np.max(
                    np.abs(
                        np.asarray(lp.battery_wh_per_dt[inv_id])
                        - np.asarray(sim_result.battery_wh_per_dt[inv_id])
                    )
                )
            )
            soc_err = max(soc_err, cur)

        report = SimulationParityReport(
            ok=(soc_err <= 1e-2 and gi_err <= 1e-2 and gf_err <= 1e-2 and cost_err <= 1e-4),
            max_abs_soc_error_wh=soc_err,
            max_abs_grid_import_error_wh=gi_err,
            max_abs_feedin_error_wh=gf_err,
            max_abs_cost_error_eur=cost_err,
        )
        return report, sim_result
