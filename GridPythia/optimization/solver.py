"""Linear (MILP) energy-management optimizer using CVXPY + HiGHS.

This module builds a topology-aware mathematical model from the same
core signals that drive GridSimulation (load, PV, battery, prices) and
solves it with HiGHS. Home appliances are intentionally excluded from
this LP model for now.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from platform import machine
from typing import cast

if machine() in ("armv7l", "armv6l"):
    import os

    os.environ["LD_PRELOAD"] = "usr/lib/arm-linux-gnueabihf/libatomic.so.1"

import cvxpy as cp
import numpy as np
from structlog import get_logger

from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.inverterbase import InverterBase
from GridPythia.simulation.grid_interpolator import FraunhoferSCModel
from GridPythia.simulation.grid_simulation import GridSimulation, SimulationResult

logger = get_logger(__name__)


class OptimizationObjective(str, Enum):
    """Supported objective functions for :class:`LinearOptimizer`."""

    MINIMIZE_COST = "cost"
    MAXIMIZE_SELF_CONSUMPTION = "self_consumption"


@dataclass
class SimulationParityReport:
    """Difference report between LP result and GridSimulation replay."""

    ok: bool
    max_abs_soc_error_wh: float
    max_abs_grid_import_error_wh: float
    max_abs_feedin_error_wh: float
    max_abs_cost_error_eur: float


@dataclass(frozen=True, slots=True)
class InverterPlan:
    """Typed inverter schedule exported by :class:`LinearOptimizer`."""

    device_id: str
    modes: np.ndarray
    rates: np.ndarray
    charge_ac_wh: np.ndarray
    discharge_ac_wh: np.ndarray
    pv_to_ac_wh: np.ndarray
    pv_to_battery_wh: np.ndarray
    battery_soc_wh: np.ndarray | None = None

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def get(self, key: str, default: object | None = None) -> object | None:
        return getattr(self, key, default)


@dataclass
class LinearSolution:
    """Solution produced by :class:`LinearOptimizer`."""

    result: SimulationResult
    objective: OptimizationObjective
    solver_status: str
    solve_time_s: float
    inverter_plans: list[InverterPlan]
    parity_report: SimulationParityReport | None = None
    simulation_result: SimulationResult | None = None
    prediction: dict | None = None


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
    p_ch: cp.Expression
    p_dc: cp.Expression
    pv_ac: cp.Expression
    pv_to_bat: cp.Expression
    soc: cp.Variable | None
    mode_ch_activity: cp.Expression | None
    mode_dc_activity: cp.Expression | None


class LinearOptimizer:
    """MILP optimizer with a compiled reusable CVXPY model.

    The problem structure (variables + constraints) is compiled once per optimizer
    instance and subsequent solves only update runtime Parameters (prediction arrays,
    battery start SoC, and initial inverter mode).
    """

    _MIN_ACTIVE_AC_RATE_PCT = 1.0

    def __init__(
        self,
        inverters: list[InverterBase],
        prediction: PredictionData,
    ) -> None:
        self.inverters = inverters
        self.prediction = prediction
        self._log = logger.bind(
            component="optimizer",
            inverter_ids=[inv.device_id for inv in inverters],
            steps=prediction.steps,
        )

        min_load_wh = float(min(prediction.load_wh)) if prediction.steps > 0 else 1.0
        self._sc_model = FraunhoferSCModel(
            baseload_wh=max(min_load_wh, 1e-6),
            dt=prediction.dt_hours,
        )

        self._compiled = False
        self._compile_problem(self._prepare_inputs())

    def solve(
        self,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST,
        solver_opts: dict | None = None,
        validate_with_simulation: bool = False,
        initial_modes: Mapping[str, InverterMode | int] | None = None,
    ) -> LinearSolution:
        """Solve the compiled MILP with updated runtime data."""
        prep = self._prepare_inputs()
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

        self._update_runtime_parameters(prep, normalized_initial_modes)

        problem = self._problems[objective]
        size = problem.size_metrics
        self._log.info(
            "optimizer_solve_start",
            objective=objective.value,
            num_variables=size.num_scalar_variables,
            num_constraints=size.num_scalar_eq_constr + size.num_scalar_leq_constr,
        )

        opts = {
            "verbose": False,
            "warm_start": True,
            "time_limit": 30,
            "mip_rel_gap": 0.02,
            **(solver_opts or {}),
        }

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
            objective=objective.value,
            objective_value=round(float(problem.value), 4) if problem.value is not None else None,
        )

        solution = self._build_solution(
            prep=prep,
            blocks=self._blocks,
            g_import_val=np.asarray(self._g_import.value, dtype=float),
            g_feedin_val=np.asarray(self._g_feedin.value, dtype=float),
            objective=objective,
            solver_status=status,
            solve_time=solve_time,
            prediction=self.prediction.to_dict(),
        )

        if validate_with_simulation:
            parity, sim_res = self._validate_with_simulation(solution, prep)
            solution.parity_report = parity
            solution.simulation_result = sim_res
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

    def _prepare_inputs(self) -> _PreparedInputs:
        solver_view = self.prediction.to_solver_view(dtype=np.float64)
        return _PreparedInputs(
            T=solver_view.steps,
            dt=solver_view.dt_hours,
            load_wh=solver_view.load_wh,
            price=solver_view.electricprice_eur_wh,
            feedin_tariff=solver_view.feedintariff_eur_wh,
            pv_by_source=solver_view.pv_by_inverter,
        )

    def _compile_problem(self, prep: _PreparedInputs) -> None:
        self._T = prep.T
        self._dt = prep.dt

        self._constraints: list[cp.Constraint] = []
        self._mode_switch_costs: list[cp.Expression] = []

        self._load_param = cp.Parameter(self._T, name="load_wh")
        self._price_param = cp.Parameter(self._T, name="price")
        self._feedin_param = cp.Parameter(self._T, name="feedin_tariff")
        self._c_pv_param = cp.Parameter(self._T, name="c_pv")
        self._rhs_sc_param = cp.Parameter(self._T, name="rhs_sc")
        self._terminal_value_param = cp.Parameter(nonneg=True, name="terminal_value")

        self._pv_params: dict[str, cp.Parameter] = {
            inv.device_id: cp.Parameter(self._T, nonneg=True, name=f"pv_pred_{inv.device_id}")
            for inv in self.inverters
        }
        self._pv_external_param = cp.Parameter(self._T, nonneg=True, name="pv_pred_external")

        self._soc_init_params: dict[str, cp.Parameter] = {}
        self._init_mode_ch_params: dict[str, cp.Parameter] = {}
        self._init_mode_dc_params: dict[str, cp.Parameter] = {}

        self._g_import = cp.Variable(self._T, nonneg=True, name="g_import")
        self._g_feedin = cp.Variable(self._T, nonneg=True, name="g_feedin")
        self._pv_self = cp.Variable(self._T, nonneg=True, name="pv_self")

        total_p_ch_terms: list[object] = []
        total_p_dc_terms: list[object] = []
        total_pv_ac_terms: list[object] = [self._pv_external_param]
        total_pv_ac_decision_terms: list[object] = []
        total_inv_consumption_terms: list[object] = []

        blocks = [self._build_inverter_block(inv) for inv in self.inverters]
        for inv, block in zip(self.inverters, blocks, strict=False):
            total_p_ch_terms.append(cast(cp.Expression, block.p_ch))
            total_p_dc_terms.append(cast(cp.Expression, block.p_dc))
            total_pv_ac_terms.append(cast(cp.Expression, block.pv_ac))
            total_pv_ac_decision_terms.append(cast(cp.Expression, block.pv_ac))

            active_w = inv.parameters.active_inverter_consumption_w
            if active_w > 0.0:
                active_wh = active_w * self._dt
                if block.mode_ch_activity is not None and block.mode_dc_activity is not None:
                    is_active = block.mode_ch_activity + block.mode_dc_activity
                elif block.mode_ch_activity is not None:
                    is_active = block.mode_ch_activity
                elif block.mode_dc_activity is not None:
                    is_active = block.mode_dc_activity
                else:
                    is_active = None
                if is_active is not None:
                    total_inv_consumption_terms.append(cp.multiply(active_wh, is_active))

        total_p_ch = self._sum_terms(total_p_ch_terms, self._T)
        total_p_dc = self._sum_terms(total_p_dc_terms, self._T)
        total_pv_ac = self._sum_terms(total_pv_ac_terms, self._T)
        total_pv_ac_decision = self._sum_terms(total_pv_ac_decision_terms, self._T)
        total_inv_consumption = self._sum_terms(total_inv_consumption_terms, self._T)

        self._constraints.extend(
            [
                self._pv_self - cp.multiply(self._c_pv_param, total_pv_ac_decision)
                <= self._rhs_sc_param,
                self._pv_self <= total_pv_ac,
                self._pv_self <= self._load_param,
            ]
        )

        self._constraints.append(
            self._g_import - self._g_feedin
            == self._load_param
            + total_p_ch
            + total_inv_consumption
            - total_p_dc
            + total_pv_ac
            - self._pv_self
        )

        terminal_terms = [block.soc[-1] for block in blocks if block.soc is not None]
        self._terminal_soc_sum: cp.Expression | float
        if terminal_terms:
            self._terminal_soc_sum = cp.sum(cp.hstack(terminal_terms))
        else:
            self._terminal_soc_sum = 0.0

        mode_switch_costs_term = cp.sum(self._mode_switch_costs) if self._mode_switch_costs else 0.0
        terminal_reward = self._terminal_reward_expr()

        objective_cost = (
            cp.sum(
                cp.multiply(self._g_import, self._price_param)
                - cp.multiply(self._g_feedin, self._feedin_param)
            )
            + mode_switch_costs_term
            - terminal_reward
        )

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

        idle_modes = {inv.device_id: InverterMode.IDLE for inv in self.inverters}
        self._update_runtime_parameters(prep, idle_modes)
        self._compiled = True

    def _ensure_compiled_layout(self, prep: _PreparedInputs) -> None:
        if not self._compiled or prep.T != self._T or prep.dt != self._dt:
            self._sc_model = FraunhoferSCModel(
                baseload_wh=max(float(np.min(prep.load_wh)) if prep.T > 0 else 1.0, 1e-6),
                dt=prep.dt,
            )
            self._compile_problem(prep)

    def _update_runtime_parameters(
        self,
        prep: _PreparedInputs,
        initial_modes: Mapping[str, InverterMode],
    ) -> None:
        self._load_param.value = prep.load_wh
        self._price_param.value = prep.price
        self._feedin_param.value = prep.feedin_tariff

        c_pv_arr = np.empty(prep.T, dtype=float)
        rhs_arr = np.empty(prep.T, dtype=float)
        for t in range(prep.T):
            load0 = max(float(prep.load_wh[t]), 1e-6)
            lc = self._sc_model.linearize(pv_0=load0, load_0=load0)
            c_pv_arr[t] = lc.c_pv
            rhs_arr[t] = lc.rhs_fixed_load(load=load0)
        self._c_pv_param.value = c_pv_arr

        for inv in self.inverters:
            inv_id = inv.device_id
            pv_arr = prep.pv_by_source.get(inv_id, np.zeros(prep.T, dtype=float))
            self._pv_params[inv_id].value = pv_arr

            if inv_id in self._soc_init_params:
                if inv.battery is None:
                    self._soc_init_params[inv_id].value = 0.0
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
        self._rhs_sc_param.value = rhs_arr + c_pv_arr * external

        terminal_value = self._estimate_terminal_value(prep, self._blocks)
        self._terminal_value_param.value = terminal_value

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

        if inv.battery is not None and inv.is_optimizable and has_charge_mode:
            max_ch_wh = float(inv.battery.max_charge_power_w * dt)
            if max_ch_wh > 0:
                mode_ch_activity = cp.Variable(T, boolean=True, name=f"mode_ch_{inv_id}")
                p_ch = cp.Variable(T, nonneg=True, name=f"p_ch_{inv_id}")
                min_ch_wh = max_ch_wh * (self._MIN_ACTIVE_AC_RATE_PCT / 100.0)
                self._constraints.append(p_ch >= min_ch_wh * mode_ch_activity)
                self._constraints.append(p_ch <= max_ch_wh * mode_ch_activity)
            else:
                p_ch = cp.Constant(np.zeros(T, dtype=float))
        else:
            p_ch = cp.Constant(np.zeros(T, dtype=float))

        if inv.battery is not None and inv.is_optimizable and has_discharge_mode:
            max_dc_wh = float(inv.battery.max_discharge_power_w * dt)
            if max_dc_wh > 0:
                mode_dc_activity = cp.Variable(T, boolean=True, name=f"mode_dc_{inv_id}")
                p_dc = cp.Variable(T, nonneg=True, name=f"p_dc_{inv_id}")
                min_dc_wh = max_dc_wh * (self._MIN_ACTIVE_AC_RATE_PCT / 100.0)
                self._constraints.append(p_dc >= min_dc_wh * mode_dc_activity)
                self._constraints.append(p_dc <= max_dc_wh * mode_dc_activity)
            else:
                p_dc = cp.Constant(np.zeros(T, dtype=float))
        else:
            p_dc = cp.Constant(np.zeros(T, dtype=float))

        if mode_ch_activity is not None and mode_dc_activity is not None:
            self._constraints.append(mode_ch_activity + mode_dc_activity <= 1)

        if inv_id in self._pv_params:
            pv_pred = self._pv_params[inv_id]
            pv_ac = cp.Variable(T, nonneg=True, name=f"pv_ac_{inv_id}")
            if inv.battery is not None:
                pv_to_bat = cp.Variable(T, nonneg=True, name=f"pv_to_bat_{inv_id}")
                self._constraints.append(pv_ac + pv_to_bat <= pv_pred)
                if mode_dc_activity is not None:
                    self._constraints.append(
                        pv_to_bat <= cp.multiply(pv_pred, 1 - mode_dc_activity)
                    )
            else:
                pv_to_bat = cp.Constant(np.zeros(T, dtype=float))
                self._constraints.append(pv_ac <= pv_pred)
        else:
            pv_ac = cp.Constant(np.zeros(T, dtype=float))
            pv_to_bat = cp.Constant(np.zeros(T, dtype=float))

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

            delta = p_ch * eta_c_ac + pv_to_bat * eta_c_pv - p_dc / eta_d
            start_soc = (
                cp.hstack([soc_init_param, soc[:-1]]) if T > 1 else cp.hstack([soc_init_param])
            )

            self._constraints.append(p_dc / eta_d <= start_soc - bat.min_soc_wh)
            self._constraints.append(p_ch * eta_c_ac <= bat.max_soc_wh - start_soc)
            self._constraints.append(pv_to_bat * eta_c_pv <= bat.max_soc_wh - start_soc)

            self._constraints.append(soc == start_soc + delta)
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
            soc=soc,
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

    @staticmethod
    def _sanitize_rate_percent(rate_pct: float) -> int:
        return int(max(0, min(100, round(float(rate_pct)))))

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
        prediction: dict | None = None,
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
        inv_modes_per_dt: dict[str, np.ndarray] = {}
        inv_rates_per_dt: dict[str, np.ndarray] = {}
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

            modes, rates = self._extract_modes_and_rates(block, p_ch, p_dc, dt)
            inv_modes_per_dt[inv_id] = np.asarray(modes, dtype=np.int8)
            inv_rates_per_dt[inv_id] = np.asarray(rates, dtype=np.int32)
            inverter_plans.append(
                InverterPlan(
                    device_id=inv_id,
                    modes=np.asarray(modes, dtype=np.int8),
                    rates=np.asarray(rates, dtype=np.int32),
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
                is_active = np.array([m != int(InverterMode.IDLE) for m in modes], dtype=float)
                losses_arr += active_wh * is_active

        result = SimulationResult(
            costs_per_dt=np.asarray(costs, dtype=np.float32),
            revenue_per_dt=np.asarray(revenue, dtype=np.float32),
            grid_import_wh_per_dt=np.asarray(gi, dtype=np.float32),
            self_consumption_wh_per_dt=np.asarray(self_consumption, dtype=np.float32),
            feedin_wh_per_dt=np.asarray(gf, dtype=np.float32),
            losses_wh_per_dt=np.asarray(losses_arr, dtype=np.float32),
            inverter_modes_per_dt=inv_modes_per_dt,
            inverter_ac_rate_per_dt=inv_rates_per_dt,
            battery_wh_per_dt=battery_wh_per_dt,
            battery_soc_percentage_per_dt=battery_soc_pct,
        )

        return LinearSolution(
            result=result,
            objective=objective,
            solver_status=solver_status,
            solve_time_s=solve_time,
            inverter_plans=inverter_plans,
            prediction=prediction,
        )

    def _extract_modes_and_rates(
        self,
        block: _InverterModelBlock,
        p_ch: np.ndarray,
        p_dc: np.ndarray,
        dt: float,
    ) -> tuple[list[int], list[int]]:
        inv = block.inverter
        T = len(p_ch)
        eps = 1e-4
        modes: list[int] = []
        rates: list[int] = []

        bat = inv.battery
        max_ch_wh = bat.max_charge_power_w * dt if bat is not None else 0.0
        max_dc_wh = bat.max_discharge_power_w * dt if bat is not None else 0.0
        mode_switch_cost_active = inv.parameters.mode_switch_cost > 0.0
        mode_ch_active: np.ndarray | None = None
        mode_dc_active: np.ndarray | None = None
        if mode_switch_cost_active and block.mode_ch_activity is not None:
            mode_ch_active = self._expr_to_vec(block.mode_ch_activity, T)
        if mode_switch_cost_active and block.mode_dc_activity is not None:
            mode_dc_active = self._expr_to_vec(block.mode_dc_activity, T)

        for t in range(T):
            ch = float(p_ch[t])
            dc = float(p_dc[t])

            if ch > eps:
                mode = (
                    int(InverterMode.AC_CHARGE)
                    if InverterMode.AC_CHARGE in inv.available_modes
                    else int(InverterMode.AC_CHARGE_ZERO_FEED_IN)
                )
                rate = int(round((min(ch / max_ch_wh, 1.0) if max_ch_wh > eps else 1.0) * 100.0))
                rate = self._sanitize_rate_percent(rate)
            elif dc > eps:
                if InverterMode.DISCHARGE in inv.available_modes:
                    mode = int(InverterMode.DISCHARGE)
                else:
                    mode = (
                        int(InverterMode.DISCHARGE_ZERO_FEED_IN)
                        if inv.parameters.zero_feed_in
                        else int(InverterMode.DISCHARGE)
                    )
                if mode == int(InverterMode.DISCHARGE_ZERO_FEED_IN):
                    rate = 0
                else:
                    rate = int(
                        round((min(dc / max_dc_wh, 1.0) if max_dc_wh > eps else 1.0) * 100.0)
                    )
                    rate = self._sanitize_rate_percent(rate)
            elif mode_switch_cost_active and mode_ch_active is not None and mode_ch_active[t] > 0.5:
                mode = (
                    int(InverterMode.AC_CHARGE)
                    if InverterMode.AC_CHARGE in inv.available_modes
                    else int(InverterMode.AC_CHARGE_ZERO_FEED_IN)
                )
                rate = 0
            elif mode_switch_cost_active and mode_dc_active is not None and mode_dc_active[t] > 0.5:
                mode = (
                    int(InverterMode.DISCHARGE_ZERO_FEED_IN)
                    if inv.parameters.zero_feed_in
                    else int(InverterMode.DISCHARGE)
                )
                rate = 0
            else:
                mode = int(InverterMode.IDLE)
                rate = 0

            modes.append(mode)
            rates.append(rate)

        return modes, rates

    def _validate_with_simulation(
        self,
        solution: LinearSolution,
        prep: _PreparedInputs,
    ) -> tuple[SimulationParityReport, SimulationResult | None]:
        sim = GridSimulation(
            prediction=self.prediction, inverters=self.inverters, home_appliances=None
        )

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
                rates[inv_id] = np.asarray(plan.rates, dtype=np.int32)
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
