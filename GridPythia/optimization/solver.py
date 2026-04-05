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


@dataclass
class LinearSolution:
    """Solution produced by :class:`LinearOptimizer`."""

    result: SimulationResult
    objective: OptimizationObjective
    solver_status: str
    solve_time_s: float
    inverter_plans: list[dict]
    parity_report: SimulationParityReport | None = None
    simulation_result: SimulationResult | None = None
    prediction: dict | None = None


@dataclass
class _RateSelector:
    """Per-inverter binary selectors for discrete charge/discharge rate states."""

    charge_rates: tuple[int, ...]
    discharge_rates: tuple[int, ...]
    y_ch: cp.Variable | None
    y_dc: cp.Variable | None


@dataclass
class _PreparedInputs:
    """Numerical input series for one optimization horizon."""

    T: int
    dt: float
    load_wh: np.ndarray
    price: np.ndarray
    feedin_tariff: np.ndarray
    pv_by_source: dict[str, np.ndarray]


@dataclass
class _InverterModelBlock:
    """Decision and helper expressions for one inverter block."""

    inverter: InverterBase
    p_ch: cp.Expression
    p_dc: cp.Expression
    pv_ac: cp.Expression
    pv_to_bat: cp.Expression
    soc: cp.Variable | None
    selector: _RateSelector | None
    zero_feed_discharge_continuous: bool = False


class LinearOptimizer:
    """MILP optimizer that builds a modular math model from system topology.

    Args:
        inverters: All inverters in the system. Topology determines which
            variables and constraints are instantiated per inverter.
        prediction: Aligned prediction channels for the horizon.
    """

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

    def solve(
        self,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST,
        solver_opts: dict | None = None,
        validate_with_simulation: bool = False,
        initial_modes: Mapping[str, InverterMode | int] | None = None,
    ) -> LinearSolution:
        """Build MILP, solve with HiGHS, and return a :class:`LinearSolution`."""
        prep = self._prepare_inputs()
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

        blocks, g_import, g_feedin, pv_self = self._build_problem(prep, normalized_initial_modes)

        terminal_value = self._estimate_terminal_value(prep, blocks)
        terminal_reward = self._terminal_reward(blocks, terminal_value)
        mode_switch_costs_term = cp.sum(self._mode_switch_costs) if self._mode_switch_costs else 0.0

        if objective == OptimizationObjective.MINIMIZE_COST:
            obj_expr = (
                cp.sum(
                    cp.multiply(g_import, prep.price) - cp.multiply(g_feedin, prep.feedin_tariff)
                )
                + mode_switch_costs_term
                - terminal_reward
            )
        else:
            obj_expr = cp.sum(g_feedin) - terminal_reward

        problem = cp.Problem(cp.Minimize(obj_expr), self._constraints)
        size = problem.size_metrics
        self._log.info(
            "optimizer_solve_start",
            objective=objective.value,
            num_variables=size.num_scalar_variables,
            num_constraints=size.num_scalar_eq_constr + size.num_scalar_leq_constr,
        )

        opts = {
            "verbose": False,
            "time_limit": 30,
            "mip_rel_gap": 0.02,
            **(solver_opts or {}),
        }

        t0 = time.perf_counter()
        try:
            problem.solve(solver=cp.HIGHS, **opts)
        except cp.SolverError as exc:
            raise RuntimeError(f"CVXPY/HiGHS solver error: {exc}") from exc
        solve_time = time.perf_counter() - t0

        status = problem.status
        accepted_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if status == "user_limit" and g_import.value is not None and g_feedin.value is not None:
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

        if g_import.value is None or g_feedin.value is None:
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
            blocks=blocks,
            g_import_val=np.asarray(g_import.value, dtype=float),
            g_feedin_val=np.asarray(g_feedin.value, dtype=float),
            pv_self_val=np.asarray(pv_self.value, dtype=float),
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
        pred = self.prediction
        T = pred.steps
        dt = pred.dt_hours

        price_series = pred.electricprice
        if price_series is None:
            price = np.zeros(T, dtype=float)
        else:
            price = np.asarray(price_series, dtype=float)

        tariff_series = pred.feedintariff
        if tariff_series is None:
            feedin_tariff = np.zeros(T, dtype=float)
        else:
            feedin_tariff = np.asarray(tariff_series, dtype=float)

        load_wh = np.asarray(pred.load_wh, dtype=float)
        pv_by_source = {k: np.asarray(v, dtype=float) for k, v in pred.pv_by_inverter.items()}

        return _PreparedInputs(
            T=T,
            dt=dt,
            load_wh=load_wh,
            price=price,
            feedin_tariff=feedin_tariff,
            pv_by_source=pv_by_source,
        )

    def _build_problem(
        self,
        prep: _PreparedInputs,
        initial_modes: Mapping[str, InverterMode],
    ) -> tuple[list[_InverterModelBlock], cp.Variable, cp.Variable, cp.Variable]:
        T = prep.T

        self._constraints: list[cp.Constraint] = []
        self._mode_switch_costs: list[cp.Expression] = []  # Track mode switch costs
        blocks: list[_InverterModelBlock] = []

        g_import = cp.Variable(T, nonneg=True, name="g_import")
        g_feedin = cp.Variable(T, nonneg=True, name="g_feedin")
        pv_self = cp.Variable(T, nonneg=True, name="pv_self")

        total_p_ch_terms: list[cp.Expression] = []
        total_p_dc_terms: list[cp.Expression] = []
        total_pv_ac_terms: list[cp.Expression] = []

        mapped_pv_sources: set[str] = set()

        for inv in self.inverters:
            block = self._build_inverter_block(inv, prep, initial_modes)
            blocks.append(block)

            total_p_ch_terms.append(block.p_ch)
            total_p_dc_terms.append(block.p_dc)
            total_pv_ac_terms.append(block.pv_ac)

            # Map PV by inverter id (PV planes reference inverter_id)
            if inv.device_id in prep.pv_by_source:
                mapped_pv_sources.add(inv.device_id)

        for inv_id, arr in prep.pv_by_source.items():
            if inv_id not in mapped_pv_sources:
                total_pv_ac_terms.append(cp.Constant(arr))

        total_p_ch = self._sum_terms(total_p_ch_terms, prep.T)
        total_p_dc = self._sum_terms(total_p_dc_terms, prep.T)
        total_pv_ac = self._sum_terms(total_pv_ac_terms, prep.T)

        # Precompute per-step linearisation coefficients.
        # total_pv_ac.value is None before solve, so pv_0 always uses the load operating point.
        c_pv_arr = np.empty(T, dtype=float)
        rhs_arr = np.empty(T, dtype=float)
        for t in range(T):
            load0 = max(float(prep.load_wh[t]), 1e-6)
            lc = self._sc_model.linearize(pv_0=load0, load_0=load0)
            c_pv_arr[t] = lc.c_pv
            rhs_arr[t] = lc.rhs_fixed_load(load=load0)

        # 3 vectorized array constraints instead of 3*T individual scalar constraints.
        self._constraints.extend(
            [
                pv_self - cp.multiply(c_pv_arr, total_pv_ac) <= rhs_arr,
                pv_self <= total_pv_ac,
                pv_self <= prep.load_wh,
            ]
        )

        # Mirrors current simulation correction logic:
        # corrected_end_load = load + p_ch - p_dc + pv_ac - pv_self
        self._constraints.append(
            g_import - g_feedin == prep.load_wh + total_p_ch - total_p_dc + total_pv_ac - pv_self
        )

        return blocks, g_import, g_feedin, pv_self

    def _build_inverter_block(
        self,
        inv: InverterBase,
        prep: _PreparedInputs,
        initial_modes: Mapping[str, InverterMode],
    ) -> _InverterModelBlock:
        T = prep.T
        dt = prep.dt
        inv_id = inv.device_id

        has_zero_feed_discharge = InverterMode.DISCHARGE_ZERO_FEED_IN in inv.available_modes
        has_rate_discharge = InverterMode.DISCHARGE in inv.available_modes

        selector: _RateSelector | None = None
        zero_feed_discharge_continuous = False

        if inv.battery is not None and inv.is_optimizable:
            charge_rates = self._get_charge_rates(inv)
            discharge_rates = self._get_discharge_rates(inv)

            y_ch = (
                cp.Variable((len(charge_rates), T), boolean=True, name=f"y_ch_{inv_id}")
                if charge_rates
                else None
            )
            y_dc = (
                cp.Variable((len(discharge_rates), T), boolean=True, name=f"y_dc_{inv_id}")
                if discharge_rates
                else None
            )
            max_ch_wh = inv.battery.max_charge_power_w * dt
            max_dc_wh = inv.battery.max_discharge_power_w * dt

            if y_ch is not None:
                p_ch = max_ch_wh * ((np.array(charge_rates, dtype=float) / 100.0) @ y_ch)
            else:
                p_ch = cp.Constant(np.zeros(T, dtype=float))

            if y_dc is not None:
                p_dc = max_dc_wh * ((np.array(discharge_rates, dtype=float) / 100.0) @ y_dc)
            elif has_zero_feed_discharge and not has_rate_discharge:
                # Zero-feed discharge is energy-target driven in simulation,
                # so model discharge continuously (bounded only by physics).
                p_dc = cp.Variable(T, nonneg=True, name=f"p_dc_{inv_id}")
                self._constraints.append(p_dc <= max_dc_wh)
                zero_feed_discharge_continuous = True
            else:
                p_dc = cp.Constant(np.zeros(T, dtype=float))

            selector = _RateSelector(
                charge_rates=charge_rates,
                discharge_rates=discharge_rates,
                y_ch=y_ch,
                y_dc=y_dc,
            )

            if y_ch is not None and y_dc is not None:
                self._constraints.extend(
                    [
                        cp.sum(y_ch, axis=0) <= 1,
                        cp.sum(y_dc, axis=0) <= 1,
                        cp.sum(y_ch, axis=0) + cp.sum(y_dc, axis=0) <= 1,
                    ]
                )
            elif y_ch is not None:
                self._constraints.append(cp.sum(y_ch, axis=0) <= 1)
                if zero_feed_discharge_continuous:
                    # Prevent simultaneous AC-charge and discharge.
                    self._constraints.append(p_dc <= max_dc_wh * (1 - cp.sum(y_ch, axis=0)))
            elif y_dc is not None:
                self._constraints.append(cp.sum(y_dc, axis=0) <= 1)

            # Add mode switch cost constraints
            # Pass p_ch and p_dc so we can track continuous discharge too
            self._add_mode_switch_costs(
                inv,
                inv_id,
                initial_modes.get(inv_id, InverterMode.IDLE),
                y_ch,
                y_dc,
                p_ch,
                p_dc,
                T,
            )

        else:
            p_ch = cp.Constant(np.zeros(T, dtype=float))
            p_dc = cp.Constant(np.zeros(T, dtype=float))

        if inv_id in prep.pv_by_source:
            pv_pred = prep.pv_by_source.get(inv_id, np.zeros(T, dtype=float))
            pv_ac = cp.Variable(T, nonneg=True, name=f"pv_ac_{inv_id}")
            if inv.battery is not None:
                pv_to_bat = cp.Variable(T, nonneg=True, name=f"pv_to_bat_{inv_id}")
                self._constraints.append(pv_ac + pv_to_bat <= pv_pred)
            else:
                pv_to_bat = cp.Constant(np.zeros(T, dtype=float))
                self._constraints.append(pv_ac <= pv_pred)
        else:
            pv_ac = cp.Constant(np.zeros(T, dtype=float))
            pv_to_bat = cp.Constant(np.zeros(T, dtype=float))

        max_ac_out = inv.parameters.max_ac_output_power_w * dt
        self._constraints.append(pv_ac + p_dc <= max_ac_out)

        if inv.battery is not None:
            bat = inv.battery
            if bat is None:
                raise RuntimeError(f"Battery block build failed for inverter '{inv_id}'")

            soc = cp.Variable(T, nonneg=True, name=f"soc_{inv_id}")
            eta_c_ac = inv.parameters.ac_to_dc_efficiency * bat.charging_efficiency
            eta_c_pv = bat.charging_efficiency
            eta_d = bat.discharging_efficiency * inv.parameters.dc_to_ac_efficiency

            delta = p_ch * eta_c_ac + pv_to_bat * eta_c_pv - p_dc / eta_d
            soc_init = float(bat.soc_wh)

            start_soc = cp.hstack([soc_init, soc[:-1]]) if T > 1 else cp.hstack([soc_init])

            # Enforce mode feasibility from the SoC at the start of each step.
            # This prevents activating discharge at min SoC or AC charge at max SoC,
            # even if opposite-direction flows in the same step would keep end SoC feasible.
            self._constraints.append(p_dc / eta_d <= start_soc - bat.min_soc_wh)
            self._constraints.append(p_ch * eta_c_ac <= bat.max_soc_wh - start_soc)

            # start_soc is already shaped (T,); one vectorized equality covers t=0..T-1.
            self._constraints.append(soc == start_soc + delta)
            self._constraints.extend([soc >= bat.min_soc_wh, soc <= bat.max_soc_wh])
        else:
            soc = None

        return _InverterModelBlock(
            inverter=inv,
            p_ch=p_ch,
            p_dc=p_dc,
            pv_ac=pv_ac,
            pv_to_bat=pv_to_bat,
            soc=soc,
            selector=selector,
            zero_feed_discharge_continuous=zero_feed_discharge_continuous,
        )

    def _add_mode_switch_costs(
        self,
        inv: InverterBase,
        inv_id: str,
        initial_mode: InverterMode,
        y_ch: cp.Variable | None,
        y_dc: cp.Variable | None,
        p_ch: cp.Expression,
        p_dc: cp.Expression,
        T: int,
    ) -> None:
        """Add mode switch cost constraints and track them in objective."""
        if not inv.battery:
            return

        mode_switch_cost = inv.parameters.mode_switch_cost

        if mode_switch_cost <= 0.0:
            return

        def _mode_flags(mode: InverterMode) -> tuple[int, int]:
            if mode in (InverterMode.AC_CHARGE, InverterMode.AC_CHARGE_ZERO_FEED_IN):
                return 1, 0
            if mode in (InverterMode.DISCHARGE, InverterMode.DISCHARGE_ZERO_FEED_IN):
                return 0, 1
            return 0, 0

        init_ch, init_dc = _mode_flags(initial_mode)

        # Derive mode-active indicators without extra binary variables when possible.
        # y_ch/y_dc already encode mode state (sum in {0,1}); reuse them directly
        # instead of introducing redundant mode_ch/mode_dc binaries (saves 2*T binaries).
        is_ch: cp.Expression | np.ndarray
        is_dc: cp.Expression | np.ndarray

        if y_ch is not None:
            is_ch = cp.sum(y_ch, axis=0)
        elif isinstance(p_ch, cp.Variable):
            is_ch_var = cp.Variable(T, boolean=True, name=f"mode_ch_{inv_id}")
            max_p_ch = inv.battery.max_charge_power_w * self.prediction.dt_hours * 1.1
            self._constraints.append(p_ch <= max_p_ch * is_ch_var)
            is_ch = is_ch_var
        else:
            is_ch = np.zeros(T, dtype=float)

        if y_dc is not None:
            is_dc = cp.sum(y_dc, axis=0)
        elif isinstance(p_dc, cp.Variable):
            is_dc_var = cp.Variable(T, boolean=True, name=f"mode_dc_{inv_id}")
            max_p_dc = inv.battery.max_discharge_power_w * self.prediction.dt_hours * 1.1
            self._constraints.append(p_dc <= max_p_dc * is_dc_var)
            is_dc = is_dc_var
        else:
            is_dc = np.zeros(T, dtype=float)

        # PASS 2 OPT: delta_mode continuous instead of binary.
        # The constraints delta_mode >= |mode_change| naturally limit delta_mode in [0,1]
        # since each mode_change is at most 1. This eliminates binary branching overhead.
        delta_mode = cp.Variable(T, nonneg=True, name=f"delta_mode_{inv_id}")

        # t=0: compare against prior mode (4 scalar constraints).
        self._constraints.extend(
            [
                delta_mode[0] >= is_ch[0] - init_ch,
                delta_mode[0] >= init_ch - is_ch[0],
                delta_mode[0] >= is_dc[0] - init_dc,
                delta_mode[0] >= init_dc - is_dc[0],
            ]
        )

        # t>0: 4 vectorized array constraints instead of 4*(T-1) individual scalar constraints.
        if T > 1:
            self._constraints.extend(
                [
                    delta_mode[1:] >= is_ch[1:] - is_ch[:-1],
                    delta_mode[1:] >= is_ch[:-1] - is_ch[1:],
                    delta_mode[1:] >= is_dc[1:] - is_dc[:-1],
                    delta_mode[1:] >= is_dc[:-1] - is_dc[1:],
                ]
            )

        # Add mode switch costs to objective (cost is EUR per mode change, not per Wh)
        mode_switch_cost_expr = cp.sum(delta_mode) * mode_switch_cost
        self._mode_switch_costs.append(mode_switch_cost_expr)

    def _estimate_terminal_value(
        self,
        prep: _PreparedInputs,
        blocks: list[_InverterModelBlock],
    ) -> float:
        """Estimate EUR/Wh terminal value for battery SoC.

        Uses the mean price over the last min(T, 6) steps, weighted by the
        mean discharge efficiency, to approximate what 1 Wh of stored energy
        would save in the near future.
        """
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

    def _terminal_reward(
        self,
        blocks: list[_InverterModelBlock],
        terminal_value: float,
    ) -> cp.Expression | float:
        if terminal_value == 0.0:
            return 0.0

        terminal_terms: list[cp.Expression] = [
            block.soc[-1] for block in blocks if block.soc is not None
        ]
        if not terminal_terms:
            return 0.0

        return terminal_value * cp.sum(cp.hstack(terminal_terms))

    @staticmethod
    def _sum_terms(terms: list[cp.Expression], T: int) -> cp.Expression:
        if not terms:
            return cp.Constant(np.zeros(T, dtype=float))
        if len(terms) == 1:
            return terms[0]
        return cp.sum(cp.vstack(terms), axis=0)

    def _get_charge_rates(self, inv: InverterBase) -> tuple[int, ...]:
        if InverterMode.AC_CHARGE not in inv.available_modes:
            return tuple()
        raw = tuple(getattr(inv, "charge_rates", tuple())) or tuple(
            getattr(inv.parameters, "ac_rates_pct", tuple())
        )
        rates = tuple(sorted({int(r) for r in raw if 0 < int(r) <= 100}))
        return rates or (100,)

    def _get_discharge_rates(self, inv: InverterBase) -> tuple[int, ...]:
        # DISCHARGE_ZERO_FEED_IN is handled as continuous discharge variable,
        # not as discrete-rate states.
        if InverterMode.DISCHARGE not in inv.available_modes:
            return tuple()

        raw = tuple(getattr(inv, "discharge_rates", tuple())) or tuple(
            getattr(inv.parameters, "ac_rates_pct", tuple())
        )
        rates = tuple(sorted({int(r) for r in raw if 0 < int(r) <= 100}))
        return rates or (100,)

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
    def _sanitize_rate_percent(
        rate_pct: float,
        *,
        allowed_levels: tuple[int, ...] = tuple(),
    ) -> int:
        r = float(rate_pct)
        if allowed_levels:
            return int(min(allowed_levels, key=lambda lvl: abs(float(lvl) - r)))
        return int(max(0, min(100, round(r))))

    def _build_solution(
        self,
        *,
        prep: _PreparedInputs,
        blocks: list[_InverterModelBlock],
        g_import_val: np.ndarray,
        g_feedin_val: np.ndarray,
        pv_self_val: np.ndarray,
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
        solar_gen_per_dt: dict[str, np.ndarray] = {}
        inv_modes_per_dt: dict[str, np.ndarray] = {}
        inv_rates_per_dt: dict[str, np.ndarray] = {}
        inverter_plans: list[dict] = []

        for block in blocks:
            inv = block.inverter
            inv_id = inv.device_id

            p_ch = self._expr_to_vec(block.p_ch, T)
            p_dc = self._expr_to_vec(block.p_dc, T)
            pv_to_bat = self._expr_to_vec(block.pv_to_bat, T)

            if inv.battery is not None and block.soc is not None:
                bat = inv.battery
                soc_vals = np.maximum(np.asarray(block.soc.value, dtype=float), 0.0)
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

            # Build mode/rate series for simulation replay and UI output.
            modes, rates = self._extract_modes_and_rates(block, p_ch, p_dc, dt)
            inv_modes_per_dt[inv_id] = np.asarray(modes, dtype=np.int8)
            inv_rates_per_dt[inv_id] = np.asarray(rates, dtype=np.int32)
            inverter_plans.append({"device_id": inv_id, "modes": modes, "rates": rates})

        # Use prediction PV series directly for reporting.
        if prep.pv_by_source:
            solar_gen_per_dt = {
                inv_id: np.asarray(arr, dtype=np.float32)
                for inv_id, arr in prep.pv_by_source.items()
            }

        result = SimulationResult(
            costs_per_dt=np.asarray(costs, dtype=np.float32),
            revenue_per_dt=np.asarray(revenue, dtype=np.float32),
            grid_import_wh_per_dt=np.asarray(gi, dtype=np.float32),
            self_consumption_wh_per_dt=np.asarray(self_consumption, dtype=np.float32),
            feedin_wh_per_dt=np.asarray(gf, dtype=np.float32),
            losses_wh_per_dt=np.asarray(losses_arr, dtype=np.float32),
            electricity_price_per_dt=np.asarray(prep.price, dtype=np.float32),
            inverter_modes_per_dt=inv_modes_per_dt,
            inverter_ac_rate_per_dt=inv_rates_per_dt,
            solar_generation_wh_per_dt=solar_gen_per_dt,
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

        y_ch_vals = None
        y_dc_vals = None
        ch_rate_levels: tuple[int, ...] = tuple()
        dc_rate_levels: tuple[int, ...] = tuple()
        if block.selector is not None:
            ch_rate_levels = block.selector.charge_rates
            dc_rate_levels = block.selector.discharge_rates
            if block.selector.y_ch is not None:
                y_ch_vals = np.asarray(block.selector.y_ch.value, dtype=float)
            if block.selector.y_dc is not None:
                y_dc_vals = np.asarray(block.selector.y_dc.value, dtype=float)

        for t in range(T):
            ch = float(p_ch[t])
            dc = float(p_dc[t])

            if ch > eps:
                mode = (
                    int(InverterMode.AC_CHARGE)
                    if InverterMode.AC_CHARGE in inv.available_modes
                    else int(InverterMode.AC_CHARGE_ZERO_FEED_IN)
                )
                if y_ch_vals is not None and y_ch_vals.shape[1] > t and ch_rate_levels:
                    k = int(np.argmax(y_ch_vals[:, t]))
                    rate = int(ch_rate_levels[k])
                else:
                    rate = int(
                        round((min(ch / max_ch_wh, 1.0) if max_ch_wh > eps else 1.0) * 100.0)
                    )
                rate = self._sanitize_rate_percent(rate, allowed_levels=ch_rate_levels)
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
                    # Zero-feed discharge uses energy_wh in simulation; no ac_rate_pct.
                    rate = 0
                elif y_dc_vals is not None and y_dc_vals.shape[1] > t and dc_rate_levels:
                    k = int(np.argmax(y_dc_vals[:, t]))
                    rate = int(dc_rate_levels[k])
                else:
                    rate = int(
                        round((min(dc / max_dc_wh, 1.0) if max_dc_wh > eps else 1.0) * 100.0)
                    )
                rate = self._sanitize_rate_percent(rate, allowed_levels=dc_rate_levels)
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

        for inv in self.inverters:
            inv_id = inv.device_id
            plan = next((p for p in solution.inverter_plans if p["device_id"] == inv_id), None)
            if plan is None:
                modes[inv_id] = np.full(prep.T, int(InverterMode.IDLE), dtype=np.int32)
                rates[inv_id] = np.zeros(prep.T, dtype=np.int32)
            else:
                modes[inv_id] = np.asarray(plan["modes"], dtype=np.int32)
                rates[inv_id] = np.asarray([int(x) for x in plan["rates"]], dtype=np.int32)

        sim_result = sim.simulate(
            inverter_modes=modes,
            inverter_ac_rates=rates,
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
