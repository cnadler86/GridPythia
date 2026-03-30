"""Linear (MILP) energy-management optimizer using CVXPY + HiGHS.

This module builds a topology-aware mathematical model from the same
core signals that drive GridSimulation (load, PV, battery, prices) and
solves it with HiGHS. Home appliances are intentionally excluded from
this LP model for now.
"""

from __future__ import annotations

import time
from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import cvxpy as cp
import numpy as np
from loguru import logger

from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.inverterbase import InverterBase
from GridPythia.simulation.grid_interpolator import FraunhoferSCModel
from GridPythia.simulation.grid_simulation import GridSimulation, SimulationResult


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
        initial_modes: Prior inverter mode per device ID for correct t=0
            mode-switch cost accounting.
    """

    def __init__(
        self,
        inverters: list[InverterBase],
        prediction: PredictionData,
        initial_modes: Mapping[str, InverterMode | int] | None = None,
    ) -> None:
        self.inverters = inverters
        self.prediction = prediction
        self.initial_modes: dict[str, InverterMode] = {}
        for inv in self.inverters:
            raw_mode = (
                initial_modes.get(inv.device_id, InverterMode.IDLE)
                if initial_modes is not None
                else InverterMode.IDLE
            )
            self.initial_modes[inv.device_id] = (
                raw_mode if isinstance(raw_mode, InverterMode) else InverterMode(int(raw_mode))
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
    ) -> LinearSolution:
        """Build MILP, solve with HiGHS, and return a :class:`LinearSolution`."""
        prep = self._prepare_inputs()
        blocks, g_import, g_feedin, pv_self = self._build_problem(prep)

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
        logger.info(
            "Solving modular linear problem with objective '{}' and {} scalar variables ({} constraints)",
            objective.value,
            size.num_scalar_variables,
            size.num_scalar_eq_constr + size.num_scalar_leq_constr,
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
            raise RuntimeError(
                f"Optimisation did not converge: solver status='{status}'. "
                "Check feasibility (battery bounds, rates, and capacities)."
            )

        if g_import.value is None or g_feedin.value is None:
            raise RuntimeError("Solver returned no values for grid variables")

        solution = self._build_solution(
            prep=prep,
            blocks=blocks,
            g_import_val=np.asarray(g_import.value, dtype=float),
            g_feedin_val=np.asarray(g_feedin.value, dtype=float),
            pv_self_val=np.asarray(pv_self.value, dtype=float),
            objective=objective,
            solver_status=status,
            solve_time=solve_time,
        )

        if validate_with_simulation:
            parity, sim_res = self._validate_with_simulation(solution, prep)
            solution.parity_report = parity
            solution.simulation_result = sim_res

        return solution

    def _prepare_inputs(self) -> _PreparedInputs:
        pred = self.prediction
        T = pred.steps
        dt = pred.dt_hours

        price_series = pred.electricprice
        if price_series is None:
            price = np.zeros(T, dtype=float)
        else:
            price = np.array(price_series.to_list(), dtype=float)

        tariff_series = pred.feedintariff
        if tariff_series is None:
            feedin_tariff = np.zeros(T, dtype=float)
        else:
            feedin_tariff = np.array(tariff_series.to_list(), dtype=float)

        load_wh = np.array(pred.load_wh.to_list(), dtype=float)
        pv_by_source = {
            k: np.array(v.to_list(), dtype=float) for k, v in pred.pv_by_inverter.items()
        }

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
            block = self._build_inverter_block(inv, prep)
            blocks.append(block)

            total_p_ch_terms.append(block.p_ch)
            total_p_dc_terms.append(block.p_dc)
            total_pv_ac_terms.append(block.pv_ac)

            pv_src = inv.parameters.pv_source
            if pv_src:
                mapped_pv_sources.add(pv_src)

        for GridPythia, arr in prep.pv_by_source.items():
            if GridPythia not in mapped_pv_sources:
                total_pv_ac_terms.append(cp.Constant(arr))

        total_p_ch = self._sum_terms(total_p_ch_terms, prep.T)
        total_p_dc = self._sum_terms(total_p_dc_terms, prep.T)
        total_pv_ac = self._sum_terms(total_pv_ac_terms, prep.T)

        for t in range(T):
            pv0 = max(
                float(total_pv_ac.value[t]) if total_pv_ac.value is not None else prep.load_wh[t],
                1e-6,
            )
            load0 = max(float(prep.load_wh[t]), 1e-6)
            lc = self._sc_model.linearize(pv_0=pv0, load_0=load0)
            rhs_fixed = lc.rhs_fixed_load(load=load0)

            self._constraints.append(pv_self[t] - lc.c_pv * total_pv_ac[t] <= rhs_fixed)
            self._constraints.append(pv_self[t] <= total_pv_ac[t])
            self._constraints.append(pv_self[t] <= prep.load_wh[t])

        # Mirrors current simulation correction logic:
        # corrected_end_load = load + p_ch - p_dc + pv_ac - pv_self
        self._constraints.append(
            g_import - g_feedin == prep.load_wh + total_p_ch - total_p_dc + total_pv_ac - pv_self
        )

        return blocks, g_import, g_feedin, pv_self

    def _build_inverter_block(
        self, inv: InverterBase, prep: _PreparedInputs
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
            self._add_mode_switch_costs(inv, inv_id, y_ch, y_dc, p_ch, p_dc, T)

        else:
            p_ch = cp.Constant(np.zeros(T, dtype=float))
            p_dc = cp.Constant(np.zeros(T, dtype=float))

        if inv.parameters.pv_source is not None:
            pv_src = inv.parameters.pv_source
            pv_pred = prep.pv_by_source.get(pv_src, np.zeros(T, dtype=float))
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

            self._constraints.append(soc[0] == soc_init + delta[0])
            if T > 1:
                self._constraints.append(soc[1:] == soc[:-1] + delta[1:])
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

        initial_mode = self.initial_modes.get(inv_id, InverterMode.IDLE)

        def _mode_flags(mode: InverterMode) -> tuple[int, int]:
            if mode in (InverterMode.AC_CHARGE, InverterMode.AC_CHARGE_ZERO_FEED_IN):
                return 1, 0
            if mode in (InverterMode.DISCHARGE, InverterMode.DISCHARGE_ZERO_FEED_IN):
                return 0, 1
            return 0, 0

        init_ch, init_dc = _mode_flags(initial_mode)

        # Create binary variables for mode switches
        delta_mode = cp.Variable(T, boolean=True, name=f"delta_mode_{inv_id}")
        mode_ch = cp.Variable(T, boolean=True, name=f"mode_ch_{inv_id}")
        mode_dc = cp.Variable(T, boolean=True, name=f"mode_dc_{inv_id}")

        if y_ch is not None:
            y_ch_sum = cp.sum(y_ch, axis=0)
            self._constraints.extend([mode_ch >= y_ch_sum, mode_ch <= y_ch_sum])
        elif isinstance(p_ch, cp.Variable):
            # Continuous charge: mode_ch = 1 if p_ch > 0
            # Use big-M constraint: p_ch <= max_p_ch * mode_ch (where max_p_ch is arbitrarily large)
            # Since p_ch is bounded by battery capacity constraints, we can use a reasonable bound
            max_p_ch = inv.battery.max_charge_power_w * self.prediction.dt_hours * 1.1
            self._constraints.append(p_ch <= max_p_ch * mode_ch)
        else:
            self._constraints.append(mode_ch == 0)

        if y_dc is not None:
            y_dc_sum = cp.sum(y_dc, axis=0)
            self._constraints.extend([mode_dc >= y_dc_sum, mode_dc <= y_dc_sum])
        elif isinstance(p_dc, cp.Variable):
            # Continuous discharge: mode_dc = 1 if p_dc > 0
            # Use big-M constraint: p_dc <= max_p_dc * mode_dc
            max_p_dc = inv.battery.max_discharge_power_w * self.prediction.dt_hours * 1.1
            self._constraints.append(p_dc <= max_p_dc * mode_dc)
        else:
            self._constraints.append(mode_dc == 0)

        # At t=0, compare against configured initial mode (default: IDLE).
        self._constraints.append(delta_mode[0] >= mode_ch[0] - init_ch)
        self._constraints.append(delta_mode[0] >= init_ch - mode_ch[0])
        self._constraints.append(delta_mode[0] >= mode_dc[0] - init_dc)
        self._constraints.append(delta_mode[0] >= init_dc - mode_dc[0])

        # For t > 0, add constraints to detect mode changes
        if T > 1:
            # delta_mode[t] = 1 if mode changed from t-1 to t
            # This means: (mode_ch[t] != mode_ch[t-1]) or (mode_dc[t] != mode_dc[t-1])
            # Simplified: delta_mode[t] >= |mode_ch[t] - mode_ch[t-1]| + |mode_dc[t] - mode_dc[t-1]|
            # Since these are binary, |a - b| = max(a - b, b - a)
            for t in range(1, T):
                # If mode_ch changes, mode_dc changes, or both change, delta_mode[t] = 1
                self._constraints.append(delta_mode[t] >= (mode_ch[t] - mode_ch[t - 1]))
                self._constraints.append(delta_mode[t] >= (mode_ch[t - 1] - mode_ch[t]))
                self._constraints.append(delta_mode[t] >= (mode_dc[t] - mode_dc[t - 1]))
                self._constraints.append(delta_mode[t] >= (mode_dc[t - 1] - mode_dc[t]))

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
    ) -> LinearSolution:
        T = prep.T
        dt = prep.dt

        gi = np.maximum(g_import_val, 0.0)
        gf = np.maximum(g_feedin_val, 0.0)
        self_consumption = np.maximum(prep.load_wh - gi, 0.0)

        costs = gi * prep.price
        revenue = gf * prep.feedin_tariff

        losses_arr = np.zeros(T, dtype=float)
        battery_wh_per_dt: dict[str, array] = {}
        battery_soc_pct: dict[str, array] = {}
        solar_gen_per_dt: dict[str, array] = {}
        inv_modes_per_dt: dict[str, array] = {}
        inv_rates_per_dt: dict[str, array] = {}
        inverter_plans: list[dict] = []

        for block in blocks:
            inv = block.inverter
            inv_id = inv.device_id

            p_ch = self._expr_to_vec(block.p_ch, T)
            p_dc = self._expr_to_vec(block.p_dc, T)
            pv_ac = self._expr_to_vec(block.pv_ac, T)
            pv_to_bat = self._expr_to_vec(block.pv_to_bat, T)

            if inv.parameters.pv_source is not None:
                solar_gen_per_dt[inv_id] = array("f", pv_ac.tolist())

            if inv.battery is not None and block.soc is not None:
                bat = inv.battery
                soc_vals = np.maximum(np.asarray(block.soc.value, dtype=float), 0.0)
                battery_wh_per_dt[inv_id] = array("f", soc_vals.tolist())
                battery_soc_pct[inv_id] = array(
                    "f", (soc_vals * (100.0 / bat.capacity_wh)).tolist()
                )

                eta_c_ac = inv.parameters.ac_to_dc_efficiency * bat.charging_efficiency
                eta_c_pv = bat.charging_efficiency
                eta_d = bat.discharging_efficiency * inv.parameters.dc_to_ac_efficiency
                losses_arr += p_ch * (1.0 - eta_c_ac)
                losses_arr += pv_to_bat * (1.0 - eta_c_pv)
                losses_arr += p_dc * (1.0 / eta_d - 1.0)

            # Build mode/rate series for simulation replay and UI output.
            modes, rates = self._extract_modes_and_rates(block, p_ch, p_dc, dt)
            inv_modes_per_dt[inv_id] = array("b", modes)
            inv_rates_per_dt[inv_id] = array("i", rates)
            inverter_plans.append({"device_id": inv_id, "modes": modes, "rates": rates})

        result = SimulationResult(
            costs_per_dt=array("f", costs.tolist()),
            revenue_per_dt=array("f", revenue.tolist()),
            grid_import_wh_per_dt=array("f", gi.tolist()),
            self_consumption_wh_per_dt=array("f", self_consumption.tolist()),
            feedin_wh_per_dt=array("f", gf.tolist()),
            losses_wh_per_dt=array("f", losses_arr.tolist()),
            electricity_price_per_dt=array("f", prep.price.tolist()),
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

        modes: dict[str, array] = {}
        rates: dict[str, array] = {}

        for inv in self.inverters:
            inv_id = inv.device_id
            plan = next((p for p in solution.inverter_plans if p["device_id"] == inv_id), None)
            if plan is None:
                modes[inv_id] = array("i", [int(InverterMode.IDLE)] * prep.T)
                rates[inv_id] = array("i", [0] * prep.T)
            else:
                modes[inv_id] = array("i", plan["modes"])
                rates[inv_id] = array("i", [int(x) for x in plan["rates"]])

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
