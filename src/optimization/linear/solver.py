"""Linear (MILP) energy-management optimizer using CVXPY + HiGHS.

Formulation
-----------
Decision variables (per optimisable inverter *i*, timestep *t*):

* ``p_ch[i,t]`` ≥ 0 — AC energy consumed for battery charging (Wh)
* ``p_dc[i,t]`` ≥ 0 — AC energy delivered by battery discharging (Wh)
* ``b_ch[i,t]`` ∈ {0,1} — charging binary (Big-M bound)
* ``b_dc[i,t]`` ∈ {0,1} — discharging binary (Big-M bound)
* ``soc[i,t]`` ≥ 0 — battery state of charge at end of step *t* (Wh)

Grid variables (per timestep):

* ``g_import[t]`` ≥ 0 — energy imported from grid (Wh)
* ``g_feedin[t]`` ≥ 0 — energy exported to grid (Wh)

Constraints
-----------
Battery dynamics::

    soc[i,0]  = soc_initial[i] + p_ch[i,0]*η_c[i] - p_dc[i,0]/η_d[i]
    soc[i,t]  = soc[i,t-1]    + p_ch[i,t]*η_c[i] - p_dc[i,t]/η_d[i]

    η_c[i] = ac_to_dc_efficiency × charging_efficiency
    η_d[i] = discharging_efficiency × dc_to_ac_efficiency

Power bounds (Big-M)::

    p_ch[i,t] ≤ max_charge_power_w[i] × dt × b_ch[i,t]
    p_dc[i,t] ≤ max_discharge_power_w[i] × dt × b_dc[i,t]
    b_ch[i,t] + b_dc[i,t] ≤ 1          # no simultaneous charge + discharge

Energy balance (per timestep)::

    g_import[t] − g_feedin[t] = load_wh[t] − pv_total_wh[t]
                                 + Σ_i p_ch[i,t] − Σ_i p_dc[i,t]

Objectives
----------
``MINIMIZE_COST``::

    min  Σ_t [g_import[t]·price[t] − g_feedin[t]·feedin_tariff[t]]
         − battery_end_value_eur_wh · Σ_i soc[i, T-1]

``MAXIMIZE_SELF_CONSUMPTION``::

    min  Σ_t g_feedin[t]
         − battery_end_value_eur_wh · Σ_i soc[i, T-1]

The ``battery_end_value_eur_wh`` terminal reward prevents the solver
from draining batteries at the horizon just to maximise revenue or
self-consumption in the final steps.

Note on zero-feed-in
--------------------
The ``zero_feed_in`` flag on an inverter is *not* enforced as a hard LP
constraint (doing so would require knowing which inverter's discharge
causes grid export, which depends on all inverters simultaneously).
Instead, the derived inverter modes are mapped to the
``DISCHARGE_ZERO_FEED_IN`` / ``AC_CHARGE_ZERO_FEED_IN`` variants when the
flag is set, so the returned ``SimulationResult`` accurately reflects the
hardware topology.
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
from loguru import logger

from src.optimization.simulation import SimulationResult
from src.prediction.prediction import PredictionData
from src.simulation.devices import InverterMode
from src.simulation.devices.inverterbase import InverterBase

if TYPE_CHECKING:
    pass


# ── Public types ──────────────────────────────────────────────────────────────


class OptimizationObjective(str, Enum):
    """Supported objective functions for :class:`LinearOptimizer`."""

    MINIMIZE_COST = "cost"
    MAXIMIZE_SELF_CONSUMPTION = "self_consumption"


@dataclass
class LinearSolution:
    """Solution produced by :class:`LinearOptimizer`.

    Attributes:
        result: Full :class:`~src.optimization.simulation.SimulationResult`
            reconstructed from the LP optimal values.
        objective: Which objective was optimised.
        solver_status: CVXPY solver status string (e.g. ``"optimal"``).
        solve_time_s: Wall-clock solve time in seconds.
        inverter_plans: One dict per optimisable inverter with keys
            ``device_id``, ``modes`` (list of :class:`InverterMode` ints),
            and ``rates`` (list of floats in [0, 1]).
    """

    result: SimulationResult
    objective: OptimizationObjective
    solver_status: str
    solve_time_s: float
    inverter_plans: list[dict]


@dataclass
class _RateSelector:
    """Per-inverter binary selectors for discrete charge/discharge rate states."""

    charge_rates: tuple[float, ...]
    discharge_rates: tuple[float, ...]
    y_ch: cp.Variable | None
    y_dc: cp.Variable | None


# ── Optimizer ─────────────────────────────────────────────────────────────────


class LinearOptimizer:
    """MILP energy-management optimizer using CVXPY with the HiGHS backend.

    Args:
        inverters: All :class:`InverterBase` instances in the system.
            PV-only inverters contribute fixed generation; only inverters
            with batteries are decision variables.
        prediction: Fetched :class:`PredictionData` covering the
            optimisation horizon.
        battery_end_value_eur_wh: EUR value assigned to each Wh remaining
            in every battery at the end of the horizon.  A positive value
            prevents the solver from gaming the objective by unnecessarily
            draining batteries in the final steps.  Defaults to ``0.0``.
    """

    def __init__(
        self,
        inverters: list[InverterBase],
        prediction: PredictionData,
        battery_end_value_eur_wh: float = 0.0,
    ) -> None:
        self.inverters = inverters
        self.prediction = prediction
        self.battery_end_value_eur_wh = battery_end_value_eur_wh
        self._opt_inverters: list[InverterBase] = [
            inv for inv in inverters if inv.is_optimizable and inv.battery is not None
        ]

    # ── Public API ────────────────────────────────────────────────────────

    def solve(
        self,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST,
        solver_opts: dict | None = None,
    ) -> LinearSolution:
        """Build the MILP, solve it with HiGHS, and return a :class:`LinearSolution`.

        Args:
            objective: Which objective function to optimise.
            solver_opts: Optional keyword arguments forwarded to CVXPY's
                ``solve()`` call (e.g. ``{"time_limit": 60}``).

        Returns:
            A :class:`LinearSolution` containing the full simulation result
            and optimisation metadata.

        Raises:
            RuntimeError: If the solver fails to find an optimal solution.
        """
        T = self.prediction.steps
        dt = self.prediction.dt_hours
        pred = self.prediction

        # ── Extract fixed parameters from PredictionData ──────────────
        price = np.array(pred["electricprice_eur_wh"].to_list(), dtype=float)
        feedin_tariff = np.array(pred["feedintariff_eur_wh"].to_list(), dtype=float)
        load_wh = np.array(pred["load_w"].to_list(), dtype=float) * dt

        # Aggregate PV generation: W → Wh per step
        pv_total_wh = np.zeros(T, dtype=float)
        pv_per_inverter: dict[str, np.ndarray] = {}
        for col in pred.df.columns:
            if col.startswith("pv_") and col.endswith("_w"):
                arr = np.array(pred[col].to_list(), dtype=float) * dt
                pv_total_wh += arr
                # Column format: pv_{name}_{inverter}_w → key is "{name}_{inverter}"
                body = col[len("pv_") : -len("_w")]
                pv_per_inverter[body] = arr

        # ── Decision variables ────────────────────────────────────────
        # Discrete sub-states per inverter:
        #   IDLE + one of charge-rate states + one of discharge-rate states.
        p_ch: dict[str, cp.Expression] = {}
        p_dc: dict[str, cp.Expression] = {}
        soc: dict[str, cp.Variable] = {}
        mode_selectors: dict[str, _RateSelector] = {}

        for inv in self._opt_inverters:
            inv_id = inv.device_id
            bat = inv.battery
            if bat is None:
                raise RuntimeError(f"Optimizable inverter '{inv_id}' has no battery")
            max_ch_wh = bat.max_charge_power_w * dt
            max_dc_wh = bat.max_discharge_power_w * dt

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

            if y_ch is not None:
                p_ch_expr: cp.Expression = max_ch_wh * (np.array(charge_rates, dtype=float) @ y_ch)
            else:
                p_ch_expr = cp.Constant(np.zeros(T, dtype=float))
            if y_dc is not None:
                p_dc_expr: cp.Expression = max_dc_wh * (
                    np.array(discharge_rates, dtype=float) @ y_dc
                )
            else:
                p_dc_expr = cp.Constant(np.zeros(T, dtype=float))

            p_ch[inv_id] = p_ch_expr
            p_dc[inv_id] = p_dc_expr
            soc[inv_id] = cp.Variable(T, nonneg=True, name=f"soc_{inv_id}")
            mode_selectors[inv_id] = _RateSelector(
                charge_rates=charge_rates,
                discharge_rates=discharge_rates,
                y_ch=y_ch,
                y_dc=y_dc,
            )

        g_import = cp.Variable(T, nonneg=True, name="g_import")
        g_feedin = cp.Variable(T, nonneg=True, name="g_feedin")

        # ── Constraints ───────────────────────────────────────────────
        constraints: list[cp.Constraint] = []

        if self._opt_inverters:
            total_p_ch: cp.Expression = cp.sum(
                cp.vstack([p_ch[inv.device_id] for inv in self._opt_inverters]), axis=0
            )
            total_p_dc: cp.Expression = cp.sum(
                cp.vstack([p_dc[inv.device_id] for inv in self._opt_inverters]), axis=0
            )
        else:
            total_p_ch = np.zeros(T)
            total_p_dc = np.zeros(T)

        for inv in self._opt_inverters:
            inv_id = inv.device_id
            bat = inv.battery
            if bat is None:
                raise RuntimeError(f"Optimizable inverter '{inv_id}' has no battery")
            # Cascade efficiencies
            eta_c = inv.parameters.ac_to_dc_efficiency * bat.charging_efficiency
            eta_d = bat.discharging_efficiency * inv.parameters.dc_to_ac_efficiency

            selector = mode_selectors[inv_id]
            y_ch = selector.y_ch
            y_dc = selector.y_dc

            # At each step: at most one sub-state active (charge or discharge), else IDLE.
            if y_ch is not None and y_dc is not None:
                constraints += [
                    cp.sum(y_ch, axis=0) <= 1,
                    cp.sum(y_dc, axis=0) <= 1,
                    cp.sum(y_ch, axis=0) + cp.sum(y_dc, axis=0) <= 1,
                ]
            elif y_ch is not None:
                constraints += [cp.sum(y_ch, axis=0) <= 1]
            elif y_dc is not None:
                constraints += [cp.sum(y_dc, axis=0) <= 1]

            # Battery dynamics (vectorised difference constraint)
            delta = p_ch[inv_id] * eta_c - p_dc[inv_id] / eta_d
            soc_init = float(bat.soc_wh)
            constraints.append(soc[inv_id][0] == soc_init + delta[0])
            if T > 1:
                constraints.append(soc[inv_id][1:] == soc[inv_id][:-1] + delta[1:])

            # SoC bounds
            constraints += [
                soc[inv_id] >= bat.min_soc_wh,
                soc[inv_id] <= bat.max_soc_wh,
            ]

        # Energy balance (vectorised over all T timesteps)
        constraints.append(g_import - g_feedin == load_wh - pv_total_wh + total_p_ch - total_p_dc)

        # ── Objective ─────────────────────────────────────────────────
        bat_terminal_reward: cp.Expression | float = 0.0
        if self.battery_end_value_eur_wh != 0.0 and self._opt_inverters:
            bat_terminal_reward = self.battery_end_value_eur_wh * sum(
                soc[inv.device_id][-1] for inv in self._opt_inverters
            )

        if objective == OptimizationObjective.MINIMIZE_COST:
            cost_term = cp.sum(cp.multiply(g_import, price) - cp.multiply(g_feedin, feedin_tariff))
            obj_expr = cost_term - bat_terminal_reward
        else:  # MAXIMIZE_SELF_CONSUMPTION: minimise grid feed-in
            obj_expr = cp.sum(g_feedin) - bat_terminal_reward

        prob = cp.Problem(cp.Minimize(obj_expr), constraints)

        # ── Solve ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        opts: dict = {"verbose": False, **(solver_opts or {})}
        try:
            prob.solve(solver=cp.HIGHS, **opts)
        except cp.SolverError as exc:
            raise RuntimeError(f"CVXPY/HiGHS solver error: {exc}") from exc
        solve_time = time.perf_counter() - t0

        status = prob.status
        logger.info(
            "LinearOptimizer: status={} objective={} time={:.2f}s value={}",
            status,
            objective.value,
            solve_time,
            prob.value,
        )

        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Optimisation did not converge: solver status='{status}'. "
                "Check that the problem is feasible (battery capacity and SoC bounds)."
            )

        if g_import.value is None or g_feedin.value is None:
            raise RuntimeError("Solver returned no values for grid variables")

        # ── Build result ───────────────────────────────────────────────
        return self._build_solution(
            T=T,
            dt=dt,
            load_wh=load_wh,
            pv_total_wh=pv_total_wh,
            pv_per_inverter=pv_per_inverter,
            price=price,
            feedin_tariff=feedin_tariff,
            p_ch=p_ch,
            p_dc=p_dc,
            soc=soc,
            mode_selectors=mode_selectors,
            g_import_val=g_import.value,
            g_feedin_val=g_feedin.value,
            objective=objective,
            solver_status=status,
            solve_time=solve_time,
        )

    def _get_charge_rates(self, inv: InverterBase) -> tuple[float, ...]:
        """Return supported discrete AC-charge rates in (0, 1]."""
        if InverterMode.AC_CHARGE not in inv.available_modes:
            return tuple()

        raw_rates = tuple(getattr(inv, "charge_rates", tuple()))
        if not raw_rates:
            raw_rates = tuple(getattr(inv.parameters, "ac_rates", tuple()))

        rates = tuple(sorted({float(r) for r in raw_rates if 0.0 < float(r) <= 1.0}))
        return rates or (1.0,)

    def _get_discharge_rates(self, inv: InverterBase) -> tuple[float, ...]:
        """Return supported discrete AC-discharge rates in (0, 1]."""
        has_discharge_mode = (
            InverterMode.DISCHARGE in inv.available_modes
            or InverterMode.DISCHARGE_ZERO_FEED_IN in inv.available_modes
        )
        if not has_discharge_mode:
            return tuple()

        raw_rates = tuple(getattr(inv, "discharge_rates", tuple()))
        if not raw_rates:
            raw_rates = tuple(getattr(inv.parameters, "ac_rates", tuple()))

        rates = tuple(sorted({float(r) for r in raw_rates if 0.0 < float(r) <= 1.0}))
        return rates or (1.0,)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_solution(
        self,
        *,
        T: int,
        dt: float,
        load_wh: np.ndarray,
        pv_total_wh: np.ndarray,
        pv_per_inverter: dict[str, np.ndarray],
        price: np.ndarray,
        feedin_tariff: np.ndarray,
        p_ch: dict[str, cp.Expression],
        p_dc: dict[str, cp.Expression],
        soc: dict[str, cp.Variable],
        mode_selectors: dict[str, _RateSelector],
        g_import_val: np.ndarray,
        g_feedin_val: np.ndarray,
        objective: OptimizationObjective,
        solver_status: str,
        solve_time: float,
    ) -> LinearSolution:
        _EPS = 1e-4  # threshold for treating power as non-zero

        # Clamp numerical noise to non-negative
        gi = np.maximum(np.asarray(g_import_val, dtype=float), 0.0)
        gf = np.maximum(np.asarray(g_feedin_val, dtype=float), 0.0)

        costs_per_dt = array("f", (gi * price).tolist())
        revenue_per_dt = array("f", (gf * feedin_tariff).tolist())
        grid_import_wh = array("f", gi.tolist())
        feedin_wh = array("f", gf.tolist())
        self_cons_wh = array("f", np.maximum(load_wh - gi, 0.0).tolist())
        elec_price_series = array("f", price.tolist())

        losses_arr = np.zeros(T, dtype=float)
        battery_wh_per_dt: dict[str, array] = {}
        battery_soc_pct: dict[str, array] = {}
        solar_gen_per_dt: dict[str, array] = {}
        inv_modes_per_dt: dict[str, array] = {}
        inv_rates_per_dt: dict[str, array] = {}
        inverter_plans: list[dict] = []

        # PV generation per inverter (fixed, from PredictionData)
        for inv in self.inverters:
            pv_src = inv.parameters.pv_source
            if pv_src and pv_src in pv_per_inverter:
                solar_gen_per_dt[inv.device_id] = array("f", pv_per_inverter[pv_src].tolist())

        # Per-inverter battery and mode extraction
        for inv in self._opt_inverters:
            inv_id = inv.device_id
            bat = inv.battery
            if bat is None:
                raise RuntimeError(f"Optimizable inverter '{inv_id}' has no battery")
            eta_c = inv.parameters.ac_to_dc_efficiency * bat.charging_efficiency
            eta_d = bat.discharging_efficiency * inv.parameters.dc_to_ac_efficiency

            ch_vals = np.maximum(np.asarray(p_ch[inv_id].value, dtype=float), 0.0)
            dc_vals = np.maximum(np.asarray(p_dc[inv_id].value, dtype=float), 0.0)
            soc_vals = np.maximum(np.asarray(soc[inv_id].value, dtype=float), 0.0)

            # Accumulate losses
            losses_arr += ch_vals * (1.0 - eta_c) + dc_vals * (1.0 / eta_d - 1.0)

            battery_wh_per_dt[inv_id] = array("f", soc_vals.tolist())
            battery_soc_pct[inv_id] = array("f", (soc_vals * (100.0 / bat.capacity_wh)).tolist())

            # Derive inverter modes and normalised AC rate from selected sub-states
            zero_feed_in = inv.parameters.zero_feed_in
            max_ch_wh = bat.max_charge_power_w * dt
            max_dc_wh = bat.max_discharge_power_w * dt

            selector = mode_selectors[inv_id]
            ch_rate_levels = selector.charge_rates
            dc_rate_levels = selector.discharge_rates
            y_ch = selector.y_ch
            y_dc = selector.y_dc

            y_ch_vals = np.asarray(y_ch.value, dtype=float) if y_ch is not None else None
            y_dc_vals = np.asarray(y_dc.value, dtype=float) if y_dc is not None else None

            modes: list[int] = []
            rates: list[float] = []

            for t in range(T):
                ch = float(ch_vals[t])
                dc = float(dc_vals[t])
                if ch > _EPS:
                    if InverterMode.AC_CHARGE in inv.available_modes:
                        mode = int(InverterMode.AC_CHARGE)
                    else:
                        mode = int(InverterMode.AC_CHARGE_ZERO_FEED_IN)

                    if y_ch_vals is not None and y_ch_vals.shape[1] > t and ch_rate_levels:
                        k = int(np.argmax(y_ch_vals[:, t]))
                        rate = float(ch_rate_levels[k])
                    else:
                        rate = min(ch / max_ch_wh, 1.0) if max_ch_wh > _EPS else 1.0
                elif dc > _EPS:
                    if InverterMode.DISCHARGE in inv.available_modes:
                        mode = int(InverterMode.DISCHARGE)
                    else:
                        mode = (
                            int(InverterMode.DISCHARGE_ZERO_FEED_IN)
                            if zero_feed_in
                            else int(InverterMode.DISCHARGE)
                        )

                    if y_dc_vals is not None and y_dc_vals.shape[1] > t and dc_rate_levels:
                        k = int(np.argmax(y_dc_vals[:, t]))
                        rate = float(dc_rate_levels[k])
                    else:
                        rate = min(dc / max_dc_wh, 1.0) if max_dc_wh > _EPS else 1.0
                else:
                    mode = int(InverterMode.IDLE)
                    rate = 0.0

                modes.append(mode)
                rates.append(rate)

            inv_modes_per_dt[inv_id] = array("b", modes)
            inv_rates_per_dt[inv_id] = array("f", rates)
            inverter_plans.append({"device_id": inv_id, "modes": modes, "rates": rates})

        result = SimulationResult(
            costs_per_dt=costs_per_dt,
            revenue_per_dt=revenue_per_dt,
            grid_import_wh_per_dt=grid_import_wh,
            self_consumption_wh_per_dt=self_cons_wh,
            feedin_wh_per_dt=feedin_wh,
            losses_wh_per_dt=array("f", losses_arr.tolist()),
            electricity_price_per_dt=elec_price_series,
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
