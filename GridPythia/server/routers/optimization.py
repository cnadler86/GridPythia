"""POST /api/optimize – run the MILP optimizer and return solution + charts."""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.optimization.plots import SolutionPlotter
from GridPythia.optimization.solution import OptimizationObjective
from GridPythia.prediction.electricprice.energycharts import ElecPriceEnergyCharts
from GridPythia.prediction.prediction import Prediction
from GridPythia.server import services
from GridPythia.server.models import OptimizeRequest, OptimizeSummary

logger = get_logger(__name__)

router = APIRouter(tags=["optimization"])


@router.post("/optimize")
async def optimize(req: OptimizeRequest) -> JSONResponse:
    """Run the MILP energy optimizer and return the full solution.

    **Request fields**

    - ``timezone`` – IANA timezone used for the forecast start (default ``"UTC"``).
    - ``battery_soc`` – per-battery SoC overrides ``{battery_id: pct}``.
      Values are clamped to ``[min_soc, max_soc]`` from the config.
    - ``initial_modes`` – initial inverter mode ``{inverter_id: InverterMode_int}``
      at the start of the horizon. Omit to default all inverters to ``IDLE (0)``.
    - ``solver_opts`` – optional HiGHS option overrides for this call only
      (merged over config defaults). E.g. ``{"time_limit": 10, "mip_rel_gap": 0.05}``.

    **Response fields**

    - ``summary`` – solver metadata and cost / savings numbers.
    - ``inverter_plans`` – per-inverter schedule; one :class:`InverterPlanStep`
      per prediction timestep, with mode, energy flows and battery SoC.
    - ``charts`` – Plotly figure JSON keyed by tab-id (prediction tabs +
      per-inverter tabs ``"tab-inv-<device_id>"``).
    - ``status`` – human-readable summary string for the UI status bar.
    """
    # ── Config ────────────────────────────────────────────────────────
    try:
        cfg, raw_yaml = services.load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    try:
        tz = ZoneInfo(req.timezone)
    except ZoneInfoNotFoundError:
        logger.warning("unknown_timezone", tz=req.timezone)
        tz = ZoneInfo("UTC")

    # ── Prediction data: cache or fresh fetch ─────────────────────────
    cached = services.get_cached_pdata()
    if cached is not None:
        pdata, forecast_from = cached
        logger.info("optimize_using_cached_pdata")
    else:
        try:
            setup = services.get_providers(cfg, raw_yaml)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Provider build error: {exc}") from exc

        pred = Prediction(setup)
        try:
            pdata = await pred.fetch(
                start=datetime.now(tz=tz),
                hours=float(cfg.prediction.horizon),
                dt_hours=float(cfg.prediction.dt_hours),
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Prediction fetch failed: {exc}") from exc

        forecast_from = (
            setup.electricprice.last_real_ts
            if isinstance(setup.electricprice, ElecPriceEnergyCharts)
            else None
        )
        services.set_cached_pdata(pdata, forecast_from)

    # ── Optimizer ─────────────────────────────────────────────────────
    try:
        optimizer = services.get_optimizer(cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimizer setup error: {exc}") from exc

    objective = (
        OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
        if cfg.optimization.solver.objective == "self_consumption"
        else OptimizationObjective.MINIMIZE_COST
    )

    # Merge solver_opts: config defaults → per-request overrides
    solver_opts = dict(cfg.optimization.solver.solver_opts)
    if req.solver_opts:
        solver_opts.update(req.solver_opts)

    soc_wh = services.soc_overrides_wh_for_solver(optimizer, req.battery_soc)
    initial_modes = req.initial_modes or None

    try:
        async with state.get_optimizer_lock():
            solution = await asyncio.to_thread(
                lambda: optimizer.solve(
                    pdata,
                    soc=soc_wh or None,
                    initial_modes=initial_modes or None,
                    objective=objective,
                    solver_opts=solver_opts,
                    validate_with_simulation=True,
                )
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc

    # ── Serialize inverter plans ──────────────────────────────────────
    inverter_plans = [
        services.inverter_plan_to_response(plan, list(pdata.timestamps))
        for plan in solution.inverter_plans
    ]

    # ── Build charts ──────────────────────────────────────────────────
    charts: dict = {}
    charts.update(services.make_prediction_figures(pdata, forecast_from))
    plotter = SolutionPlotter()
    for inv in optimizer.inverters:
        charts[f"tab-inv-{inv.device_id}"] = services.fig_to_dict(
            plotter.plot_inverter(solution, inv.device_id)
        )

    # ── Savings vs. naive baseline (PV direct to load, no battery) ────
    pv_total = np.zeros(pdata.steps, dtype=float)
    for arr in pdata.pv_by_inverter.values():
        pv_total += np.asarray(arr, dtype=float)
    load_arr = np.asarray(pdata.load_wh, dtype=float)
    price_arr = (
        np.asarray(pdata.electricprice, dtype=float)
        if pdata.electricprice is not None
        else np.zeros(pdata.steps, dtype=float)
    )
    feedin_arr = (
        np.asarray(pdata.feedintariff, dtype=float)
        if pdata.feedintariff is not None
        else np.zeros(pdata.steps, dtype=float)
    )
    naive_net_cost = float(
        np.sum(np.maximum(0.0, load_arr - pv_total) * price_arr)
        - np.sum(np.maximum(0.0, pv_total - load_arr) * feedin_arr)
    )
    net_cost = solution.result.total_cost - solution.result.total_revenue
    savings = naive_net_cost - net_cost
    parity_ok = solution.parity_report.ok if solution.parity_report is not None else None

    summary = OptimizeSummary(
        solver_status=solution.solver_status,
        solve_time_s=round(solution.solve_time_s, 2),
        objective=objective.value,
        total_cost_eur=round(float(solution.result.total_cost), 4),
        total_revenue_eur=round(float(solution.result.total_revenue), 4),
        net_cost_eur=round(float(net_cost), 4),
        naive_net_cost_eur=round(float(naive_net_cost), 4),
        savings_eur=round(float(savings), 4),
        parity_ok=parity_ok,
    )

    parity_warn = " ⚠ parity" if parity_ok is False else ""
    status = (
        f"Solved {solution.solve_time_s:.1f}s · {solution.solver_status} · "
        f"naive: {naive_net_cost:.3f} EUR → optimized: {net_cost:.3f} EUR · "
        f"savings: {savings:.3f} EUR{parity_warn}"
    )

    logger.info(
        "optimize_done",
        solver_status=solution.solver_status,
        solve_time_s=round(solution.solve_time_s, 2),
        savings_eur=round(savings, 3),
    )

    return JSONResponse(
        {
            "summary": summary.model_dump(),
            "inverter_plans": [p.model_dump() for p in inverter_plans],
            "charts": charts,
            "status": status,
        }
    )
