"""POST /api/optimize – run the MILP optimizer and return solution + charts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.server import services
from GridPythia.server.models import OptimizeRequest, OptimizeStatusResponse

logger = get_logger(__name__)

router = APIRouter(tags=["optimization"])


@router.post("/optimize")
async def optimize(req: OptimizeRequest) -> JSONResponse:
    """Run the MILP energy optimizer and return the full solution.

    **Request fields**

    - ``timezone`` – IANA timezone for the horizon start (default ``"UTC"``).
    - ``start`` – ISO-8601 horizon start override; defaults to ``now``.
      The runner floors this to the nearest slot boundary for the prediction
      fetch and ceils it for the solver window.
    - ``battery_soc`` – per-battery SoC overrides ``{battery_id: pct}``.
    - ``initial_modes`` – initial inverter mode ``{inverter_id: InverterMode_int}``.
    - ``solver_opts`` – optional HiGHS option overrides for this call only.

    **Response fields**

    - ``summary`` – solver metadata and cost / savings numbers.
    - ``inverter_plans`` – per-inverter schedule with one step per solver slot.
    - ``charts`` – Plotly figure JSON keyed by tab-id.
    - ``status`` – human-readable summary string for the UI status bar.
    """
    try:
        cfg, raw_yaml = services.load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    try:
        tz = ZoneInfo(req.timezone)
    except ZoneInfoNotFoundError:
        logger.warning("unknown_timezone", tz=req.timezone)
        tz = ZoneInfo("UTC")

    if req.start is not None:
        try:
            start = datetime.fromisoformat(req.start)
            if start.tzinfo is None:
                start = start.replace(tzinfo=tz)
            else:
                start = start.astimezone(tz)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid start: {exc}") from exc
    else:
        start = datetime.now(tz=tz)

    end = start + timedelta(hours=float(cfg.prediction.horizon))

    try:
        response_data = await services.run_optimization_cycle(
            start=start,
            end=end,
            cfg=cfg,
            raw_yaml=raw_yaml,
            battery_soc_overrides=req.battery_soc or None,
            initial_modes_overrides=req.initial_modes or None,
            solver_opts_overrides=req.solver_opts,
            validate_with_simulation=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(response_data)


@router.get("/optimize/status")
async def optimize_status() -> OptimizeStatusResponse:
    """Return cache status and metadata (not the full solution)."""
    cached = services.get_cached_solution()
    if cached is None:
        return OptimizeStatusResponse(has_cache=False, age_s=None, ttl_s=state.SOLUTION_CACHE_TTL_S)

    age_s = (
        (datetime.now(timezone.utc) - state.solution_cache_ts).total_seconds()
        if state.solution_cache_ts is not None
        else None
    )
    solved_at = state.solution_cache_ts.isoformat() if state.solution_cache_ts is not None else None

    return OptimizeStatusResponse(
        has_cache=True, age_s=age_s, ttl_s=state.SOLUTION_CACHE_TTL_S, solved_at=solved_at
    )


@router.get("/optimize")
async def get_cached_optimize() -> Response:
    """Return the cached optimization result if available.

    Returns 204 (No Content) if no cache exists or cache is stale.
    """
    cached = services.get_cached_solution()
    if cached is None:
        return Response(status_code=204)
    return JSONResponse(cached)
