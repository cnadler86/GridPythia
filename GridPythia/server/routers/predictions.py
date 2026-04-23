"""Prediction endpoints.

POST /api/predictions/fetch  – fetch all forecast channels, return Plotly charts.
GET  /api/predictions/status – cache status (age, TTL, forecast_from).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.prediction.electricprice.energycharts import ElecPriceEnergyCharts
from GridPythia.prediction.prediction import Prediction
from GridPythia.server import services
from GridPythia.server.models import FetchRequest, PredictionsStatusResponse

logger = get_logger(__name__)

router = APIRouter(tags=["predictions"])


@router.get("/status", response_model=PredictionsStatusResponse)
async def predictions_status() -> PredictionsStatusResponse:
    """Return the current state of the server-side prediction cache."""
    if state.pdata_cache is None or state.pdata_cache_ts is None:
        return PredictionsStatusResponse(has_cache=False, ttl_s=state.PDATA_CACHE_TTL_S)
    age = (datetime.now() - state.pdata_cache_ts).total_seconds()
    return PredictionsStatusResponse(
        has_cache=age < state.PDATA_CACHE_TTL_S,
        age_s=round(age, 1),
        ttl_s=state.PDATA_CACHE_TTL_S,
        forecast_from=(
            state.pdata_forecast_from.isoformat() if state.pdata_forecast_from else None
        ),
    )


@router.post("/fetch")
async def fetch_predictions(req: FetchRequest) -> JSONResponse:
    """Fetch all prediction channels and return Plotly figure JSON.

    Response schema::

        {
          "charts": {
            "tab-elecprice": <plotly_json>,
            "tab-feedin":    <plotly_json>,
            "tab-load":      <plotly_json>,
            "tab-pv":        <plotly_json>,
            "tab-weather":   <plotly_json>   // only when weather is configured
          },
          "from_cache": true | false
        }

    The ``forecast_from`` shading (lavendel background in the electric-price
    chart) is derived from the last confirmed real timestamp of the EnergyCharts
    provider and is stored alongside the cached data.
    """
    # ── Fast path: serve from cache ───────────────────────────────────
    cached = services.get_cached_pdata()
    if cached is not None:
        pdata, forecast_from = cached
        charts = services.make_prediction_figures(pdata, forecast_from)
        logger.info("predictions_served_from_cache", charts=list(charts.keys()))
        return JSONResponse({"charts": charts, "from_cache": True})

    # ── Slow path: fetch via persistent provider singletons ───────────
    try:
        cfg, raw_yaml = services.load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    try:
        tz = ZoneInfo(req.timezone)
    except ZoneInfoNotFoundError:
        logger.warning("unknown_timezone", tz=req.timezone)
        tz = ZoneInfo("UTC")

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

    forecast_from: datetime | None = None
    if isinstance(setup.electricprice, ElecPriceEnergyCharts):
        forecast_from = setup.electricprice.last_real_ts

    services.set_cached_pdata(pdata, forecast_from)
    charts = services.make_prediction_figures(pdata, forecast_from)
    logger.info("predictions_fetched", charts=list(charts.keys()))
    return JSONResponse({"charts": charts, "from_cache": False})
