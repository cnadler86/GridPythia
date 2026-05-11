"""Prediction endpoints.

POST /api/predictions/fetch  – fetch all forecast channels, return Plotly charts.
GET  /api/predictions/status – cache status (age, TTL, forecast_from).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.prediction.base import floor_to_slot
from GridPythia.prediction.prediction import Prediction
from GridPythia.server import services
from GridPythia.server.models import FetchRequest, PredictionsStatusResponse

logger = get_logger(__name__)

router = APIRouter(tags=["predictions"])

# ── Exponential-backoff retry delays (seconds) ────────────────────────────
# Sequence: 60, 120, 240, 480, 900 (≈ 15 min maximum)
_RETRY_DELAYS_S = [60, 120, 240, 480, 900]


async def _retry_failed_providers(
    setup,
    cfg,
    tz: ZoneInfo,
    failed_provider_ids: list[str],
) -> None:
    """Background task: retry failed providers with exponential backoff.

    On each successful retry the shared prediction cache is updated with
    the new partial result merged into the existing cached data.
    """
    remaining = list(failed_provider_ids)
    attempt = 0

    while remaining and attempt < len(_RETRY_DELAYS_S):
        delay = _RETRY_DELAYS_S[attempt]
        attempt += 1
        logger.info(
            "prediction_retry_scheduled",
            providers=remaining,
            delay_s=delay,
            attempt=attempt,
        )
        await asyncio.sleep(delay)

        pred = Prediction(setup)
        try:
            pdata, errors = await pred.fetch_partial(
                start=floor_to_slot(datetime.now(tz=tz), float(cfg.prediction.dt_hours)),
                hours=float(cfg.prediction.horizon),
                dt_hours=float(cfg.prediction.dt_hours),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("prediction_retry_fetch_error", error=str(exc))
            continue

        still_failing = [pid for pid in remaining if pid in errors]
        recovered = [pid for pid in remaining if pid not in errors]

        if recovered:
            logger.info("prediction_retry_recovered", recovered=recovered)

        remaining = still_failing
        if not remaining:
            logger.info("prediction_retry_all_recovered")
            return

    if remaining:
        logger.warning("prediction_retry_exhausted", still_failing=remaining)


@router.get("/status", response_model=PredictionsStatusResponse)
async def predictions_status() -> PredictionsStatusResponse:
    """Return prediction provider metadata (last real data timestamp)."""
    providers = state.providers
    forecast_from: str | None = None
    if providers is not None and providers.electricprice is not None:
        lrt = providers.electricprice.last_real_ts
        forecast_from = lrt.isoformat() if lrt is not None else None
    return PredictionsStatusResponse(forecast_from=forecast_from)


@router.post("/fetch")
async def fetch_predictions(req: FetchRequest) -> JSONResponse:
    """Fetch all prediction channels and return Plotly figure JSON.

    Uses partial fetching: if one provider fails the others are still returned
    and a background retry task is started for the failed ones (exponential
    backoff: 60 s, 120 s, 240 s, 480 s, 900 s).

    Implements prediction result caching by timestamp slot to avoid redundant
    HTTP calls: if the same time slot is requested multiple times within a
    15-minute window, the cached result is returned without network fetch.

    When include_charts=False, skips all chart generation to save significant
    compute (~30-50% of latency).

    Response schema::

        {
          "charts": {
            "tab-elecprice": <plotly_json>,
            ...
          },
          "from_cache": false|true,
          "errors": {"EnergyCharts": "..."}   // only present when partial failure
        }
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

    try:
        setup = services.get_providers(cfg, raw_yaml)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Provider build error: {exc}") from exc

    # ── Compute start time slot ────────────────────────────────────────────
    pred = Prediction(setup)
    start_ts = floor_to_slot(datetime.now(tz=tz), float(cfg.prediction.dt_hours))

    # ── Check prediction result cache before fetching ──────────────────────
    cfg_mtime = services._config_mtime_for_singletons()
    cache_key = services._prediction_cache_key(start_ts, cfg_mtime)
    cached_result = services._prediction_cache_get(cache_key)

    if cached_result is not None:
        logger.info("prediction_cache_hit", cache_key=cache_key)
        pdata = cached_result["pdata"]
        errors = cached_result["errors"]
        forecast_from = cached_result["forecast_from"]
        from_cache = True
    else:
        # Cache miss: fetch from all providers
        logger.info("prediction_cache_miss", cache_key=cache_key)
        try:
            pdata, errors = await pred.fetch_partial(
                start=start_ts,
                hours=float(cfg.prediction.horizon),
                dt_hours=float(cfg.prediction.dt_hours),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Prediction fetch failed: {exc}") from exc

        # Hard error only when *all* core providers failed (no useful data at all).
        core_failed = {"electricprice", "feedintariff", "load"}
        if errors.keys() >= core_failed:
            raise HTTPException(
                status_code=502,
                detail=f"All core prediction providers failed: {errors}",
            )

        forecast_from = (
            setup.electricprice.last_real_ts if setup.electricprice is not None else None
        )

        # Cache the fetched result for future requests in this time slot
        cache_data = {
            "pdata": pdata,
            "errors": errors,
            "forecast_from": forecast_from,
        }
        services._prediction_cache_put(cache_key, cache_data)
        from_cache = False

    # Keep latest prediction snapshot up-to-date for lazy chart loading,
    # even when charts are not requested in this call.
    pdata = services.apply_appliance_loads(pdata)
    pred_scope = services.prediction_chart_scope(cfg, pdata, forecast_from)

    state.latest_prediction_data = pdata
    state.latest_prediction_forecast_from = forecast_from
    state.latest_prediction_chart_scope = pred_scope

    await state.ws_hub.broadcast(
        {
            "type": "predictions_updated",
            "payload": {
                "chart_scope": pred_scope,
                "from_cache": from_cache,
                "errors": errors,
            },
        }
    )

    # Start background retry for failed providers regardless of chart mode.
    if errors:
        failed_ids = list(errors.keys())
        logger.warning("predictions_partial_failure", failed=failed_ids)
        if state._retry_task is not None and not state._retry_task.done():
            state._retry_task.cancel()
        state._retry_task = asyncio.create_task(
            _retry_failed_providers(setup, cfg, tz, failed_ids),
            name="prediction_retry",
        )

    # ── Early exit if no charts needed: skip all generation work ───────────
    if not req.include_charts:
        # Return minimal response without chart generation/serialization
        response: dict = {
            "from_cache": from_cache,
            "chart_scope": pred_scope,
            "summary": {
                "steps": pdata.steps,
                "start": pdata.timestamps[0].isoformat() if pdata.timestamps else None,
                "end": pdata.timestamps[-1].isoformat() if pdata.timestamps else None,
                "dt_hours": pdata.dt_hours,
                "forecast_from": forecast_from.isoformat() if forecast_from is not None else None,
            },
        }
        if errors:
            response["errors"] = errors
        return JSONResponse(response)

    # ── Generate charts only when include_charts=True ─────────────────────
    response: dict = {
        "from_cache": from_cache,
        "chart_scope": pred_scope,
        "summary": {
            "steps": pdata.steps,
            "start": pdata.timestamps[0].isoformat() if pdata.timestamps else None,
            "end": pdata.timestamps[-1].isoformat() if pdata.timestamps else None,
            "dt_hours": pdata.dt_hours,
            "forecast_from": forecast_from.isoformat() if forecast_from is not None else None,
        },
    }
    response["charts"] = services.make_prediction_figures(
        pdata,
        forecast_from,
        visible_tabs=services.visible_prediction_tabs(cfg, raw_yaml),
    )
    if errors:
        response["errors"] = errors
    return JSONResponse(response)


@router.get("/charts/{tab_id}")
async def fetch_prediction_chart(tab_id: str) -> JSONResponse:
    """Return a single prediction chart for lazy dashboard tab loading."""
    try:
        cfg, raw_yaml = services.load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    if tab_id not in services.visible_prediction_tabs(cfg, raw_yaml):
        raise HTTPException(
            status_code=404, detail=f"Unknown or disabled prediction tab '{tab_id}'"
        )

    pdata = state.latest_prediction_data
    if pdata is None:
        raise HTTPException(status_code=409, detail="No prediction snapshot available yet")

    forecast_from = state.latest_prediction_forecast_from
    scope = state.latest_prediction_chart_scope or services.prediction_chart_scope(
        cfg,
        pdata,
        forecast_from,
    )
    chart = services.get_or_build_prediction_chart(
        tab_id=tab_id,
        cfg=cfg,
        pdata=pdata,
        forecast_from=forecast_from,
        scope=scope,
    )
    if chart is None:
        raise HTTPException(status_code=404, detail=f"Chart not available for tab '{tab_id}'")
    return JSONResponse({"tab_id": tab_id, "chart_scope": scope, "chart": chart})
