"""Prediction endpoints.

POST /api/predictions/fetch  – fetch all forecast channels, return Plotly charts.
GET  /api/predictions/status – cache status (age, TTL, forecast_from).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from structlog import get_logger

import GridPythia.server.state as state
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
                start=datetime.now(tz=tz),
                hours=float(cfg.prediction.horizon),
                dt_hours=float(cfg.prediction.dt_hours),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("prediction_retry_fetch_error", error=str(exc))
            continue

        still_failing = [pid for pid in remaining if pid in errors]
        recovered = [pid for pid in remaining if pid not in errors]

        if recovered:
            forecast_from: datetime | None = None
            if setup.electricprice is not None:
                forecast_from = setup.electricprice.last_real_ts
            services.set_cached_pdata(pdata, forecast_from)
            logger.info("prediction_retry_recovered", recovered=recovered)

        remaining = still_failing
        if not remaining:
            logger.info("prediction_retry_all_recovered")
            return

    if remaining:
        logger.warning("prediction_retry_exhausted", still_failing=remaining)


@router.get("/status", response_model=PredictionsStatusResponse)
async def predictions_status() -> PredictionsStatusResponse:
    """Return the current state of the server-side prediction cache."""
    if state.pdata_cache is None or state.pdata_cache_ts is None:
        return PredictionsStatusResponse(has_cache=False, ttl_s=state.PDATA_CACHE_TTL_S)
    age = (datetime.now(timezone.utc) - state.pdata_cache_ts).total_seconds()
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

    Uses partial fetching: if one provider fails the others are still returned
    and a background retry task is started for the failed ones (exponential
    backoff: 60 s, 120 s, 240 s, 480 s, 900 s).

    Response schema::

        {
          "charts": {
            "tab-elecprice": <plotly_json>,
            ...
          },
          "from_cache": false,
          "errors": {"EnergyCharts": "..."}   // only present when partial failure
        }
    """
    # ── Fast path: serve from cache ───────────────────────────────────
    cached = services.get_cached_pdata()
    if cached is not None:
        pdata, forecast_from = cached
        pdata = services.apply_appliance_loads(pdata)
        charts = services.make_prediction_figures(pdata, forecast_from)
        logger.info("predictions_served_from_cache", charts=list(charts.keys()))
        return JSONResponse({"charts": charts, "from_cache": True})

    stale_cached = services.get_cached_pdata_any_age()

    def _stale_fallback(reason: str) -> JSONResponse | None:
        if stale_cached is None:
            return None
        pdata, forecast_from = stale_cached
        pdata = services.apply_appliance_loads(pdata)
        charts = services.make_prediction_figures(pdata, forecast_from)
        age_s = services.get_cached_pdata_age_s()
        logger.warning(
            "predictions_stale_cache_fallback",
            reason=reason,
            cache_age_s=(round(age_s, 1) if age_s is not None else None),
        )
        return JSONResponse(
            {
                "charts": charts,
                "from_cache": True,
                "stale_cache": True,
                "stale_age_s": (round(age_s, 1) if age_s is not None else None),
            }
        )

    # ── Slow path: fetch via persistent provider singletons ───────────
    try:
        cfg, raw_yaml = services.load_config()
    except Exception as exc:
        fallback = _stale_fallback(f"config_error: {exc}")
        if fallback is not None:
            return fallback
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    try:
        tz = ZoneInfo(req.timezone)
    except ZoneInfoNotFoundError:
        logger.warning("unknown_timezone", tz=req.timezone)
        tz = ZoneInfo("UTC")

    try:
        setup = services.get_providers(cfg, raw_yaml)
    except Exception as exc:
        fallback = _stale_fallback(f"provider_build_error: {exc}")
        if fallback is not None:
            return fallback
        raise HTTPException(status_code=500, detail=f"Provider build error: {exc}") from exc

    pred = Prediction(setup)
    try:
        pdata, errors = await pred.fetch_partial(
            start=datetime.now(tz=tz),
            hours=float(cfg.prediction.horizon),
            dt_hours=float(cfg.prediction.dt_hours),
        )
    except Exception as exc:  # noqa: BLE001
        fallback = _stale_fallback(f"prediction_fetch_error: {exc}")
        if fallback is not None:
            return fallback
        raise HTTPException(status_code=502, detail=f"Prediction fetch failed: {exc}") from exc

    # All providers failed → load_wh would be zeros; treat as hard error only
    # when *all* core providers failed (no useful data at all).
    core_failed = {"electricprice", "feedintariff", "load"}
    if errors.keys() >= core_failed:
        fallback = _stale_fallback(f"all_core_failed: {errors}")
        if fallback is not None:
            return fallback
        raise HTTPException(
            status_code=502,
            detail=f"All core prediction providers failed: {errors}",
        )

    forecast_from: datetime | None = None
    if setup.electricprice is not None:
        forecast_from = setup.electricprice.last_real_ts

    services.set_cached_pdata(pdata, forecast_from)
    pdata = services.apply_appliance_loads(pdata)
    charts = services.make_prediction_figures(pdata, forecast_from)

    await state.ws_hub.broadcast(
        {
            "type": "predictions_updated",
            "payload": {
                "charts": charts,
                "from_cache": False,
                "errors": errors,
            },
        }
    )

    # ── Start background retry for any failed providers ───────────────
    if errors:
        failed_ids = list(errors.keys())
        logger.warning("predictions_partial_failure", failed=failed_ids)
        # Cancel any previously running retry task
        if state._retry_task is not None and not state._retry_task.done():
            state._retry_task.cancel()
        state._retry_task = asyncio.create_task(
            _retry_failed_providers(setup, cfg, tz, failed_ids),
            name="prediction_retry",
        )

    logger.info("predictions_fetched", charts=list(charts.keys()), errors=list(errors.keys()))
    response: dict = {"charts": charts, "from_cache": False}
    if errors:
        response["errors"] = errors
    return JSONResponse(response)
