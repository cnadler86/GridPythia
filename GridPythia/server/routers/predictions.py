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
                start=services.snap_to_dt_grid(datetime.now(tz=tz), float(cfg.prediction.dt_hours)),
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
        pdata, errors = await pred.fetch_partial(
            start=services.snap_to_dt_grid(datetime.now(tz=tz), float(cfg.prediction.dt_hours)),
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

    forecast_from: datetime | None = (
        setup.electricprice.last_real_ts if setup.electricprice is not None else None
    )
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
