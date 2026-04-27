"""Appliance load forecast endpoints.

GET    /api/appliance_load                    – list all active appliance forecasts
POST   /api/appliance_load/{appliance_id}     – submit / update forecast for one appliance
DELETE /api/appliance_load/{appliance_id}     – clear forecast for one appliance

The payload format is intentionally identical to the MQTT retained message so
that dishwasher-style controllers can use either transport interchangeably.

**MQTT equivalent**

    Topic:   ``{prefix}/appliance_load/forecast/{appliance_id}``
    Payload: same JSON array ``[{"time": "<iso>", "load_wh": 150.0}, ...]``
             Empty string / empty array clears the forecast.

**HTTP payload** (``POST``)::

    {
      "slots": [
        {"time": "2026-04-27T14:00:00+02:00", "load_wh": 150.0},
        ...
      ]
    }
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.server.models import (
    ApplianceForecastInfo,
    ApplianceForecastRequest,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/appliance_load", tags=["appliance_load"])


@router.get("", response_model=list[ApplianceForecastInfo])
async def list_appliance_forecasts() -> list[ApplianceForecastInfo]:
    """Return a summary of all currently active appliance load forecasts."""
    result: list[ApplianceForecastInfo] = []
    for appliance_id, slots in state.appliance_forecasts.items():
        times = [s.get("time") for s in slots if s.get("time")]
        result.append(
            ApplianceForecastInfo(
                appliance_id=appliance_id,
                slot_count=len(slots),
                first_slot=times[0] if times else None,
                last_slot=times[-1] if times else None,
            )
        )
    return result


@router.post("/{appliance_id}")
async def update_appliance_forecast(
    appliance_id: str,
    req: ApplianceForecastRequest,
) -> JSONResponse:
    """Submit or replace the load forecast for one appliance.

    The forecast is stored in memory and injected into the next optimisation
    and prediction fetch.  Slots in the past are silently ignored during
    solver ingestion but are stored as-is here.

    Pass an empty ``slots`` list to clear the forecast (equivalent to
    ``DELETE``).
    """
    if not req.slots:
        state.appliance_forecasts.pop(appliance_id, None)
        logger.info("appliance_forecast_cleared_via_post", appliance_id=appliance_id)
        return JSONResponse({"appliance_id": appliance_id, "slot_count": 0, "status": "cleared"})

    # Store raw dicts compatible with the MQTT payload format
    raw_slots = [{"time": s.time, "load_wh": s.load_wh} for s in req.slots]
    state.appliance_forecasts[appliance_id] = raw_slots
    logger.info(
        "appliance_forecast_updated_via_http",
        appliance_id=appliance_id,
        slots=len(raw_slots),
    )
    return JSONResponse(
        {"appliance_id": appliance_id, "slot_count": len(raw_slots), "status": "updated"}
    )


@router.delete("/{appliance_id}")
async def clear_appliance_forecast(appliance_id: str) -> JSONResponse:
    """Remove the active load forecast for one appliance."""
    existed = appliance_id in state.appliance_forecasts
    state.appliance_forecasts.pop(appliance_id, None)
    if not existed:
        raise HTTPException(
            status_code=404,
            detail=f"No active forecast found for appliance '{appliance_id}'",
        )
    logger.info("appliance_forecast_deleted", appliance_id=appliance_id)
    return JSONResponse({"appliance_id": appliance_id, "status": "deleted"})
