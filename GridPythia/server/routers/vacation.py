"""Vacation-mode endpoint.

GET  /api/vacation_mode  – return current vacation-mode state.
PUT  /api/vacation_mode  – enable or disable vacation mode.

When vacation mode is active the load forecast uses the minimum (vacation)
profile from the configured CSV instead of the regular weekday / weekend
profile.  This reflects reduced household consumption during periods when
nobody is home.

Toggling vacation mode invalidates the prediction result cache so that the
next fetch picks up the new load profile immediately.
"""

from __future__ import annotations

from fastapi import APIRouter
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.server.models import VacationModeRequest, VacationModeResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/vacation_mode", tags=["vacation_mode"])


@router.get("", response_model=VacationModeResponse)
async def get_vacation_mode() -> VacationModeResponse:
    """Return the current vacation-mode state."""
    return VacationModeResponse(
        enabled=state.vacation_mode,
        description="vacation profile active" if state.vacation_mode else "normal profile active",
    )


@router.put("", response_model=VacationModeResponse)
async def set_vacation_mode(req: VacationModeRequest) -> VacationModeResponse:
    """Enable or disable vacation mode.

    When the value changes the prediction result cache is invalidated so that
    the next optimization or prediction fetch uses the updated load profile.
    """
    changed = state.vacation_mode != req.enabled
    state.vacation_mode = req.enabled

    if changed:
        logger.info("vacation_mode_changed", enabled=req.enabled)

    return VacationModeResponse(
        enabled=state.vacation_mode,
        description="vacation profile active" if state.vacation_mode else "normal profile active",
    )
