"""Inverter status endpoints.

POST /api/inverters/{device_id}/status   – report current SoC and mode
GET  /api/inverters/status               – list all known inverter states
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.server import services
from GridPythia.server.models import InverterStatusRequest, InverterStatusResponse
from GridPythia.simulation.devices import InverterMode

logger = get_logger(__name__)

router = APIRouter(prefix="/inverters", tags=["inverters"])

_MODE_NAMES: dict[int, str] = {m.value: m.name for m in InverterMode}


@router.post("/{device_id}/status", response_model=InverterStatusResponse)
async def report_inverter_status(
    device_id: str, req: InverterStatusRequest
) -> InverterStatusResponse:
    """Report the current state-of-charge and mode for one inverter.

    Call this whenever the inverter publishes a new status (e.g. every minute
    from your MQTT bridge or home-automation system).  The server uses the most
    recent status to seed the battery SoC and initial mode for the next
    optimisation run.

    **Optimization is blocked** when any optimisable inverter's status is
    older than ``server.inverter_status_max_age_s`` (default 300 s).
    """
    # Validate device_id against configured inverters so typos are caught early.
    try:
        cfg, _ = services.load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    known_ids = {inv.device_id for inv in cfg.optimization.inverters}
    if device_id not in known_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown inverter '{device_id}'. Configured: {sorted(known_ids)}",
        )

    inv_state = state.coordinator.update_status(
        device_id=device_id,
        soc=req.soc,
        mode=req.mode,
    )

    max_age = cfg.server.inverter_status_max_age_s
    return InverterStatusResponse(
        device_id=inv_state.device_id,
        soc=inv_state.soc,
        mode=inv_state.mode.value,
        mode_name=inv_state.mode.name,
        reported_at=inv_state.reported_at.isoformat(),
        age_s=round(inv_state.age_s(), 1),
        is_fresh=inv_state.is_fresh(max_age),
    )


@router.get("/status", response_model=list[InverterStatusResponse])
async def get_all_inverter_status() -> list[InverterStatusResponse]:
    """Return the last-known status for all inverters that have reported.

    Inverters that have never reported are omitted.
    """
    try:
        cfg, _ = services.load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    max_age = cfg.server.inverter_status_max_age_s
    result = []
    for inv_state in state.coordinator.snapshot().values():
        result.append(
            InverterStatusResponse(
                device_id=inv_state.device_id,
                soc=inv_state.soc,
                mode=inv_state.mode.value,
                mode_name=inv_state.mode.name,
                reported_at=inv_state.reported_at.isoformat(),
                age_s=round(inv_state.age_s(), 1),
                is_fresh=inv_state.is_fresh(max_age),
            )
        )
    return result
