"""FastAPI router for the load learning plugin.

Endpoints
---------
* POST /ingest/power                       – Submit a power measurement (W).
* POST /ingest/energy                      – Submit an energy measurement (Wh).
* GET  /stats                              – Learning statistics.
* GET  /vacation                           – Get vacation mode status.
* POST /vacation/enable                    – Enable vacation mode.
* POST /vacation/disable                   – Disable vacation mode.
* POST /appliances/{id}/active             – Mark appliance as started.
* POST /appliances/{id}/inactive           – Mark appliance as stopped.
* POST /appliances/{id}/scheduled          – Announce a scheduled start time.
* DELETE /appliances/{id}/announcement     – Clear a pending announcement.
* POST /maintenance                        – Trigger manual TSDB maintenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from GridPythia.server.plugins.load_learning.service import LoadLearningService


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------


class PowerMeasurement(BaseModel):
    watts: float = Field(..., description="Instantaneous power in watts")
    timestamp: float | None = Field(None, description="Unix timestamp (optional, defaults to now)")


class EnergyMeasurement(BaseModel):
    wh: float = Field(..., description="Energy in watt-hours")
    duration_h: float = Field(..., gt=0, description="Measurement duration in hours")
    timestamp: float | None = Field(None, description="Unix timestamp of period start")


class ApplianceInactiveRequest(BaseModel):
    avg_power_w: float = Field(default=0.0, ge=0, description="Average power during the run (W)")
    timestamp: float | None = None


class ApplianceScheduledRequest(BaseModel):
    scheduled_start_ts: float = Field(..., description="Planned start as Unix timestamp")


class VacationStatus(BaseModel):
    active: bool
    message: str


class MaintenanceResponse(BaseModel):
    compacted: int = 0
    deleted: int = 0


# ------------------------------------------------------------------
# Router factory
# ------------------------------------------------------------------


def create_router(service: LoadLearningService) -> APIRouter:
    """Create the load learning API router bound to a service instance."""
    router = APIRouter(tags=["load-learning"])

    @router.post("/ingest/power", status_code=202)
    async def ingest_power(measurement: PowerMeasurement) -> dict:
        """Ingest an instantaneous power measurement (W)."""
        service.ingest_power(measurement.watts, measurement.timestamp)
        return {"status": "accepted"}

    @router.post("/ingest/energy", status_code=202)
    async def ingest_energy(measurement: EnergyMeasurement) -> dict:
        """Ingest an energy measurement (Wh over a duration)."""
        service.ingest_energy(measurement.wh, measurement.duration_h, measurement.timestamp)
        return {"status": "accepted"}

    @router.get("/stats")
    async def get_stats() -> dict:
        """Return learning statistics and provider state."""
        return service.get_stats()

    # ------------------------------------------------------------------
    # Vacation mode
    # ------------------------------------------------------------------

    @router.get("/vacation", response_model=VacationStatus)
    async def get_vacation() -> VacationStatus:
        """Get current vacation mode status (runtime flag, not persisted)."""
        return VacationStatus(
            active=service.vacation_mode,
            message="Vacation mode active" if service.vacation_mode else "Normal mode",
        )

    @router.post("/vacation/enable", response_model=VacationStatus)
    async def enable_vacation() -> VacationStatus:
        """Activate vacation mode.

        forecast uses 10th-percentile baseline;
        ingested data is discarded (not used for learning).
        """
        service.vacation_mode = True
        return VacationStatus(active=True, message="Vacation mode enabled")

    @router.post("/vacation/disable", response_model=VacationStatus)
    async def disable_vacation() -> VacationStatus:
        """Deactivate vacation mode – resume normal adaptive forecasting."""
        service.vacation_mode = False
        return VacationStatus(active=False, message="Vacation mode disabled")

    # ------------------------------------------------------------------
    # Appliance tracker
    # ------------------------------------------------------------------

    @router.post("/appliances/{appliance_id}/active", status_code=202)
    async def appliance_active(appliance_id: str, timestamp: float | None = None) -> dict:
        """Notify that *appliance_id* just started running.

        Use this when an appliance controller reports activation so the
        learner can correctly attribute the additional load and subtract it
        from the base-load metric.
        """
        service.notify_appliance_active(appliance_id, timestamp)
        return {"status": "recorded", "appliance_id": appliance_id}

    @router.post("/appliances/{appliance_id}/inactive", status_code=202)
    async def appliance_inactive(appliance_id: str, body: ApplianceInactiveRequest) -> dict:
        """Notify that *appliance_id* finished its run."""
        service.notify_appliance_inactive(appliance_id, body.avg_power_w, body.timestamp)
        return {"status": "recorded", "appliance_id": appliance_id}

    @router.post("/appliances/{appliance_id}/scheduled", status_code=202)
    async def appliance_scheduled(appliance_id: str, body: ApplianceScheduledRequest) -> dict:
        """Record a scheduled future start time for *appliance_id*.

        When the learner has previously predicted a different run time,
        its prediction is superseded by this announcement.  The optimizer's
        appliance forecast (submitted via ``/api/appliance_load``) is used
        for the actual optimization; this endpoint is only for updating the
        learning system's expectations.
        """
        service.notify_appliance_scheduled(appliance_id, body.scheduled_start_ts)
        return {"status": "recorded", "appliance_id": appliance_id}

    @router.delete("/appliances/{appliance_id}/announcement", status_code=200)
    async def clear_announcement(appliance_id: str) -> dict:
        """Clear a pending scheduled announcement for *appliance_id*."""
        if service.provider is not None:
            service.provider.appliance_tracker.clear_announcement(appliance_id)
        return {"status": "cleared", "appliance_id": appliance_id}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @router.post("/maintenance", response_model=MaintenanceResponse)
    async def run_maintenance() -> MaintenanceResponse:
        """Manually flush accumulators and run TSDB compaction / retention."""
        if service.provider is None:
            raise HTTPException(status_code=503, detail="Provider not initialized")
        stats = service.provider.run_maintenance()
        return MaintenanceResponse(**stats)

    return router
