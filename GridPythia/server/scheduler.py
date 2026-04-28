"""Server-side periodic scheduler for prediction refresh and optimization.

Runs independently from the browser dashboard so plans keep refreshing and
publishing even when no client window is open.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import HTTPException
from structlog import get_logger

from GridPythia.coordination import next_optimization_slot
from GridPythia.server import services
from GridPythia.server.models import FetchRequest, OptimizeRequest
from GridPythia.server.routers.optimization import optimize
from GridPythia.server.routers.predictions import fetch_predictions

logger = get_logger(__name__)


async def run_scheduler() -> None:
    """Run prediction+optimization cycles at configured slot boundaries.

    Flow per slot:
    1) refresh prediction cache (fast-path returns cached data)
    2) run optimization (publishes plans via MQTT when enabled)
    """
    while True:
        try:
            cfg, _ = services.load_config()
            interval_min = int(cfg.server.scheduler.optimization_interval_minutes)
            server_tz = cfg.server.timezone or "UTC"
            now = datetime.now(tz=timezone.utc)
            slot = next_optimization_slot(now, interval_min)
            sleep_s = max(0.5, (slot - now).total_seconds())
            logger.debug(
                "scheduler_waiting",
                interval_min=interval_min,
                next_slot=slot.isoformat(),
                sleep_s=round(sleep_s, 2),
            )
            await asyncio.sleep(sleep_s)

            logger.info("scheduler_slot_reached", slot=slot.isoformat(), interval_min=interval_min)

            try:
                await fetch_predictions(FetchRequest(timezone=server_tz))
            except HTTPException as exc:
                logger.warning(
                    "scheduler_prediction_refresh_failed",
                    status_code=exc.status_code,
                    detail=str(exc.detail),
                )
                continue

            try:
                await optimize(OptimizeRequest(timezone=server_tz))
            except HTTPException as exc:
                logger.warning(
                    "scheduler_optimization_failed",
                    status_code=exc.status_code,
                    detail=str(exc.detail),
                )
                continue

            logger.info("scheduler_cycle_complete", slot=slot.isoformat())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("scheduler_cycle_error", error=str(exc))
            await asyncio.sleep(5)
