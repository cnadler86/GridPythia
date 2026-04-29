"""Server-side periodic scheduler for prediction refresh and optimization.

Runs independently from the browser dashboard so plans keep refreshing and
publishing even when no client window is open.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.coordination import next_optimization_slot
from GridPythia.server import services
from GridPythia.server.models import FetchRequest, OptimizeRequest
from GridPythia.server.routers.optimization import optimize
from GridPythia.server.routers.predictions import fetch_predictions

logger = get_logger(__name__)

# Exponential-backoff delays for the startup fetch (seconds).
# First entry is 0 → immediate first attempt; subsequent entries add wait time.
_STARTUP_BACKOFF_DELAYS_S = [0, 5, 10, 20, 40, 60, 120, 240, 480]


async def run_startup_fetch() -> None:
    """Fetch all prediction channels on server startup with exponential backoff.

    Runs immediately on first attempt.  If the fetch fails (all core providers
    down) or returns only partial data, it retries using the backoff sequence
    above until either a complete (error-free) result is obtained or all
    attempts are exhausted.

    A *partial* result (some providers failed) is stored in the cache with the
    short fallback TTL so the periodic scheduler can replace it quickly on the
    next successful cycle.  A *complete* result uses the normal TTL.

    Stops early if another task (e.g. a manual /api/predictions/fetch) already
    populated a fresh, non-fallback cache.
    """
    from GridPythia.prediction.prediction import Prediction

    for attempt, delay in enumerate(_STARTUP_BACKOFF_DELAYS_S):
        if delay > 0:
            logger.info("startup_fetch_retry_wait", delay_s=delay, attempt=attempt)
            await asyncio.sleep(delay)

        # Yield if a full (non-fallback) cache was set by another task in the meantime.
        if services.get_cached_pdata() is not None and not state.pdata_is_fallback:
            logger.info("startup_fetch_superseded_by_fresh_cache", attempt=attempt)
            return

        try:
            cfg, raw_yaml = services.load_config()
        except Exception as exc:
            logger.warning("startup_fetch_config_error", attempt=attempt, error=str(exc))
            continue

        try:
            tz = ZoneInfo(cfg.server.timezone or "UTC")
        except Exception:
            tz = ZoneInfo("UTC")

        try:
            setup = services.get_providers(cfg, raw_yaml)
        except Exception as exc:
            logger.warning("startup_fetch_provider_error", attempt=attempt, error=str(exc))
            continue

        pred = Prediction(setup)
        try:
            pdata, errors = await pred.fetch_partial(
                start=services.snap_to_dt_grid(datetime.now(tz=tz), float(cfg.prediction.dt_hours)),
                hours=float(cfg.prediction.horizon),
                dt_hours=float(cfg.prediction.dt_hours),
            )
        except Exception as exc:
            logger.warning("startup_fetch_failed", attempt=attempt, error=str(exc))
            continue

        # If all core providers failed there is nothing useful to cache – retry.
        core_failed = {"electricprice", "feedintariff", "load"}
        if errors.keys() >= core_failed:
            logger.warning(
                "startup_fetch_all_core_failed",
                attempt=attempt,
                errors=list(errors.keys()),
            )
            continue

        forecast_from = getattr(setup.electricprice, "last_real_ts", None)
        is_fallback = bool(errors)
        services.set_cached_pdata(pdata, forecast_from, is_fallback=is_fallback)

        if not errors:
            logger.info("startup_fetch_success", attempt=attempt)
            return

        logger.warning(
            "startup_fetch_partial_success",
            attempt=attempt,
            failed=list(errors.keys()),
        )

    logger.error(
        "startup_fetch_exhausted_retries",
        max_attempts=len(_STARTUP_BACKOFF_DELAYS_S),
    )


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
