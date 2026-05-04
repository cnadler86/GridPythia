"""Server-side periodic scheduler for prediction refresh and optimization.

Runs independently from the browser dashboard so plans keep refreshing and
publishing even when no client window is open.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
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

_NEXT_SLOT_EPSILON_S = 1.1

# Exponential-backoff delays for the startup fetch (seconds).
# First entry is 0 → immediate first attempt; subsequent entries add wait time.
_STARTUP_BACKOFF_DELAYS_S = [0, 5, 10, 20, 40, 60, 120, 240, 480]


def _solver_time_limit_seconds(cfg) -> float:
    """Return the configured solver time limit used to size the scheduler lead."""
    raw = cfg.optimization.solver.solver_opts.get("time_limit", 30.0)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 30.0


def _adaptive_dispatch_buffer_seconds(cfg, publish_lateness_s: float = 0.0) -> float:
    """Return the current publish safety buffer.

    The base buffer comes from config. If the previous cycle finished after its
    target dispatch slot, the lateness delta is added on top, capped to the
    configured maximum.
    """
    initial_buffer_s = float(cfg.server.scheduler.dispatch_buffer_seconds)
    max_buffer_s = float(cfg.server.scheduler.dispatch_buffer_max_seconds)
    lateness_s = max(0.0, float(publish_lateness_s))
    return min(max_buffer_s, initial_buffer_s + lateness_s)


def _scheduler_lead_seconds(cfg, publish_lateness_s: float = 0.0) -> float:
    """Return how long before the dispatch slot the scheduler should fire."""
    return _solver_time_limit_seconds(cfg) + _adaptive_dispatch_buffer_seconds(
        cfg, publish_lateness_s
    )


def _next_future_dispatch_slot(reference: datetime, interval_minutes: int) -> datetime:
    """Return the first dispatch slot strictly after *reference*."""
    return next_optimization_slot(
        reference + timedelta(seconds=_NEXT_SLOT_EPSILON_S),
        interval_minutes,
    )


def _next_scheduler_trigger(
    now: datetime,
    cfg,
    publish_lateness_s: float = 0.0,
    last_dispatch_slot: datetime | None = None,
) -> tuple[datetime, datetime, float]:
    """Return ``(dispatch_slot, run_at, lead_s)`` for the next scheduler cycle."""
    interval_min = int(cfg.server.scheduler.optimization_interval_minutes)
    reference = last_dispatch_slot if last_dispatch_slot is not None else now
    dispatch_slot = _next_future_dispatch_slot(reference, interval_min)
    lead_s = _scheduler_lead_seconds(cfg, publish_lateness_s)
    run_at = dispatch_slot - timedelta(seconds=lead_s)
    return dispatch_slot, run_at, lead_s


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
    1) refresh prediction cache if ``prediction_refresh_minutes`` elapsed
       (fast-path returns cached data otherwise – avoids shifting the
       prediction window on every 15-min cycle and thus prevents the
       "plan shift" bug where the optimal charging slot drifts by one slot
       whenever the prediction window advances)
    2) run optimization (publishes plans via MQTT when enabled)
    """
    publish_lateness_s = 0.0
    last_dispatch_slot: datetime | None = None
    last_prediction_refresh_slot: datetime | None = None  # tracks when we last fetched fresh data

    while True:
        try:
            cfg, _ = services.load_config()
            server_tz = cfg.server.timezone or "UTC"
            now = datetime.now(tz=timezone.utc)
            slot, run_at, lead_s = _next_scheduler_trigger(
                now,
                cfg,
                publish_lateness_s=publish_lateness_s,
                last_dispatch_slot=last_dispatch_slot,
            )
            sleep_s = max(0.0, (run_at - now).total_seconds())
            logger.debug(
                "scheduler_waiting",
                interval_min=int(cfg.server.scheduler.optimization_interval_minutes),
                next_slot=slot.isoformat(),
                run_at=run_at.isoformat(),
                lead_s=round(lead_s, 2),
                publish_buffer_s=round(
                    _adaptive_dispatch_buffer_seconds(cfg, publish_lateness_s),
                    2,
                ),
                sleep_s=round(sleep_s, 2),
            )
            if sleep_s > 0.0:
                await asyncio.sleep(sleep_s)

            last_dispatch_slot = slot

            # ── Decide whether to refresh predictions ─────────────────────
            # Only re-fetch from providers when prediction_refresh_minutes
            # have elapsed since the last refresh.  Skipping intermediate
            # fetches is the key fix for the "15-minute plan shift":
            # without this guard every 15-min cycle fetches new prediction
            # data starting from *now* (e.g. 11:15) instead of the cached
            # data starting from the previous refresh (e.g. 11:00), which
            # shifts the optimizer's index-based optimal charging step by
            # one slot on every cycle.
            refresh_interval_s = float(cfg.server.scheduler.prediction_refresh_minutes) * 60.0
            if last_prediction_refresh_slot is None:
                seconds_since_refresh = float("inf")
            else:
                seconds_since_refresh = (slot - last_prediction_refresh_slot).total_seconds()

            needs_prediction_refresh = seconds_since_refresh >= refresh_interval_s
            cached_pdata_info = services.get_cached_pdata_any_age()
            if cached_pdata_info is None:
                needs_prediction_refresh = True  # always fetch when cache is empty

            logger.info(
                "scheduler_slot_optimization_starting",
                slot=slot.isoformat(),
                interval_min=int(cfg.server.scheduler.optimization_interval_minutes),
                lead_s=round(lead_s, 2),
                needs_prediction_refresh=needs_prediction_refresh,
                seconds_since_last_refresh=(
                    round(seconds_since_refresh, 0)
                    if seconds_since_refresh != float("inf")
                    else None
                ),
                refresh_interval_s=refresh_interval_s,
                prediction_cache_start=(
                    cached_pdata_info[0].timestamps[0].isoformat()
                    if cached_pdata_info is not None
                    else None
                ),
            )

            if needs_prediction_refresh:
                try:
                    await fetch_predictions(FetchRequest(timezone=server_tz))
                    last_prediction_refresh_slot = slot
                    # Log new cache start for observability
                    new_pdata_info = services.get_cached_pdata_any_age()
                    if new_pdata_info is not None:
                        logger.info(
                            "scheduler_prediction_refreshed",
                            slot=slot.isoformat(),
                            prediction_start=new_pdata_info[0].timestamps[0].isoformat(),
                            prediction_steps=new_pdata_info[0].steps,
                        )
                except HTTPException as exc:
                    logger.warning(
                        "scheduler_prediction_refresh_failed",
                        status_code=exc.status_code,
                        detail=str(exc.detail),
                    )
                    continue
            else:
                logger.debug(
                    "scheduler_prediction_cache_reused",
                    slot=slot.isoformat(),
                    seconds_since_refresh=round(seconds_since_refresh, 0),
                    prediction_start=(
                        cached_pdata_info[0].timestamps[0].isoformat()
                        if cached_pdata_info is not None
                        else None
                    ),
                )

            try:
                # Pass prediction_start = dispatch slot so that optimize()
                # slices the cached prediction from exactly the dispatch slot
                # instead of using a freshly-fetched prediction starting at
                # snap(now).  This is the fix for the 15-minute slot-shift:
                # prices stay anchored to their absolute timestamps across
                # consecutive optimisation cycles.
                await optimize(
                    OptimizeRequest(
                        timezone=server_tz,
                        prediction_start=slot.isoformat(),
                    )
                )
            except HTTPException as exc:
                logger.warning(
                    "scheduler_optimization_failed",
                    status_code=exc.status_code,
                    detail=str(exc.detail),
                )
                continue

            completed_at = datetime.now(tz=timezone.utc)
            publish_lateness_s = max(0.0, (completed_at - slot).total_seconds())
            logger.info(
                "scheduler_cycle_complete",
                slot=slot.isoformat(),
                completed_at=completed_at.isoformat(),
                publish_lateness_s=round(publish_lateness_s, 2),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("scheduler_cycle_error", error=str(exc))
            await asyncio.sleep(5)
