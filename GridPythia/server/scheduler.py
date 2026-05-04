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

from GridPythia.coordination import next_optimization_slot
from GridPythia.server import services
from GridPythia.server.models import OptimizeRequest
from GridPythia.server.routers.optimization import optimize

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

    The goal is to prime the provider-internal caches so the first scheduled
    optimization does not block on a cold-start network fetch under tight timing.
    """
    from GridPythia.prediction.prediction import Prediction

    for attempt, delay in enumerate(_STARTUP_BACKOFF_DELAYS_S):
        if delay > 0:
            logger.info("startup_fetch_retry_wait", delay_s=delay, attempt=attempt)
            await asyncio.sleep(delay)

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
            _, errors = await pred.fetch_partial(
                start=services.snap_to_dt_grid(datetime.now(tz=tz), float(cfg.prediction.dt_hours)),
                hours=float(cfg.prediction.horizon),
                dt_hours=float(cfg.prediction.dt_hours),
            )
        except Exception as exc:
            logger.warning("startup_fetch_failed", attempt=attempt, error=str(exc))
            continue

        # If all core providers failed there is nothing useful – retry.
        core_failed = {"electricprice", "feedintariff", "load"}
        if errors.keys() >= core_failed:
            logger.warning(
                "startup_fetch_all_core_failed",
                attempt=attempt,
                errors=list(errors.keys()),
            )
            continue

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
    """Run optimization cycles at configured slot boundaries.

    Each slot: call optimize() with prediction_start=slot.  The optimize
    endpoint fetches directly from providers (providers own cache TTL via
    cache_ttl_hours) with dispatch_slot as start, eliminating the slot-shift bug.
    """
    publish_lateness_s = 0.0
    last_dispatch_slot: datetime | None = None

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

            logger.info(
                "scheduler_slot_optimization_starting",
                slot=slot.isoformat(),
                interval_min=int(cfg.server.scheduler.optimization_interval_minutes),
                lead_s=round(lead_s, 2),
            )

            try:
                # prediction_start = dispatch slot so optimize() anchors the
                # prediction window at the same start across consecutive cycles
                # and slices from this slot.  Providers own their cache TTL
                # (cache_ttl_hours) and decide whether to hit the network.
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
