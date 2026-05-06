"""Shared utilities for stitching inverter plan steps across slot boundaries.

Both the MQTT gateway and the optimization-cycle service need to prepend the
currently-active slot from a previously published plan when the newly solved
plan starts at the *next* slot boundary.  This module provides the canonical
dict-based implementation used by both.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import floor


def parse_step_timestamp(step: dict) -> datetime | None:
    """Parse a plan-step ``timestamp`` field, returning ``None`` for malformed entries."""
    raw_ts = step.get("timestamp")
    if not isinstance(raw_ts, str):
        return None
    try:
        return datetime.fromisoformat(raw_ts)
    except ValueError:
        return None


def current_slot_start(published_at: datetime, dt_hours: float) -> datetime:
    """Return the start timestamp of the slot containing *published_at*."""
    step_seconds = max(1.0, float(dt_hours) * 3600.0)
    slot_epoch = floor(published_at.timestamp() / step_seconds) * step_seconds
    return datetime.fromtimestamp(slot_epoch, tz=published_at.tzinfo or timezone.utc)


def stitch_current_slot_from_previous_plan(
    steps: list[dict],
    previous_steps: list[dict],
    *,
    published_at: datetime,
    dt_hours: float,
) -> list[dict]:
    """Prepend the active slot from *previous_steps* to *steps* when needed.

    If a newly solved plan starts at the next slot boundary because the solve
    ran shortly before dispatch, downstream consumers still need the currently
    active slot from the previously retained plan until the boundary is reached.

    Args:
        steps:          Raw plan steps (list of dicts) for a single device.
        previous_steps: Previously published/cached steps for the same device.
        published_at:   Timestamp of the publish or solve event.  Used to
                        determine the current slot via ``floor(t / dt_hours)``.
        dt_hours:       Slot duration in hours (e.g. 0.25 for 15-minute slots).

    Returns:
        ``steps`` unchanged when no stitching is needed, or a new list with
        the matching previous step prepended.
    """
    stitched_steps = [dict(step) for step in steps]
    if not stitched_steps or not previous_steps:
        return stitched_steps

    first_step_ts = parse_step_timestamp(stitched_steps[0])
    if first_step_ts is None or published_at >= first_step_ts:
        return stitched_steps

    slot = current_slot_start(published_at, dt_hours)
    if first_step_ts <= slot:
        return stitched_steps

    for prev_step in reversed(previous_steps):
        prev_step_ts = parse_step_timestamp(prev_step)
        if prev_step_ts is None:
            continue
        if prev_step_ts == slot:
            if parse_step_timestamp(stitched_steps[0]) == slot:
                return stitched_steps
            return [dict(prev_step), *stitched_steps]

    return stitched_steps
