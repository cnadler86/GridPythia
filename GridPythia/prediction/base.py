"""Core utilities for the prediction framework."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timedelta
from math import floor

import numpy as np


def make_timestamps(start: datetime, hours: float, dt_hours: float) -> list[datetime]:
    """Create a uniformly-spaced list of datetime values.

    Args:
        start:     First timestamp.
        hours:     Total window length in hours.
        dt_hours:  Step width in hours.

    Returns:
        ``list[datetime]`` with length ``max(1, round(hours / dt_hours))``.
    """
    if dt_hours <= 0:
        raise ValueError(f"dt_hours must be > 0, got {dt_hours}")

    n = max(1, round(hours / dt_hours))
    step = timedelta(hours=dt_hours)
    return [start + i * step for i in range(n)]


def floor_to_slot(dt: datetime, dt_hours: float) -> datetime:
    """Return the largest slot boundary that is <= *dt*.

    Equivalent to ``floor(dt / dt_hours) * dt_hours``.

    Args:
        dt:       Timezone-aware datetime to floor.
        dt_hours: Slot duration in hours (e.g. 0.25 for 15 min).

    Raises:
        ValueError: If *dt* is naive or *dt_hours* is not positive.
    """
    if dt.tzinfo is None:
        raise ValueError("floor_to_slot requires a timezone-aware datetime")
    if dt_hours <= 0:
        raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
    step_s = dt_hours * 3600.0
    epoch = dt.timestamp()
    return datetime.fromtimestamp(floor(epoch / step_s) * step_s, tz=dt.tzinfo)


def ceil_to_slot(dt: datetime, dt_hours: float) -> datetime:
    """Return the smallest slot boundary that is >= *dt*.

    If *dt* already falls exactly on a boundary, it is returned unchanged.

    Args:
        dt:       Timezone-aware datetime to ceil.
        dt_hours: Slot duration in hours (e.g. 0.25 for 15 min).

    Raises:
        ValueError: If *dt* is naive or *dt_hours* is not positive.
    """
    if dt.tzinfo is None:
        raise ValueError("ceil_to_slot requires a timezone-aware datetime")
    if dt_hours <= 0:
        raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
    step_s = dt_hours * 3600.0
    epoch = dt.timestamp()
    floor_epoch = floor(epoch / step_s) * step_s
    if abs(epoch - floor_epoch) <= 1e-9:
        return datetime.fromtimestamp(floor_epoch, tz=dt.tzinfo)
    return datetime.fromtimestamp(floor_epoch + step_s, tz=dt.tzinfo)


def round_to_slot(dt: datetime, dt_hours: float) -> datetime:
    """Return the nearest slot boundary (ties round to the next slot).

    For a 15-minute grid:
    * 14:47 → 14:45  (7 min into slot, before midpoint 14:52:30)
    * 14:55 → 15:00  (10 min into slot, after midpoint)
    * 14:52:30 → 15:00  (exactly at midpoint, rounds up)

    Args:
        dt:       Timezone-aware datetime to round.
        dt_hours: Slot duration in hours (e.g. 0.25 for 15 min).

    Raises:
        ValueError: If *dt* is naive or *dt_hours* is not positive.
    """
    if dt.tzinfo is None:
        raise ValueError("round_to_slot requires a timezone-aware datetime")
    if dt_hours <= 0:
        raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
    step_s = dt_hours * 3600.0
    epoch = dt.timestamp()
    return datetime.fromtimestamp(round(epoch / step_s) * step_s, tz=dt.tzinfo)


def resample_to_timestamps(
    values: Sequence[float],
    source_dt_hours: float,
    timestamps: list[datetime],
    pad_value: float | None = None,
) -> np.ndarray:
    """Map fixed-interval source *values* to *timestamps* via linear interpolation.

    Source values are assumed to start at ``timestamps[0]`` with a fixed step
    of *source_dt_hours*.  Values beyond the last source point are held constant
    by default, or replaced with *pad_value* when explicitly specified.

    Args:
        values:           Source values.
        source_dt_hours:  Step width of the source data in hours.
        timestamps:       Target datetime series (uniform or non-uniform).
        pad_value:        Fill value beyond the last source point.  ``None`` = hold last.

    Returns:
        ``np.ndarray`` of ``float32`` with the same length as *timestamps*.
    """
    if source_dt_hours <= 0:
        raise ValueError(f"source_dt_hours must be > 0, got {source_dt_hours}")

    n_src = len(values)
    n_tgt = len(timestamps)
    if n_tgt == 0:
        return np.empty(0, dtype=np.float32)
    if n_src == 0:
        fill = 0.0 if pad_value is None else pad_value
        return np.full(n_tgt, fill, dtype=np.float32)

    start_ts = timestamps[0]
    src_dt_s: float = source_dt_hours * 3_600.0

    delta_s = np.array([(ts - start_ts).total_seconds() for ts in timestamps], dtype=np.float64)
    t = delta_s / src_dt_s
    lo = np.floor(t).astype(np.int64)
    frac = t - lo

    src_arr = np.asarray(values, dtype=np.float64)
    result = np.empty(n_tgt, dtype=np.float64)

    mask_before = lo < 0
    mask_end = lo >= n_src - 1
    mask_interp = ~mask_before & ~mask_end

    result[mask_before] = src_arr[0]

    if pad_value is not None:
        mask_pad = lo >= n_src
        result[mask_end & ~mask_pad] = src_arr[-1]
        result[mask_pad] = pad_value
    else:
        result[mask_end] = src_arr[-1]

    if mask_interp.any():
        lo_i = lo[mask_interp]
        fi = frac[mask_interp]
        result[mask_interp] = src_arr[lo_i] * (1.0 - fi) + src_arr[lo_i + 1] * fi

    return np.asarray(result, dtype=np.float32)


class PredictionProvider(ABC):
    """Interface every prediction provider must implement."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Short unique identifier, e.g. ``"EnergyCharts"``."""
        ...
