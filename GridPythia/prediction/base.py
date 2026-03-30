"""Core utilities for the prediction framework."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

import polars as pl


def make_timestamps(start: datetime, hours: float, dt_hours: float) -> pl.Series:
    """Create a uniformly-spaced datetime Series.

    Args:
        start:     First timestamp.
        hours:     Total window length in hours.
        dt_hours:  Step width in hours.

    Returns:
        ``pl.Series`` of datetime values with length ``max(1, round(hours / dt_hours))``.
    """
    n = max(1, round(hours / dt_hours))
    dt_us = int(dt_hours * 3_600_000_000)  # microseconds per step

    if start.tzinfo is not None:
        # Build as epoch-microsecond integers to avoid polars tz-aware list crash
        start_us = int(start.timestamp() * 1_000_000)
        # Normalise tz name for polars (e.g. "UTC", "Europe/Berlin")
        tz_name: str = getattr(start.tzinfo, "key", None) or str(start.tzinfo)
        if tz_name in ("UTC", "utc"):
            tz_name = "UTC"
        vals = [start_us + i * dt_us for i in range(n)]
        return pl.Series(vals, dtype=pl.Int64).cast(pl.Datetime("us", tz_name))
    else:
        # Naive datetimes: build directly as Int64 epoch-microseconds
        epoch = datetime(1970, 1, 1)
        start_us = int((start - epoch).total_seconds() * 1_000_000)
        vals = [start_us + i * dt_us for i in range(n)]
        return pl.Series(vals, dtype=pl.Int64).cast(pl.Datetime("us"))


def resample_to_timestamps(
    values: Sequence[float],
    source_dt_hours: float,
    timestamps: pl.Series,
    pad_value: float | None = None,
) -> pl.Series:
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
        ``pl.Series(Float32)`` with the same length as *timestamps*.
    """
    n_src = len(values)
    n_tgt = len(timestamps)
    if n_tgt == 0:
        return pl.Series([], dtype=pl.Float32)
    if n_src == 0:
        fill = 0.0 if pad_value is None else pad_value
        return pl.Series([fill] * n_tgt, dtype=pl.Float32)

    ts_list: list[datetime] = timestamps.to_list()
    start_ts = ts_list[0]
    src_dt_s: float = source_dt_hours * 3_600.0

    result: list[float] = []
    for ts in ts_list:
        delta_s = (ts - start_ts).total_seconds()
        t = delta_s / src_dt_s
        lo = int(t)
        frac = t - lo
        if lo < 0:
            result.append(float(values[0]))
        elif lo >= n_src - 1:
            # At or beyond last source point
            if lo >= n_src and pad_value is not None:
                result.append(pad_value)
            else:
                result.append(float(values[n_src - 1]))
        else:
            v = float(values[lo]) * (1.0 - frac) + float(values[lo + 1]) * frac
            result.append(v)

    return pl.Series(result, dtype=pl.Float32)


class PredictionProvider(ABC):
    """Interface every prediction provider must implement."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Short unique identifier, e.g. ``"EnergyCharts"``."""
        ...
