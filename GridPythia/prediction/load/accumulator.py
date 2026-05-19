"""In-memory measurement accumulator with periodic flush to TSDB.

Rationale
---------
Measurements arrive frequently (every few seconds from MQTT or REST).
Writing each sample directly to SQLite would create excessive I/O.
This accumulator buffers samples in RAM and flushes every
``flush_interval_s`` seconds (default 300 s / 5 min) for resilience.

On flush, samples are pre-aggregated into **15-minute aligned** averages
before being stored in the TSDB at level 0.  The TSDB compaction will
later promote those level-0 entries to level 1 (once the window is old
enough), keeping a consistent two-level pipeline.

Alignment rule::

    bucket_start = (unix_ts // 900) * 900   # 900 s = 15 min

Energy measurements (Wh over a duration) are converted to average power
(W) before storage so that the TSDB always stores power values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from structlog import get_logger

if TYPE_CHECKING:
    from GridPythia.tsdb.storage import TimeSeriesDB

logger = get_logger(__name__)

_BUCKET_S = 900  # 15-minute buckets


@dataclass
class _Bucket:
    """Running sum / count for one 15-min bucket."""

    sum_w: float = 0.0
    count: int = 0
    watt_seconds: float = 0.0  # for energy-based inputs

    def avg_w(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_w / self.count


class MeasurementAccumulator:
    """Buffers measurements in RAM and flushes to TSDB every 5 minutes.

    Args:
        db: Target :class:`~GridPythia.tsdb.storage.TimeSeriesDB`.
        metric: Metric name to write (e.g. ``"load_w"``).
        flush_interval_s: Seconds between automatic DB flushes (default 300).
    """

    def __init__(
        self,
        db: TimeSeriesDB,
        metric: str,
        flush_interval_s: int = 300,
    ) -> None:
        self._db = db
        self._metric = metric
        self._flush_interval_s = flush_interval_s
        # buckets: {bucket_start_ts: _Bucket}
        self._buckets: dict[int, _Bucket] = {}
        self._last_flush: float = 0.0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_power(self, watts: float, ts: float | None = None) -> None:
        """Add an instantaneous power measurement (W).

        The sample is placed into the 15-min bucket that contains *ts*.
        A flush is triggered automatically after ``flush_interval_s``.

        Args:
            watts: Instantaneous load in watts.
            ts: Unix timestamp.  Defaults to ``time.time()``.
        """
        if ts is None:
            ts = time.time()
        bucket = (int(ts) // _BUCKET_S) * _BUCKET_S
        b = self._buckets.setdefault(bucket, _Bucket())
        b.sum_w += watts
        b.count += 1
        self._maybe_flush()

    def add_energy(self, wh: float, duration_h: float, ts: float | None = None) -> None:
        """Add an energy measurement (Wh over a known duration).

        Converts to average power and distributes across the covered 15-min
        buckets proportionally.

        Args:
            wh: Energy consumed in watt-hours.
            duration_h: Duration of the measurement window in hours.
            ts: Unix timestamp of the **start** of the measurement window.
        """
        if duration_h <= 0 or wh < 0:
            return
        if ts is None:
            ts = time.time()

        avg_w = wh / duration_h
        duration_s = duration_h * 3600.0
        end_ts = ts + duration_s

        # Distribute across covered 15-min buckets
        t = ts
        while t < end_ts:
            bucket = (int(t) // _BUCKET_S) * _BUCKET_S
            bucket_end = float(bucket + _BUCKET_S)
            overlap_s = min(bucket_end, end_ts) - t
            weight = overlap_s / duration_s
            b = self._buckets.setdefault(bucket, _Bucket())
            b.sum_w += avg_w * weight * (duration_s / overlap_s) * overlap_s / overlap_s
            # Simpler: accumulate power weighted by overlap fraction
            b.watt_seconds += avg_w * overlap_s
            b.count += 1
            t = bucket_end

        self._maybe_flush()

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def _maybe_flush(self) -> None:
        now = time.time()
        if now - self._last_flush >= self._flush_interval_s:
            self.flush()

    def flush(self, force_all: bool = False) -> int:
        """Write accumulated buckets to the TSDB.

        Only buckets that are fully closed (their 15-min window has ended)
        are flushed by default.  Pass ``force_all=True`` to flush the
        current (potentially open) bucket too (e.g. on shutdown).

        Returns the number of buckets written.
        """
        if not self._buckets:
            return 0

        now = int(time.time())
        current_bucket = (now // _BUCKET_S) * _BUCKET_S

        samples: list[tuple[float, float]] = []
        flushed_keys: list[int] = []

        for bucket_ts, b in sorted(self._buckets.items()):
            if not force_all and bucket_ts >= current_bucket:
                # Bucket still open – don't flush yet
                continue
            avg = self._bucket_avg_w(b)
            if avg >= 0:
                samples.append((float(bucket_ts), avg))
            flushed_keys.append(bucket_ts)

        if samples:
            self._db.insert_batch(self._metric, samples, level=0)
            logger.debug(
                "accumulator_flushed",
                metric=self._metric,
                buckets=len(samples),
            )

        for k in flushed_keys:
            del self._buckets[k]

        self._last_flush = time.time()
        return len(samples)

    @staticmethod
    def _bucket_avg_w(b: _Bucket) -> float:
        """Compute average W for a bucket from either power or energy inputs."""
        if b.count == 0:
            return 0.0
        # If energy-based (watt_seconds populated): use that
        if b.watt_seconds > 0:
            return b.watt_seconds / _BUCKET_S
        return b.sum_w / b.count

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def pending_buckets(self) -> int:
        """Number of in-memory buckets not yet flushed."""
        return len(self._buckets)
