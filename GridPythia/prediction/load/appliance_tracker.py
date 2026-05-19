"""Appliance-aware pattern tracker for the adaptive load forecast.

This module separates household **base load** from **appliance contributions**
and learns the typical run patterns for each appliance independently.

Concepts
--------
Base load
    The steady household consumption that remains when no appliances are
    running (background devices, standby, lighting, fridge, etc.).

Appliance contribution
    The additional load caused by one named appliance while it is active.

Pattern learning
    For each appliance and day type, the tracker builds robust statistics of:

    * **Start hour** – when does the appliance typically start (hour of day)?
    * **Duration** – how long does a typical run last?
    * **Power** – average watts drawn during a run.

    Outliers are rejected using the **MAD-based robust Z-score** before
    computing means and standard deviations.

Forecast generation
    At prediction time the tracker adds expected appliance load to each slot:

    1. If an appliance has an *active announcement* (scheduled start time
       received via :meth:`notify_scheduled`), that announcement is used
       and pattern-based prediction for that appliance on that day is
       suppressed (avoid double-counting with ``state.appliance_forecasts``).
    2. If no announcement exists and the pattern has sufficient data,
       the learned pattern is used to estimate a probability-weighted
       contribution.
    3. If the optimizer already has an explicit appliance forecast (via
       ``active_forecast_appliances`` passed to :meth:`predict_contributions`),
       pattern prediction for that appliance is also suppressed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from structlog import get_logger

from GridPythia.prediction.load.provider import DayType, day_type_for_date

if TYPE_CHECKING:
    from GridPythia.tsdb.storage import TimeSeriesDB

logger = get_logger(__name__)

# How far back to look for historical runs
_LOOKBACK_DAYS = 180
# Minimum number of historical runs needed to build a pattern
_MIN_RUNS = 5
# Robust Z-score threshold for outlier rejection
_OUTLIER_SIGMA = 3.0


def _mad_filter(values: np.ndarray, n_sigma: float = _OUTLIER_SIGMA) -> np.ndarray:
    """Return *values* with outliers removed using MAD-based Z-score.

    The scale factor 1.4826 makes the MAD estimate consistent with the
    standard deviation of a Gaussian distribution.
    """
    if len(values) < 4:
        return values
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return values
    robust_z = 0.6745 * np.abs(values - median) / mad
    return values[robust_z <= n_sigma]


@dataclass
class AppliancePattern:
    """Learned run statistics for one appliance on one day type.

    Attributes:
        n_samples: Number of historical runs used.
        start_hour_mean: Mean start hour of day (0–24).
        start_hour_std: Standard deviation of start hour.
        duration_h_mean: Mean run duration in hours.
        duration_h_std: Standard deviation of duration.
        power_w_mean: Mean average power during a run (watts).
    """

    n_samples: int = 0
    start_hour_mean: float = 0.0
    start_hour_std: float = 1.0
    duration_h_mean: float = 1.0
    duration_h_std: float = 0.5
    power_w_mean: float = 0.0


@dataclass
class _PendingRun:
    """Tracks a currently-active (not yet closed) appliance run."""

    appliance: str
    start_ts: float
    power_samples: list[float] = field(default_factory=list)


@dataclass
class _Announcement:
    """A scheduled start time announced by an appliance."""

    scheduled_ts: float
    announced_at: float = field(default_factory=time.time)
    # Expected duration in hours (from learned pattern, or default)
    expected_duration_h: float = 1.0
    expected_power_w: float = 0.0


class ApplianceTracker:
    """Tracks appliance run patterns and generates appliance load forecasts.

    Args:
        db: :class:`~GridPythia.tsdb.storage.TimeSeriesDB` instance.
        country: ISO-3166-1 country code for holiday detection.
        subdivision: Country subdivision (e.g. ``"BW"``).
        min_runs: Minimum historical runs needed to use a pattern.
        outlier_sigma: MAD Z-score threshold for outlier rejection.
    """

    def __init__(
        self,
        db: TimeSeriesDB,
        country: str | None = None,
        subdivision: str | None = None,
        min_runs: int = _MIN_RUNS,
        outlier_sigma: float = _OUTLIER_SIGMA,
    ) -> None:
        self._db = db
        self._country = country
        self._subdivision = subdivision
        self._min_runs = min_runs
        self._outlier_sigma = outlier_sigma

        # In-flight (still running) appliances
        self._active_runs: dict[str, _PendingRun] = {}

        # Latest announcement per appliance
        self._announcements: dict[str, _Announcement] = {}

        # Learned pattern cache: {appliance_id: {DayType: AppliancePattern}}
        self._patterns: dict[str, dict[DayType, AppliancePattern]] = {}
        self._pattern_cache_ts: float = 0.0
        self._pattern_cache_ttl: float = 3600.0  # refresh hourly

    # ------------------------------------------------------------------
    # Runtime notifications – appliance state
    # ------------------------------------------------------------------

    def notify_active(
        self,
        appliance: str,
        ts: float | None = None,
    ) -> None:
        """Record that *appliance* just started a run.

        Args:
            appliance: Appliance identifier.
            ts: Unix timestamp of activation.  Defaults to now.
        """
        if ts is None:
            ts = time.time()
        if appliance in self._active_runs:
            logger.debug("appliance_already_active", appliance=appliance)
            return
        self._active_runs[appliance] = _PendingRun(appliance=appliance, start_ts=ts)
        logger.debug("appliance_activated", appliance=appliance)

    def notify_inactive(
        self,
        appliance: str,
        avg_power_w: float = 0.0,
        ts: float | None = None,
    ) -> None:
        """Record that *appliance* finished a run.

        Persists the run to the TSDB and invalidates the pattern cache.

        Args:
            appliance: Appliance identifier.
            avg_power_w: Average power drawn during the run (watts).
                         Used only if no per-sample power was recorded.
            ts: Unix timestamp of deactivation.  Defaults to now.
        """
        if ts is None:
            ts = time.time()
        run = self._active_runs.pop(appliance, None)
        if run is None:
            logger.debug("appliance_was_not_active", appliance=appliance)
            return

        # Determine average power
        if run.power_samples:
            power = float(np.mean(run.power_samples))
        else:
            power = avg_power_w

        self._db.record_appliance_run(
            appliance=appliance,
            start_ts=run.start_ts,
            end_ts=ts,
            avg_power_w=power,
        )

        # Invalidate cached patterns so the next forecast uses fresh data
        self._patterns.pop(appliance, None)

        logger.debug(
            "appliance_deactivated",
            appliance=appliance,
            duration_h=round((ts - run.start_ts) / 3600, 2),
            avg_power_w=round(power, 1),
        )

    def add_power_sample(self, appliance: str, watts: float) -> None:
        """Record a power sample while an appliance is running.

        Useful for computing a more accurate average when the run finishes.
        """
        run = self._active_runs.get(appliance)
        if run is not None:
            run.power_samples.append(watts)

    # ------------------------------------------------------------------
    # Runtime notifications – scheduled announcements
    # ------------------------------------------------------------------

    def notify_scheduled(
        self,
        appliance: str,
        scheduled_start_ts: float,
    ) -> None:
        """Record an announced scheduled start time for *appliance*.

        If an existing announcement for this appliance exists at a
        **different** time, it is overwritten and a log entry is emitted
        (the optimizer should clear its old prediction for that appliance).

        Args:
            appliance: Appliance identifier.
            scheduled_start_ts: Planned start as unix timestamp.
        """
        existing = self._announcements.get(appliance)
        pat = self._get_pattern_for_appliance(appliance)

        if existing is not None and abs(existing.scheduled_ts - scheduled_start_ts) > 900:
            logger.info(
                "appliance_announcement_updated",
                appliance=appliance,
                old_h=datetime.fromtimestamp(existing.scheduled_ts, tz=timezone.utc).strftime(
                    "%H:%M"
                ),
                new_h=datetime.fromtimestamp(scheduled_start_ts, tz=timezone.utc).strftime("%H:%M"),
            )

        # Look up expected duration / power from learned pattern
        dur_h = 1.0
        power_w = 0.0
        if pat is not None:
            dur_h = pat.duration_h_mean
            power_w = pat.power_w_mean

        self._announcements[appliance] = _Announcement(
            scheduled_ts=scheduled_start_ts,
            expected_duration_h=dur_h,
            expected_power_w=power_w,
        )

    def clear_announcement(self, appliance: str) -> None:
        """Clear any pending announcement for *appliance*."""
        self._announcements.pop(appliance, None)

    # ------------------------------------------------------------------
    # Base-load separation
    # ------------------------------------------------------------------

    def appliance_power_at(self, ts: float) -> float:
        """Return the total appliance power (W) active at *ts* from the TSDB."""
        rows = self._db.get_active_appliances(ts)
        return sum(pw for _, pw in rows)

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def predict_contributions(
        self,
        timestamps: list[datetime],
        active_forecast_appliances: set[str] | None = None,
    ) -> np.ndarray:
        """Return expected appliance load (W) per timestamp.

        The contribution is expressed in average watts so that the caller
        can multiply by the slot duration (h) to get watt-hours.

        For slots already covered by the optimizer's appliance forecasts
        (``active_forecast_appliances``), pattern-based contribution is
        suppressed to avoid double-counting.

        Args:
            timestamps: Ordered list of forecast timestamps.
            active_forecast_appliances: Set of appliance IDs that already
                have an explicit forecast in the optimizer.  Their load will
                be added by the optimizer separately; don't predict it here.
        """
        if not timestamps:
            return np.zeros(0, dtype=np.float32)

        n = len(timestamps)
        result = np.zeros(n, dtype=np.float64)
        excluded = active_forecast_appliances or set()

        # Refresh patterns if stale
        self._refresh_patterns()

        now = time.time()

        for appliance, patterns in self._patterns.items():
            if appliance in excluded:
                continue

            # Check for an active announcement
            ann = self._announcements.get(appliance)
            if ann is not None:
                # Announcement expires if its scheduled time + 2×expected_duration is past
                expiry = ann.scheduled_ts + ann.expected_duration_h * 7200
                if now > expiry:
                    del self._announcements[appliance]
                    ann = None

            for i, ts in enumerate(timestamps):
                d_type = day_type_for_date(ts.date(), self._country, self._subdivision)
                pat = patterns.get(d_type)
                if pat is None or pat.n_samples < self._min_runs:
                    continue

                if ann is not None:
                    # Use announcement-based prediction
                    contribution = self._announcement_contribution(ann, ts)
                else:
                    # Use pattern-based prediction
                    contribution = self._pattern_contribution(pat, ts)

                result[i] += contribution

        return result.astype(np.float32)

    def _announcement_contribution(self, ann: _Announcement, ts: datetime) -> float:
        """Power contribution (W) from an announced run at timestamp *ts*."""
        ts_unix = ts.timestamp()
        end_unix = ann.scheduled_ts + ann.expected_duration_h * 3600.0
        if ann.scheduled_ts <= ts_unix < end_unix:
            return ann.expected_power_w
        return 0.0

    def _pattern_contribution(self, pat: AppliancePattern, ts: datetime) -> float:
        """Expected power contribution (W) from a learned pattern at *ts*.

        Uses a Gaussian probability density around the expected start time
        multiplied by the expected power and duration to give an average
        expected watt value for the slot.
        """
        hour = ts.hour + ts.minute / 60.0
        # Probability that the run starts within a 15-min slot around this hour
        # P(start in [h - 0.125, h + 0.125]) under Gaussian start distribution
        sigma = max(pat.start_hour_std, 0.25)
        z_lo = (hour - 0.125 - pat.start_hour_mean) / sigma
        z_hi = (hour + 0.125 - pat.start_hour_mean) / sigma
        # Simple normal CDF approximation (fast, no scipy)
        p_start = _norm_cdf(z_hi) - _norm_cdf(z_lo)

        # Expected contribution: P(running) × power
        # A run that starts at hour h and lasts duration_h covers duration_h / dt slots.
        # Probability of being active during this slot ≈ p_start × duration_h / 24
        # (very rough – sufficient for a soft contribution estimate)
        p_active = p_start * pat.duration_h_mean / 24.0
        return p_active * pat.power_w_mean

    # ------------------------------------------------------------------
    # Pattern learning
    # ------------------------------------------------------------------

    def _refresh_patterns(self) -> None:
        """Rebuild learned patterns if the cache is stale."""
        now = time.time()
        if now - self._pattern_cache_ts < self._pattern_cache_ttl:
            return

        appliances = self._db.known_appliances()
        lookback = now - _LOOKBACK_DAYS * 86400

        for appliance in appliances:
            runs = self._db.get_appliance_runs(appliance, start_ts=lookback)
            if len(runs) < self._min_runs:
                continue
            self._patterns[appliance] = self._build_patterns(runs)

        self._pattern_cache_ts = now

    def _build_patterns(
        self, runs: list[tuple[int, int, float]]
    ) -> dict[DayType, AppliancePattern]:
        """Build per-day-type patterns from a list of ``(start_ts, end_ts, power_w)``."""
        by_type: dict[DayType, list[tuple[float, float, float]]] = {}
        for start_ts, end_ts, power_w in runs:
            dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
            d_type = day_type_for_date(dt.date(), self._country, self._subdivision)
            hour = dt.hour + dt.minute / 60.0
            duration_h = (end_ts - start_ts) / 3600.0
            by_type.setdefault(d_type, []).append((hour, duration_h, power_w))

        patterns: dict[DayType, AppliancePattern] = {}
        for d_type, samples in by_type.items():
            if len(samples) < self._min_runs:
                continue
            arr = np.array(samples)  # shape (N, 3): [hour, duration_h, power_w]

            hours = _mad_filter(arr[:, 0], self._outlier_sigma)
            durs = _mad_filter(arr[:, 1], self._outlier_sigma)
            powers = _mad_filter(arr[:, 2], self._outlier_sigma)

            if len(hours) < self._min_runs:
                continue

            patterns[d_type] = AppliancePattern(
                n_samples=len(hours),
                start_hour_mean=float(np.mean(hours)),
                start_hour_std=float(np.std(hours)) if len(hours) > 1 else 1.0,
                duration_h_mean=float(np.mean(durs)),
                duration_h_std=float(np.std(durs)) if len(durs) > 1 else 0.5,
                power_w_mean=float(np.mean(powers)),
            )

        return patterns

    def _get_pattern_for_appliance(self, appliance: str) -> AppliancePattern | None:
        """Return the most representative pattern for an appliance (across day types)."""
        self._refresh_patterns()
        day_patterns = self._patterns.get(appliance)
        if not day_patterns:
            return None
        # Return weekday pattern if available, otherwise first found
        return day_patterns.get(DayType.WEEKDAY) or next(iter(day_patterns.values()))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return tracker statistics for diagnostics."""
        self._refresh_patterns()
        return {
            "active_runs": list(self._active_runs.keys()),
            "announcements": {
                a: datetime.fromtimestamp(ann.scheduled_ts, tz=timezone.utc).isoformat()
                for a, ann in self._announcements.items()
            },
            "appliances_with_patterns": {
                a: {dt.value: p.n_samples for dt, p in pts.items()}
                for a, pts in self._patterns.items()
            },
        }


# ---------------------------------------------------------------------------
# Lightweight normal CDF approximation (no scipy dependency)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF using Abramowitz & Stegun 26.2.17."""
    # Accurate to ~1.5e-7
    if x < -6.0:
        return 0.0
    if x > 6.0:
        return 1.0
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = k * (
        0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429)))
    )
    val = 1.0 - (1.0 / (2.506628274631 * 1.0)) * (2.718281828 ** (-0.5 * x * x)) * poly
    return val if x >= 0 else 1.0 - val
