"""Adaptive load provider that learns from actual consumption data.

This module implements a learning load prediction system that wraps the
base :class:`~GridPythia.prediction.load.profilecsv.LoadProfileCSV` provider
and improves its forecasts using historical measurements stored in the TSDB.

Learning approach
-----------------
1. Raw measurements (W or Wh) arrive via MQTT or REST and are buffered by the
   :class:`~GridPythia.prediction.load.accumulator.MeasurementAccumulator`.
2. Every 5 minutes the accumulator flushes 15-min aligned averages to the TSDB.
3. At query time, the base-load metric (``base_load_w``) is used for learning.
   When an ingested sample arrives, any active appliance contribution is
   subtracted first, so the learned profile represents *standby + background*
   load only (washing machine, dishwasher etc. do not inflate the baseline).
4. Per day-type (weekday / saturday / sunday) the system builds an
   exponentially-decayed weighted profile.  Recent days are weighted more
   heavily (configurable half-life).
5. The forecast blends the static CSV baseline with the learned profile using
   a confidence factor that grows with the number of observed days.
6. Appliance contributions from :class:`~GridPythia.prediction.load.appliance_tracker.ApplianceTracker`
   are added on top — but only for appliances that have **not** submitted an
   explicit forecast to the optimizer (to avoid double-counting).

Vacation mode
-------------
Runtime-only flag (not persisted in config).  When active the provider
returns the 10th percentile of the static CSV profiles (minimal standby
load).  Data ingested during vacation mode is silently discarded so it does
not pollute the learned profile.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from structlog import get_logger

from GridPythia.prediction.load.accumulator import MeasurementAccumulator
from GridPythia.prediction.load.appliance_tracker import ApplianceTracker
from GridPythia.prediction.load.config import AdaptiveLoadConfig, LoadProfileConfig
from GridPythia.prediction.load.profilecsv import LoadProfileCSV
from GridPythia.prediction.load.provider import DayType, LoadProvider, day_type_for_date
from GridPythia.tsdb.storage import TimeSeriesDB

logger = get_logger(__name__)

METRIC_TOTAL_LOAD_W = "load_w"
METRIC_BASE_LOAD_W = "base_load_w"


class AdaptiveLoadProvider(LoadProvider):
    """Learning load provider that improves on the static CSV profile.

    Wraps a :class:`LoadProfileCSV` as baseline and blends in learned
    day-type profiles derived from historical base-load measurements.

    Args:
        config: Load profile configuration (for the base CSV provider).
        adaptive_config: Adaptive-specific settings.
    """

    def __init__(
        self,
        config: LoadProfileConfig,
        adaptive_config: AdaptiveLoadConfig,
    ) -> None:
        super().__init__(country=config.country, subdivision=config.subdivision)
        self._base_provider = LoadProfileCSV(config)
        self._adaptive_cfg = adaptive_config

        # Vacation mode is a pure runtime flag (not loaded from config)
        self._vacation_mode: bool = False

        # TSDB
        self._db = TimeSeriesDB(db_path=Path(adaptive_config.db_path))

        # Accumulator for raw measurements → 15-min aligned buckets → TSDB
        self._total_acc = MeasurementAccumulator(
            self._db,
            metric=METRIC_TOTAL_LOAD_W,
            flush_interval_s=adaptive_config.flush_interval_s,
        )
        self._base_acc = MeasurementAccumulator(
            self._db,
            metric=METRIC_BASE_LOAD_W,
            flush_interval_s=adaptive_config.flush_interval_s,
        )

        # Appliance tracker
        self._appliance_tracker = ApplianceTracker(
            db=self._db,
            country=config.country,
            subdivision=config.subdivision,
        )

        # Active optimizer appliance forecasts (updated before fetch)
        self._active_forecast_appliances: set[str] = set()

        # Cache for learned profiles
        self._learned_cache: dict[DayType, np.ndarray | None] = {}
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 300.0

        # Vacation percentile profile (lazy computed)
        self._vacation_profile_w: np.ndarray | None = None
        self._vacation_source_dt: float | None = None

    # ------------------------------------------------------------------
    # PredictionProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        return f"AdaptiveLoad(base={self._base_provider.provider_id})"

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------

    @property
    def vacation_mode(self) -> bool:
        return self._vacation_mode

    @vacation_mode.setter
    def vacation_mode(self, active: bool) -> None:
        self._vacation_mode = active
        logger.info("vacation_mode_changed", active=active)

    @property
    def db(self) -> TimeSeriesDB:
        return self._db

    @property
    def appliance_tracker(self) -> ApplianceTracker:
        return self._appliance_tracker

    def set_active_forecast_appliances(self, appliance_ids: set[str]) -> None:
        """Inform the provider which appliances have explicit optimizer forecasts.

        Their pattern-based contribution will be suppressed during :meth:`fetch`
        to avoid double-counting with the optimizer's appliance load.
        """
        self._active_forecast_appliances = appliance_ids

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest_power(self, watts: float, ts: float | None = None) -> None:
        """Record an instantaneous load measurement (W).

        The appliance contribution is subtracted before storing the base-load
        metric, so the learned profile represents background load only.

        Args:
            watts: Total household load in watts.
            ts: Unix timestamp.  Defaults to now.
        """
        if self._vacation_mode:
            return
        if ts is None:
            ts = time.time()
        self._total_acc.add_power(watts, ts)
        appliance_w = self._appliance_tracker.appliance_power_at(ts)
        base_w = max(0.0, watts - appliance_w)
        self._base_acc.add_power(base_w, ts)

    def ingest_energy(self, wh: float, duration_h: float, ts: float | None = None) -> None:
        """Record an energy measurement (Wh over *duration_h* hours).

        Converts to average power, subtracts the appliance contribution for
        the window midpoint, and stores both total and base load metrics.

        Args:
            wh: Energy consumed in watt-hours.
            duration_h: Duration of the measurement window in hours.
            ts: Unix timestamp of the window **start**.
        """
        if self._vacation_mode:
            return
        if duration_h <= 0 or wh < 0:
            return
        if ts is None:
            ts = time.time()
        self._total_acc.add_energy(wh, duration_h, ts)
        avg_w = wh / duration_h
        mid_ts = ts + duration_h * 1800.0  # midpoint of the window
        appliance_w = self._appliance_tracker.appliance_power_at(mid_ts)
        base_wh = max(0.0, avg_w - appliance_w) * duration_h
        self._base_acc.add_energy(base_wh, duration_h, ts)

    def flush_accumulators(self, force_all: bool = False) -> None:
        """Manually flush both accumulators to the TSDB."""
        self._total_acc.flush(force_all=force_all)
        self._base_acc.flush(force_all=force_all)

    def run_maintenance(self) -> dict[str, int]:
        """Flush accumulators and run TSDB compaction / retention cleanup."""
        self.flush_accumulators()
        return self._db.run_maintenance()

    # ------------------------------------------------------------------
    # LoadProvider implementation
    # ------------------------------------------------------------------

    def _get_day_profile_w(self, day_type: DayType) -> tuple[list[float], float]:
        """Return blended profile (base learned + appliance) for a day type."""
        if day_type == DayType.VACATIONS or self._vacation_mode:
            return self._get_vacation_profile()

        base_profile, source_dt = self._base_provider._get_day_profile_w(day_type)
        learned = self._get_learned_profile(day_type, len(base_profile), source_dt)

        if learned is None:
            return base_profile, source_dt

        blend = self._adaptive_cfg.blend_factor
        blended = [
            base_profile[i] * (1.0 - blend) + learned[i] * blend for i in range(len(base_profile))
        ]
        return blended, source_dt

    async def fetch(self, timestamps: list, *, use_vacation_profile: bool = False) -> np.ndarray:
        """Override to append appliance tracker contributions after base fetch."""
        result = await super().fetch(timestamps, use_vacation_profile=use_vacation_profile)

        if use_vacation_profile or self._vacation_mode or not timestamps:
            return result

        ts_list = list(timestamps)
        target_dt_h = (
            (ts_list[1] - ts_list[0]).total_seconds() / 3600.0 if len(ts_list) >= 2 else 0.25
        )

        appliance_w = self._appliance_tracker.predict_contributions(
            ts_list,
            active_forecast_appliances=self._active_forecast_appliances,
        )
        result = result + (appliance_w * target_dt_h).astype(np.float32)
        return result

    # ------------------------------------------------------------------
    # Learning logic
    # ------------------------------------------------------------------

    def _get_learned_profile(
        self, day_type: DayType, n_slots: int, source_dt_h: float
    ) -> list[float] | None:
        """Compute the learned base-load profile for a day type."""
        now = time.time()
        if now - self._cache_ts < self._cache_ttl and day_type in self._learned_cache:
            cached = self._learned_cache[day_type]
            return list(cached) if cached is not None else None

        lookback_ts = now - self._adaptive_cfg.decay_days * 2 * 86400
        rows = self._db.query(METRIC_BASE_LOAD_W, start_ts=lookback_ts)

        if not rows:
            self._learned_cache[day_type] = None
            self._cache_ts = now
            return None

        day_slots: dict[date, list[tuple[float, float]]] = {}
        for ts_val, value in rows:
            dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            d = dt.date()
            d_type = day_type_for_date(d, self._country, self._subdivision)
            if d_type == day_type:
                day_slots.setdefault(d, []).append((ts_val, value))

        if len(day_slots) < self._adaptive_cfg.min_samples:
            self._learned_cache[day_type] = None
            self._cache_ts = now
            return None

        half_life = self._adaptive_cfg.decay_days * 86400
        decay_lambda = 0.693147 / half_life

        slot_sums = np.zeros(n_slots, dtype=np.float64)
        slot_weights = np.zeros(n_slots, dtype=np.float64)

        for d, samples in day_slots.items():
            day_ts = datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()
            age = now - day_ts
            weight = np.exp(-decay_lambda * age)

            for ts_val, value in samples:
                dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                hour_frac = dt.hour + dt.minute / 60.0
                slot_idx = int(hour_frac / source_dt_h) % n_slots
                slot_sums[slot_idx] += value * weight
                slot_weights[slot_idx] += weight

        mask = slot_weights > 0
        profile = np.zeros(n_slots, dtype=np.float64)
        profile[mask] = slot_sums[mask] / slot_weights[mask]

        if not mask.all():
            base_profile, _ = self._base_provider._get_day_profile_w(day_type)
            for i in range(n_slots):
                if not mask[i]:
                    profile[i] = base_profile[i]

        self._learned_cache[day_type] = profile
        self._cache_ts = now
        return list(profile)

    # ------------------------------------------------------------------
    # Vacation profile
    # ------------------------------------------------------------------

    def _get_vacation_profile(self) -> tuple[list[float], float]:
        """Return 10th-percentile profile from base CSV data."""
        if self._vacation_profile_w is not None and self._vacation_source_dt is not None:
            return list(self._vacation_profile_w), self._vacation_source_dt

        weekday_profile, source_dt = self._base_provider._get_day_profile_w(DayType.WEEKDAY)
        sat_profile, _ = self._base_provider._get_day_profile_w(DayType.SATURDAY)
        sun_profile, _ = self._base_provider._get_day_profile_w(DayType.SUNDAY)

        all_profiles = np.array([weekday_profile, sat_profile, sun_profile])
        vacation = np.percentile(all_profiles, 10, axis=0)

        self._vacation_profile_w = vacation
        self._vacation_source_dt = source_dt
        return list(vacation), source_dt

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return learning statistics for diagnostics."""
        total = self._db.count(METRIC_TOTAL_LOAD_W)
        base = self._db.count(METRIC_BASE_LOAD_W)
        latest = self._db.latest(METRIC_TOTAL_LOAD_W)
        return {
            "total_load_samples": total,
            "base_load_samples": base,
            "latest_ts": latest[0] if latest else None,
            "latest_value_w": latest[1] if latest else None,
            "vacation_mode": self._vacation_mode,
            "pending_accumulator_buckets": (
                self._total_acc.pending_buckets + self._base_acc.pending_buckets
            ),
            "learned_day_types": [
                dt.value for dt, v in self._learned_cache.items() if v is not None
            ],
            "appliance_tracker": self._appliance_tracker.get_stats(),
        }
