"""Pydantic v2 configuration model for load profile providers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class AdaptiveLoadConfig(BaseModel):
    """Configuration for the adaptive (learning) load provider.

    Args:
        enabled:          Whether adaptive learning is active.
        decay_days:       Exponential decay half-life in days.  Recent days
                          are weighted more heavily.
        min_samples:      Minimum number of days of data required before the
                          learned profile is blended in.
        blend_factor:     Maximum weight for the learned profile (0–1).
                          ``0`` = base only, ``1`` = learned only.
        db_path:          Path to the SQLite TSDB file.
        flush_interval_s: How often the in-memory accumulator is flushed to
                          the TSDB (seconds).  Default 300 s (5 min).
        mqtt_topic:       MQTT topic for receiving load measurements.

    Note:
        Compaction and retention are defined by the TSDB's built-in
        ``LOAD_METRIC_POLICY``:  raw → 15-min after 1 h,  15-min → 1 h
        after 1 year,  delete after 2 years.  These are not configurable
        here to keep the config surface small.

        Vacation mode is a **runtime** flag only; it is not persisted in
        the config file.  Use the REST endpoint to toggle it.
    """

    enabled: bool = False
    decay_days: int = Field(default=30, ge=7, le=365)
    min_samples: int = Field(default=7, ge=1)
    blend_factor: float = Field(default=0.7, ge=0.0, le=1.0)
    db_path: str = "data/tsdb_load.sqlite"
    flush_interval_s: int = Field(default=300, ge=30, le=3600)
    mqtt_topic: str = "gridpythia/sensors/load_w"


class LoadProfileConfig(BaseModel):
    """Configuration for a file-based load profile provider.

    The file format is determined from the path suffix.

    Args:
        path:                 Path to the profile file (``.csv``).
        country:              ISO-3166-1 alpha-2 country code used to look up
                              public holidays (e.g. ``"DE"``).  When ``None``,
                              holiday detection is disabled.
        subdivision:          Country-specific subdivision code (e.g. ``"BW"``
                              for Baden-Württemberg).  Only used when *country*
                              is set.
        adaptive:             Adaptive learning configuration.  When
                              ``adaptive.enabled`` is ``True``, the factory
                              returns an :class:`AdaptiveLoadProvider`.
    """

    path: Path
    country: str | None = None
    subdivision: str | None = None
    adaptive: AdaptiveLoadConfig = Field(default_factory=AdaptiveLoadConfig)

    @model_validator(mode="after")
    def _validate_path_suffix(self) -> "LoadProfileConfig":
        suffix = self.path.suffix.lower()
        if suffix != ".csv":
            raise ValueError(f"Unsupported profile file extension: {suffix!r} (use .csv)")
        return self
