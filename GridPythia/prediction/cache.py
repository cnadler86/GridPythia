"""Reusable in-memory cache primitives for time-series prediction providers."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def to_utc(dt: datetime) -> datetime:
    """Normalize *dt* to timezone-aware UTC."""
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class TimeBucketCache:
    """Cache float values keyed by fixed-size UTC time buckets.

    Attributes:
        bucket_seconds: Width of one bucket in seconds.
        values: Mapping ``bucket -> value``.
        coverage_start: First covered UTC timestamp (inclusive).
        coverage_end: Last covered UTC timestamp (inclusive).
        source_valid_until: Validity end timestamp inherited from fetch data.
    """

    bucket_seconds: int
    values: dict[int, float] = field(default_factory=dict)
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None
    source_valid_until: datetime | None = None

    def bucket_of(self, dt: datetime) -> int:
        """Return integer bucket id for *dt* (UTC-normalized)."""
        return int(to_utc(dt).timestamp()) // self.bucket_seconds

    def value_at(self, dt: datetime) -> float | None:
        """Return cached value at *dt* or ``None`` when missing."""
        return self.values.get(self.bucket_of(dt))

    def has_data(self) -> bool:
        """Return True when the cache currently contains values."""
        return bool(self.values)

    def covers(self, start: datetime, end: datetime) -> bool:
        """Return True when cache fully covers ``[start, end]``."""
        if not self.values or self.coverage_start is None or self.coverage_end is None:
            return False
        start_utc = to_utc(start)
        end_utc = to_utc(end)
        return start_utc >= self.coverage_start and end_utc <= self.coverage_end

    def update(
        self,
        values: dict[int, float],
        coverage_start: datetime,
        coverage_end: datetime,
        source_valid_until: datetime | None,
    ) -> None:
        """Replace cache content atomically with new values and metadata."""
        self.values = values
        self.coverage_start = to_utc(coverage_start)
        self.coverage_end = to_utc(coverage_end)
        self.source_valid_until = to_utc(source_valid_until) if source_valid_until else None
