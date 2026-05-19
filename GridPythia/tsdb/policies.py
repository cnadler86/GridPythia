"""Per-metric storage and compaction policies for the TSDB.

Each metric can define independent rules for:
- How quickly raw samples are compacted into 15-min aligned averages.
- How quickly 15-min data is further compacted to 1-hour averages.
- How long data is retained before deletion.

Load metrics should use :data:`LOAD_METRIC_POLICY`.
All other metrics fall back to :data:`DEFAULT_POLICY`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricPolicy:
    """Compaction and retention rules for one metric.

    Attributes:
        compact_raw_after_s: Compact level-0 (raw) samples that are older
            than this many seconds into 15-min aligned averages (level=1).
        compact_15min_after_s: Compact level-1 (15-min) rows that are older
            than this many seconds into 1-hour averages (level=2).
            Ignored when ``has_15min_level`` is ``False``.
        retention_s: Delete **all** rows for this metric that are older than
            this many seconds, regardless of level.
        has_15min_level: When ``True`` the two-stage pipeline
            raw → 15-min → 1h is used.  When ``False`` raw is compacted
            directly to 1h (useful for coarser-grained metrics).
    """

    compact_raw_after_s: int = 86_400  # default: 1 day
    compact_15min_after_s: int = 365 * 86_400  # default: 1 year
    retention_s: int = 365 * 86_400  # default: 1 year
    has_15min_level: bool = False


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------

#: Policy for household load metrics (``load_w``, ``base_load_w``).
#: - Raw (5-min flush) → 15-min after 1 hour (guarantees window closure)
#: - 15-min → 1h after 1 year (full-year high-res then downsampled)
#: - Retention 2 years
LOAD_METRIC_POLICY = MetricPolicy(
    compact_raw_after_s=3_600,  # 1 hour
    compact_15min_after_s=365 * 86_400,  # 1 year
    retention_s=730 * 86_400,  # 2 years
    has_15min_level=True,
)

#: Default policy for all other metrics (coarser resolution).
DEFAULT_POLICY = MetricPolicy(
    compact_raw_after_s=86_400,  # 1 day
    compact_15min_after_s=365 * 86_400,
    retention_s=365 * 86_400,  # 1 year
    has_15min_level=False,
)

#: Convenience mapping used by :class:`~GridPythia.tsdb.storage.TimeSeriesDB`
#: when no explicit ``policies`` argument is supplied.
BUILTIN_POLICIES: dict[str, MetricPolicy] = {
    "load_w": LOAD_METRIC_POLICY,
    "base_load_w": LOAD_METRIC_POLICY,
}
