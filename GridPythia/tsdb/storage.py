"""SQLite-backed time-series storage with two-level compaction and TTL.

This module is the sole persistence layer for the adaptive load learning system.
It uses SQLite (stdlib) to remain dependency-free and ARM-friendly.

Storage levels
--------------
+-------+-------+-----------------------------------------------+
| level |  size | meaning                                       |
+=======+=======+===============================================+
|   0   |  raw  | written by the accumulator every ~5 minutes   |
|   1   | 15min | compacted from level-0 (aligned bucket)       |
|   2   |   1h  | compacted from level-1 (aligned bucket)       |
+-------+-------+-----------------------------------------------+

Each metric has its own :class:`~GridPythia.tsdb.policies.MetricPolicy`
controlling when each level transition occurs and when rows are deleted.

Appliance runs
--------------
A second table ``appliance_runs`` records start/end timestamps and average
power for individual appliance cycles.  This data is used by the learning
algorithm to separate base load from appliance contributions.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from structlog import get_logger

from GridPythia.tsdb.policies import BUILTIN_POLICIES, DEFAULT_POLICY, MetricPolicy

logger = get_logger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS measurements (
    ts         INTEGER NOT NULL,
    metric     TEXT    NOT NULL,
    value      REAL    NOT NULL,
    level      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ts_metric_level ON measurements (metric, ts, level);

CREATE TABLE IF NOT EXISTS appliance_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    appliance   TEXT    NOT NULL,
    start_ts    INTEGER NOT NULL,
    end_ts      INTEGER NOT NULL,
    avg_power_w REAL    NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_appliance_start ON appliance_runs (appliance, start_ts);
"""

_15MIN = 900
_1HOUR = 3_600


class TimeSeriesDB:
    """Lightweight time-series store backed by a single SQLite file.

    Args:
        db_path: Path to the SQLite database file.  Created if missing.
        policies: Per-metric overrides for compaction and retention.
            Merged with :data:`~GridPythia.tsdb.policies.BUILTIN_POLICIES`;
            explicit entries win.
    """

    def __init__(
        self,
        db_path: Path | str,
        policies: dict[str, MetricPolicy] | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        merged: dict[str, MetricPolicy] = dict(BUILTIN_POLICIES)
        if policies:
            merged.update(policies)
        self._policies = merged

        self._init_db()

    # ------------------------------------------------------------------
    # Policy lookup
    # ------------------------------------------------------------------

    def policy(self, metric: str) -> MetricPolicy:
        """Return the effective :class:`MetricPolicy` for *metric*."""
        return self._policies.get(metric, DEFAULT_POLICY)

    # ------------------------------------------------------------------
    # Connection context manager
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Write – measurements
    # ------------------------------------------------------------------

    def insert(self, metric: str, value: float, ts: float | None = None) -> None:
        """Insert a single raw measurement (level 0).

        Args:
            metric: Metric name (e.g. ``"load_w"``).
            value: Numeric value.
            ts: Unix timestamp.  Defaults to ``time.time()``.
        """
        if ts is None:
            ts = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO measurements (ts, metric, value, level) VALUES (?, ?, ?, 0)",
                (int(ts), metric, value),
            )

    def insert_batch(
        self,
        metric: str,
        samples: list[tuple[float, float]],
        level: int = 0,
    ) -> None:
        """Insert multiple ``(timestamp, value)`` pairs.

        Args:
            metric: Metric name.
            samples: List of ``(unix_ts, value)`` tuples.
            level: Storage level (default 0 = raw).
        """
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO measurements (ts, metric, value, level) VALUES (?, ?, ?, ?)",
                [(int(t), metric, v, level) for t, v in samples],
            )

    # ------------------------------------------------------------------
    # Write – appliance runs
    # ------------------------------------------------------------------

    def record_appliance_run(
        self,
        appliance: str,
        start_ts: float,
        end_ts: float,
        avg_power_w: float,
    ) -> int:
        """Persist a completed appliance run and return its row id."""
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO appliance_runs (appliance, start_ts, end_ts, avg_power_w)"
                " VALUES (?, ?, ?, ?)",
                (appliance, int(start_ts), int(end_ts), avg_power_w),
            )
            return cursor.lastrowid or 0

    # ------------------------------------------------------------------
    # Read – measurements
    # ------------------------------------------------------------------

    def query(
        self,
        metric: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        min_level: int = 0,
    ) -> list[tuple[int, float]]:
        """Return ``[(unix_ts, value), ...]`` sorted by time.

        Rows from all storage levels >= *min_level* are returned.

        Args:
            metric: Metric to query.
            start_ts: Lower bound (inclusive).  ``None`` = no lower bound.
            end_ts: Upper bound (exclusive).  ``None`` = no upper bound.
            min_level: Include only rows at or above this level.
        """
        clauses = ["metric = ?", "level >= ?"]
        params: list = [metric, min_level]
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(int(start_ts))
        if end_ts is not None:
            clauses.append("ts < ?")
            params.append(int(end_ts))

        sql = f"SELECT ts, value FROM measurements WHERE {' AND '.join(clauses)} ORDER BY ts"
        with self._conn() as conn:
            return conn.execute(sql, params).fetchall()

    def query_15min_averages(
        self,
        metric: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> list[tuple[int, float]]:
        """Return 15-minute-bucket averages as ``[(bucket_start_ts, avg_w), ...]``.

        Groups all levels together so the result is consistent regardless of
        what compaction has already run.
        """
        clauses = ["metric = ?"]
        params: list = [metric]
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(int(start_ts))
        if end_ts is not None:
            clauses.append("ts < ?")
            params.append(int(end_ts))

        where = " AND ".join(clauses)
        sql = (
            f"SELECT (ts / {_15MIN}) * {_15MIN} AS bucket, AVG(value) "
            f"FROM measurements WHERE {where} "
            f"GROUP BY bucket ORDER BY bucket"
        )
        with self._conn() as conn:
            return conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------
    # Read – appliance runs
    # ------------------------------------------------------------------

    def get_active_appliances(self, ts: float) -> list[tuple[str, float]]:
        """Return ``[(appliance_id, avg_power_w)]`` for runs active at *ts*."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT appliance, avg_power_w FROM appliance_runs"
                " WHERE start_ts <= ? AND end_ts >= ?",
                (int(ts), int(ts)),
            ).fetchall()
            return list(rows)

    def get_appliance_runs(
        self,
        appliance: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """Return ``[(start_ts, end_ts, avg_power_w)]`` for an appliance."""
        clauses = ["appliance = ?"]
        params: list = [appliance]
        if start_ts is not None:
            clauses.append("start_ts >= ?")
            params.append(int(start_ts))
        if end_ts is not None:
            clauses.append("start_ts < ?")
            params.append(int(end_ts))
        sql = (
            f"SELECT start_ts, end_ts, avg_power_w FROM appliance_runs"
            f" WHERE {' AND '.join(clauses)} ORDER BY start_ts"
        )
        with self._conn() as conn:
            return conn.execute(sql, params).fetchall()

    def known_appliances(self) -> list[str]:
        """Return distinct appliance IDs that have at least one run recorded."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT appliance FROM appliance_runs ORDER BY appliance"
            ).fetchall()
            return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def run_maintenance(self, metrics: list[str] | None = None) -> dict[str, int]:
        """Run compaction and retention for all (or specified) metrics.

        Returns aggregated stats: ``{"compacted": N, "deleted": N}``.
        """
        if metrics is None:
            with self._conn() as conn:
                rows = conn.execute("SELECT DISTINCT metric FROM measurements").fetchall()
            metrics = [r[0] for r in rows]

        total_compacted = 0
        total_deleted = 0
        for metric in metrics:
            c, d = self._run_metric_maintenance(metric)
            total_compacted += c
            total_deleted += d

        if total_compacted > 0 or total_deleted > 0:
            logger.info("tsdb_maintenance", compacted=total_compacted, deleted=total_deleted)
        return {"compacted": total_compacted, "deleted": total_deleted}

    def _run_metric_maintenance(self, metric: str) -> tuple[int, int]:
        pol = self.policy(metric)
        now = int(time.time())
        compacted = 0
        deleted = 0

        if pol.has_15min_level:
            # Stage 1: raw (level=0) -> 15-min (level=1)
            raw_cutoff = now - pol.compact_raw_after_s
            compacted += self._compact_to_bucket(
                metric, src_level=0, bucket_s=_15MIN, cutoff=raw_cutoff
            )
            # Stage 2: 15-min (level=1) -> 1h (level=2)
            min15_cutoff = now - pol.compact_15min_after_s
            compacted += self._compact_to_bucket(
                metric, src_level=1, bucket_s=_1HOUR, cutoff=min15_cutoff
            )
        else:
            # Single stage: raw (level=0) -> 1h (level=2)
            raw_cutoff = now - pol.compact_raw_after_s
            compacted += self._compact_to_bucket(
                metric, src_level=0, bucket_s=_1HOUR, cutoff=raw_cutoff
            )

        # Retention: delete all rows older than retention_s
        retention_cutoff = now - pol.retention_s
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM measurements WHERE metric = ? AND ts < ?",
                (metric, retention_cutoff),
            )
            deleted += cursor.rowcount

        return compacted, deleted

    def _compact_to_bucket(self, metric: str, src_level: int, bucket_s: int, cutoff: int) -> int:
        """Aggregate level-*src_level* rows older than *cutoff* into bucket averages."""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT (ts / {bucket_s}) * {bucket_s} AS bucket, AVG(value), COUNT(*) "
                f"FROM measurements "
                f"WHERE metric = ? AND level = ? AND ts < ? "
                f"GROUP BY bucket",
                (metric, src_level, cutoff),
            ).fetchall()

            if not rows:
                return 0

            target_level = 1 if bucket_s == _15MIN else 2
            conn.executemany(
                "INSERT INTO measurements (ts, metric, value, level) VALUES (?, ?, ?, ?)",
                [(r[0], metric, r[1], target_level) for r in rows],
            )
            conn.execute(
                "DELETE FROM measurements WHERE metric = ? AND level = ? AND ts < ?",
                (metric, src_level, cutoff),
            )
            return sum(r[2] for r in rows)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self, metric: str | None = None) -> int:
        """Return total row count, optionally filtered by metric."""
        if metric:
            sql = "SELECT COUNT(*) FROM measurements WHERE metric = ?"
            params: tuple = (metric,)
        else:
            sql = "SELECT COUNT(*) FROM measurements"
            params = ()
        with self._conn() as conn:
            return conn.execute(sql, params).fetchone()[0]

    def metrics(self) -> list[str]:
        """Return all distinct metric names stored."""
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT metric FROM measurements ORDER BY metric")
            return [r[0] for r in rows]

    def latest(self, metric: str) -> tuple[int, float] | None:
        """Return the most recent ``(ts, value)`` for a metric, or ``None``."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT ts, value FROM measurements WHERE metric = ? ORDER BY ts DESC LIMIT 1",
                (metric,),
            ).fetchone()
            return row if row else None

    @staticmethod
    def unix_ts(dt: datetime) -> float:
        """Convert a datetime to unix timestamp (UTC)."""
        return dt.replace(tzinfo=timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()
