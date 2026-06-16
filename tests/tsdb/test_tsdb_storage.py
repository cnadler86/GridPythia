"""Tests for the SQLite-backed TimeSeriesDB (queries, compaction, retention)."""

from __future__ import annotations

import time

import pytest

from GridPythia.tsdb.policies import DEFAULT_POLICY, LOAD_METRIC_POLICY
from GridPythia.tsdb.storage import TimeSeriesDB

_DAY = 86_400


@pytest.fixture
def db(tmp_path) -> TimeSeriesDB:
    return TimeSeriesDB(tmp_path / "tsdb.sqlite")


# ---------------------------------------------------------------------------
# Basic read / write
# ---------------------------------------------------------------------------


def test_insert_and_query_roundtrip(db: TimeSeriesDB) -> None:
    db.insert("load_w", 123.0, ts=1000)
    db.insert("load_w", 456.0, ts=2000)
    assert db.query("load_w") == [(1000, 123.0), (2000, 456.0)]


def test_query_time_bounds_are_half_open(db: TimeSeriesDB) -> None:
    db.insert_batch("m", [(100, 1.0), (200, 2.0), (300, 3.0)])
    # start inclusive, end exclusive
    assert db.query("m", start_ts=200, end_ts=300) == [(200, 2.0)]


def test_query_15min_averages_groups_into_buckets(db: TimeSeriesDB) -> None:
    base = 0
    db.insert_batch("m", [(base + 0, 100.0), (base + 60, 300.0), (base + 900, 50.0)])
    result = db.query_15min_averages("m")
    assert result == [(0, 200.0), (900, 50.0)]


def test_count_metrics_and_latest(db: TimeSeriesDB) -> None:
    db.insert("a", 1.0, ts=10)
    db.insert("a", 2.0, ts=20)
    db.insert("b", 9.0, ts=15)
    assert db.count() == 3
    assert db.count("a") == 2
    assert db.metrics() == ["a", "b"]
    assert db.latest("a") == (20, 2.0)
    assert db.latest("missing") is None


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_policy_lookup(db: TimeSeriesDB) -> None:
    assert db.policy("load_w") is LOAD_METRIC_POLICY
    assert db.policy("base_load_w") is LOAD_METRIC_POLICY
    assert db.policy("something_else") is DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Appliance runs
# ---------------------------------------------------------------------------


def test_appliance_runs_record_and_query(db: TimeSeriesDB) -> None:
    db.record_appliance_run("dishwasher", start_ts=1000, end_ts=4600, avg_power_w=1800.0)
    db.record_appliance_run("washer", start_ts=5000, end_ts=8000, avg_power_w=900.0)

    assert db.known_appliances() == ["dishwasher", "washer"]
    runs = db.get_appliance_runs("dishwasher")
    assert runs == [(1000, 4600, 1800.0)]


def test_get_active_appliances(db: TimeSeriesDB) -> None:
    db.record_appliance_run("dishwasher", start_ts=1000, end_ts=4600, avg_power_w=1800.0)
    db.record_appliance_run("washer", start_ts=2000, end_ts=3000, avg_power_w=900.0)

    active = dict(db.get_active_appliances(2500))
    assert active == {"dishwasher": 1800.0, "washer": 900.0}

    assert db.get_active_appliances(4700) == []


# ---------------------------------------------------------------------------
# Maintenance: compaction + retention
# ---------------------------------------------------------------------------


def test_compaction_raw_to_15min(db: TimeSeriesDB) -> None:
    now = int(time.time())
    # Two raw samples in the same 15-min bucket, older than the 1 h compaction
    # threshold of LOAD_METRIC_POLICY.
    old = now - 2 * 3600
    bucket = (old // 900) * 900
    db.insert_batch("load_w", [(bucket + 10, 100.0), (bucket + 20, 300.0)], level=0)

    stats = db.run_maintenance(["load_w"])

    assert stats["compacted"] == 2
    # Raw rows replaced by a single level-1 average row.
    assert db.count("load_w") == 1
    level1 = db.query("load_w", min_level=1)
    assert level1 == [(bucket, 200.0)]


def test_recent_raw_not_compacted(db: TimeSeriesDB) -> None:
    now = int(time.time())
    db.insert("load_w", 500.0, ts=now - 60)  # within the 1 h window
    stats = db.run_maintenance(["load_w"])
    assert stats["compacted"] == 0
    assert db.query("load_w", min_level=1) == []


def test_retention_deletes_old_rows(db: TimeSeriesDB) -> None:
    now = int(time.time())
    # Older than LOAD_METRIC_POLICY retention (2 years).
    db.insert("load_w", 500.0, ts=now - 800 * _DAY)
    stats = db.run_maintenance(["load_w"])
    assert stats["deleted"] >= 1
    assert db.count("load_w") == 0
