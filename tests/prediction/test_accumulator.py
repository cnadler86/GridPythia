"""Tests for the MeasurementAccumulator (15-min bucketing, energy, threading)."""

from __future__ import annotations

import threading
import time

import pytest

from GridPythia.prediction.load.accumulator import _BUCKET_S, MeasurementAccumulator
from GridPythia.tsdb.storage import TimeSeriesDB

# A bucket-aligned timestamp safely in the past so flush() treats it as closed.
_BASE = 1_000_000 // _BUCKET_S * _BUCKET_S  # 1970-01-12, bucket aligned


@pytest.fixture
def db(tmp_path) -> TimeSeriesDB:
    return TimeSeriesDB(tmp_path / "acc.sqlite")


def _accumulator(db: TimeSeriesDB) -> MeasurementAccumulator:
    """Build an accumulator that never auto-flushes during a test.

    ``_last_flush`` defaults to 0.0, which makes the very first ingest trigger
    an immediate flush (harmless in production where the current bucket is
    still open).  Seed it with "now" so tests control flushing explicitly.
    """
    acc = MeasurementAccumulator(db, "load_w", flush_interval_s=10**9)
    acc._last_flush = time.time()
    return acc


def _rows(db: TimeSeriesDB, metric: str = "load_w") -> list[tuple[int, float]]:
    return db.query(metric, min_level=0)


def test_add_power_averages_samples_in_bucket(db: TimeSeriesDB) -> None:
    acc = _accumulator(db)
    acc.add_power(100.0, ts=_BASE + 10)
    acc.add_power(300.0, ts=_BASE + 20)

    assert acc.flush(force_all=True) == 1
    rows = _rows(db)
    assert rows == [(_BASE, 200.0)]


def test_open_bucket_not_flushed_without_force(db: TimeSeriesDB) -> None:
    acc = _accumulator(db)
    # A bucket containing "now" is still open and must not be flushed.
    acc.add_power(500.0, ts=time.time())
    assert acc.flush(force_all=False) == 0
    assert acc.pending_buckets == 1
    assert _rows(db) == []


def test_add_energy_full_bucket_is_average_power(db: TimeSeriesDB) -> None:
    acc = _accumulator(db)
    # 50 Wh over 0.25 h fully covering one 15-min bucket → 200 W average.
    acc.add_energy(wh=50.0, duration_h=0.25, ts=_BASE)
    acc.flush(force_all=True)
    assert _rows(db) == [(_BASE, 200.0)]


def test_add_energy_partial_bucket_uses_covered_seconds(db: TimeSeriesDB) -> None:
    """Regression: a window covering only part of a bucket must report the
    average power over the *covered* seconds, not diluted over the full 900 s."""
    acc = _accumulator(db)
    # 20 Wh over 0.1 h (360 s) → 200 W; covers only 360/900 of the bucket.
    acc.add_energy(wh=20.0, duration_h=0.1, ts=_BASE)
    acc.flush(force_all=True)
    rows = _rows(db)
    assert rows[0][1] == pytest.approx(200.0)  # not 200 * 360/900 = 80


def test_add_energy_spanning_multiple_buckets(db: TimeSeriesDB) -> None:
    acc = _accumulator(db)
    # Start 600 s into bucket0, run 0.5 h (1800 s): covers bucket0 (300 s),
    # bucket1 (900 s) and bucket2 (600 s).  Power is constant at 400 W.
    acc.add_energy(wh=200.0, duration_h=0.5, ts=_BASE + 600)
    acc.flush(force_all=True)
    rows = dict(_rows(db))
    assert rows[_BASE] == pytest.approx(400.0)
    assert rows[_BASE + _BUCKET_S] == pytest.approx(400.0)
    assert rows[_BASE + 2 * _BUCKET_S] == pytest.approx(400.0)


def test_add_energy_rejects_invalid_input(db: TimeSeriesDB) -> None:
    acc = _accumulator(db)
    acc.add_energy(wh=-5.0, duration_h=0.25, ts=_BASE)
    acc.add_energy(wh=10.0, duration_h=0.0, ts=_BASE)
    assert acc.pending_buckets == 0


def test_concurrent_add_and_flush_no_corruption(db: TimeSeriesDB) -> None:
    """Adding from one thread while flushing from another must not raise
    (e.g. 'dictionary changed size during iteration') nor lose buckets."""
    acc = _accumulator(db)
    n_buckets = 2000
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for i in range(n_buckets):
                acc.add_power(100.0, ts=_BASE + i * _BUCKET_S)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(300):
            acc.flush(force_all=False)
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)
    t.join()
    acc.flush(force_all=True)

    assert not errors
    # Each bucket receives exactly one sample → exactly n_buckets rows.
    assert db.count("load_w") == n_buckets
