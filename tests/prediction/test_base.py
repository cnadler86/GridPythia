"""Tests for GridPythia.prediction.base utilities."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from GridPythia.prediction.base import make_timestamps, resample_to_timestamps

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


def test_make_timestamps_hourly():
    ts = make_timestamps(START, hours=3, dt_hours=1.0)
    assert len(ts) == 3
    assert ts[0] == START
    assert ts[1] == START + timedelta(hours=1)
    assert ts[2] == START + timedelta(hours=2)


def test_make_timestamps_quarter_hour():
    ts = make_timestamps(START, hours=1, dt_hours=0.25)
    assert len(ts) == 4


def test_make_timestamps_zero_hours():
    ts = make_timestamps(START, hours=0, dt_hours=1.0)
    assert len(ts) == 1  # at least one step


def test_make_timestamps_24h():
    ts = make_timestamps(START, hours=24, dt_hours=1.0)
    assert len(ts) == 24


def test_resample_identity():
    ts = make_timestamps(START, hours=4, dt_hours=1.0)
    out = resample_to_timestamps([1.0, 2.0, 3.0, 4.0], 1.0, ts)
    assert list(out) == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_resample_upsample():
    """1h -> 0.5h should interpolate at midpoint."""
    ts = make_timestamps(START, hours=1, dt_hours=0.5)  # [0h, 0.5h]
    out = resample_to_timestamps([0.0, 10.0], 1.0, ts)
    assert len(out) == 2
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(5.0)


def test_resample_pad_value():
    ts = make_timestamps(START, hours=4, dt_hours=1.0)
    out = resample_to_timestamps([1.0, 2.0], 1.0, ts, pad_value=99.0)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(99.0)
    assert out[3] == pytest.approx(99.0)


def test_resample_hold_last_without_pad():
    ts = make_timestamps(START, hours=4, dt_hours=1.0)
    out = resample_to_timestamps([5.0, 10.0], 1.0, ts)
    assert out[3] == pytest.approx(10.0)


def test_resample_empty_source():
    ts = make_timestamps(START, hours=2, dt_hours=1.0)
    out = resample_to_timestamps([], 1.0, ts)
    assert list(out) == pytest.approx([0.0, 0.0])


def test_resample_dtype():
    ts = make_timestamps(START, hours=2, dt_hours=1.0)
    out = resample_to_timestamps([1.0, 2.0], 1.0, ts)
    assert out.dtype == np.float32
