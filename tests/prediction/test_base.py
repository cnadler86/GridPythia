"""Tests for GridPythia.prediction.base utilities."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from GridPythia.prediction.base import (
    ceil_to_slot,
    floor_to_slot,
    make_timestamps,
    resample_to_timestamps,
)

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


def test_make_timestamps_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt_hours must be > 0"):
        make_timestamps(START, hours=24, dt_hours=0.0)


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


def test_resample_rejects_non_positive_source_dt():
    ts = make_timestamps(START, hours=2, dt_hours=1.0)
    with pytest.raises(ValueError, match="source_dt_hours must be > 0"):
        resample_to_timestamps([1.0, 2.0], 0.0, ts)


# ── floor_to_slot / ceil_to_slot ──────────────────────────────────────────


_DT = 0.25  # 15-minute slot
_TS_11_15 = datetime(2025, 6, 15, 11, 15, tzinfo=timezone.utc)
_TS_11_18 = datetime(2025, 6, 15, 11, 18, tzinfo=timezone.utc)
_TS_11_30 = datetime(2025, 6, 15, 11, 30, tzinfo=timezone.utc)
_TS_19_24 = datetime(2025, 6, 15, 19, 24, tzinfo=timezone.utc)
_TS_19_30 = datetime(2025, 6, 15, 19, 30, tzinfo=timezone.utc)


def test_floor_to_slot_already_aligned():
    assert floor_to_slot(_TS_11_15, _DT) == _TS_11_15


def test_floor_to_slot_mid_slot():
    result = floor_to_slot(_TS_11_18, _DT)
    assert result == _TS_11_15


def test_floor_to_slot_near_boundary():
    # 11:14:59 → 11:00 (the slot before 11:15)
    ts = datetime(2025, 6, 15, 11, 14, 59, tzinfo=timezone.utc)
    result = floor_to_slot(ts, _DT)
    assert result == datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc)


def test_floor_to_slot_requires_tz():
    naive = datetime(2025, 6, 15, 11, 18)
    with pytest.raises(ValueError, match="timezone-aware"):
        floor_to_slot(naive, _DT)


def test_floor_to_slot_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt_hours must be > 0"):
        floor_to_slot(_TS_11_18, 0.0)


def test_ceil_to_slot_already_aligned():
    assert ceil_to_slot(_TS_11_15, _DT) == _TS_11_15


def test_ceil_to_slot_mid_slot():
    result = ceil_to_slot(_TS_11_18, _DT)
    assert result == _TS_11_30


def test_ceil_to_slot_end_example():
    result = ceil_to_slot(_TS_19_24, _DT)
    assert result == _TS_19_30


def test_ceil_to_slot_requires_tz():
    naive = datetime(2025, 6, 15, 11, 18)
    with pytest.raises(ValueError, match="timezone-aware"):
        ceil_to_slot(naive, _DT)


def test_ceil_to_slot_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt_hours must be > 0"):
        ceil_to_slot(_TS_11_18, 0.0)


def test_floor_ceil_inverse_property():
    """floor(t) <= t <= ceil(t), both on slot boundaries."""
    for minute_offset in [0, 1, 7, 14, 15, 22, 29]:
        ts = _TS_11_15 + timedelta(minutes=minute_offset)
        f = floor_to_slot(ts, _DT)
        c = ceil_to_slot(ts, _DT)
        assert f <= ts <= c, f"{f} <= {ts} <= {c}"
        # Both must be on 15-min boundaries
        assert f.minute % 15 == 0 and f.second == 0
        assert c.minute % 15 == 0 and c.second == 0

