"""Tests for src.prediction.base utilities."""

from array import array

import pytest

from src.prediction.base import make_array, n_steps, resample

# ── make_array ────────────────────────────────────────────────────────


def test_make_array_from_values():
    a = make_array([1.0, 2.0, 3.0])
    assert list(a) == pytest.approx([1.0, 2.0, 3.0])
    assert a.typecode == "f"


def test_make_array_zero_filled():
    a = make_array(size=4)
    assert list(a) == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_make_array_empty():
    a = make_array(size=0)
    assert len(a) == 0


# ── n_steps ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hours, dt, expected",
    [
        (24, 1.0, 24),
        (24, 0.25, 96),
        (1, 0.25, 4),
        (0.5, 0.25, 2),
        (0, 1.0, 1),  # at least 1 step
    ],
)
def test_n_steps(hours, dt, expected):
    assert n_steps(hours, dt) == expected


# ── resample ──────────────────────────────────────────────────────────


def test_resample_identity():
    src = array("f", [1.0, 2.0, 3.0, 4.0])
    out = resample(src, 1.0, 1.0)
    assert list(out) == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_resample_upsample_2x():
    """1h → 0.5h should double the number of points with linear interp."""
    src = array("f", [0.0, 10.0])
    out = resample(src, 1.0, 0.5)
    assert len(out) == 4
    assert list(out) == pytest.approx([0.0, 5.0, 10.0, 10.0])


def test_resample_upsample_4x():
    """1h → 0.25h should quadruple."""
    src = array("f", [0.0, 4.0])
    out = resample(src, 1.0, 0.25)
    assert len(out) == 8
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_resample_downsample():
    """0.5h → 1h should halve."""
    src = array("f", [0.0, 5.0, 10.0, 15.0])
    out = resample(src, 0.5, 1.0)
    assert len(out) == 2
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(10.0)


def test_resample_empty():
    out = resample(array("f"), 1.0, 0.5)
    assert len(out) == 0
