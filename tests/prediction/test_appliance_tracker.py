"""Tests for appliance_tracker helpers (normal CDF, MAD outlier filter)."""

from __future__ import annotations

import numpy as np
import pytest

from GridPythia.prediction.load.appliance_tracker import _mad_filter, _norm_cdf


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0.5),
        (1.0, 0.8413447),
        (-1.0, 0.1586553),
        (1.96, 0.9750021),
        (-1.96, 0.0249979),
        (2.5758, 0.9950002),
    ],
)
def test_norm_cdf_matches_known_values(x: float, expected: float) -> None:
    # Abramowitz & Stegun 26.2.17 has an inherent absolute error up to ~1e-6.
    assert _norm_cdf(x) == pytest.approx(expected, abs=1e-6)


def test_norm_cdf_saturates_in_tails() -> None:
    assert _norm_cdf(-7.0) == 0.0
    assert _norm_cdf(7.0) == 1.0


def test_norm_cdf_is_symmetric() -> None:
    for x in (0.3, 1.1, 2.4):
        assert _norm_cdf(x) + _norm_cdf(-x) == pytest.approx(1.0, abs=1e-6)


def test_mad_filter_removes_outliers() -> None:
    values = np.array([10.0, 11.0, 9.0, 10.5, 100.0])
    filtered = _mad_filter(values)
    assert 100.0 not in filtered
    assert len(filtered) == 4


def test_mad_filter_keeps_small_samples_untouched() -> None:
    values = np.array([1.0, 50.0, 2.0])
    # Fewer than 4 samples: no filtering applied.
    assert np.array_equal(_mad_filter(values), values)
