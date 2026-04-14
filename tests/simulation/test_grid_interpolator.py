"""Tests for Fraunhofer self-consumption interpolation/model behavior."""

from datetime import datetime, timedelta
import warnings

import numpy as np
import pytest

from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.grid_interpolator import FraunhoferSCModel
from GridPythia.simulation.grid_simulation import GridSimulation


def test_sc_ratio_near_one_for_small_pv_at_baseload() -> None:
    """If PV is much smaller than load, nearly all PV should be self-consumed."""
    model = FraunhoferSCModel(baseload_wh=500.0, dt=0.25)

    sc = model.sc_ratio(pv_wh=300.0, load_wh=800.0)

    assert 0.90 <= sc <= 1.0


def test_sc_ratio_decreases_with_higher_pv_for_fixed_load() -> None:
    """For constant load, SCR should decrease as PV/load ratio increases."""
    model = FraunhoferSCModel(baseload_wh=500.0, dt=1.0)

    sc_low = model.sc_ratio(pv_wh=50.0, load_wh=500.0)
    sc_mid = model.sc_ratio(pv_wh=500.0, load_wh=500.0)
    sc_high = model.sc_ratio(pv_wh=2000.0, load_wh=500.0)

    assert 0.0 <= sc_high <= sc_mid <= sc_low <= 1.0


def test_self_consumed_and_grid_feed_in_match_sc_definition() -> None:
    """Energy helper methods should be consistent with SC ratio definition."""
    model = FraunhoferSCModel(baseload_wh=600.0, dt=1.0)

    pv_wh = 900.0
    load_wh = 600.0
    sc = model.sc_ratio(pv_wh=pv_wh, load_wh=load_wh)
    e_self = model.self_consumed_wh(pv_wh=pv_wh, load_wh=load_wh)
    e_feed = model.grid_feed_in_wh(pv_wh=pv_wh, load_wh=load_wh)

    assert e_self == pytest.approx(sc * pv_wh, abs=1e-9)
    assert e_feed == pytest.approx((1.0 - sc) * pv_wh, abs=1e-9)
    assert e_self + e_feed == pytest.approx(pv_wh, abs=1e-9)


def test_sc_ratio_returns_one_for_zero_pv() -> None:
    model = FraunhoferSCModel(baseload_wh=500.0, dt=1.0)

    sc = model.sc_ratio(pv_wh=0.0, load_wh=500.0)

    assert sc == pytest.approx(1.0)

def test_sc_ratio_returns_one_for_pv_below_baseload() -> None:
    model = FraunhoferSCModel(baseload_wh=500.0, dt=0.25)

    sc = model.sc_ratio(pv_wh=300.0, load_wh=1000.0)

    assert sc == pytest.approx(1)


def test_sc_ratio_increases_with_higher_baseload_for_same_inputs() -> None:
    """For same PV and dt, a higher baseload should yield a higher SCR."""
    low_baseload_model = FraunhoferSCModel(baseload_wh=400.0, dt=1.0)
    high_baseload_model = FraunhoferSCModel(baseload_wh=1200.0, dt=1.0)

    pv_wh = 600.0
    sc_low_baseload = low_baseload_model.sc_ratio(pv_wh=pv_wh)
    sc_high_baseload = high_baseload_model.sc_ratio(pv_wh=pv_wh)

    assert 0.0 <= sc_low_baseload <= 1.0
    assert 0.0 <= sc_high_baseload <= 1.0
    assert sc_high_baseload > sc_low_baseload


def test_grid_simulation_uses_prediction_dt_and_min_load_for_fraunhofer_init() -> None:
    """GridSimulation should initialize Fraunhofer model from prediction metadata."""
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(3)]
    prediction = PredictionData(
        timestamps=timestamps,
        dt_hours=0.25,
        load_wh=np.array([850.0, 400.0, 620.0], dtype=np.float32),
        electricprice_eur_wh=np.array([0.0003, 0.0003, 0.0003], dtype=np.float32),
        feedintariff_eur_wh=np.array([0.0001, 0.0001, 0.0001], dtype=np.float32),
    )

    sim = GridSimulation(
        prediction=prediction,
        inverters=None,
        home_appliances=None,
    )

    assert sim._fraunhofer_sc_model.dt == pytest.approx(0.25)
    assert sim._fraunhofer_sc_model.baseload_wh == pytest.approx(400.0)

    # The initialized model should deliver bounded SCR values.
    scr = sim._fraunhofer_sc_model.sc_ratio(pv_wh=100.0, load_wh=400.0)
    assert 0.0 <= scr <= 1.0


def test_linearize_batch_handles_zero_load_without_runtime_warning() -> None:
    """Batch linearization must not emit invalid-divide warnings for zero-load steps."""
    model = FraunhoferSCModel(baseload_wh=500.0, dt=0.25)
    pv_0 = np.array([0.0, 50.0, 300.0, 900.0], dtype=np.float64)
    load_0 = np.array([0.0, 0.0, 400.0, 800.0], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        c_const, c_pv, c_load = model.linearize_batch(pv_0=pv_0, load_0=load_0)

    assert c_const.shape == pv_0.shape
    assert c_pv.shape == pv_0.shape
    assert c_load.shape == pv_0.shape
    assert np.all(np.isfinite(c_const))
    assert np.all(np.isfinite(c_pv))
    assert np.all(np.isfinite(c_load))


def test_linearize_batch_matches_scalar_linearize_coefficients() -> None:
    """Vectorized linearization coefficients should match per-step scalar linearization."""
    model = FraunhoferSCModel(baseload_wh=500.0, dt=0.25)
    pv_0 = np.array([0.0, 20.0, 150.0, 500.0, 1200.0], dtype=np.float64)
    load_0 = np.array([0.0, 50.0, 200.0, 800.0, 1500.0], dtype=np.float64)

    c_const_b, c_pv_b, c_load_b = model.linearize_batch(pv_0=pv_0, load_0=load_0)

    c_const_s = []
    c_pv_s = []
    c_load_s = []
    for pv, load in zip(pv_0, load_0, strict=False):
        lc = model.linearize(pv_0=float(pv), load_0=float(load))
        c_const_s.append(lc.c_const)
        c_pv_s.append(lc.c_pv)
        c_load_s.append(lc.c_load)

    np.testing.assert_allclose(c_const_b, np.array(c_const_s), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(c_pv_b, np.array(c_pv_s), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(c_load_b, np.array(c_load_s), rtol=0.0, atol=1e-12)


def test_linearization_envelope_is_tight_at_operating_point_and_upper_bounds_locally() -> None:
    """The LP envelope (linearized row + physical bounds) should conservatively bound exact model."""
    model = FraunhoferSCModel(baseload_wh=500.0, dt=0.25)
    # Choose interior operating points where SC clipping is inactive.
    pv_0 = np.array([300.0, 500.0, 1200.0, 2000.0], dtype=np.float64)
    load_0 = np.array([200.0, 500.0, 900.0, 1200.0], dtype=np.float64)

    c_const, c_pv, c_load = model.linearize_batch(pv_0=pv_0, load_0=load_0)
    e_exact_0 = np.asarray(model.self_consumed_wh(pv_0, load_0), dtype=np.float64)
    e_lin_0 = c_const + c_pv * pv_0 + c_load * load_0

    # Tangency at operating point.
    np.testing.assert_allclose(e_lin_0, e_exact_0, rtol=0.0, atol=1e-9)

    pv_factors = np.array([0.5, 0.8, 1.0, 1.2, 1.5], dtype=np.float64)
    load_factors = np.array([0.6, 0.9, 1.0, 1.1, 1.4], dtype=np.float64)

    for i in range(pv_0.size):
        pv_test = np.maximum(0.0, pv_0[i] * pv_factors)
        load_test = np.maximum(1e-6, load_0[i] * load_factors)
        exact = np.asarray(model.self_consumed_wh(pv_test, load_test), dtype=np.float64)
        lin = c_const[i] + c_pv[i] * pv_test + c_load[i] * load_test

        # The optimization model applies all three constraints simultaneously.
        envelope = np.minimum(np.minimum(lin, pv_test), load_test)
        assert np.all(envelope + 1e-8 >= exact)
