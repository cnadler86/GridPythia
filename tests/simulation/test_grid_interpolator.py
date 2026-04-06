"""Tests for Fraunhofer self-consumption interpolation/model behavior."""

from datetime import datetime, timedelta

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
