"""Tests for GridPythia.optimization.runner – the optimization orchestration layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from GridPythia.optimization.runner import OptimizationResult, run_optimization
from GridPythia.prediction.base import ceil_to_slot, floor_to_slot, round_to_slot
from GridPythia.prediction.prediction import PredictionData, PredictionSetup


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_pdata(steps: int, start: datetime, dt_hours: float = 0.25) -> PredictionData:
    ts = [start + timedelta(hours=i * dt_hours) for i in range(steps)]
    return PredictionData(
        timestamps=ts,
        dt_hours=dt_hours,
        load_wh=np.ones(steps, dtype=np.float32) * 100.0,
        electricprice_eur_wh=np.ones(steps, dtype=np.float32) * 0.0003,
        feedintariff_eur_wh=np.zeros(steps, dtype=np.float32),
        pv_by_inverter={},
    )


def _make_mock_prediction(fetch_pdata: PredictionData):
    """Return a mock Prediction that returns *fetch_pdata* from fetch()."""
    mock = MagicMock()
    mock.fetch = AsyncMock(return_value=fetch_pdata)
    return mock


def _make_mock_optimizer(solution=None):
    """Return a mock LinearOptimizer that returns a dummy solution from solve()."""
    if solution is None:
        solution = MagicMock()
        solution.solver_status = "optimal"
        solution.solve_time_s = 0.1
        solution.inverter_plans = []
    mock = MagicMock()
    mock.solve = MagicMock(return_value=solution)
    mock.inverters = []
    return mock


# ── TZ enforcement ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_optimization_rejects_naive_start():
    with pytest.raises(ValueError, match="timezone-aware"):
        await run_optimization(
            start=datetime(2025, 6, 15, 11, 0),  # naive
            end=datetime(2025, 6, 15, 19, 0, tzinfo=timezone.utc),
            prediction=_make_mock_prediction(_make_pdata(32, datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc))),
            optimizer=_make_mock_optimizer(),
            dt_hours=0.25,
        )


@pytest.mark.asyncio
async def test_run_optimization_rejects_naive_end():
    with pytest.raises(ValueError, match="timezone-aware"):
        await run_optimization(
            start=datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc),
            end=datetime(2025, 6, 15, 19, 0),  # naive
            prediction=_make_mock_prediction(_make_pdata(32, datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc))),
            optimizer=_make_mock_optimizer(),
            dt_hours=0.25,
        )


# ── Slot alignment ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_optimization_floors_start_for_fetch():
    """fetch() is called with floor(start), not the raw start."""
    dt = 0.25
    start = datetime(2025, 6, 15, 11, 18, tzinfo=timezone.utc)  # not aligned
    end = start + timedelta(hours=8)

    expected_fetch_start = floor_to_slot(start, dt)  # 11:15

    # build pdata starting at expected_fetch_start
    fetch_pdata = _make_pdata(33, expected_fetch_start, dt)
    mock_pred = _make_mock_prediction(fetch_pdata)
    mock_opt = _make_mock_optimizer()

    result = await run_optimization(
        start=start,
        end=end,
        prediction=mock_pred,
        optimizer=mock_opt,
        dt_hours=dt,
    )

    # Verify fetch was called with floored start
    call_kwargs = mock_pred.fetch.call_args
    fetched_start = call_kwargs.kwargs.get("start") or call_kwargs.args[0]
    assert fetched_start == expected_fetch_start


@pytest.mark.asyncio
async def test_run_optimization_solver_start_rounds_to_nearest_second_half():
    """When start is in the second half of a slot, solver_start = next slot (like ceil)."""
    dt = 0.25
    # 11:23 is 8 min into the 11:15 slot (midpoint = 11:22:30) → round → 11:30
    start = datetime(2025, 6, 15, 11, 23, tzinfo=timezone.utc)
    end = start + timedelta(hours=8)

    expected_solver_start = round_to_slot(start, dt)  # 11:30 (ceil equivalent)
    expected_fetch_start = floor_to_slot(start, dt)   # 11:15

    # solver_start must equal ceil here
    assert expected_solver_start == ceil_to_slot(start, dt)

    fetch_pdata = _make_pdata(33, expected_fetch_start, dt)
    mock_pred = _make_mock_prediction(fetch_pdata)
    mock_opt = _make_mock_optimizer()

    result = await run_optimization(
        start=start,
        end=end,
        prediction=mock_pred,
        optimizer=mock_opt,
        dt_hours=dt,
    )

    assert result.solver_start == expected_solver_start
    assert result.solver_pdata.timestamps[0] == expected_solver_start
    # fetch_pdata starts at floor(start)
    assert result.fetch_pdata.timestamps[0] == expected_fetch_start


@pytest.mark.asyncio
async def test_run_optimization_solver_start_rounds_to_nearest_first_half():
    """When start is in the first half of a slot, solver_start = same slot (like floor)."""
    dt = 0.25
    # 11:18 is 3 min into the 11:15 slot (midpoint = 11:22:30) → round → 11:15
    start = datetime(2025, 6, 15, 11, 18, tzinfo=timezone.utc)
    end = start + timedelta(hours=8)

    expected_solver_start = round_to_slot(start, dt)  # 11:15 (floor equivalent)
    expected_fetch_start = floor_to_slot(start, dt)   # 11:15

    # solver_start and fetch_start must be the same
    assert expected_solver_start == expected_fetch_start

    fetch_pdata = _make_pdata(33, expected_fetch_start, dt)
    mock_pred = _make_mock_prediction(fetch_pdata)
    mock_opt = _make_mock_optimizer()

    result = await run_optimization(
        start=start,
        end=end,
        prediction=mock_pred,
        optimizer=mock_opt,
        dt_hours=dt,
    )

    assert result.solver_start == expected_solver_start
    assert result.solver_pdata.timestamps[0] == expected_solver_start
    assert result.fetch_pdata.timestamps[0] == expected_fetch_start


@pytest.mark.asyncio
async def test_run_optimization_aligned_start_no_slice():
    """When start is already aligned, solver_start == start (no data dropped)."""
    dt = 0.25
    start = datetime(2025, 6, 15, 11, 15, tzinfo=timezone.utc)  # exactly aligned
    end = start + timedelta(hours=8)

    fetch_pdata = _make_pdata(32, start, dt)
    mock_pred = _make_mock_prediction(fetch_pdata)
    mock_opt = _make_mock_optimizer()

    result = await run_optimization(
        start=start,
        end=end,
        prediction=mock_pred,
        optimizer=mock_opt,
        dt_hours=dt,
    )

    assert result.solver_start == start
    # When start is aligned: fetch_pdata and solver_pdata have same first timestamp
    assert result.solver_pdata.timestamps[0] == start
    assert result.fetch_pdata.timestamps[0] == start
    assert result.solver_pdata.steps == result.fetch_pdata.steps


@pytest.mark.asyncio
async def test_run_optimization_pdata_transform_applied_before_slice():
    """pdata_transform is called on fetch_pdata before slicing to solver_pdata."""
    dt = 0.25
    start = datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=4)

    fetch_pdata = _make_pdata(16, start, dt)
    mock_pred = _make_mock_prediction(fetch_pdata)
    mock_opt = _make_mock_optimizer()

    transformed_marker = []

    def transform(pd: PredictionData) -> PredictionData:
        transformed_marker.append(True)
        return pd  # identity transform

    await run_optimization(
        start=start,
        end=end,
        prediction=mock_pred,
        optimizer=mock_opt,
        dt_hours=dt,
        pdata_transform=transform,
    )

    assert transformed_marker == [True], "pdata_transform must be called exactly once"


@pytest.mark.asyncio
async def test_run_optimization_returns_result_dataclass():
    dt = 0.25
    start = datetime(2025, 6, 15, 11, 15, tzinfo=timezone.utc)
    end = start + timedelta(hours=4)

    fetch_pdata = _make_pdata(16, start, dt)
    mock_pred = _make_mock_prediction(fetch_pdata)
    mock_opt = _make_mock_optimizer()

    result = await run_optimization(
        start=start, end=end,
        prediction=mock_pred, optimizer=mock_opt, dt_hours=dt,
    )

    assert isinstance(result, OptimizationResult)
    assert result.solution is not None
    assert isinstance(result.fetch_pdata, PredictionData)
    assert isinstance(result.solver_pdata, PredictionData)
    assert result.solver_start.tzinfo is not None
