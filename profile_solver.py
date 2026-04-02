"""Profiling script to identify performance bottlenecks in LinearOptimizer."""

import cProfile
import io
import json
import pstats
from pathlib import Path

import numpy as np

from GridPythia.config.models import BatteryParameters, InverterParameters
from GridPythia.optimization.solver import LinearOptimizer, OptimizationObjective
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


def load_result_json(json_path: str | Path) -> dict:
    """Load result JSON from file."""
    with open(json_path, "r") as f:
        return json.load(f)


def create_prediction_from_result(result: dict) -> PredictionData:
    """Create PredictionData from result JSON."""
    res = result["result"]

    # Get the minimum length from all time-series arrays
    costs_len = len(res.get("costs_per_dt", []))
    revenue_len = len(res.get("revenue_per_dt", []))
    feedin_len = len(res.get("feedin_wh_per_dt", []))
    grid_import_len = len(res.get("grid_import_wh_per_dt", []))
    pv_len = len(res.get("solar_generation_wh_per_dt", {}).get("inverter1", []))

    n = min(costs_len, revenue_len, feedin_len, grid_import_len, pv_len)

    # Use electricity_price_per_dt if available, otherwise use a default price
    price_data = res.get("electricity_price_per_dt", [])
    if len(price_data) < n:
        price_data = [0.0003] * n  # Default price
    else:
        price_data = price_data[:n]

    data = {
        "electricprice_eur_wh": np.array(price_data, dtype=np.float32),
        "feedintariff_eur_wh": np.array(res["revenue_per_dt"][:n], dtype=np.float32),
        "load_wh": np.array(res["grid_import_wh_per_dt"][:n], dtype=np.float32),
        "pv_inverter1_wh": np.array(
            res["solar_generation_wh_per_dt"]["inverter1"][:n], dtype=np.float32
        ),
    }

    from datetime import datetime, timedelta

    start = datetime(2020, 1, 1)
    timestamps = [start + i * timedelta(minutes=15) for i in range(n)]
    return PredictionData(_timestamps=timestamps, _arrays=data, dt_hours=0.25)


def create_inverter() -> InverterBase:
    """Create the inverter from result config."""
    battery = Battery(
        BatteryParameters(
            device_id="inverter1",
            capacity_wh=1920,
            charging_efficiency=0.98,
            discharging_efficiency=0.98,
            max_charge_power_w=1000,
            max_discharge_power_w=800,
            initial_soc_percentage=50,
            min_soc_percentage=20,
            max_soc_percentage=100,
        ),
        prediction_hours=int(256 * 0.25),  # 64 hours
    )

    inv = InverterBase(
        InverterParameters(
            device_id="inverter1",
            battery_id="inverter1",
            pv_source="inverter1",
            max_ac_output_power_w=800,
            max_ac_charge_power_w=1000,
            dc_to_ac_efficiency=0.95,
            ac_to_dc_efficiency=0.95,
            zero_feed_in=True,
            ac_rates_pct=(50, 100),
            mode_switch_cost=0.005,
        ),
        battery=battery,
    )
    return inv


def profile_solver():
    """Profile the LinearOptimizer with real data from result."""
    # Load result JSON
    result_path = Path(__file__).parent / "result.json"
    if not result_path.exists():
        print(f"Result JSON not found at {result_path}")
        print("Please save the result JSON first.")
        return

    result = load_result_json(result_path)
    pred = create_prediction_from_result(result)
    inv = create_inverter()

    print(f"Prediction steps: {pred.steps}")
    print(f"Prediction dt: {pred.dt_hours} hours")
    print()

    # Profile solve
    pr = cProfile.Profile()
    pr.enable()

    optimizer = LinearOptimizer([inv], pred)
    solution = optimizer.solve(
        OptimizationObjective.MINIMIZE_COST,
        validate_with_simulation=False,
    )

    pr.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())

    # Summary
    print("\n" + "=" * 80)
    print("SOLVER RESULT:")
    print(f"Status: {solution.solver_status}")
    print(f"Solve time: {solution.solve_time_s:.2f}s")
    print(f"Objective value: {solution.result.total_cost:.6f} EUR")
    print(f"Total cost: {solution.result.total_cost:.6f} EUR")


if __name__ == "__main__":
    profile_solver()
