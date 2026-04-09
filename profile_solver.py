"""Profiling script to identify performance bottlenecks in LinearOptimizer."""

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import numpy as np
import yaml

from GridPythia.config.optimization import BatteryParameters, InverterParameters, OptimizationConfig
from GridPythia.optimization.solver import LinearOptimizer, OptimizationObjective
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


def load_result_json(json_path: str | Path) -> dict:
    """Load JSON from file (either a result wrapper or a raw prediction payload)."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_prediction_from_result(result: dict) -> PredictionData:
    """Create PredictionData from either a result JSON (with `prediction`).

    or directly from a prediction payload (contains `load_wh`, `electricprice_eur_wh`, etc.).
    """
    # Accept either {"prediction": {...}} or the prediction dict itself.
    if isinstance(result.get("prediction"), dict):
        pred = result["prediction"]
    elif isinstance(result.get("result", {}).get("prediction"), dict):
        pred = result["result"]["prediction"]
    else:
        # treat the whole file as a prediction payload
        pred = result

    if not isinstance(pred, dict):
        raise RuntimeError("Missing prediction payload in JSON input.")

    # Extract channels
    price = list(pred.get("electricprice_eur_wh", []))
    feedin = list(pred.get("feedintariff_eur_wh", []))
    load = list(pred.get("load_wh", []))

    # pv series may be provided as 'pv_by_inverter' dict or as pv_<id>_wh keys
    pv_series = []
    if "pv_by_inverter" in pred and isinstance(pred["pv_by_inverter"], dict):
        # use first inverter series
        first = next(iter(pred["pv_by_inverter"].values()), [])
        pv_series = list(first)
    else:
        pv_keys = [k for k in pred.keys() if k.startswith("pv_") and k.endswith("_wh")]
        if pv_keys:
            pv_series = list(pred.get(pv_keys[0], []))

    # Determine horizon length n (min of available non-empty channels)
    lens = [len(x) for x in (price, feedin, load, pv_series) if len(x) > 0]
    n = min(lens) if lens else max(len(price), len(load), len(pv_series), 0)
    if n == 0:
        raise RuntimeError("Prediction payload contains no time series data.")

    price_arr = np.array(price[:n] if price else [0.0003] * n, dtype=np.float32)
    feedin_arr = np.array(feedin[:n] if feedin else [0.0] * n, dtype=np.float32)
    load_arr = np.array(load[:n] if load else [0.0] * n, dtype=np.float32)
    pv_arr = np.array(pv_series[:n] if pv_series else [0.0] * n, dtype=np.float32)

    # Use provided timestamps if available
    from datetime import datetime, timedelta

    if (
        "timestamps" in pred
        and isinstance(pred["timestamps"], list)
        and len(pred["timestamps"]) >= n
    ):
        # parse provided ISO timestamps
        try:
            timestamps = [datetime.fromisoformat(ts) for ts in pred["timestamps"][:n]]
        except Exception:
            start = datetime(2020, 1, 1)
            timestamps = [start + i * timedelta(minutes=15) for i in range(n)]
    else:
        start = datetime(2020, 1, 1)
        timestamps = [start + i * timedelta(minutes=15) for i in range(n)]

    return PredictionData(
        timestamps=timestamps,
        dt_hours=float(pred.get("dt_hours", 0.25)),
        load_wh=load_arr,
        electricprice_eur_wh=price_arr,
        feedintariff_eur_wh=feedin_arr,
        pv_by_inverter={"inverter1": pv_arr},
    )


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
        )
    )

    inv = InverterBase(
        InverterParameters(
            device_id="inverter1",
            battery_id="inverter1",
            has_pv=True,
            max_ac_output_power_w=800,
            max_ac_charge_power_w=1000,
            dc_to_ac_efficiency=0.95,
            ac_to_dc_efficiency=0.95,
            zero_feed_in=True,
            mode_switch_cost=0.005,
        ),
        battery=battery,
    )
    return inv


def profile_solver():
    """Profile the LinearOptimizer with real data from result.

    Measures:
    - build time (construct optimizer)
    - first solve time
    - mutate parameters on same inverter/battery instances
    - second solve time
    """
    # Determine input path: default to result.json in script dir
    parser = argparse.ArgumentParser(
        description="Profile LinearOptimizer using a prediction/result JSON"
    )
    parser.add_argument("input", nargs="?", help="Path to result or prediction JSON (fixture)")
    args = parser.parse_args()

    if args.input:
        result_path = Path(args.input)
    else:
        result_path = Path(__file__).parent / "result.json"

    # If the chosen path doesn't exist, try to fall back to the first fixture
    if not result_path.exists():
        fixtures_dir = Path(__file__).parent / "tests" / "optimization" / "fixtures"
        if fixtures_dir.exists() and fixtures_dir.is_dir():
            json_files = sorted(fixtures_dir.glob("*.json"))
            if json_files:
                result_path = json_files[0]
                print(f"No input provided — using fixture {result_path}")
            else:
                print(f"No JSON fixtures found in {fixtures_dir}")
                print("Provide a path to a prediction/result JSON as the first argument.")
                return
        else:
            print(f"Input JSON not found at {result_path}")
            print("Provide a path to a prediction/result JSON as the first argument.")
            return

    result = load_result_json(result_path)
    pred = create_prediction_from_result(result)

    print(f"Prediction steps: {pred.steps}")
    print(f"Prediction dt: {pred.dt_hours} hours")
    print()

    # Load YAML config (optimization section) if available
    cfg_path = Path(__file__).parent / "config.yaml"
    optimization_cfg = None
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        try:
            optimization_cfg = OptimizationConfig.model_validate(cfg.get("optimization", {}))
        except Exception:
            optimization_cfg = None

    # Build inverters from config or fallback
    if optimization_cfg and optimization_cfg.inverters:
        # create Battery objects first
        bats: dict[str, Battery] = {}
        for bat_p in optimization_cfg.batteries:
            b = Battery(bat_p)
            bats[bat_p.device_id] = b

        inverters: list[InverterBase] = []
        for inv_p in optimization_cfg.inverters:
            bat = bats.get(inv_p.battery_id) if inv_p.battery_id else None
            inv = InverterBase(inv_p, battery=bat)
            inverters.append(inv)
    else:
        inverters = [create_inverter()]

    # Profile building + solve + re-solve with modified params on same instance
    pr = cProfile.Profile()
    pr.enable()

    t_build_start = time.perf_counter()
    optimizer = LinearOptimizer(inverters, pred)
    t_build = time.perf_counter() - t_build_start

    # First solve
    t_solve1_start = time.perf_counter()
    solution1 = optimizer.solve(
        OptimizationObjective.MINIMIZE_COST,
        validate_with_simulation=False,
    )
    t_solve1 = time.perf_counter() - t_solve1_start

    # Modify parameters on the same instance: update only first battery initial SoC
    if optimization_cfg and optimization_cfg.batteries and inverters:
        # update battery initial SoC (example change)
        bat_cfg = optimization_cfg.batteries[0]
        new_init = min(95, max(5, bat_cfg.initial_soc_percentage + 5))
        new_bat_params = bat_cfg.model_copy(update={"initial_soc_percentage": new_init})
        # find corresponding Battery object
        target_bat = None
        for inv in inverters:
            if inv.battery is not None:
                target_bat = inv.battery
                break
        if target_bat is not None:
            target_bat.parameters = new_bat_params
            target_bat._setup()

    # Second solve (same optimizer instance)
    t_solve2_start = time.perf_counter()
    solution2 = optimizer.solve(
        OptimizationObjective.MINIMIZE_COST,
        validate_with_simulation=False,
    )
    t_solve2 = time.perf_counter() - t_solve2_start

    pr.disable()

    # Print profiler stats (top 30)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    # Summary
    print("\n" + "=" * 80)
    print("PROFILE SUMMARY:")
    print(f"Build time: {t_build:.4f} s")
    print(
        f"First solve wall time: {t_solve1:.4f} s (solver reported {solution1.solve_time_s:.4f} s)"
    )
    print(
        f"Second solve wall time: {t_solve2:.4f} s (solver reported {solution2.solve_time_s:.4f} s)"
    )
    print()
    print("SOLUTION 1:")
    print(f"Status: {solution1.solver_status}")
    print(f"Objective value (problem): {getattr(solution1.result, 'total_cost', 'N/A')}")
    print()
    print("SOLUTION 2:")
    print(f"Status: {solution2.solver_status}")
    print(f"Objective value (problem): {getattr(solution2.result, 'total_cost', 'N/A')}")


if __name__ == "__main__":
    profile_solver()
