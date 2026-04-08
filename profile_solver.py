"""Profiling script to identify performance bottlenecks in LinearOptimizer."""

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
from GridPythia.simulation.devices import SystemTopology
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


def load_result_json(json_path: str | Path) -> dict:
    """Load result JSON from file."""
    with open(json_path, "r") as f:
        return json.load(f)


def create_prediction_from_result(result: dict) -> PredictionData:
    """Create PredictionData from result JSON."""
    # This function now expects a top-level 'prediction' payload produced by
    # the optimizer/solver. Do not rely on legacy 'solar_generation_wh_per_dt'
    # or 'electricity_price_per_dt' fields which were removed from SimulationResult.
    pred = result.get("prediction") or result.get("result", {}).get("prediction")
    if not isinstance(pred, dict):
        raise RuntimeError(
            "Missing 'prediction' payload in result JSON. Provide 'prediction' with "
            "channels like 'electricprice_eur_wh', 'feedintariff_eur_wh', 'load_wh', 'pv_<id>_wh'."
        )

    # Extract arrays from prediction; prefer explicit channel names.
    price = list(pred.get("electricprice_eur_wh", []))
    feedin = list(pred.get("feedintariff_eur_wh", []))
    load = list(pred.get("load_wh", []))
    pv_keys = [k for k in pred.keys() if k.startswith("pv_") and k.endswith("_wh")]
    pv_series = list(pred.get(pv_keys[0], [])) if pv_keys else []

    # Determine horizon length n (min of available non-empty channels)
    lens = [len(x) for x in (price, feedin, load, pv_series) if len(x) > 0]
    n = min(lens) if lens else len(price)
    if n == 0:
        raise RuntimeError("Prediction payload contains no time series data.")

    price_arr = np.array(price[:n] if price else [0.0003] * n, dtype=np.float32)
    feedin_arr = np.array(feedin[:n] if feedin else [0.0] * n, dtype=np.float32)
    load_arr = np.array(load[:n] if load else [0.0] * n, dtype=np.float32)
    pv_arr = np.array(pv_series[:n] if pv_series else [0.0] * n, dtype=np.float32)

    from datetime import datetime, timedelta

    start = datetime(2020, 1, 1)
    timestamps = [start + i * timedelta(minutes=15) for i in range(n)]
    return PredictionData(
        timestamps=timestamps,
        dt_hours=0.25,
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
    # Load result JSON
    result_path = Path(__file__).parent / "result.json"
    if not result_path.exists():
        print(f"Result JSON not found at {result_path}")
        print("Please save the result JSON first.")
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

    # Modify parameters on the same instance: update first battery initial SoC and inverter mode cost
    if optimization_cfg and optimization_cfg.batteries and inverters:
        # update battery initial SoC (example change)
        bat_cfg = optimization_cfg.batteries[0]
        new_init = min(95, max(5, bat_cfg.initial_soc_percentage + 30))
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

        # update inverter params (increase mode_switch_cost) on first inverter
        inv0 = inverters[0]
        try:
            new_inv_params = inv0.parameters.model_copy(
                update={"mode_switch_cost": inv0.parameters.mode_switch_cost * 2.0}
            )
        except Exception:
            new_inv_params = inv0.parameters

        # apply new inverter params (update derived attrs)
        inv0.parameters = new_inv_params
        inv0._max_ac_output_power_w = new_inv_params.max_ac_output_power_w
        inv0._max_ac_charge_power_w = new_inv_params.max_ac_charge_power_w
        inv0._dc_to_ac_efficiency = new_inv_params.dc_to_ac_efficiency
        inv0._ac_to_dc_efficiency = new_inv_params.ac_to_dc_efficiency
        inv0._zero_feed_in = new_inv_params.zero_feed_in
        inv0._has_pv = getattr(new_inv_params, "has_pv", False)
        inv0.topology = inv0._resolve_topology()
        inv0.available_modes = inv0._resolve_available_modes()
        inv0.is_optimizable = (
            inv0.topology != SystemTopology.PV_ONLY and len(inv0.available_modes) > 1
        )

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
