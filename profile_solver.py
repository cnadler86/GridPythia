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

    # Rolling-Horizon: 15 Durchläufe, jeweils PredictionData um 1 Sample kürzen, initial SoC übernehmen
    num_rolls = 15

    soc_percent = None  # Startwert: aus Config oder Default
    inverter_mode = None  # Startwert: None (IDLE)
    pred_current = pred
    results_table = []

    for roll in range(num_rolls):
        print(f"\n{'=' * 30} ROLLING HORIZON STEP {roll + 1} / {num_rolls} {'=' * 30}")

        # Setze initial SoC und initial Mode für alle Batterien

        for inv in inverters:
            if inv.battery is not None:
                params = inv.battery.parameters.model_copy(
                    update={"initial_soc_percentage": soc_percent}
                    if soc_percent is not None
                    else {}
                )
                inv.battery.parameters = params
                inv.battery._setup()
                # Setze explizit den aktuellen SoC in Wh
                if soc_percent is not None:
                    inv.battery.soc_wh = inv.battery.capacity_wh * soc_percent / 100.0

        # initial_modes für Optimizer vorbereiten (immer mit dem mode aus dem letzten Run, analog zu soc_percent)
        initial_modes = None
        if inverter_mode is not None:
            initial_modes = {inv.device_id: inverter_mode for inv in inverters}

        # Build and solve once per rolling step, using carried-over initial modes
        pr = cProfile.Profile()
        pr.enable()

        t_build_start = time.perf_counter()
        optimizer = LinearOptimizer(inverters, pred_current)
        t_build = time.perf_counter() - t_build_start

        t_solve_start = time.perf_counter()
        solution = optimizer.solve(
            OptimizationObjective.MINIMIZE_COST,
            validate_with_simulation=False,
            initial_modes=initial_modes,
        )
        t_solve = time.perf_counter() - t_solve_start

        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(15)
        print(s.getvalue())

        # Summary
        print(f"Build time: {t_build:.4f} s")
        print(f"Solve wall time: {t_solve:.4f} s (solver reported {solution.solve_time_s:.4f} s)")
        print(f"Status: {solution.solver_status}")
        print(f"Objective value (problem): {getattr(solution.result, 'total_cost', 'N/A')}")

        # Rolling-Horizon: Hole SoC und Mode für nächsten Schritt (Index 1)
        soc_dict = getattr(solution.result, "battery_soc_percentage_per_dt", None)
        modes_dict = getattr(solution.result, "inverter_modes_per_dt", None)
        soc_used = None
        mode_used = None
        inv_id = None
        # prefer index 1 (next step) — fall back to index 0 if horizon==1
        if soc_dict and isinstance(soc_dict, dict):
            inv_ids = list(soc_dict.keys())
            if inv_ids:
                inv_id = inv_ids[0]
                arr = soc_dict[inv_id]
                if isinstance(arr, (list, np.ndarray)) and len(arr) > 1:
                    soc_used = float(arr[1])
                elif isinstance(arr, (list, np.ndarray)) and len(arr) > 0:
                    soc_used = float(arr[0])
                    print("WARN: SoC-Array nur Länge 1, verwende Index 0 für nächsten Start.")
                else:
                    print("WARN: SoC-Array zu kurz, initial SoC bleibt unverändert.")
            else:
                print("WARN: Keine Inverter-IDs im SoC-Ergebnis.")
        else:
            print("WARN: Ergebnis hat kein battery_soc_percentage_per_dt-Attribut.")

        if modes_dict and isinstance(modes_dict, dict):
            inv_ids = list(modes_dict.keys())
            if inv_ids:
                inv_id = inv_ids[0]
                arr = modes_dict[inv_id]
                if isinstance(arr, (list, np.ndarray)) and len(arr) > 1:
                    mode_used = int(arr[1])
                elif isinstance(arr, (list, np.ndarray)) and len(arr) > 0:
                    mode_used = int(arr[0])
                    print("WARN: Mode-Array nur Länge 1, verwende Index 0 für nächsten Start.")
                else:
                    print("WARN: Mode-Array zu kurz, initial Mode bleibt unverändert.")
            else:
                print("WARN: Keine Inverter-IDs im Mode-Ergebnis.")
        else:
            print("WARN: Ergebnis hat kein inverter_modes_per_dt-Attribut.")

        if soc_used is not None:
            print(f"Initial SoC für nächsten Schritt: {soc_used:.2f}% (aus {inv_id})")
            soc_percent = soc_used
        if mode_used is not None:
            print(f"Initial Mode für nächsten Schritt: {mode_used} (aus {inv_id})")
            inverter_mode = mode_used

        # Ergebnisse für Tabelle sammeln
        results_table.append(
            {
                "roll": roll + 1,
                "build_time": t_build,
                "solve_time": t_solve,
                "solver_time": solution.solve_time_s,
                "status": solution.solver_status,
                "objective": getattr(solution.result, "total_cost", None),
                "soc_next": soc_used,
                "mode_next": mode_used,
                "steps": pred_current.steps,
            }
        )

        # PredictionData für nächsten Schritt: entferne erstes Sample
        if pred_current.steps > 1:
            pred_current = PredictionData(
                timestamps=pred_current.timestamps[1:],
                dt_hours=pred_current.dt_hours,
                load_wh=pred_current.load_wh[1:],
                electricprice_eur_wh=pred_current.electricprice[1:]
                if pred_current.electricprice is not None
                else None,
                feedintariff_eur_wh=pred_current.feedintariff[1:]
                if pred_current.feedintariff is not None
                else None,
                pv_by_inverter={k: v[1:] for k, v in pred_current.pv_by_inverter.items()},
                weather_by_channel={k: v[1:] for k, v in pred_current.weather_by_channel.items()}
                if hasattr(pred_current, "weather_by_channel") and pred_current.weather_by_channel
                else None,
            )
        else:
            print("Rolling horizon beendet: PredictionData leer.")

    # Am Ende: Tabelle ausgeben
    print("\n==================== ROLLING HORIZON SUMMARY ====================")
    print(
        f"{'Step':>4} | {'Steps':>5} | {'Build [s]':>9} | {'Solve [s]':>9} | {'Solver [s]':>10} | {'Status':>8} | {'Obj':>12} | {'SoC_next':>9} | {'Mode_next':>9}"
    )
    print("-" * 100)
    for row in results_table:
        print(
            f"{row['roll']:>4} | {row['steps']:>5} | {row['build_time']:9.4f} | {row['solve_time']:9.4f} | {row['solver_time']:10.4f} | {row['status']:>8} | {row['objective']:12.6f} | {row['soc_next']:.2f} | {str(row['mode_next']):>9}"
        )


if __name__ == "__main__":
    profile_solver()
