"""Profiling script to identify performance bottlenecks in LinearOptimizer."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from GridPythia.config.optimization import BatteryParameters, InverterParameters, OptimizationConfig
from GridPythia.optimization.solver import LinearOptimizer, OptimizationObjective
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase

# Constants for tuning candidates and run duration
TUNING_CANDIDATES = [
    {},
    {"mip_rel_gap": 0.02},
    {"mip_rel_gap": 0.03, "presolve": "on"},
    {"mip_rel_gap": 0.05, "presolve": "on"},
    {"mip_lp_solver": "simplex"},
    {"mip_lp_solver": "ipm", "mip_rel_gap": 0.02},
    {"mip_heuristic_run_rens": False, "mip_heuristic_run_rins": True},
    {"mip_heuristic_run_rens": True, "mip_heuristic_run_rins": True},
    {"mip_heuristic_run_root_reduced_cost": True, "mip_rel_gap": 0.03},
    {"time_limit": 45, "mip_rel_gap": 0.05},
]

# Constants for default argument values
DEFAULT_ROLLING_STEPS = 15
DEFAULT_WINDOW_STEPS = 0
DEFAULT_WARM_START = True
DEFAULT_TUNE_ITERS = 10


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
        pv_series = list(first) if isinstance(first, (list, tuple, np.ndarray)) else []
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


def slice_prediction_window(
    prediction: PredictionData,
    *,
    start_idx: int,
    length: int,
) -> PredictionData:
    """Return a fixed-length rolling-horizon window from prediction data."""
    end_idx = start_idx + length
    return PredictionData(
        timestamps=prediction.timestamps[start_idx:end_idx],
        dt_hours=prediction.dt_hours,
        load_wh=prediction.load_wh[start_idx:end_idx],
        electricprice_eur_wh=prediction.electricprice[start_idx:end_idx]
        if prediction.electricprice is not None
        else None,
        feedintariff_eur_wh=prediction.feedintariff[start_idx:end_idx]
        if prediction.feedintariff is not None
        else None,
        pv_by_inverter={
            inverter_id: values[start_idx:end_idx]
            for inverter_id, values in prediction.pv_by_inverter.items()
        },
        weather_by_channel={
            channel: values[start_idx:end_idx]
            for channel, values in prediction.weather_by_channel.items()
        }
        if prediction.weather_by_channel
        else None,
    )


def profile_solver():
    """Profile the LinearOptimizer with real data from result.

    Measures:
    - build time (construct optimizer)
    - first solve time
    - mutate parameters on same inverter/battery instances
    - second solve time
    """
    # Determine input path: default to result.json in script dir
    parser = argparse.ArgumentParser(description="Profile LinearOptimizer on rolling horizon")
    parser.add_argument("input", nargs="?", help="Path to result or prediction JSON (fixture)")
    parser.add_argument(
        "--rolling-steps",
        type=int,
        default=DEFAULT_ROLLING_STEPS,
        help="Number of rolling-horizon solves",
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=DEFAULT_WINDOW_STEPS,
        help="Window length in steps (0 -> pred.steps - rolling_steps)",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        default=DEFAULT_WARM_START,
        help="Enable plan-shift warm start between rolling runs",
    )
    parser.add_argument(
        "--tune-iters",
        type=int,
        default=DEFAULT_TUNE_ITERS,
        help="Try up to N predefined solver option sets (max 10)",
    )
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

    solver_provider = optimization_cfg.solver.provider if optimization_cfg else "highs"
    if str(solver_provider).lower() != "highs":
        raise RuntimeError(
            f"This profiler currently supports only provider='highs'. Got '{solver_provider}'."
        )

    objective = OptimizationObjective.MINIMIZE_COST
    if optimization_cfg and optimization_cfg.solver.objective == "self_consumption":
        objective = OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION

    base_solver_opts = dict(optimization_cfg.solver.solver_opts) if optimization_cfg else {}

    # Rolling horizon layout
    requested_rolls = max(int(args.rolling_steps), 1)
    num_rolls = min(requested_rolls, 20)
    if pred.steps <= num_rolls:
        raise RuntimeError(
            f"Prediction horizon ({pred.steps}) must be longer than num_rolls ({num_rolls})."
        )
    if args.window_steps > 0:
        window_steps = max(1, min(int(args.window_steps), pred.steps - 1))
        num_rolls = min(num_rolls, pred.steps - window_steps)
    else:
        window_steps = pred.steps - num_rolls

    print(f"Rolling horizon windows: {num_rolls}")
    print(f"Window length per solve: {window_steps} steps")
    print(f"Solver provider: {solver_provider}")
    print(f"Objective: {objective.value}")
    print(f"Warm start enabled: {bool(args.warm_start)}")
    print()

    candidate_overrides = TUNING_CANDIDATES
    best_opts = dict(base_solver_opts)
    best_avg = float("inf")
    best_idx = -1
    if candidate_overrides:
        print("Tuning solver options...")
        tuning_rolls = min(num_rolls, 2)
        print(f"Tuning horizon subset: {tuning_rolls} rolling windows")
        for idx, override in enumerate(candidate_overrides, start=1):
            merged = {**base_solver_opts, **override}
            summary = _run_rolling_horizon(
                prediction=pred,
                inverters=inverters,
                objective=objective,
                solver_opts=merged,
                num_rolls=tuning_rolls,
                window_steps=window_steps,
                use_warm_start=bool(args.warm_start),
                verbose=False,
            )
            print(
                f"  candidate {idx}: avg_solve={summary['avg_solve_time']:.4f}s total={summary['total_solve_time']:.4f}s opts={override}"
            )
            if summary["avg_solve_time"] < best_avg:
                best_avg = float(summary["avg_solve_time"])
                best_opts = merged
                best_idx = idx

        print(f"Best candidate: {best_idx} with avg_solve={best_avg:.4f}s")
        print(f"Best solver_opts: {best_opts}")
        print()

    final = _run_rolling_horizon(
        prediction=pred,
        inverters=inverters,
        objective=objective,
        solver_opts=best_opts,
        num_rolls=num_rolls,
        window_steps=window_steps,
        use_warm_start=bool(args.warm_start),
        verbose=True,
    )

    # Am Ende: Tabelle ausgeben
    print("\n==================== ROLLING HORIZON SUMMARY ====================")
    print(
        f"{'Step':>4} | {'Steps':>5} | {'Build [s]':>9} | {'Solve [s]':>9} | {'Solver [s]':>10} | {'Status':>12} | {'Obj':>12}"
    )
    print("-" * 92)
    for row in final["rows"]:
        print(
            f"{row['roll']:>4} | {row['steps']:>5} | {row['build_time']:9.4f} | {row['solve_time']:9.4f} | {row['solver_time']:10.4f} | {row['status']:>12} | {row['objective']:12.6f}"
        )
    print(
        f"TOTAL solve wall time: {final['total_solve_time']:.4f}s (avg {final['avg_solve_time']:.4f}s)"
    )


def _run_rolling_horizon(
    *,
    prediction: PredictionData,
    inverters: list[InverterBase],
    objective: OptimizationObjective,
    solver_opts: dict,
    num_rolls: int,
    window_steps: int,
    use_warm_start: bool,
    verbose: bool,
) -> dict:
    rows: list[dict] = []
    optimizer: LinearOptimizer | None = None
    initial_modes: dict[str, InverterMode] | None = None
    initial_soc_wh: dict[str, float] | None = None
    warm_start_plan: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    total_solve_time = 0.0

    for roll in range(num_rolls):
        pred_current = slice_prediction_window(prediction, start_idx=roll, length=window_steps)
        if verbose:
            print(f"\n{'=' * 30} ROLLING HORIZON STEP {roll + 1} / {num_rolls} {'=' * 30}")

        # Apply runtime start state to physical objects
        if initial_soc_wh:
            for inv in inverters:
                if inv.battery is None:
                    continue
                start = initial_soc_wh.get(inv.device_id)
                if start is not None:
                    inv.battery.soc_wh = float(
                        np.clip(start, inv.battery.min_soc_wh, inv.battery.max_soc_wh)
                    )

        t_build = 0.0
        if optimizer is None:
            t0_build = time.perf_counter()
            optimizer = LinearOptimizer(inverters, pred_current)
            t_build = time.perf_counter() - t0_build
        else:
            optimizer.prediction = pred_current

        t0 = time.perf_counter()
        solution = optimizer.solve(
            objective=objective,
            solver_opts=solver_opts,
            validate_with_simulation=False,
            initial_modes=initial_modes,
            warm_start_plan=warm_start_plan if use_warm_start else None,
        )
        t_solve = time.perf_counter() - t0
        total_solve_time += t_solve

        # next initial state is state right after first control interval
        next_modes: dict[str, InverterMode] = {}
        next_soc_wh: dict[str, float] = {}
        for plan in solution.inverter_plans:
            if plan.modes.size > 0:
                next_modes[plan.device_id] = InverterMode(int(plan.modes[0]))
            soc_trace = solution.result.battery_wh_per_dt.get(plan.device_id)
            if soc_trace is not None and len(soc_trace) > 0:
                next_soc_wh[plan.device_id] = float(soc_trace[0])

        initial_modes = next_modes or None
        initial_soc_wh = next_soc_wh or None

        if use_warm_start:
            warm_start_plan = LinearOptimizer.shift_solution_for_next_horizon(
                solution,
                horizon_steps=window_steps,
                shift_steps=1,
            )

        objective_value = (
            float(solution.result.total_cost)
            if objective == OptimizationObjective.MINIMIZE_COST
            else float(solution.result.total_self_consumption)
        )
        rows.append(
            {
                "roll": roll + 1,
                "build_time": t_build,
                "solve_time": t_solve,
                "solver_time": solution.solve_time_s,
                "status": solution.solver_status,
                "objective": objective_value,
                "steps": pred_current.steps,
            }
        )

        if verbose:
            print(f"Build time: {t_build:.4f} s")
            print(
                f"Solve wall time: {t_solve:.4f} s (solver reported {solution.solve_time_s:.4f} s)"
            )
            print(f"Status: {solution.solver_status}")
            print(f"Objective value: {objective_value:.6f}")

    return {
        "rows": rows,
        "total_solve_time": total_solve_time,
        "avg_solve_time": total_solve_time / max(len(rows), 1),
    }


if __name__ == "__main__":
    profile_solver()
