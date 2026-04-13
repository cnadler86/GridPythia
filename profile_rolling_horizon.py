"""Profile rolling-horizon behavior on fixture prediction data.

This script is focused on diagnosing warm-start quality for repeated MILP solves.
It uses the project config for topology, fixture JSON as prediction input, and
prints per-roll timing and initial MIP bound/gap diagnostics parsed from HiGHS logs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from GridPythia.config import AppConfig
from GridPythia.config.optimization import OptimizationConfig
from GridPythia.optimization.solver import LinearOptimizer, OptimizationObjective
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase

DEFAULT_CONFIG = Path("config.yaml")
DEFAULT_FIXTURE = Path("tests/optimization/fixtures/prediction_2026_04_06_48h_15m.json")
DEFAULT_ROLLING_HOURS = 4.0


@dataclass
class StartMipStats:
    first_best_bound: float | None
    first_best_sol: float | None
    first_gap_pct: float | None
    root_best_bound: float | None
    root_best_sol: float | None
    root_gap_pct: float | None


@dataclass
class RollStats:
    roll: int
    start_idx: int
    solve_wall_s: float
    solver_reported_s: float
    status: str
    objective: float
    start_mip: StartMipStats


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts[:-1] + "+00:00")
        raise


def load_fixture_prediction(path: Path) -> PredictionData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Fixture payload must be a JSON object")

    timestamps_raw = payload.get("timestamps")
    load_wh = payload.get("load_wh")
    price = payload.get("electricprice_eur_wh")
    feedin = payload.get("feedintariff_eur_wh")
    dt_hours = float(payload.get("dt_hours", 0.25))

    if not isinstance(timestamps_raw, list) or not timestamps_raw:
        raise RuntimeError("Fixture must contain non-empty 'timestamps'")
    if not isinstance(load_wh, list) or not load_wh:
        raise RuntimeError("Fixture must contain non-empty 'load_wh'")

    timestamps = [_parse_iso(str(x)) for x in timestamps_raw]
    steps = min(len(timestamps), len(load_wh))

    pv_by_inverter: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key.startswith("pv_") and key.endswith("_wh") and isinstance(value, list):
            inv_id = key[len("pv_") : -len("_wh")]
            pv_by_inverter[inv_id] = np.asarray(value[:steps], dtype=np.float32)

    return PredictionData(
        timestamps=timestamps[:steps],
        dt_hours=dt_hours,
        load_wh=np.asarray(load_wh[:steps], dtype=np.float32),
        electricprice_eur_wh=np.asarray((price or [0.0] * steps)[:steps], dtype=np.float32),
        feedintariff_eur_wh=np.asarray((feedin or [0.0] * steps)[:steps], dtype=np.float32),
        pv_by_inverter=pv_by_inverter,
    )


def build_inverters(opt_cfg: OptimizationConfig) -> list[InverterBase]:
    batteries: dict[str, Battery] = {b.device_id: Battery(b) for b in opt_cfg.batteries}
    inverters: list[InverterBase] = []
    for inv in opt_cfg.inverters:
        bat = batteries.get(inv.battery_id) if inv.battery_id else None
        inverters.append(InverterBase(inv, battery=bat))
    if not inverters:
        raise RuntimeError("No inverters found in optimization config")
    return inverters


def slice_prediction(pred: PredictionData, *, start_idx: int, length: int) -> PredictionData:
    end_idx = start_idx + length
    return PredictionData(
        timestamps=pred.timestamps[start_idx:end_idx],
        dt_hours=pred.dt_hours,
        load_wh=pred.load_wh[start_idx:end_idx],
        electricprice_eur_wh=pred.electricprice[start_idx:end_idx]
        if pred.electricprice is not None
        else None,
        feedintariff_eur_wh=pred.feedintariff[start_idx:end_idx]
        if pred.feedintariff is not None
        else None,
        pv_by_inverter={
            inv_id: series[start_idx:end_idx] for inv_id, series in pred.pv_by_inverter.items()
        },
    )


def _parse_float_token(token: str) -> float | None:
    t = token.strip().lower().replace(",", "")
    if t in {"inf", "+inf", "-inf"}:
        return None
    if t.endswith("%"):
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        return None


def parse_highs_start_stats(log_path: Path) -> StartMipStats:
    if not log_path.exists():
        return StartMipStats(
            first_best_bound=None,
            first_best_sol=None,
            first_gap_pct=None,
            root_best_bound=None,
            root_best_sol=None,
            root_gap_pct=None,
        )

    # Approximate parsing of the branch-and-bound table lines.
    # We use the first line that reports a finite BestSol as the "start" quality.
    row_re = re.compile(
        r"^\s*[A-Za-z]?\s*\d+\s+\d+\s+\d+\s+[0-9.]+%\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)",
    )

    first: tuple[float | None, float | None, float | None] | None = None
    root_like: tuple[float | None, float | None, float | None] | None = None

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = row_re.match(line)
        if not m:
            continue
        bnd = _parse_float_token(m.group(1))
        sol = _parse_float_token(m.group(2))
        gap = _parse_float_token(m.group(3))
        if sol is None:
            continue
        if first is None:
            first = (bnd, sol, gap)
        if root_like is None and bnd is not None and bnd >= 0.0:
            root_like = (bnd, sol, gap)

    return StartMipStats(
        first_best_bound=first[0] if first else None,
        first_best_sol=first[1] if first else None,
        first_gap_pct=first[2] if first else None,
        root_best_bound=root_like[0] if root_like else None,
        root_best_sol=root_like[1] if root_like else None,
        root_gap_pct=root_like[2] if root_like else None,
    )


def run_profile(
    config_path: Path, fixture_path: Path, rolling_hours: float, output_dir: Path
) -> None:
    app_cfg = AppConfig.from_yaml_file(config_path)
    pred = load_fixture_prediction(fixture_path)
    inverters = build_inverters(app_cfg.optimization)

    objective = (
        OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
        if app_cfg.optimization.solver.objective == "self_consumption"
        else OptimizationObjective.MINIMIZE_COST
    )

    dt = float(pred.dt_hours)
    rolling_steps = max(1, int(round(rolling_hours / dt)))
    if pred.steps <= rolling_steps + 1:
        raise RuntimeError(f"Prediction too short ({pred.steps}) for rolling_steps={rolling_steps}")

    window_steps = pred.steps - rolling_steps
    output_dir.mkdir(parents=True, exist_ok=True)

    base_opts = dict(app_cfg.optimization.solver.solver_opts)
    base_opts.update(
        {
            "verbose": True,
            "warm_start": True,
            "output_flag": True,
            "log_to_console": True,
            # Keep logs parseable and compact for each roll.
            "mip_min_logging_interval": 0.0,
        }
    )

    optimizer: LinearOptimizer | None = None
    current_modes: dict[str, InverterMode] | None = None
    current_soc: dict[str, float] | None = None
    rows: list[RollStats] = []

    print(f"Fixture steps: {pred.steps}, dt_hours: {pred.dt_hours}")
    print(f"Rolling horizon time: {rolling_hours} h -> {rolling_steps} steps")
    print(f"Window length per solve: {window_steps} steps")
    print(f"Objective: {objective.value}")
    print("Solver verbose enabled: yes")
    print()

    for roll in range(rolling_steps):
        pred_window = slice_prediction(pred, start_idx=roll, length=window_steps)

        if optimizer is None:
            optimizer = LinearOptimizer(
                inverters=inverters,
                horizon=pred_window.steps,
                dt_hours=pred_window.dt_hours,
                objective=objective,
                solver_opts=base_opts,
            )

        log_file = output_dir / f"roll_{roll + 1:03d}.log"

        t0 = time.perf_counter()
        sol = optimizer.solve(
            pred_window,
            soc=current_soc,
            initial_modes=current_modes,
            solver_opts={"log_file": str(log_file)},
        )
        wall_s = time.perf_counter() - t0

        current_modes = {
            plan.device_id: InverterMode(int(plan.modes[0]))
            for plan in sol.inverter_plans
            if plan.modes.size > 0
        } or None
        current_soc = {
            plan.device_id: float(soc_trace[0])
            for plan in sol.inverter_plans
            if (soc_trace := sol.result.battery_wh_per_dt.get(plan.device_id)) is not None
            and len(soc_trace) > 0
        } or None

        start_mip = parse_highs_start_stats(log_file)
        objective_value = (
            float(sol.result.total_cost)
            if objective == OptimizationObjective.MINIMIZE_COST
            else float(sol.result.total_self_consumption)
        )
        rows.append(
            RollStats(
                roll=roll + 1,
                start_idx=roll,
                solve_wall_s=wall_s,
                solver_reported_s=float(sol.solve_time_s),
                status=sol.solver_status,
                objective=objective_value,
                start_mip=start_mip,
            )
        )

    print("=== Rolling Horizon Summary ===")
    print(
        f"{'Roll':>4} {'Idx':>4} {'Wall[s]':>9} {'Solver[s]':>10} {'FirstGap[%]':>12} {'RootGap[%]':>11} {'RootLB':>11} {'RootUB':>11} {'Status':>12}"
    )
    for r in rows:
        first_gap_txt = (
            f"{r.start_mip.first_gap_pct:.2f}" if r.start_mip.first_gap_pct is not None else "n/a"
        )
        root_gap_txt = (
            f"{r.start_mip.root_gap_pct:.2f}" if r.start_mip.root_gap_pct is not None else "n/a"
        )
        root_lb_txt = (
            f"{r.start_mip.root_best_bound:.4f}"
            if r.start_mip.root_best_bound is not None
            else "n/a"
        )
        root_ub_txt = (
            f"{r.start_mip.root_best_sol:.4f}" if r.start_mip.root_best_sol is not None else "n/a"
        )
        print(
            f"{r.roll:4d} {r.start_idx:4d} {r.solve_wall_s:9.4f} {r.solver_reported_s:10.4f} {first_gap_txt:>12} {root_gap_txt:>11} {root_lb_txt:>11} {root_ub_txt:>11} {r.status:>12}"
        )

    first = rows[0]
    follow = rows[1:] if len(rows) > 1 else []
    follow_avg = sum(r.solve_wall_s for r in follow) / len(follow) if follow else math.nan
    speedup = (first.solve_wall_s / follow_avg) if follow and follow_avg > 0 else math.nan
    finite_first_gaps = [
        r.start_mip.first_gap_pct for r in rows if r.start_mip.first_gap_pct is not None
    ]
    finite_root_gaps = [
        r.start_mip.root_gap_pct for r in rows if r.start_mip.root_gap_pct is not None
    ]
    avg_first_gap = (
        sum(finite_first_gaps) / len(finite_first_gaps) if finite_first_gaps else math.nan
    )
    avg_root_gap = sum(finite_root_gaps) / len(finite_root_gaps) if finite_root_gaps else math.nan

    print()
    print("=== Diagnostics ===")
    print(f"First solve wall time: {first.solve_wall_s:.4f} s")
    if follow:
        print(f"Avg follow-up wall time: {follow_avg:.4f} s")
        print(f"Speedup first/follow-up: {speedup:.2f}x")
    else:
        print("Avg follow-up wall time: n/a")
    if not math.isnan(avg_first_gap):
        print(f"Average first-incumbent MIP gap: {avg_first_gap:.2f}%")
    else:
        print("Average first-incumbent MIP gap: n/a")
    if not math.isnan(avg_root_gap):
        print(f"Average root-like MIP gap (bound >= 0): {avg_root_gap:.2f}%")
    else:
        print("Average root-like MIP gap (bound >= 0): n/a")
    print(f"HiGHS logs written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile rolling-horizon warm-start behavior")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Fixture prediction JSON path",
    )
    parser.add_argument(
        "--rolling-hours",
        type=float,
        default=DEFAULT_ROLLING_HOURS,
        help="Rolling horizon shift in hours (default: 4h)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rolling_profile"),
        help="Directory for HiGHS log files",
    )
    args = parser.parse_args()

    run_profile(
        config_path=args.config,
        fixture_path=args.fixture,
        rolling_hours=args.rolling_hours,
        output_dir=args.output_dir,
    )
