"""Tune follow-up rolling-horizon solver options while keeping first run unchanged.

Goal:
- First roll uses the base configuration exactly as configured.
- Follow-up rolls use candidate overrides.
- Keep mip_rel_gap equal to first run for comparability.
- Iterate through sensible HiGHS options (doc-based) and report latency.
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from GridPythia.config import AppConfig
from GridPythia.optimization.solver import LinearOptimizer, OptimizationObjective
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.inverterbase import InverterBase
from profile_rolling_horizon import build_inverters, load_fixture_prediction, slice_prediction

DEFAULT_CONFIG = Path("config.yaml")
DEFAULT_FIXTURE = Path("tests/optimization/fixtures/prediction_2026_04_06_48h_15m.json")
DEFAULT_ROLLING_HOURS = 4.0
DEFAULT_TARGET_MS_LOW = 200.0
DEFAULT_TARGET_MS_HIGH = 300.0
DEFAULT_LOG_DIR = Path("artifacts/followup_tune_logs")


@dataclass
class TrialResult:
    name: str
    follow_opts: dict[str, Any]
    first_wall_s: float
    avg_follow_wall_s: float
    med_follow_wall_s: float
    best_follow_wall_s: float
    worst_follow_wall_s: float
    status_counts: dict[str, int]
    rolls: int
    avg_nodes: float
    avg_lp_iterations: float
    avg_separation_work: float
    avg_heuristics_work: float


def _objective_from_cfg(app_cfg: AppConfig) -> OptimizationObjective:
    if app_cfg.optimization.solver.objective == "self_consumption":
        return OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
    return OptimizationObjective.MINIMIZE_COST


def _count_statuses(statuses: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in statuses:
        out[s] = out.get(s, 0) + 1
    return out


def _prepare_inverters(inverters: list[InverterBase]) -> list[InverterBase]:
    # Rebuild objects for each trial so warm/cached state cannot leak across trials.
    rebuilt: list[InverterBase] = []
    for inv in inverters:
        battery = None
        if inv.battery is not None:
            battery = type(inv.battery)(inv.battery.parameters)
        rebuilt.append(type(inv)(inv.parameters, battery=battery))
    return rebuilt


def _parse_log_metric(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_highs_work_metrics(log_file: Path) -> dict[str, float]:
    if not log_file.exists():
        return {"nodes": 0.0, "lp_iterations": 0.0, "separation": 0.0, "heuristics": 0.0}

    txt = log_file.read_text(encoding="utf-8", errors="ignore")
    nodes = _parse_log_metric(r"^\s*Nodes\s+(\d+)", txt) or 0.0
    lp_it = _parse_log_metric(r"^\s*LP iterations\s+(\d+)", txt) or 0.0
    sep = _parse_log_metric(r"^\s*(\d+)\s*\(separation\)", txt) or 0.0
    heu = _parse_log_metric(r"^\s*(\d+)\s*\(heuristics\)", txt) or 0.0
    return {
        "nodes": nodes,
        "lp_iterations": lp_it,
        "separation": sep,
        "heuristics": heu,
    }


def run_trial(
    *,
    name: str,
    pred,
    inverters_template: list[InverterBase],
    objective: OptimizationObjective,
    base_opts: dict[str, Any],
    follow_opts: dict[str, Any],
    rolling_steps: int,
    window_steps: int,
    log_dir: Path,
) -> TrialResult:
    inverters = _prepare_inverters(inverters_template)

    optimizer: LinearOptimizer | None = None
    initial_modes: dict[str, InverterMode] | None = None
    initial_soc_wh: dict[str, float] | None = None
    warm_start_plan: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    wall_times: list[float] = []
    statuses: list[str] = []
    nodes_arr: list[float] = []
    lp_it_arr: list[float] = []
    sep_arr: list[float] = []
    heu_arr: list[float] = []

    # Keep mip_rel_gap fixed to first run setting for comparability.
    fixed_rel_gap = base_opts.get("mip_rel_gap", None)
    follow_opts_effective = dict(follow_opts)
    if fixed_rel_gap is not None:
        follow_opts_effective["mip_rel_gap"] = fixed_rel_gap

    for roll in range(rolling_steps):
        pred_window = slice_prediction(pred, start_idx=roll, length=window_steps)

        if initial_soc_wh:
            for inv in inverters:
                if inv.battery is None:
                    continue
                start = initial_soc_wh.get(inv.device_id)
                if start is not None:
                    inv.battery.soc_wh = float(
                        np.clip(start, inv.battery.min_soc_wh, inv.battery.max_soc_wh)
                    )

        if optimizer is None:
            optimizer = LinearOptimizer(
                inverters=inverters,
                horizon=pred_window.steps,
                dt_hours=pred_window.dt_hours,
            )

        opts = dict(base_opts) if roll == 0 else {**base_opts, **follow_opts_effective}
        opts["output_flag"] = True
        opts["log_to_console"] = True
        log_file = log_dir / f"{name}_roll_{roll + 1:03d}.log"
        opts["log_file"] = str(log_file)

        t0 = time.perf_counter()
        solution = optimizer.solve(
            objective=objective,
            solver_opts=opts,
            validate_with_simulation=False,
            initial_modes=initial_modes,
            warm_start_plan=warm_start_plan,
            prediction=pred_window,
        )
        wall = time.perf_counter() - t0

        statuses.append(str(solution.solver_status))
        wall_times.append(wall)
        metrics = parse_highs_work_metrics(log_file)
        nodes_arr.append(metrics["nodes"])
        lp_it_arr.append(metrics["lp_iterations"])
        sep_arr.append(metrics["separation"])
        heu_arr.append(metrics["heuristics"])

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
        warm_start_plan = LinearOptimizer.shift_solution_for_next_horizon(
            solution,
            horizon_steps=window_steps,
            shift_steps=1,
        )

    if not wall_times:
        raise RuntimeError("No rolling runs executed")

    follow_times = wall_times[1:] if len(wall_times) > 1 else wall_times
    return TrialResult(
        name=name,
        follow_opts=follow_opts_effective,
        first_wall_s=wall_times[0],
        avg_follow_wall_s=float(sum(follow_times) / len(follow_times)),
        med_follow_wall_s=float(statistics.median(follow_times)),
        best_follow_wall_s=float(min(follow_times)),
        worst_follow_wall_s=float(max(follow_times)),
        status_counts=_count_statuses(statuses),
        rolls=len(wall_times),
        avg_nodes=float(sum(nodes_arr) / len(nodes_arr)) if nodes_arr else 0.0,
        avg_lp_iterations=float(sum(lp_it_arr) / len(lp_it_arr)) if lp_it_arr else 0.0,
        avg_separation_work=float(sum(sep_arr) / len(sep_arr)) if sep_arr else 0.0,
        avg_heuristics_work=float(sum(heu_arr) / len(heu_arr)) if heu_arr else 0.0,
    )


def candidate_follow_opts() -> list[tuple[str, dict[str, Any]]]:
    # Based on HiGHS docs: compare LP engine and heuristic intensity.
    # No time limits are used to keep comparisons fair.
    return [
        ("follow_baseline", {}),
        (
            "follow_light_heur",
            {
                "mip_heuristic_effort": 0.01,
                "mip_heuristic_run_rens": False,
                "mip_heuristic_run_rins": False,
                "mip_heuristic_run_feasibility_jump": False,
                "mip_heuristic_run_root_reduced_cost": False,
                "mip_lp_solver": "simplex",
                "parallel": "on",
                "threads": 0,
            },
        ),
        (
            "follow_fast_root",
            {
                "mip_heuristic_effort": 0.0,
                "mip_heuristic_run_rens": False,
                "mip_heuristic_run_rins": False,
                "mip_heuristic_run_feasibility_jump": False,
                "mip_heuristic_run_root_reduced_cost": False,
                "mip_allow_restart": False,
                "mip_detect_symmetry": True,
                "mip_lp_solver": "simplex",
                "parallel": "on",
                "threads": 0,
            },
        ),
        (
            "follow_ipm_light_heur",
            {
                "mip_heuristic_effort": 0.01,
                "mip_heuristic_run_rens": False,
                "mip_heuristic_run_rins": False,
                "mip_heuristic_run_feasibility_jump": False,
                "mip_heuristic_run_root_reduced_cost": False,
                "mip_lp_solver": "ipm",
                "parallel": "on",
                "threads": 0,
            },
        ),
        (
            "follow_simplex_less_parallel",
            {
                "mip_heuristic_effort": 0.0,
                "mip_heuristic_run_rens": False,
                "mip_heuristic_run_rins": False,
                "mip_heuristic_run_feasibility_jump": False,
                "mip_heuristic_run_root_reduced_cost": False,
                "mip_lp_solver": "simplex",
                "parallel": "off",
                "threads": 0,
            },
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune follow-up rolling-horizon options with fixed first-run config"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--rolling-hours", type=float, default=DEFAULT_ROLLING_HOURS)
    parser.add_argument("--target-ms-low", type=float, default=DEFAULT_TARGET_MS_LOW)
    parser.add_argument("--target-ms-high", type=float, default=DEFAULT_TARGET_MS_HIGH)
    parser.add_argument(
        "--max-rolls",
        type=int,
        default=16,
        help="Limit number of rolling solves for faster tuning",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory where per-roll HiGHS logs are written",
    )
    args = parser.parse_args()

    app_cfg = AppConfig.from_yaml_file(args.config)
    pred = load_fixture_prediction(args.fixture)
    objective = _objective_from_cfg(app_cfg)
    inverters = build_inverters(app_cfg.optimization)

    dt = float(pred.dt_hours)
    rolling_steps = max(1, int(round(float(args.rolling_hours) / dt)))
    if args.max_rolls > 0:
        rolling_steps = min(rolling_steps, int(args.max_rolls))
    if pred.steps <= rolling_steps + 1:
        raise RuntimeError(f"Prediction too short ({pred.steps}) for rolling_steps={rolling_steps}")
    window_steps = pred.steps - rolling_steps

    base_opts = dict(app_cfg.optimization.solver.solver_opts)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print("=== Setup ===")
    print(f"Fixture: {args.fixture}")
    print(f"Steps: {pred.steps}, dt_hours: {pred.dt_hours}")
    print(f"Rolling hours: {args.rolling_hours} -> rolling_steps={rolling_steps}")
    print(f"Window steps: {window_steps}")
    print(f"Objective: {objective.value}")
    print(f"Base first-run opts: {base_opts}")
    print()

    trials: list[TrialResult] = []
    for name, opts in candidate_follow_opts():
        trial = run_trial(
            name=name,
            pred=pred,
            inverters_template=inverters,
            objective=objective,
            base_opts=base_opts,
            follow_opts=opts,
            rolling_steps=rolling_steps,
            window_steps=window_steps,
            log_dir=args.log_dir,
        )
        trials.append(trial)
        print(
            f"{name:>22}: first={trial.first_wall_s * 1000:7.1f} ms | "
            f"follow_avg={trial.avg_follow_wall_s * 1000:7.1f} ms | "
            f"follow_med={trial.med_follow_wall_s * 1000:7.1f} ms | "
            f"best={trial.best_follow_wall_s * 1000:7.1f} ms | "
            f"nodes={trial.avg_nodes:5.1f} | lp_it={trial.avg_lp_iterations:7.1f} | "
            f"sep={trial.avg_separation_work:6.1f} | heur={trial.avg_heuristics_work:7.1f} | "
            f"status={trial.status_counts}"
        )

    print()
    print("=== Ranking (by follow-up avg latency) ===")
    ranked = sorted(trials, key=lambda t: t.avg_follow_wall_s)
    for i, trial in enumerate(ranked, start=1):
        print(
            f"{i:2d}. {trial.name:>22} -> avg={trial.avg_follow_wall_s * 1000:7.1f} ms, "
            f"median={trial.med_follow_wall_s * 1000:7.1f} ms, lp_it={trial.avg_lp_iterations:7.1f}, "
            f"sep={trial.avg_separation_work:6.1f}, heur={trial.avg_heuristics_work:7.1f}, "
            f"status={trial.status_counts}"
        )

    lo = float(args.target_ms_low)
    hi = float(args.target_ms_high)
    target = [t for t in ranked if lo <= t.avg_follow_wall_s * 1000.0 <= hi]

    print()
    print("=== Target Check ===")
    if target:
        best = target[0]
        print(
            f"Reached target {lo:.0f}-{hi:.0f} ms with '{best.name}' "
            f"(avg follow-up {best.avg_follow_wall_s * 1000:.1f} ms)."
        )
        print(f"Suggested follow-up opts: {best.follow_opts}")
    else:
        best = ranked[0]
        print(
            f"Target {lo:.0f}-{hi:.0f} ms not reached. Best was '{best.name}' "
            f"with avg follow-up {best.avg_follow_wall_s * 1000:.1f} ms."
        )
        print(f"Best follow-up opts: {best.follow_opts}")
    print(f"HiGHS logs are available in: {args.log_dir}")


if __name__ == "__main__":
    main()
