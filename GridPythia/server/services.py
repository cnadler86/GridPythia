"""Business-logic helpers for the GridPythia API server.

All provider / inverter construction, singleton management, chart building
and plan serialisation live here.  FastAPI router handlers import from this
module rather than embedding logic inline.
"""

from __future__ import annotations

import bisect
import json
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# plotly.graph_objects is imported lazily in fig_to_dict() to avoid pulling
# narwhals (56 modules) via _plotly_utils at server startup.
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.config import AppConfig
from GridPythia.optimization.solution import OptimizationObjective
from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.prediction.electricprice.energycharts import (
    ElecPriceEnergyCharts,
    EnergyChartsConfig,
)
from GridPythia.prediction.electricprice.epexpredictor import (
    ElecPriceEpexPredictor,
    EpexPredictorConfig,
)
from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
from GridPythia.prediction.electricprice.provider import ElecPriceFallbackChain
from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed
from GridPythia.prediction.load.config import LoadProfileConfig
from GridPythia.prediction.load.provider import load_provider_from_config

# ElecPricePlotter, FeedInTariffPlotter, LoadPlotter, PVForecastPlotter,
# WeatherPlotter are imported lazily inside make_prediction_figures() to
# avoid pulling plotly → narwhals at server startup.
from GridPythia.prediction.prediction import PredictionData, PredictionSetup
from GridPythia.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor
from GridPythia.prediction.pvforecast.openmeteo import PVForecastOpenMeteo
from GridPythia.prediction.pvforecast.provider import PVPlaneConfig
from GridPythia.prediction.weather.brightsky import WeatherBrightSky
from GridPythia.prediction.weather.openmeteo import WeatherOpenMeteo
from GridPythia.server.models import InverterPlanResponse, InverterPlanStep
from GridPythia.server.plan_utils import stitch_current_slot_from_previous_plan
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase

logger = get_logger(__name__)

_MODE_NAMES: dict[int, str] = {m.value: m.name for m in InverterMode}


# ── Config loader ─────────────────────────────────────────────────────────


def snap_to_dt_grid(dt: datetime, dt_hours: float) -> datetime:
    """Round *dt* to the nearest dt_hours grid boundary.

    E.g. 13:07 with dt_hours=0.25 → 13:15; 13:04 → 13:00.
    Ensures pred.fetch() always receives an aligned start so the solver gets
    exactly round(hours/dt_hours) steps instead of +1.
    """
    step_s = dt_hours * 3600.0
    epoch = dt.timestamp()
    rounded_epoch = floor(epoch / step_s + 0.5) * step_s
    return datetime.fromtimestamp(rounded_epoch, tz=dt.tzinfo)


def load_config() -> tuple[AppConfig, dict[str, Any]]:
    """Parse the YAML config file; return ``(AppConfig, raw_dict)``."""
    path = state.config_path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.from_dict(raw), raw


# ── Pure builder helpers ──────────────────────────────────────────────────


def build_providers(cfg: AppConfig, raw_yaml: dict[str, Any]) -> PredictionSetup:
    """Instantiate all prediction providers from *AppConfig*."""
    pred_cfg = cfg.prediction

    # Electric price
    ep = pred_cfg.electricprice
    if ep.provider == "EpexPredictor":
        primary = ElecPriceEpexPredictor(
            EpexPredictorConfig(
                region=ep.epexpredictor.region,
                charges_kwh=ep.charges_kwh,
                vat_rate=ep.vat_rate,
                base_url=ep.epexpredictor.base_url,
            )
        )
        fallback = ElecPriceEnergyCharts(
            EnergyChartsConfig(
                bidding_zone=ep.energycharts.bidding_zone,
                charges_kwh=ep.charges_kwh,
                vat_rate=ep.vat_rate,
            )
        )
        electricprice = ElecPriceFallbackChain(primary=primary, fallback=fallback)
    elif ep.provider == "EnergyCharts":
        electricprice = ElecPriceEnergyCharts(
            EnergyChartsConfig(
                bidding_zone=ep.energycharts.bidding_zone,
                charges_kwh=ep.charges_kwh,
                vat_rate=ep.vat_rate,
            )
        )
    else:
        electricprice = ElecPriceFixed(
            price_kwh=ep.charges_kwh,
            charges_kwh=ep.charges_kwh,
            vat_rate=ep.vat_rate,
        )

    feedintariff = FeedInTariffFixed(tariff_kwh=pred_cfg.feedintariff.tariff_kwh)

    raw_load_path = Path(pred_cfg.load.path)
    load_path = (
        raw_load_path if raw_load_path.is_absolute() else (state.config_path.parent / raw_load_path)
    )
    load_provider = load_provider_from_config(
        LoadProfileConfig(
            path=load_path,
            country=pred_cfg.load.country or None,
            subdivision=pred_cfg.load.subdivision or None,
        )
    )

    plane_cfg = pred_cfg.pvforecast.plane
    om_cfg = pred_cfg.pvforecast.openmeteo
    plane = PVPlaneConfig(
        peak_kw=plane_cfg.peak_kw,
        tilt=plane_cfg.tilt,
        azimuth=plane_cfg.azimuth,
        userhorizon=tuple(plane_cfg.userhorizon) if plane_cfg.userhorizon else None,
        loss_pct=plane_cfg.loss_pct,
        damping_morning=om_cfg.damping_morning,
        damping_evening=om_cfg.damping_evening,
        partial_shading=om_cfg.partial_shading,
        inverter_id=plane_cfg.inverter_id,
    )
    if pred_cfg.pvforecast.provider == "OpenMeteo":
        pv_ttl_s = (
            float(pred_cfg.pvforecast.cache_ttl_hours) * 3600.0
            if pred_cfg.pvforecast.cache_ttl_hours is not None
            else None
        )
        pv_provider = PVForecastOpenMeteo(
            planes=[plane],
            latitude=pred_cfg.latitude,
            longitude=pred_cfg.longitude,
            api_key=om_cfg.api_key or None,
            weather_model=om_cfg.weather_model or None,
            cache_ttl_s=pv_ttl_s,
        )
    else:
        pv_provider = PVForecastAkkudoktor(
            planes=[plane],
            latitude=pred_cfg.latitude,
            longitude=pred_cfg.longitude,
        )

    weather_provider = None
    if "weather" in raw_yaml.get("prediction", {}):
        w_cfg = pred_cfg.weather
        if w_cfg.provider == "BrightSky":
            weather_provider = WeatherBrightSky(
                latitude=pred_cfg.latitude, longitude=pred_cfg.longitude
            )
        else:
            weather_provider = WeatherOpenMeteo(
                latitude=pred_cfg.latitude, longitude=pred_cfg.longitude
            )

    return PredictionSetup(
        electricprice=electricprice,
        feedintariff=feedintariff,
        load=load_provider,
        pv={plane.inverter_id: pv_provider},
        weather=weather_provider,
    )


def build_inverters(cfg: AppConfig) -> list[InverterBase]:
    """Build *Battery* + *InverterBase* instances from config (no SoC overrides)."""
    batteries: dict[str, Battery] = {p.device_id: Battery(p) for p in cfg.optimization.batteries}
    inverters: list[InverterBase] = []
    for inv_params in cfg.optimization.inverters:
        bat = batteries.get(inv_params.battery_id) if inv_params.battery_id else None
        inverters.append(InverterBase(inv_params, battery=bat))
    if not inverters:
        raise RuntimeError("No inverters configured in optimization.inverters")
    return inverters


# ── Singleton management ──────────────────────────────────────────────────


def get_providers(cfg: AppConfig, raw_yaml: dict[str, Any]) -> PredictionSetup:
    """Return the persistent provider singleton, rebuilding only when the config changes.

    The singleton keeps the internal ``TimeBucketCache`` of ``ElecPriceEnergyCharts``
    alive across requests, avoiding redundant HTTP fetches.
    """
    try:
        mtime = state.config_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if state.providers is None or mtime != state.providers_config_mtime:
        state.providers = build_providers(cfg, raw_yaml)
        state.providers_config_mtime = mtime
        logger.info("providers_rebuilt")
    return state.providers


def get_optimizer(cfg: AppConfig) -> LinearOptimizer:
    """Return the optimizer singleton, rebuilding only when the config changes.

    Reusing a ``LinearOptimizer`` instance allows CVXPY model reuse: the problem
    structure is compiled once and only runtime Parameters (price arrays, SoC
    start values) are updated on each call to ``solve()``.
    """
    try:
        mtime = state.config_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if state.optimizer is None or mtime != state.optimizer_config_mtime:
        objective = (
            OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
            if cfg.optimization.solver.objective == "self_consumption"
            else OptimizationObjective.MINIMIZE_COST
        )
        state.optimizer = LinearOptimizer(
            inverters=build_inverters(cfg),
            objective=objective,
            solver_opts=dict(cfg.optimization.solver.solver_opts),
        )
        state.optimizer_config_mtime = mtime
        # Sync coordinator max-age from server config
        state.coordinator._max_age_s = cfg.server.inverter_status_max_age_s
        logger.info("optimizer_rebuilt")
    return state.optimizer


# ── Solution cache ───────────────────────────────────────────────────────


def get_cached_solution() -> dict | None:
    """Return cached solution when fresh, else None."""
    if state.solution_cache is None or state.solution_cache_ts is None:
        return None
    age = (datetime.now(timezone.utc) - state.solution_cache_ts).total_seconds()
    if age >= state.SOLUTION_CACHE_TTL_S:
        return None
    return state.solution_cache


def set_cached_solution(solution_dict: dict) -> None:
    """Store solution JSON and timestamp."""
    state.solution_cache = solution_dict
    state.solution_cache_ts = datetime.now(timezone.utc)


# ── SoC override mapping ──────────────────────────────────────────────────


def soc_overrides_wh_for_solver(
    optimizer: LinearOptimizer,
    battery_soc_pct_overrides: dict[str, float],
) -> dict[str, float]:
    """Map ``battery_id → %`` overrides to ``inverter_id → Wh`` for the solver.

    The solver accepts SoC keyed by *inverter* device-id (Wh absolute), while
    the API accepts it keyed by *battery* device-id (% relative).  This function
    performs the mapping and clamps values to [min_soc, max_soc].
    """
    if not battery_soc_pct_overrides:
        return {}
    result: dict[str, float] = {}
    for inv in optimizer.inverters:
        if inv.battery is None or not inv.parameters.battery_id:
            continue
        bat_id = inv.parameters.battery_id
        if bat_id not in battery_soc_pct_overrides:
            continue
        raw_pct = float(battery_soc_pct_overrides[bat_id])
        pct = float(
            np.clip(raw_pct, inv.battery.min_soc_percentage, inv.battery.max_soc_percentage)
        )
        result[inv.device_id] = (pct / 100.0) * float(inv.battery.capacity_wh)
    return result


def cap_runtime_soc_wh_for_solver(
    optimizer: LinearOptimizer,
    runtime_soc_wh: dict[str, float],
) -> dict[str, float]:
    """Clamp runtime SoC overrides (Wh) to each inverter battery limits.

    Accepts a mapping keyed by inverter device-id and returns a new mapping
    containing only inverters with batteries, clipped to [min_soc_wh, max_soc_wh].
    """
    if not runtime_soc_wh:
        return {}

    capped: dict[str, float] = {}
    for inv in optimizer.inverters:
        if inv.battery is None:
            continue
        raw = runtime_soc_wh.get(inv.device_id)
        if raw is None:
            continue
        capped[inv.device_id] = float(np.clip(raw, inv.battery.min_soc_wh, inv.battery.max_soc_wh))
    return capped


# ── Chart builders ────────────────────────────────────────────────────────


def fig_to_dict(fig: Any) -> dict[str, Any]:
    return json.loads(fig.to_json())


def make_prediction_figures(
    pdata: PredictionData,
    forecast_from: datetime | None = None,
) -> dict[str, Any]:
    """Build Plotly figure dicts for all available prediction channels.

    Returns a ``{tab_id: plotly_json_dict}`` mapping.
    """
    # Lazy imports: plotly is only loaded on the first chart request, not at startup.
    from GridPythia.prediction.plots.electricprice import ElecPricePlotter  # noqa: PLC0415
    from GridPythia.prediction.plots.feedintariff import FeedInTariffPlotter  # noqa: PLC0415
    from GridPythia.prediction.plots.load import LoadPlotter  # noqa: PLC0415
    from GridPythia.prediction.plots.pvforecast import PVForecastPlotter  # noqa: PLC0415
    from GridPythia.prediction.plots.weather import WeatherPlotter  # noqa: PLC0415

    ts = pdata.timestamps
    figs: dict[str, Any] = {}
    if pdata.electricprice is not None:
        figs["tab-elecprice"] = fig_to_dict(
            ElecPricePlotter().plot(pdata.electricprice, ts, forecast_from=forecast_from)
        )
    if pdata.feedintariff is not None:
        figs["tab-feedin"] = fig_to_dict(FeedInTariffPlotter().plot(pdata.feedintariff, ts))
    figs["tab-load"] = fig_to_dict(
        LoadPlotter().plot(
            pdata.base_load_wh,
            ts,
            appliance_load_by_id=pdata.appliance_load_by_id,
        )
    )
    if pdata.pv_by_inverter:
        figs["tab-pv"] = fig_to_dict(
            PVForecastPlotter().plot(pdata.pv_by_inverter, ts, dt_hours=pdata.dt_hours)
        )
    if pdata.weather_by_channel:
        figs["tab-weather"] = fig_to_dict(WeatherPlotter().plot(pdata.weather_by_channel, ts))
    return figs


# ── Appliance load helpers ─────────────────────────────────────────────────


def snap_appliance_forecasts_to_grid(
    forecasts: dict[str, list[dict]],
    timestamps: list,
    dt_hours: float,
) -> dict[str, np.ndarray]:
    """Snap raw appliance forecast slots to the prediction time grid.

    For each slot, the nearest prediction timestamp within ``dt_hours / 2`` is
    found via binary search.  Energy (Wh) from slots that fall outside the
    horizon or before *now* is silently dropped.  Multiple slots mapping to the
    same grid point are summed.
    """
    from datetime import timezone as _tz

    if not timestamps:
        return {}

    half_dt_s = dt_hours * 3600.0 / 2.0
    now = datetime.now(tz=_tz.utc)
    ts_epochs = [ts.timestamp() for ts in timestamps]

    result: dict[str, np.ndarray] = {}
    for appliance_id, slots in forecasts.items():
        arr = np.zeros(len(timestamps), dtype=np.float32)
        for slot in slots:
            try:
                t_raw = slot["time"]
                wh = float(slot["load_wh"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                t = datetime.fromisoformat(t_raw)
            except ValueError:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=_tz.utc)
            else:
                t = t.astimezone(_tz.utc)
            if t < now:
                continue
            t_epoch = t.timestamp()
            idx = bisect.bisect_left(ts_epochs, t_epoch)
            if idx == 0:
                nearest = 0
            elif idx >= len(ts_epochs):
                nearest = len(ts_epochs) - 1
            else:
                nearest = (
                    idx
                    if abs(ts_epochs[idx] - t_epoch) < abs(ts_epochs[idx - 1] - t_epoch)
                    else idx - 1
                )
            if abs(ts_epochs[nearest] - t_epoch) <= half_dt_s:
                arr[nearest] += wh
        result[appliance_id] = arr
    return result


def apply_appliance_loads(pdata: "PredictionData") -> "PredictionData":
    """Return a new :class:`PredictionData` with active appliance forecasts injected.

    When no appliance forecasts are registered the original *pdata* is returned
    unchanged (zero-copy fast path).
    """
    if not state.appliance_forecasts:
        return pdata
    snapped = snap_appliance_forecasts_to_grid(
        state.appliance_forecasts,
        pdata.timestamps,
        pdata.dt_hours,
    )
    snapped = {k: v for k, v in snapped.items() if v.any()}
    if not snapped:
        return pdata
    return PredictionData(
        requested_start=pdata.requested_start,
        timestamps=pdata.timestamps,
        dt_hours=pdata.dt_hours,
        load_wh=pdata.base_load_wh,
        electricprice_eur_wh=pdata.electricprice,
        feedintariff_eur_wh=pdata.feedintariff,
        pv_by_inverter=pdata.pv_by_inverter,
        weather_by_channel=pdata.weather_by_channel,
        appliance_load_by_id=snapped,
    )


# ── Inverter plan serialisation ───────────────────────────────────────────


def inverter_plan_to_response(
    plan: Any,  # InverterPlan – avoid circular import via TYPE_CHECKING
    timestamps: list[datetime],
) -> InverterPlanResponse:
    """Convert an *InverterPlan* + prediction timestamps into a JSON-serialisable model."""
    steps: list[InverterPlanStep] = []
    for i, ts in enumerate(timestamps):
        if i >= plan.steps:
            break
        steps.append(
            InverterPlanStep(
                timestamp=ts.isoformat(),
                mode=int(plan.modes[i]),
                mode_name=_MODE_NAMES.get(int(plan.modes[i]), "UNKNOWN"),
                charge_ac_wh=float(plan.charge_ac_wh[i]),
                discharge_ac_wh=float(plan.discharge_ac_wh[i]),
                pv_to_ac_wh=float(plan.pv_to_ac_wh[i]),
                pv_to_battery_wh=float(plan.pv_to_battery_wh[i]),
                battery_soc_wh=(
                    float(plan.battery_soc_wh[i]) if plan.battery_soc_wh is not None else None
                ),
            )
        )
    return InverterPlanResponse(device_id=plan.device_id, steps=steps)


# ── Optimization cycle orchestration ─────────────────────────────────────


async def run_optimization_cycle(
    start: datetime,
    end: datetime,
    cfg: "AppConfig",
    raw_yaml: dict[str, Any],
    *,
    battery_soc_overrides: dict[str, float] | None = None,
    initial_modes_overrides: dict[str, int] | None = None,
    solver_opts_overrides: dict[str, Any] | None = None,
    validate_with_simulation: bool = True,
) -> dict[str, Any]:
    """Orchestrate a full optimization cycle including server-side effects.

    This function is the shared entry point for the HTTP router and the
    scheduler.  Both pass a timezone-aware *start* / *end* pair and receive
    the same serialisable response dict.

    Side effects performed here (in addition to fetch + solve):
    - Applies active appliance forecasts to the prediction data.
    - Checks inverter readiness (raises ``ValueError`` if any inverter is stale).
    - Guards against missing PV forecast data.
    - Builds Plotly chart dicts for the dashboard.
    - Caches the result via :func:`set_cached_solution`.
    - Broadcasts the result over the WebSocket hub.
    - Publishes inverter plans via MQTT (when the gateway is running).

    Args:
        start:  Horizon start (timezone-aware).  Floored to slot by the runner.
        end:    Horizon end (timezone-aware).  Covered by the runner.
        cfg:    Parsed :class:`~GridPythia.config.AppConfig`.
        raw_yaml: Raw YAML dict (needed to rebuild providers on config change).
        battery_soc_overrides:  ``battery_id → %`` SoC overrides from the caller.
        initial_modes_overrides: ``inverter_id → int`` mode overrides from the caller.
        solver_opts_overrides:   HiGHS option overrides merged on top of config defaults.
        validate_with_simulation: Attach simulation parity report to solution.

    Returns:
        Serialisable response dict with keys ``summary``, ``inverter_plans``,
        ``charts``, and ``status``.

    Raises:
        RuntimeError:  Config / provider / optimizer setup failed.
        ValueError:    *start* or *end* are naive, or inverter status is stale.
        Exception:     Prediction fetch or solver error (re-raised as-is).
    """
    # Lazy imports to avoid circular import at module level
    from GridPythia.optimization.plots import SolutionPlotter  # noqa: PLC0415
    from GridPythia.optimization.runner import run_optimization  # noqa: PLC0415
    from GridPythia.optimization.solution import OptimizationObjective  # noqa: PLC0415
    from GridPythia.prediction.prediction import Prediction  # noqa: PLC0415
    from GridPythia.server.models import OptimizeSummary  # noqa: PLC0415

    dt_hours = float(cfg.prediction.dt_hours)

    setup = get_providers(cfg, raw_yaml)
    optimizer = get_optimizer(cfg)

    # ── Inverter readiness check ──────────────────────────────────────
    ready, missing = state.coordinator.all_optimizable_ready(
        optimizer.inverters,
        now=start,
    )
    if not ready:
        raise ValueError(
            f"Optimization blocked: missing or stale inverter status for {sorted(missing)}"
        )

    # ── Objective ────────────────────────────────────────────────────
    objective = (
        OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
        if cfg.optimization.solver.objective == "self_consumption"
        else OptimizationObjective.MINIMIZE_COST
    )

    # ── SoC and initial modes ─────────────────────────────────────────
    soc_wh = state.coordinator.get_soc_overrides_wh(optimizer.inverters)
    soc_wh.update(soc_overrides_wh_for_solver(optimizer, battery_soc_overrides or {}))
    soc_wh = cap_runtime_soc_wh_for_solver(optimizer, soc_wh)

    initial_modes = state.coordinator.get_initial_modes(optimizer.inverters)
    if initial_modes_overrides:
        from GridPythia.simulation.devices import InverterMode as _IM  # noqa: PLC0415

        initial_modes.update(
            {inv_id: _IM(int(mode)) for inv_id, mode in initial_modes_overrides.items()}
        )

    # ── Solver opts ───────────────────────────────────────────────────
    solver_opts = dict(cfg.optimization.solver.solver_opts)
    if solver_opts_overrides:
        solver_opts.update(solver_opts_overrides)

    logger.info(
        "optimization_cycle_initial_conditions",
        start=start.isoformat(),
        end=end.isoformat(),
        soc_wh={k: round(v, 1) for k, v in soc_wh.items()},
        initial_modes={k: v.name for k, v in initial_modes.items()},
    )

    # ── Core: fetch predictions + solve ──────────────────────────────
    prediction = Prediction(setup)
    async with state.get_optimizer_lock():
        result = await run_optimization(
            start=start,
            end=end,
            prediction=prediction,
            optimizer=optimizer,
            dt_hours=dt_hours,
            soc=soc_wh or None,
            initial_modes=initial_modes or None,
            solver_opts=solver_opts,
            objective=objective,
            validate_with_simulation=validate_with_simulation,
            pdata_transform=apply_appliance_loads,
        )

    solution = result.solution
    fetch_pdata = result.fetch_pdata
    solver_pdata = result.solver_pdata

    # ── PV data integrity guard ───────────────────────────────────────
    pv_missing = [
        inv.device_id
        for inv in optimizer.inverters
        if inv.parameters.has_pv and inv.device_id not in solver_pdata.pv_by_inverter
    ]
    if pv_missing:
        raise ValueError(
            f"PV forecast missing for inverter(s) {pv_missing}. "
            "Refusing to optimize without PV data to avoid planning as if there is no solar "
            "generation. Fix the PV provider or wait for the retry task to recover."
        )

    # ── Inverter plans ────────────────────────────────────────────────
    inverter_plans = [
        inverter_plan_to_response(plan, list(solver_pdata.timestamps))
        for plan in solution.inverter_plans
    ]

    # ── Stitch current-slot step from previous plan when solver skipped it ──
    # Happens when now > slot midpoint (e.g. 14:55 → solver starts at 15:00).
    fetch_start = result.fetch_pdata.timestamps[0]
    if result.solver_start > fetch_start:
        prev_solution = get_cached_solution()
        if prev_solution is not None:
            prev_by_device = {
                p["device_id"]: p["steps"] for p in prev_solution.get("inverter_plans", [])
            }
            stitched = []
            for plan in inverter_plans:
                new_steps = stitch_current_slot_from_previous_plan(
                    [s.model_dump() for s in plan.steps],
                    prev_by_device.get(plan.device_id, []),
                    published_at=fetch_start,
                    dt_hours=dt_hours,
                )
                stitched.append(
                    InverterPlanResponse(
                        device_id=plan.device_id,
                        steps=[InverterPlanStep(**s) for s in new_steps],
                    )
                )
            inverter_plans = stitched
            logger.debug(
                "optimization_stitched_current_slot",
                fetch_start=fetch_start.isoformat(),
                solver_start=result.solver_start.isoformat(),
            )

    # ── Charts ────────────────────────────────────────────────────────
    forecast_from: datetime | None = (
        setup.electricprice.last_real_ts if setup.electricprice is not None else None
    )
    charts: dict[str, Any] = {}
    charts.update(make_prediction_figures(fetch_pdata, forecast_from))
    plotter = SolutionPlotter()
    for inv in optimizer.inverters:
        charts[f"tab-inv-{inv.device_id}"] = fig_to_dict(
            plotter.plot_inverter(solution, inv.device_id)
        )

    # ── Savings vs. naive baseline ────────────────────────────────────
    import numpy as np  # noqa: PLC0415

    pv_total = np.zeros(solver_pdata.steps, dtype=float)
    for arr in solver_pdata.pv_by_inverter.values():
        pv_total += np.asarray(arr, dtype=float)
    load_arr = np.asarray(solver_pdata.load_wh, dtype=float)
    price_arr = (
        np.asarray(solver_pdata.electricprice, dtype=float)
        if solver_pdata.electricprice is not None
        else np.zeros(solver_pdata.steps, dtype=float)
    )
    feedin_arr = (
        np.asarray(solver_pdata.feedintariff, dtype=float)
        if solver_pdata.feedintariff is not None
        else np.zeros(solver_pdata.steps, dtype=float)
    )
    naive_net_cost = float(
        np.sum(np.maximum(0.0, load_arr - pv_total) * price_arr)
        - np.sum(np.maximum(0.0, pv_total - load_arr) * feedin_arr)
    )
    net_cost = solution.result.total_cost - solution.result.total_revenue
    savings = naive_net_cost - net_cost
    parity_ok = solution.parity_report.ok if solution.parity_report is not None else None

    # ── Summary ───────────────────────────────────────────────────────
    from GridPythia.server.models import OptimizeSummary  # noqa: PLC0415, F811

    summary = OptimizeSummary(
        solver_status=solution.solver_status,
        solve_time_s=round(solution.solve_time_s, 2),
        objective=objective.value,
        total_cost_eur=round(float(solution.result.total_cost), 4),
        total_revenue_eur=round(float(solution.result.total_revenue), 4),
        net_cost_eur=round(float(net_cost), 4),
        naive_net_cost_eur=round(float(naive_net_cost), 4),
        savings_eur=round(float(savings), 4),
        parity_ok=parity_ok,
        solved_at=datetime.now(timezone.utc).isoformat(),
    )
    parity_warn = " ⚠ parity" if parity_ok is False else ""
    status = (
        f"Solved {solution.solve_time_s:.1f}s · {solution.solver_status} · "
        f"naive: {naive_net_cost:.3f} EUR → optimized: {net_cost:.3f} EUR · "
        f"savings: {savings:.3f} EUR{parity_warn}"
    )

    logger.info(
        "optimization_cycle_done",
        solver_status=solution.solver_status,
        solve_time_s=round(solution.solve_time_s, 2),
        savings_eur=round(savings, 3),
        solver_start=result.solver_start.isoformat(),
        solver_steps=solver_pdata.steps,
    )

    response_data: dict[str, Any] = {
        "summary": summary.model_dump(),
        "inverter_plans": [p.model_dump() for p in inverter_plans],
        "charts": charts,
        "status": status,
    }

    # ── Cache + broadcast + MQTT ──────────────────────────────────────
    set_cached_solution(response_data)

    await state.ws_hub.broadcast({"type": "optimization_updated", "payload": response_data})

    if state.mqtt_gateway is not None:
        state.mqtt_gateway.publish_plans(
            [p.model_dump() for p in inverter_plans],
            dt_hours=dt_hours,
        )

    return response_data
