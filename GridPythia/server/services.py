"""Business-logic helpers for the GridPythia API server.

All provider / inverter construction, singleton management, chart building
and plan serialisation live here.  FastAPI router handlers import from this
module rather than embedding logic inline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import yaml
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
from GridPythia.prediction.plots.electricprice import ElecPricePlotter
from GridPythia.prediction.plots.feedintariff import FeedInTariffPlotter
from GridPythia.prediction.plots.load import LoadPlotter
from GridPythia.prediction.plots.pvforecast import PVForecastPlotter
from GridPythia.prediction.plots.weather import WeatherPlotter
from GridPythia.prediction.prediction import PredictionData, PredictionSetup
from GridPythia.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor
from GridPythia.prediction.pvforecast.openmeteo import PVForecastOpenMeteo
from GridPythia.prediction.pvforecast.provider import PVPlaneConfig
from GridPythia.prediction.weather.brightsky import WeatherBrightSky
from GridPythia.prediction.weather.openmeteo import WeatherOpenMeteo
from GridPythia.server.models import InverterPlanResponse, InverterPlanStep
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase

logger = get_logger(__name__)

_MODE_NAMES: dict[int, str] = {m.value: m.name for m in InverterMode}


# ── Config loader ─────────────────────────────────────────────────────────


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
        pv_provider = PVForecastOpenMeteo(
            planes=[plane],
            latitude=pred_cfg.latitude,
            longitude=pred_cfg.longitude,
            api_key=om_cfg.api_key or None,
            weather_model=om_cfg.weather_model or None,
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


# ── PData cache ───────────────────────────────────────────────────────────


def get_cached_pdata() -> tuple[PredictionData, datetime | None] | None:
    """Return ``(pdata, forecast_from)`` when the cache is still fresh, else ``None``."""
    if state.pdata_cache is None or state.pdata_cache_ts is None:
        return None
    age = (datetime.now() - state.pdata_cache_ts).total_seconds()
    if age >= state.PDATA_CACHE_TTL_S:
        return None
    return state.pdata_cache, state.pdata_forecast_from


def get_cached_pdata_any_age() -> tuple[PredictionData, datetime | None] | None:
    """Return cached ``(pdata, forecast_from)`` regardless of TTL freshness."""
    if state.pdata_cache is None or state.pdata_cache_ts is None:
        return None
    return state.pdata_cache, state.pdata_forecast_from


def get_cached_pdata_age_s() -> float | None:
    """Return age of current prediction cache in seconds, or None when absent."""
    if state.pdata_cache_ts is None:
        return None
    return (datetime.now() - state.pdata_cache_ts).total_seconds()


def set_cached_pdata(pdata: PredictionData, forecast_from: datetime | None) -> None:
    state.pdata_cache = pdata
    state.pdata_cache_ts = datetime.now()
    state.pdata_forecast_from = forecast_from


# ── Solution cache ───────────────────────────────────────────────────────


def get_cached_solution() -> dict | None:
    """Return cached solution when fresh, else None."""
    if state.solution_cache is None or state.solution_cache_ts is None:
        return None
    age = (datetime.now() - state.solution_cache_ts).total_seconds()
    if age >= state.SOLUTION_CACHE_TTL_S:
        return None
    return state.solution_cache


def set_cached_solution(solution_dict: dict) -> None:
    """Store solution JSON and timestamp."""
    state.solution_cache = solution_dict
    state.solution_cache_ts = datetime.now()


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


def fig_to_dict(fig: go.Figure) -> dict[str, Any]:
    return json.loads(fig.to_json())


def make_prediction_figures(
    pdata: PredictionData,
    forecast_from: datetime | None = None,
) -> dict[str, Any]:
    """Build Plotly figure dicts for all available prediction channels.

    Returns a ``{tab_id: plotly_json_dict}`` mapping.
    """
    ts = pdata.timestamps
    figs: dict[str, Any] = {}
    if pdata.electricprice is not None:
        figs["tab-elecprice"] = fig_to_dict(
            ElecPricePlotter().plot(pdata.electricprice, ts, forecast_from=forecast_from)
        )
    if pdata.feedintariff is not None:
        figs["tab-feedin"] = fig_to_dict(FeedInTariffPlotter().plot(pdata.feedintariff, ts))
    figs["tab-load"] = fig_to_dict(LoadPlotter().plot(pdata.load_wh, ts))
    if pdata.pv_by_inverter:
        figs["tab-pv"] = fig_to_dict(
            PVForecastPlotter().plot(pdata.pv_by_inverter, ts, dt_hours=pdata.dt_hours)
        )
    if pdata.weather_by_channel:
        figs["tab-weather"] = fig_to_dict(WeatherPlotter().plot(pdata.weather_by_channel, ts))
    return figs


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
