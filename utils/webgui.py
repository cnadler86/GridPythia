"""Lightweight web-based GUI for GridPythia energy management.

Runs a FastAPI/uvicorn server on port 8080.  Config is loaded automatically
from ``config.yaml`` in the repository root; override with ``--config``.

Usage::

    uv run python -m utils.webgui
    uv run python -m utils.webgui --config /path/to/config.yaml --port 8080
    # then open http://localhost:8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import plotly.graph_objects as go
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from structlog import get_logger

from GridPythia.config import AppConfig
from GridPythia.optimization.plots import SolutionPlotter
from GridPythia.optimization.solution import OptimizationObjective
from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.prediction.electricprice.energycharts import (
    ElecPriceEnergyCharts,
    EnergyChartsConfig,
)
from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed
from GridPythia.prediction.load.config import LoadProfileConfig
from GridPythia.prediction.load.provider import load_provider_from_config
from GridPythia.prediction.plots.electricprice import ElecPricePlotter
from GridPythia.prediction.plots.feedintariff import FeedInTariffPlotter
from GridPythia.prediction.plots.load import LoadPlotter
from GridPythia.prediction.plots.pvforecast import PVForecastPlotter
from GridPythia.prediction.plots.weather import WeatherPlotter
from GridPythia.prediction.prediction import Prediction, PredictionData, PredictionSetup
from GridPythia.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor
from GridPythia.prediction.pvforecast.openmeteo import PVForecastOpenMeteo
from GridPythia.prediction.pvforecast.provider import PVPlaneConfig
from GridPythia.prediction.weather.brightsky import WeatherBrightSky
from GridPythia.prediction.weather.openmeteo import WeatherOpenMeteo
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase

logger = get_logger(__name__)

# ── app-level mutable state (set before app starts) ───────────────────────
_config_path: Path = Path(__file__).resolve().parent.parent / "config.yaml"

app = FastAPI(title="GridPythia Web GUI", docs_url=None, redoc_url=None)

# ── persistent provider singleton (keeps internal caches alive) ───────────
# Rebuilt only when the config changes. This avoids re-fetching EnergyCharts
# on every request because the internal TimeBucketCache survives across calls.
_providers: PredictionSetup | None = None
_providers_config_mtime: float = 0.0

# ── optimizer singleton (reuses compiled CVXPY model across requests) ─────
_optimizer: LinearOptimizer | None = None
_optimizer_config_mtime: float = 0.0
_optimizer_lock = asyncio.Lock()


def _get_providers(cfg: AppConfig, raw_yaml: dict[str, Any]) -> PredictionSetup:
    """Return the persistent provider singleton, rebuilding only when config changed."""
    global _providers, _providers_config_mtime  # noqa: PLW0603
    try:
        mtime = _config_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _providers is None or mtime != _providers_config_mtime:
        _providers = _build_providers(cfg, raw_yaml)
        _providers_config_mtime = mtime
    return _providers


def _get_optimizer(cfg: AppConfig) -> LinearOptimizer:
    """Return optimizer singleton, rebuilding only when config changed.

    Reusing a LinearOptimizer instance allows CVXPY model reuse (compiled once,
    runtime parameters updated per solve), which reduces repeated setup overhead.
    """
    global _optimizer, _optimizer_config_mtime  # noqa: PLW0603
    try:
        mtime = _config_path.stat().st_mtime
    except OSError:
        mtime = 0.0

    if _optimizer is None or mtime != _optimizer_config_mtime:
        inverters = _build_inverters(cfg)
        objective = (
            OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
            if cfg.optimization.solver.objective == "self_consumption"
            else OptimizationObjective.MINIMIZE_COST
        )
        solver_opts = dict(cfg.optimization.solver.solver_opts)
        _optimizer = LinearOptimizer(
            inverters=inverters,
            objective=objective,
            solver_opts=solver_opts,
        )
        _optimizer_config_mtime = mtime

    return _optimizer


def _soc_overrides_wh_for_solver(
    optimizer: LinearOptimizer,
    battery_soc_pct_overrides: dict[str, float],
) -> dict[str, float]:
    """Map GUI battery SoC overrides (battery_id -> %) to solver SoC map (inverter_id -> Wh)."""
    if not battery_soc_pct_overrides:
        return {}

    soc_wh_by_inverter: dict[str, float] = {}
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
        soc_wh_by_inverter[inv.device_id] = (pct / 100.0) * float(inv.battery.capacity_wh)

    return soc_wh_by_inverter


# ── prediction data cache (avoids double-fetch within one optimize cycle) ─
# Also stores forecast_from (last_real_ts from EnergyCharts) so plots can
# shade the statistical forecast region in lavendel.
_pdata_cache: PredictionData | None = None
_pdata_cache_ts: datetime | None = None
_pdata_forecast_from: datetime | None = None  # last real API timestamp
_PDATA_CACHE_TTL_S: float = 300.0  # 5 minutes


def _get_cached_pdata() -> "tuple[PredictionData, datetime | None] | None":
    """Return (PredictionData, forecast_from) if still fresh, else None."""
    if _pdata_cache is None or _pdata_cache_ts is None:
        return None
    age = (datetime.now() - _pdata_cache_ts).total_seconds()
    if age >= _PDATA_CACHE_TTL_S:
        return None
    return _pdata_cache, _pdata_forecast_from


def _set_cached_pdata(pdata: "PredictionData", forecast_from: "datetime | None") -> None:
    global _pdata_cache, _pdata_cache_ts, _pdata_forecast_from  # noqa: PLW0603
    _pdata_cache = pdata
    _pdata_cache_ts = datetime.now()
    _pdata_forecast_from = forecast_from


# ── config helpers ────────────────────────────────────────────────────────


def _load_config() -> tuple[AppConfig, dict[str, Any]]:
    """Parse the YAML config; returns (AppConfig, raw_yaml_dict)."""
    if not _config_path.exists():
        raise FileNotFoundError(f"Config file not found: {_config_path}")
    text = _config_path.read_text(encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load(text) or {}
    cfg = AppConfig.from_dict(raw)
    return cfg, raw


# ── provider builder ──────────────────────────────────────────────────────


def _build_providers(cfg: AppConfig, raw_yaml: dict[str, Any]) -> PredictionSetup:
    """Build all prediction providers from AppConfig."""
    pred_cfg = cfg.prediction

    # Electric price
    ep = pred_cfg.electricprice
    if ep.provider == "EnergyCharts":
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

    # Feed-in tariff
    feedintariff = FeedInTariffFixed(tariff_kwh=pred_cfg.feedintariff.tariff_kwh)

    # Load
    raw_load_path = Path(pred_cfg.load.path)
    load_path = (
        raw_load_path if raw_load_path.is_absolute() else (_config_path.parent / raw_load_path)
    )
    load_provider = load_provider_from_config(
        LoadProfileConfig(
            path=load_path,
            country=pred_cfg.load.country or None,
            subdivision=pred_cfg.load.subdivision or None,
        )
    )

    # PV forecast
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

    # Weather – only when explicitly present in the YAML
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


# ── inverter builder ──────────────────────────────────────────────────────


def _build_inverters(
    cfg: AppConfig,
    soc_overrides: dict[str, float] | None = None,
) -> list[InverterBase]:
    """Build Battery + InverterBase from config with optional SoC overrides."""
    overrides = soc_overrides or {}
    batteries: dict[str, Battery] = {}
    for params in cfg.optimization.batteries:
        if params.device_id in overrides:
            soc = int(round(float(overrides[params.device_id])))
            params = params.model_copy(update={"initial_soc_percentage": soc})
        batteries[params.device_id] = Battery(params)

    inverters: list[InverterBase] = []
    for inv_params in cfg.optimization.inverters:
        bat = batteries.get(inv_params.battery_id) if inv_params.battery_id else None
        inverters.append(InverterBase(inv_params, battery=bat))

    if not inverters:
        raise RuntimeError("No inverters configured in optimization.inverters")
    return inverters


# ── figure builders ───────────────────────────────────────────────────────


def _fig_to_dict(fig: go.Figure) -> dict[str, Any]:
    return json.loads(fig.to_json())


def _make_prediction_figures(
    pdata: PredictionData, forecast_from: datetime | None = None
) -> dict[str, Any]:
    """Delegate to the existing per-channel prediction plotters."""
    ts = pdata.timestamps
    figs: dict[str, Any] = {}

    if pdata.electricprice is not None:
        figs["tab-elecprice"] = _fig_to_dict(
            ElecPricePlotter().plot(pdata.electricprice, ts, forecast_from=forecast_from)
        )
    if pdata.feedintariff is not None:
        figs["tab-feedin"] = _fig_to_dict(FeedInTariffPlotter().plot(pdata.feedintariff, ts))
    figs["tab-load"] = _fig_to_dict(LoadPlotter().plot(pdata.load_wh, ts))
    if pdata.pv_by_inverter:
        figs["tab-pv"] = _fig_to_dict(
            PVForecastPlotter().plot(pdata.pv_by_inverter, ts, dt_hours=pdata.dt_hours)
        )
    if pdata.weather_by_channel:
        figs["tab-weather"] = _fig_to_dict(WeatherPlotter().plot(pdata.weather_by_channel, ts))
    return figs


# ── HTML page generator ───────────────────────────────────────────────────


def _build_html(cfg: AppConfig, has_weather: bool) -> str:
    """Generate the full single-page HTML application."""
    bat_lookup = {b.device_id: b for b in cfg.optimization.batteries}

    # ── Battery SoC input cards ───────────────────────────────────────
    battery_cards_html = ""
    for bat in cfg.optimization.batteries:
        battery_cards_html += f"""
      <div class="d-flex flex-column me-3">
        <label class="form-label mb-1 small fw-semibold text-white-50">
          {bat.device_id} SoC&nbsp;(%)
        </label>
        <div class="input-group input-group-sm" style="width:130px">
          <input type="number" class="form-control bat-soc-input"
            id="soc-{bat.device_id}" data-battery="{bat.device_id}"
            min="{bat.min_soc_percentage}" max="{bat.max_soc_percentage}"
            value="{bat.initial_soc_percentage}" step="1">
          <span class="input-group-text">%</span>
        </div>
        <span class="small text-white-50">{bat.min_soc_percentage}–{bat.max_soc_percentage}% · {bat.capacity_wh}&nbsp;Wh</span>
      </div>"""

    # ── Inverter info badges ──────────────────────────────────────────
    inverter_cards_html = ""
    for inv in cfg.optimization.inverters:
        bat = bat_lookup.get(inv.battery_id) if inv.battery_id else None
        pv_badge = '<span class="badge bg-warning text-dark me-1">PV</span>' if inv.has_pv else ""
        bat_badge = (
            f'<span class="badge bg-info text-dark me-1">{inv.battery_id}</span>' if bat else ""
        )
        zfi = '<span class="badge bg-secondary me-1">ZFI</span>' if inv.zero_feed_in else ""
        inverter_cards_html += f"""
      <div class="me-3 small border rounded px-2 py-1 bg-white">
        <div class="fw-semibold">{inv.device_id}</div>
        <div>{pv_badge}{bat_badge}{zfi}</div>
        <div class="text-muted" style="font-size:0.75rem">
          Out {inv.max_ac_output_power_w:.0f} W · Chg {inv.max_ac_charge_power_w:.0f} W
        </div>
      </div>"""

    # ── Tab IDs (for JS) ──────────────────────────────────────────────
    prediction_tab_ids: list[str] = ["tab-elecprice", "tab-feedin", "tab-load", "tab-pv"]
    if has_weather:
        prediction_tab_ids.append("tab-weather")
    inv_tab_ids = [f"tab-inv-{inv.device_id}" for inv in cfg.optimization.inverters]
    all_tab_ids = prediction_tab_ids + inv_tab_ids

    # ── Tab navigation HTML ───────────────────────────────────────────
    tab_nav_items = ""
    for i, (tab_id, label) in enumerate(
        [
            ("tab-elecprice", "Electric Price"),
            ("tab-feedin", "Feed-in Tariff"),
            ("tab-load", "Load"),
            ("tab-pv", "PV Forecast"),
        ]
        + ([("tab-weather", "Weather")] if has_weather else [])
        + [
            (f"tab-inv-{inv.device_id}", f"⚡ {inv.device_id}")
            for inv in cfg.optimization.inverters
        ]
    ):
        active = "active" if i == 0 else ""
        tab_nav_items += (
            f'  <li class="nav-item">'
            f'<a class="nav-link {active}" id="{tab_id}-link" href="#" data-tab="{tab_id}">'
            f"{label}</a></li>\n"
        )

    # ── Tab content panes HTML ────────────────────────────────────────
    tab_panes_html = ""
    for i, tab_id in enumerate(all_tab_ids):
        active = "show active" if i == 0 else ""
        h = (
            "600px"
            if tab_id == "tab-weather"
            else ("700px" if tab_id.startswith("tab-inv-") else "450px")
        )
        tab_panes_html += f"""  <div class="tab-pane {active}" id="{tab_id}">
    <div id="chart-{tab_id}" class="gp-chart" style="height:{h}"></div>
  </div>\n"""

    tabs_json = json.dumps(all_tab_ids)
    first_inv_tab_js = f'"{inv_tab_ids[0]}"' if inv_tab_ids else '""'

    # prediction horizon & dt for display
    horizon_h = cfg.prediction.horizon
    dt_min = int(cfg.prediction.dt_hours * 60)
    objective = cfg.optimization.solver.objective

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GridPythia</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
    crossorigin="anonymous">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
  <style>
    body {{ font-size: .875rem; background: #f8f9fa; }}
    .gp-chart {{ min-height: 300px; }}
    .tab-pane {{ display: none; }}
    .tab-pane.show.active {{ display: block; }}
    .nav-link {{ cursor: pointer; padding: .35rem .75rem; }}
    #status-msg {{ min-width: 180px; font-size: .8rem; }}
    .spinner-border {{ width: 1rem; height: 1rem; border-width: .15em; }}
    #tabContent {{ background: #fff; border: 1px solid #dee2e6; border-top: none;
                   border-radius: 0 0 .375rem .375rem; padding: .75rem; }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
      .navbar {{ flex-wrap: wrap; }}
      .navbar-brand {{ font-size: 1rem; }}
      #battery-controls {{ width: 100%; margin-top: 0.5rem; order: 3; }}
      .optimize-controls {{ order: 4; margin-top: 0.5rem; }}
      #mainTabs {{ flex-wrap: nowrap; overflow-x: auto; }}
      .nav-tabs .nav-link {{ white-space: nowrap; padding: .25rem .5rem; font-size: .75rem; }}
      #tabContent {{ padding: .5rem; }}
      .gp-chart {{ min-height: 250px; }}
    }}

    @media (max-width: 576px) {{
      body {{ font-size: .75rem; }}
      .navbar {{ padding: 0.5rem; }}
      .d-flex > div {{ margin-right: 0.25rem !important; }}
      #mainTabs {{ border-bottom: 1px solid #dee2e6; overflow-x: auto; scrollbar-width: thin; }}
      #mainTabs::-webkit-scrollbar {{ height: 4px; }}
      #mainTabs::-webkit-scrollbar-track {{ background: #f1f1f1; }}
      #mainTabs::-webkit-scrollbar-thumb {{ background: #888; border-radius: 2px; }}
      .nav-tabs .nav-link {{ padding: .2rem .4rem; font-size: .65rem; }}
      #tabContent {{ padding: .25rem; }}
      .gp-chart {{ min-height: 200px; }}
    }}
  </style>
</head>
<body>

<!-- ── Navbar ────────────────────────────────────────────────── -->
<nav class="navbar navbar-dark bg-dark py-2 px-3">
  <div class="container-fluid">
    <span class="navbar-brand fw-bold">⚡ GridPythia</span>

    <!-- Right section with status and timezone -->
    <div class="d-flex align-items-center gap-2 ms-auto">
      <span id="status-msg" class="text-white-50 text-truncate" style="max-width: 200px; font-size: .8rem;"></span>
      <div id="spinner" class="spinner-border text-light d-none" role="status" style="width:1rem; height:1rem; border-width:.15em">
        <span class="visually-hidden">Loading…</span>
      </div>
      <span id="tz-display" class="badge bg-secondary ms-2" style="font-size:.65rem;"></span>
    </div>
  </div>

  <!-- Battery controls + Optimize (wrapped for responsiveness) -->
  <div class="w-100" id="battery-controls">
    <div class="d-flex flex-wrap gap-1 align-items-end mt-2">
{battery_cards_html}
      <div class="d-flex flex-column optimize-controls">
        <div class="small text-white-50 mb-1" style="font-size:.7rem;">Horizon: {horizon_h:.0f} h · Δt: {dt_min} min · {objective}</div>
        <button class="btn btn-success btn-sm px-3 fw-semibold" id="btn-optimize" onclick="runOptimize()" style="font-size:.75rem;">
          ▶ Optimize
        </button>
      </div>
    </div>
  </div>
</nav>

<!-- ── Inverter info bar (scrollable on mobile) ──────────────────── -->
<div class="px-2 py-2 bg-light border-bottom d-flex flex-wrap align-items-center gap-1 overflow-x-auto">
  <span class="small text-muted fw-semibold me-1" style="white-space:nowrap;">Inverters:</span>
{inverter_cards_html}
</div>

<!-- ── Tabs (scrollable on mobile) ──────────────────────────────── -->
<div class="px-2 pt-1">
  <ul class="nav nav-tabs d-flex flex-nowrap overflow-x-auto pb-0" id="mainTabs" style="border-bottom: 1px solid #dee2e6;">
{tab_nav_items}  </ul>
  <div class="tab-content" id="tabContent">
{tab_panes_html}  </div>
</div>

<script>
// ── Timezone ────────────────────────────────────────────────────
const TZ = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
document.getElementById('tz-display').textContent = '🕐 ' + TZ;

// ── Tab switching ───────────────────────────────────────────────
const ALL_TABS = {tabs_json};
const FIRST_INV_TAB = {first_inv_tab_js};

function showTab(tabId) {{
  ALL_TABS.forEach(id => {{
    const pane = document.getElementById(id);
    const link = document.getElementById(id + '-link');
    if (!pane || !link) return;
    const active = id === tabId;
    pane.className = 'tab-pane' + (active ? ' show active' : '');
    link.className = 'nav-link' + (active ? ' active' : '');
  }});
  // Trigger responsive resize for any plotly charts in the pane
  const chartDiv = document.getElementById('chart-' + tabId);
  if (chartDiv && chartDiv.data) {{
    Plotly.relayout(chartDiv, {{ autosize: true }});
  }}
}}

document.querySelectorAll('.nav-link[data-tab]').forEach(link => {{
  link.addEventListener('click', e => {{ e.preventDefault(); showTab(link.dataset.tab); }});
}});

// ── Optimize ────────────────────────────────────────────────
async function runOptimize() {{
  const btn    = document.getElementById('btn-optimize');
  const spinner = document.getElementById('spinner');
  const statusEl = document.getElementById('status-msg');

  btn.disabled = true;
  spinner.classList.remove('d-none');
  statusEl.textContent = 'Fetching…';

  const soc = {{}};
  document.querySelectorAll('.bat-soc-input').forEach(inp => {{
    soc[inp.dataset.battery] = parseFloat(inp.value);
  }});

  const payload = {{ timezone: TZ, battery_soc: soc }};

  try {{
    // Phase 1: Fetch predictions
    const fetchResp = await fetch('/api/fetch', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload)
    }});

    if (!fetchResp.ok) {{
      const body = await fetchResp.json().catch(() => ({{ detail: fetchResp.statusText }}));
      throw new Error('Fetch failed: ' + (body.detail || fetchResp.statusText));
    }}

    const fetchData = await fetchResp.json();

    // Render prediction charts immediately
    let rendered = 0;
    for (const [tabId, figData] of Object.entries(fetchData.charts)) {{
      const el = document.getElementById('chart-' + tabId);
      if (!el) continue;
      const layout = Object.assign({{ autosize: true }}, figData.layout || {{}});
      Plotly.react(el, figData.data || [], layout, {{ responsive: true, displayModeBar: true }});
      rendered++;
    }}

    statusEl.textContent = 'Forecasts loaded ✓ · Optimizing…';

    // Phase 2: Run optimization
    const optResp = await fetch('/api/optimize', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload)
    }});

    if (!optResp.ok) {{
      const body = await optResp.json().catch(() => ({{ detail: optResp.statusText }}));
      throw new Error('Optimization failed: ' + (body.detail || optResp.statusText));
    }}

    const optData = await optResp.json();

    // Render inverter charts
    let optRendered = 0;
    for (const [tabId, figData] of Object.entries(optData.charts)) {{
      const el = document.getElementById('chart-' + tabId);
      if (!el) continue;
      const layout = Object.assign({{ autosize: true }}, figData.layout || {{}});
      Plotly.react(el, figData.data || [], layout, {{ responsive: true, displayModeBar: true }});
      optRendered++;
    }}

    statusEl.textContent = optData.status || `Done ✓ (${{rendered + optRendered}} charts)`;

    // Jump to first inverter tab after optimization
    if (FIRST_INV_TAB) showTab(FIRST_INV_TAB);

  }} catch (err) {{
    statusEl.textContent = '✗ ' + err.message;
    console.error('Optimize error:', err);
  }} finally {{
    btn.disabled = false;
    spinner.classList.add('d-none');
  }}
}}
</script>
</body>
</html>"""


# ── FastAPI endpoints ─────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page application."""
    try:
        cfg, raw_yaml = _load_config()
    except FileNotFoundError as exc:
        return HTMLResponse(f"<h1>Config not found</h1><pre>{exc}</pre>", status_code=503)
    except Exception as exc:
        return HTMLResponse(f"<h1>Config error</h1><pre>{exc}</pre>", status_code=500)
    has_weather = "weather" in raw_yaml.get("prediction", {})
    return HTMLResponse(_build_html(cfg, has_weather))


class OptimizeRequest(BaseModel):
    timezone: str = "UTC"
    battery_soc: dict[str, float] = {}


@app.post("/api/fetch")
async def fetch_predictions(req: OptimizeRequest) -> JSONResponse:
    """Fetch predictions and return Plotly figure JSON for prediction tabs.

    If the server-side pdata cache is still fresh, the cached data is returned
    immediately without any external API calls.
    """
    # ── Fast path: serve from cache if still fresh ────────────────────
    cached = _get_cached_pdata()
    if cached is not None:
        pdata, forecast_from = cached
        charts = _make_prediction_figures(pdata, forecast_from)
        logger.info("webgui_fetch_served_from_cache", charts=list(charts.keys()))
        return JSONResponse({"charts": charts, "from_cache": True})

    # ── Slow path: build providers (singletons) and fetch ─────────────
    try:
        cfg, raw_yaml = _load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    try:
        tz = ZoneInfo(req.timezone)
    except ZoneInfoNotFoundError:
        logger.warning("unknown_timezone", tz=req.timezone)
        tz = ZoneInfo("UTC")

    try:
        setup = _get_providers(cfg, raw_yaml)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Provider build error: {exc}") from exc

    pred = Prediction(setup)
    start = datetime.now(tz=tz)
    hours = float(cfg.prediction.horizon)
    dt = float(cfg.prediction.dt_hours)

    try:
        pdata = await pred.fetch(start=start, hours=hours, dt_hours=dt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Prediction fetch failed: {exc}") from exc

    # Extract forecast_from from the EnergyCharts provider (if present)
    forecast_from: datetime | None = None
    if isinstance(setup.electricprice, ElecPriceEnergyCharts):
        forecast_from = setup.electricprice.last_real_ts

    _set_cached_pdata(pdata, forecast_from)

    charts = _make_prediction_figures(pdata, forecast_from)
    logger.info("webgui_fetch_done", charts=list(charts.keys()))
    return JSONResponse({"charts": charts, "from_cache": False})


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest) -> JSONResponse:
    """Run optimization and return Plotly figure JSON for all tabs.

    Reuses server-side cached PredictionData when fresh (populated by /api/fetch).
    Falls back to a fresh fetch if the cache has expired.
    """
    # ── Load config ───────────────────────────────────────────────────
    try:
        cfg, raw_yaml = _load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    try:
        tz = ZoneInfo(req.timezone)
    except ZoneInfoNotFoundError:
        logger.warning("unknown_timezone", tz=req.timezone)
        tz = ZoneInfo("UTC")

    # ── Prediction: use cache or re-fetch via persistent providers ────
    cached = _get_cached_pdata()
    if cached is not None:
        pdata, forecast_from = cached
        logger.info("webgui_optimize_using_cached_pdata")
    else:
        try:
            setup = _get_providers(cfg, raw_yaml)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Provider build error: {exc}") from exc

        pred = Prediction(setup)
        start = datetime.now(tz=tz)
        hours = float(cfg.prediction.horizon)
        dt = float(cfg.prediction.dt_hours)
        try:
            pdata = await pred.fetch(start=start, hours=hours, dt_hours=dt)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Prediction fetch failed: {exc}") from exc

        forecast_from = (
            setup.electricprice.last_real_ts
            if isinstance(setup.electricprice, ElecPriceEnergyCharts)
            else None
        )
        _set_cached_pdata(pdata, forecast_from)

    # ── Run optimization with singleton optimizer ─────────────────────
    try:
        optimizer = _get_optimizer(cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimizer setup error: {exc}") from exc

    objective = (
        OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
        if cfg.optimization.solver.objective == "self_consumption"
        else OptimizationObjective.MINIMIZE_COST
    )
    solver_opts = dict(cfg.optimization.solver.solver_opts)
    soc_wh_overrides = _soc_overrides_wh_for_solver(optimizer, req.battery_soc)

    try:
        async with _optimizer_lock:
            solution = await asyncio.to_thread(
                lambda: optimizer.solve(
                    pdata,
                    soc=soc_wh_overrides or None,
                    objective=objective,
                    solver_opts=solver_opts,
                    validate_with_simulation=True,
                )
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc

    # ── Build charts ──────────────────────────────────────────────────
    charts: dict[str, Any] = {}

    # Prediction tabs
    charts.update(_make_prediction_figures(pdata, forecast_from))

    # Per-inverter optimization tabs
    _plotter = SolutionPlotter()
    for inv in optimizer.inverters:
        tab_id = f"tab-inv-{inv.device_id}"
        charts[tab_id] = _fig_to_dict(_plotter.plot_inverter(solution, inv.device_id))

    # ── Savings vs. naive baseline (PV direct to load, no battery) ───
    pv_total_arr = np.zeros(pdata.steps, dtype=float)
    for _arr in pdata.pv_by_inverter.values():
        pv_total_arr += np.asarray(_arr, dtype=float)
    _load = np.asarray(pdata.load_wh, dtype=float)
    _price = (
        np.asarray(pdata.electricprice, dtype=float)
        if pdata.electricprice is not None
        else np.zeros(pdata.steps, dtype=float)
    )
    _feedin_t = (
        np.asarray(pdata.feedintariff, dtype=float)
        if pdata.feedintariff is not None
        else np.zeros(pdata.steps, dtype=float)
    )
    naive_net_cost = float(
        np.sum(np.maximum(0.0, _load - pv_total_arr) * _price)
        - np.sum(np.maximum(0.0, pv_total_arr - _load) * _feedin_t)
    )
    optimized_net_cost = solution.result.total_cost - solution.result.total_revenue
    savings = naive_net_cost - optimized_net_cost

    # ── Status message ────────────────────────────────────────────────
    parity_warn = (
        " ⚠ parity"
        if (solution.parity_report is not None and not solution.parity_report.ok)
        else ""
    )
    status = (
        f"Solved {solution.solve_time_s:.1f}s · {solution.solver_status} · "
        f"naive: {naive_net_cost:.3f} EUR → optimized: {optimized_net_cost:.3f} EUR · "
        f"savings: {savings:.3f} EUR{parity_warn}"
    )

    logger.info(
        "webgui_optimize_done",
        solver_status=solution.solver_status,
        solve_time_s=round(solution.solve_time_s, 2),
        charts=list(charts.keys()),
    )

    return JSONResponse({"charts": charts, "status": status})


# ── entry point ───────────────────────────────────────────────────────────


def run() -> None:
    """Parse CLI arguments and start the uvicorn server."""
    import uvicorn

    global _config_path  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="GridPythia web GUI")
    parser.add_argument(
        "--config",
        default=str(_config_path),
        help="Path to config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="TCP port (default: 8080)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev mode only)",
    )
    args = parser.parse_args()

    _config_path = Path(args.config).expanduser().resolve()

    # Configure structlog
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    logger.info(
        "webgui_starting",
        config=str(_config_path),
        url=f"http://localhost:{args.port}",
    )
    uvicorn.run(
        "utils.webgui:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


if __name__ == "__main__":
    run()
