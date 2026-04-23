# GridPythia Server

REST API backend for the GridPythia home energy management system.
Built with **FastAPI** and served by **uvicorn**.

---

## Architecture

```text
GridPythia/server/
├── __init__.py          # exports create_app()
├── app.py               # FastAPI app factory
├── state.py             # Module-level singleton state (providers, optimizer, pdata cache)
├── services.py          # Business logic: config loading, provider/optimizer management,
│                        #   chart building, plan serialisation
├── models.py            # Pydantic request / response schemas
├── routers/
│   ├── config.py        # GET  /api/config
│   ├── predictions.py   # GET  /api/predictions/status
│   │                    # POST /api/predictions/fetch
│   └── optimization.py  # POST /api/optimize
└── static/
    └── index.html       # Single-page frontend (pure HTML/JS/CSS, no server-side templating)
```

**Design principles**

- The frontend (`index.html`) is a fully static file.  It bootstraps by calling `GET /api/config`
  on page load and constructs all UI elements (battery inputs, inverter info, tabs) from that
  JSON response — no Python f-strings or server-side templating required.
- All business logic lives in `services.py`.  Router handlers are thin wrappers that validate
  input, call services, and format responses.
- Expensive objects (provider singletons, compiled CVXPY model) are kept alive in `state.py`
  across requests.  They are rebuilt only when the config file's mtime changes.

---

## Running

```bash
uv run python -m utils.webgui
# then open http://localhost:8080
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `config.yaml` in repo root | Path to YAML configuration file |
| `--host HOST`   | `0.0.0.0` | Bind address |
| `--port PORT`   | `8080`    | TCP port |
| `--reload`      | off       | Enable uvicorn auto-reload (dev only) |

Interactive API docs are available at **`/api/docs`** (Swagger UI) and **`/api/redoc`** (ReDoc).

---

## Endpoints

### `GET /api/config`

Returns UI bootstrap data.  Called once on page load.

**Response** `200 OK`

```json
{
  "batteries": [
    {
      "device_id": "AB2000X",
      "min_soc_percentage": 10,
      "max_soc_percentage": 95,
      "initial_soc_percentage": 50,
      "capacity_wh": 2000.0
    }
  ],
  "inverters": [
    {
      "device_id": "SolarEdge1",
      "has_pv": true,
      "battery_id": "AB2000X",
      "max_ac_output_power_w": 5000.0,
      "max_ac_charge_power_w": 3600.0,
      "zero_feed_in": true
    }
  ],
  "has_weather": false,
  "horizon_h": 48.0,
  "dt_min": 15,
  "objective": "cost"
}
```

---

### `GET /api/predictions/status`

Returns the current state of the server-side prediction cache.

**Response** `200 OK`

```json
{
  "has_cache": true,
  "age_s": 42.3,
  "ttl_s": 300.0,
  "forecast_from": "2026-04-23T14:00:00+02:00"
}
```

`forecast_from` is the last confirmed real-data timestamp from the EnergyCharts provider
(the boundary between measured and statistically forecast prices).  `null` when the
EnergyCharts provider is not configured.

---

### `POST /api/predictions/fetch`

Fetches all forecast channels and returns Plotly figure JSON.
Serves from the server-side 5-minute cache when available.

**Request body**

```json
{
  "timezone": "Europe/Berlin"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timezone` | string | `"UTC"` | IANA timezone name for the forecast start time |

**Response** `200 OK`

```json
{
  "charts": {
    "tab-elecprice": { "data": [...], "layout": {...} },
    "tab-feedin":    { "data": [...], "layout": {...} },
    "tab-load":      { "data": [...], "layout": {...} },
    "tab-pv":        { "data": [...], "layout": {...} },
    "tab-weather":   { "data": [...], "layout": {...} }
  },
  "from_cache": false
}
```

Each `charts` value is a Plotly figure serialised to JSON and can be passed directly to
`Plotly.react(element, fig.data, fig.layout)`.  The `tab-weather` key is only present
when weather is configured.

---

### `POST /api/optimize`

Runs the MILP energy optimizer and returns the full solution together with Plotly charts.

**Request body**

```json
{
  "timezone": "Europe/Berlin",
  "battery_soc": {
    "AB2000X": 65.0
  },
  "initial_modes": {
    "SolarEdge1": 0
  },
  "solver_opts": {
    "time_limit": 15,
    "mip_rel_gap": 0.02
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timezone` | string | `"UTC"` | IANA timezone for forecast start |
| `battery_soc` | `{battery_id: float}` | `{}` | Battery SoC overrides in % `[0, 100]`.  Clamped to `[min_soc, max_soc]`. |
| `initial_modes` | `{inverter_id: int}` | `{}` | Inverter mode at the start of the horizon. See [InverterMode](#invertermodes). Defaults to `IDLE (0)`. |
| `solver_opts` | object \| null | `null` | HiGHS option overrides for this call only, merged over config defaults. |

**Response** `200 OK`

```json
{
  "summary": {
    "solver_status": "optimal",
    "solve_time_s": 2.4,
    "objective": "cost",
    "total_cost_eur": 1.234,
    "total_revenue_eur": 0.210,
    "net_cost_eur": 1.024,
    "naive_net_cost_eur": 1.437,
    "savings_eur": 0.413,
    "parity_ok": true
  },
  "inverter_plans": [
    {
      "device_id": "SolarEdge1",
      "steps": [
        {
          "timestamp": "2026-04-23T08:00:00+02:00",
          "mode": 1,
          "mode_name": "DISCHARGE",
          "charge_ac_wh": 0.0,
          "discharge_ac_wh": 312.5,
          "pv_to_ac_wh": 450.0,
          "pv_to_battery_wh": 0.0,
          "battery_soc_wh": 1687.5
        }
      ]
    }
  ],
  "charts": {
    "tab-elecprice":     { "data": [...], "layout": {...} },
    "tab-feedin":        { "data": [...], "layout": {...} },
    "tab-load":          { "data": [...], "layout": {...} },
    "tab-pv":            { "data": [...], "layout": {...} },
    "tab-inv-SolarEdge1":{ "data": [...], "layout": {...} }
  },
  "status": "Solved 2.4s · optimal · naive: 1.437 EUR → optimized: 1.024 EUR · savings: 0.413 EUR"
}
```

**`summary` fields**

| Field | Description |
|-------|-------------|
| `solver_status` | HiGHS status string (`"optimal"`, `"optimal_inaccurate"`, `"user_limit"`) |
| `solve_time_s` | Wall-clock seconds spent inside the solver |
| `objective` | Objective function used (`"cost"` or `"self_consumption"`) |
| `total_cost_eur` | Total grid import cost over the horizon [EUR] |
| `total_revenue_eur` | Total feed-in revenue over the horizon [EUR] |
| `net_cost_eur` | `total_cost_eur - total_revenue_eur` |
| `naive_net_cost_eur` | Net cost of the naive baseline (PV direct to load, no battery) |
| `savings_eur` | `naive_net_cost_eur - net_cost_eur` |
| `parity_ok` | `true` when the LP solution matches a GridSimulation replay within tolerances; `null` when parity checking was skipped |

**`inverter_plans[].steps[]` fields**

| Field | Unit | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | Wall-clock start of this time slot |
| `mode` | int | [InverterMode](#invertermodes) integer |
| `mode_name` | string | Human-readable mode name |
| `charge_ac_wh` | Wh | AC energy drawn from grid to charge battery |
| `discharge_ac_wh` | Wh | AC energy delivered from battery to home |
| `pv_to_ac_wh` | Wh | PV energy routed to the AC bus |
| `pv_to_battery_wh` | Wh | PV energy routed into the battery |
| `battery_soc_wh` | Wh | SoC at the **end** of this slot; `null` if no battery |

---

## InverterModes

| Value | Name | Description |
|-------|------|-------------|
| `0` | `IDLE` | No active charging or discharging |
| `1` | `DISCHARGE` | Battery discharges to cover home load and/or feed into grid |
| `2` | `DISCHARGE_ZERO_FEED_IN` | Discharge with zero-feed-in limit (no export to grid) |
| `3` | `AC_CHARGE` | Charge battery from grid |
| `4` | `AC_CHARGE_ZERO_FEED_IN` | Charge from grid with zero-feed-in limit |

---

## Singleton State and Caching

The server maintains three singleton objects that survive across HTTP requests:

| Object | Location | Rebuilt when |
|--------|----------|--------------|
| Provider instances | `state.providers` | Config file `mtime` changes |
| `LinearOptimizer` (compiled CVXPY model) | `state.optimizer` | Config file `mtime` changes |
| `PredictionData` cache | `state.pdata_cache` | Older than 5 minutes (TTL) |

The prediction cache avoids double-fetching from external APIs when `/api/predictions/fetch`
and `/api/optimize` are called in quick succession (the standard UI flow).

The optimizer singleton preserves the compiled CVXPY problem structure across calls.
Only the runtime Parameters (price arrays, battery start SoC, initial modes) are updated per
solve via `_update_runtime_parameters()`, which significantly reduces per-request overhead.
