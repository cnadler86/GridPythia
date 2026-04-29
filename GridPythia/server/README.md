# GridPythia Server

REST API + Dashboard for the GridPythia home energy management system.

---

## Quick start

### Dashboard first (recommended)

If you mainly want to run the server and open the dashboard, this is enough:

```bash
# From repo root (venv active)
python -m main --port 8080
```

Then open:

- Dashboard: `http://localhost:8080`
- API docs: `http://localhost:8080/api/docs`

What happens automatically in the dashboard:

- prediction status is polled continuously
- inverter status is polled continuously
- at each optimization slot, optimization runs automatically when inverter data is fresh

### Full startup examples

```bash
# From the repo root (venv already active):
python -m main

# Custom config or port:
python -m main --config /path/to/config.yaml --port 8080

# Then open: http://localhost:8080
```

Interactive API docs: **`http://localhost:8080/api/docs`**

| Flag | Default | Description |
| ------ | --------- | ----------- |
| `--config PATH` | `config.yaml` | Path to the YAML configuration file |
| `--host HOST` | `server.bind_host` | Bind address (CLI override) |
| `--port PORT` | `server.bind_port` | TCP port (CLI override) |
| `--reload` | off | Auto-reload on code changes (dev only) |

---

## Configuration reference

All settings live in a single `config.yaml`.  The three top-level sections are:

### `prediction` – forecast sources

```yaml
prediction:
  latitude: 47.99545      # Site location
  longitude: 7.83355
  horizon: 48             # Forecast window in hours
  dt_hours: 0.25          # Time-step (0.25 = 15 min)

  electricprice:
    provider: "EnergyCharts"   # EnergyCharts | Fixed
    charges_kwh: 0.1528        # Network charges added on top
    vat_rate: 0.19
    energycharts:
      bidding_zone: "DE-LU"

  feedintariff:
    provider: "Fixed"
    tariff_kwh: 0.082

  load:
    provider: "ProfileCSV"
    path: "GridPythia/prediction/load/data/profiles.csv"
    country: "DE"
    subdivision: "BW"

  pvforecast:
    provider: "OpenMeteo"      # OpenMeteo | Akkudoktor
    plane:
      inverter_id: "SF800Pro"  # Must match an inverter device_id below
      peak_kw: 0.41
      tilt: 75.0
      azimuth: 218.0           # North=0, East=90, South=180, West=270
    openmeteo:
      damping_morning: 2.0
      damping_evening: 0.2

  # Optional – uncomment to enable weather tab in the dashboard:
  # weather:
  #   provider: "OpenMeteo"    # OpenMeteo | BrightSky
```

### `optimization` – battery and inverter hardware

```yaml
optimization:
  solver:
    provider: "highs"
    objective: "cost"          # cost | self_consumption
    solver_opts:
      time_limit: 30
      mip_rel_gap: 0.03

  batteries:
    - device_id: "AB2000X"
      capacity_wh: 1920
      charging_efficiency: 0.98
      discharging_efficiency: 0.98
      initial_soc_percentage: 50   # Used when no live status is available
      min_soc_percentage: 20
      max_soc_percentage: 100

  inverters:
    - device_id: "SF800Pro"
      battery_id: "AB2000X"        # References batteries[].device_id
      has_pv: true
      max_ac_output_power_w: 800
      max_ac_charge_power_w: 1000
      dc_to_ac_efficiency: 0.95
      ac_to_dc_efficiency: 0.93
      zero_feed_in: true
      mode_switch_cost: 0.005      # EUR per mode change (wear cost)
      active_inverter_consumption_w: 15
```

### `server` – runtime behaviour

```yaml
server:
  # Web server bind. Set to your LAN IP for direct device access,
  # or use 0.0.0.0 to listen on all interfaces.
  bind_host: "127.0.0.1"
  bind_port: 8080

  # Optimization is blocked when any optimisable inverter's last status
  # report is older than this value (seconds).
  inverter_status_max_age_s: 300   # 5 minutes

  scheduler:
    optimization_interval_minutes: 15   # Must be a divisor of 60
    prediction_refresh_minutes: 30      # How often to refresh forecast data
    dispatch_buffer_seconds: 5          # Extra safety margin on top of solver time_limit
    dispatch_buffer_max_seconds: 30     # Cap for adaptive late-publish compensation

  mqtt:
    enabled: false
    broker: "mqtt://localhost:1883"
    client_id: "gridpythia"
    topic_prefix: "gridpythia"
```

> **Note on optimization slots:** The scheduler fires at fixed boundaries aligned
> to the top of the hour — e.g., every 15 min → 00:00, 00:15, 00:30, 00:45.
> It starts each solve *before* the upcoming slot using
> `solver time_limit + dispatch_buffer_seconds`, and if the previous cycle
> published late, that lateness delta is added to the 5 s buffer up to
> `dispatch_buffer_max_seconds`.
> When a solve finishes shortly before dispatch and the new plan starts at the
> next slot, the MQTT publisher prepends the still-active current slot from the
> previously retained plan so downstream consumers can react correctly until the
> boundary is reached.

---

## Feeding inverter status to the server

Before the optimizer runs it needs to know the **current battery SoC** and the
**active operating mode** of each inverter.  You push this data via a simple
REST call — from your home-automation system, a cron job, or an MQTT bridge.

### `POST /api/inverters/{device_id}/status`

Call this whenever your inverter publishes a new state (once per minute is
usually sufficient).

### Request

```http
POST /api/inverters/SF800Pro/status
Content-Type: application/json

{
  "soc": 63.5,
  "mode": 0
}
```

| Field | Type | Required | Description |
| ------- | ------ | ---------- | ----------- |
| `soc` | float | ✅ | Battery SoC in % `[0–100]` |
| `mode` | int | — | Active mode (default `0` = IDLE). See [Inverter modes](#inverter-modes). |

### Response `200 OK`

```json
{
  "device_id": "SF800Pro",
  "soc": 63.5,
  "mode": 0,
  "mode_name": "IDLE",
  "reported_at": "2026-04-23T14:07:00+02:00",
  "age_s": 0.1,
  "is_fresh": true
}
```

`is_fresh` is `true` when `age_s` ≤ `server.inverter_status_max_age_s`.
When `is_fresh` is `false` for any optimisable inverter, the `POST /api/optimize`
call will be rejected with a `409` error.

### `GET /api/inverters/status`

Returns the last-known state for all inverters that have reported at least once.
Useful for checking from the dashboard or a monitoring script.

```json
[
  {
    "device_id": "SF800Pro",
    "soc": 63.5,
    "mode": 0,
    "mode_name": "IDLE",
    "reported_at": "2026-04-23T14:07:00+02:00",
    "age_s": 42.3,
    "is_fresh": true
  }
]
```

### Example: push from shell

```bash
curl -s -X POST http://localhost:8080/api/inverters/SF800Pro/status \
  -H "Content-Type: application/json" \
  -d '{"soc": 63.5, "mode": 0}'
```

### Example: Home Assistant automation (REST)

```yaml
action:
  - service: rest_command.gridpythia_status
    data:
      device_id: "SF800Pro"
      soc: "{{ states('sensor.battery_soc') | float }}"
      mode: 0

rest_command:
  gridpythia_status:
    url: "http://gridpythia.local:8080/api/inverters/{{ device_id }}/status"
    method: POST
    content_type: "application/json"
    payload: '{"soc": {{ soc }}, "mode": {{ mode }}}'
```

---

## Endpoints overview

| Method | Path | Description |
| -------- | ------ | ----------- |
| `GET` | `/api/config` | UI bootstrap: batteries, inverters, horizon settings |
| `GET` | `/api/predictions/status` | Cache age and TTL |
| `POST` | `/api/predictions/fetch` | Fetch all forecast channels; returns Plotly charts |
| `POST` | `/api/inverters/{device_id}/status` | Report inverter SoC + mode |
| `GET` | `/api/inverters/status` | All known inverter states |
| `GET` | `/api/optimize/status` | Optimization cache age and TTL |
| `GET` | `/api/optimize` | Get last cached optimization result |
| `POST` | `/api/optimize` | Run MILP optimizer; returns schedule + charts |

---

## `POST /api/predictions/fetch`

Fetches electricity price, PV forecast, load, and (optionally) weather.
Results are cached for `prediction_cache_ttl_s` (default 5 min) and reused by
`/api/optimize` if called shortly after.

### Fetch request

```json
{ "timezone": "Europe/Berlin" }
```

### Fetch response

`charts` maps tab-id → Plotly figure JSON:

```json
{
  "charts": {
    "tab-elecprice": { "data": [...], "layout": {...} },
    "tab-feedin":    { "data": [...], "layout": {...} },
    "tab-load":      { "data": [...], "layout": {...} },
    "tab-pv":        { "data": [...], "layout": {...} }
  },
  "from_cache": false
}
```

---

## `POST /api/optimize`

Runs the MILP optimizer and returns the full schedule.

### Optimizer request

```json
{
  "timezone": "Europe/Berlin",
  "battery_soc": { "AB2000X": 65.0 },
  "initial_modes": { "SF800Pro": 0 },
  "solver_opts": { "time_limit": 15 }
}
```

| Field | Description |
| ------- | ----------- |
| `battery_soc` | Override SoC per battery in % (keyed by `battery_id`). If omitted, uses the last value from `/api/inverters/{id}/status`, or `initial_soc_percentage` from config as fallback. |
| `initial_modes` | Override starting mode per inverter (keyed by `inverter_id`). Defaults to `IDLE (0)`. |
| `solver_opts` | Per-call HiGHS options, merged over config defaults. |

### Optimizer response

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
      "device_id": "SF800Pro",
      "steps": [
        {
          "timestamp": "2026-04-23T14:15:00+02:00",
          "mode": 2,
          "mode_name": "DISCHARGE_ZERO_FEED_IN",
          "charge_ac_wh": 0.0,
          "discharge_ac_wh": 62.5,
          "pv_to_ac_wh": 112.0,
          "pv_to_battery_wh": 0.0,
          "battery_soc_wh": 1157.5
        }
      ]
    }
  ],
  "charts": { ... },
  "status": "Solved 2.4s · optimal · savings: 0.413 EUR"
}
```

### How to read the schedule

Each step covers one `dt_hours` slot (e.g. 15 min = 0.25 h).  All energy values are
in **Wh for that slot** (not W).

| Field | What it means |
| ------- | ------------- |
| `mode` | What the inverter should do in this slot. See table below. |
| `discharge_ac_wh` | Energy drawn from battery, delivered to home load (after inverter losses). |
| `charge_ac_wh` | Energy drawn from the grid to charge the battery (before battery losses). |
| `pv_to_ac_wh` | PV energy that flows directly to the home / grid. |
| `pv_to_battery_wh` | PV energy stored in the battery. |
| `battery_soc_wh` | Battery state-of-charge **at the end** of this slot. |

The `summary.savings_eur` is the gain vs. a naive baseline (PV direct to load, no
battery charging/discharging control).  A negative value means the optimizer
found no improvement for this horizon (can happen with zero feed-in tariff and a
flat price profile).

---

## Inverter modes

| Value | Name | Description |
| ------- | ------ | ------------- |
| `0` | `IDLE` | Battery is passive; PV flows directly to load |
| `1` | `DISCHARGE` | Battery discharges; excess may feed into the grid |
| `2` | `DISCHARGE_ZERO_FEED_IN` | Battery discharges; zero export to grid |
| `3` | `AC_CHARGE` | Grid charges the battery; excess PV may export |
| `4` | `AC_CHARGE_ZERO_FEED_IN` | Grid charges the battery; no grid export |

---

## Architecture notes

```text
GridPythia/server/
├── app.py               # FastAPI app factory
├── state.py             # Request-shared singletons (providers, optimizer, coordinator)
├── services.py          # Business logic: config, provider/optimizer lifecycle, charts
├── models.py            # Pydantic request/response schemas
├── routers/
│   ├── config.py        # GET  /api/config
│   ├── predictions.py   # GET  /api/predictions/status  POST /api/predictions/fetch
│   ├── inverters.py     # POST /api/inverters/{id}/status  GET /api/inverters/status
│   └── optimization.py  # POST /api/optimize
└── static/
    └── index.html       # Single-page dashboard (pure HTML/JS/CSS)
```

- **Expensive objects** (CVXPY compiled model, provider HTTP caches) are kept in `state.py`
  and rebuilt only when `config.yaml` changes on disk.
- **Inverter state** is held by the `InverterCoordinator` singleton in `state.coordinator`.
  It is updated by `POST /api/inverters/{id}/status` and read by `POST /api/optimize`.
- The **prediction cache** avoids double-fetching when the dashboard calls
  `/predictions/fetch` and then `/optimize` in quick succession.

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

### Design principles

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
# start the server (venv active)
python -m main
# then open http://localhost:8080
```

Options:

| Flag | Default | Description |
| ------------------- | ------------------------ | ------------- |
| `--config PATH` | `config.yaml` in repo root | Path to YAML configuration file |
| `--host HOST` | `server.bind_host` | Bind address (CLI override) |
| `--port PORT` | `server.bind_port` | TCP port (CLI override) |
| `--reload` | off | Enable uvicorn auto-reload (dev only) |

Interactive API docs are available at **`/api/docs`** (Swagger UI) and **`/api/redoc`** (ReDoc).

---

## Endpoints

### `GET /api/config`

Returns UI bootstrap data.  Called once on page load.

#### Config response `200 OK`

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

#### Status response `200 OK`

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

### `POST /api/predictions/fetch` (detailed)

Fetches all forecast channels and returns Plotly figure JSON.
Serves from the server-side 5-minute cache when available.

#### Fetch request body

```json
{
  "timezone": "Europe/Berlin"
}
```

| Field | Type | Default | Description |
| ------- | ------ | --------- | ----------- |
| `timezone` | string | `"UTC"` | IANA timezone name for the forecast start time |

#### Fetch response body

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

### `POST /api/optimize` (detailed)

Runs the MILP energy optimizer and returns the full solution together with Plotly charts.

#### Optimizer request body

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
| ------- | ------ | --------- | ----------- |
| `timezone` | string | `"UTC"` | IANA timezone for forecast start |
| `battery_soc` | `{battery_id: float}` | `{}` | Battery SoC overrides in % `[0, 100]`.  Clamped to `[min_soc, max_soc]`. |
| `initial_modes` | `{inverter_id: int}` | `{}` | Inverter mode at the start of the horizon. See [InverterMode](#inverter-modes). Defaults to `IDLE (0)`. |
| Field | Type | Default | Description |

#### Optimizer response body

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

#### `summary` fields

| Field | Description |
| ------- | ----------- |
| `solver_status` | HiGHS status string (`"optimal"`, `"optimal_inaccurate"`, `"user_limit"`) |
| `solve_time_s` | Wall-clock seconds spent inside the solver |
| `objective` | Objective function used (`"cost"` or `"self_consumption"`) |
| `total_cost_eur` | Total grid import cost over the horizon [EUR] |
| `total_revenue_eur` | Total feed-in revenue over the horizon [EUR] |
| `net_cost_eur` | `total_cost_eur - total_revenue_eur` |
| `naive_net_cost_eur` | Net cost of the naive baseline (PV direct to load, no battery) |
| `savings_eur` | `naive_net_cost_eur - net_cost_eur` |
| `parity_ok` | `true` when the LP solution matches a GridSimulation replay within tolerances; `null` when parity checking was skipped |

#### `inverter_plans[].steps[]` fields

| Field | Unit | Description |
| ------- | ------ | ----------- |
| `timestamp` | ISO 8601 | Wall-clock start of this time slot |
| `mode` | int | [InverterMode](#inverter-modes) integer |
| `mode_name` | string | Human-readable mode name |
| `charge_ac_wh` | Wh | AC energy drawn from grid to charge battery |
| `discharge_ac_wh` | Wh | AC energy delivered from battery to home |
| `pv_to_ac_wh` | Wh | PV energy routed to the AC bus |
| `pv_to_battery_wh` | Wh | PV energy routed into the battery |
| `battery_soc_wh` | Wh | SoC at the **end** of this slot; `null` if no battery |

---

## Singleton State and Caching

The server maintains three singleton objects that survive across HTTP requests:

| Object | Location | Rebuilt when |
| -------- | ---------- | -------------- |
| Provider instances | `state.providers` | Config file `mtime` changes |
| `LinearOptimizer` (compiled CVXPY model) | `state.optimizer` | Config file `mtime` changes |
| `PredictionData` cache | `state.pdata_cache` | Older than 5 minutes (TTL) |

The prediction cache avoids double-fetching from external APIs when `/api/predictions/fetch`
and `/api/optimize` are called in quick succession (the standard UI flow).

Optimization results are cached server-side as well, so the dashboard can
restore the last solution after reopening. Use `/api/optimize/status` to
inspect cache metadata and `/api/optimize` (GET) to retrieve the cached payload.

The optimizer singleton preserves the compiled CVXPY problem structure across calls.
Only the runtime Parameters (price arrays, battery start SoC, initial modes) are updated per
solve via `_update_runtime_parameters()`, which significantly reduces per-request overhead.
