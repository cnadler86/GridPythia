# GridPythia

**Home energy prediction and optimization system for battery storage with PV.**

GridPythia continuously forecasts electricity prices, PV generation, and household load, then computes an optimal battery charge/discharge schedule using a MILP solver. It publishes the schedule via MQTT so connected inverter controllers (e.g. [ZeroPythia](https://github.com/cnadler86/ZeroPythia)) can execute it automatically.

---

## How it works

1. **Prediction** – fetches up to 48 h of electricity price, feed-in tariff, PV forecast, and load forecast from configurable providers (OpenMeteo, EpexPredictor, EnergyCharts, CSV profiles).
2. **Optimization** – runs a Mixed-Integer Linear Program (HiGHS solver via CVXPY) that minimises cost or maximises self-consumption over the forecast horizon, respecting battery SoC limits and inverter constraints.
3. **Scheduling** – fires at configurable fixed-boundary intervals (default: every 15 min) and dispatches the resulting plan to inverters via MQTT.
4. **Dashboard** – a built-in web UI shows live forecasts, optimization results, and inverter status.

---

## Entry Points

| Script | Purpose |
| --- | --- |
| `main.py` | Start the web server + scheduler (production) |
| `utils/profile_rolling_horizon.py` | Offline profile-rolling batch run |
| `utils/gui.py` | Local plotting utilities |

---

## Quick Start

**Requirements:** Python 3.13 or newer and [uv](https://docs.astral.sh/uv/).

### 1. Install dependencies

```bash
# Recommended (uv)
uv sync --no-dev

# Alternative (pip)
pip install -e .
```

> **ARM note (Raspberry Pi):** GridPythia uses HiGHS as its MILP solver.
> On ARMv7/ARMv6 targets, install `libatomic1` first:
> ```bash
> sudo apt-get install libatomic1
> ```

### 2. Configure

Copy and edit the configuration file:

```bash
cp config.yaml my-config.yaml   # optional – or edit config.yaml in-place
```

At minimum, set your site location and PV panel parameters:

```yaml
prediction:
  latitude: 47.995
  longitude: 7.834

  pvforecast:
    plane:
      peak_kw: 0.41
      tilt: 75.0
      azimuth: 218.0    # North=0, East=90, South=180, West=270

optimization:
  batteries:
    - device_id: "AB2000X"
      capacity_wh: 1920
      # ...

  inverters:
    - device_id: "SF800Pro"
      battery_id: "AB2000X"
      max_ac_output_power_w: 800
      # ...
```

See [Configuration reference](#configuration-reference) for all options.

### 3. Start the server

```bash
. .venv/bin/activate
python main.py
# → Dashboard:  http://127.0.0.1:8080
# → API docs:   http://127.0.0.1:8080/api/docs
```

For LAN access, override the bind address:

```bash
python main.py --host 0.0.0.0 --port 8080
```

### 4. All CLI options

```text
--config PATH   Path to config.yaml   (default: config.yaml next to main.py)
--host   HOST   Bind address          (default: value from config server.bind_host)
--port   PORT   TCP port              (default: value from config server.bind_port)
--reload        Auto-reload on code changes (dev only)
```

---

## Feeding inverter status

Before the optimizer runs it needs the **current battery SoC** and the **active mode**
of each configured inverter. Push this via the REST API whenever your device reports
(once per minute is sufficient):

```bash
curl -s -X POST http://localhost:8080/api/inverters/SF800Pro/status \
  -H "Content-Type: application/json" \
  -d '{"soc": 63.5, "mode": 0}'
```

If MQTT is enabled in the config, ZeroPythia (running in AUTO mode) pushes this
automatically.

---

## MQTT integration

Enable MQTT in `config.yaml`:

```yaml
server:
  mqtt:
    enabled: true
    broker: "mqtt://localhost:1883"
    client_id: "gridpythia"
    topic_prefix: "gridpythia"
```

After each successful optimization the scheduler publishes the plan:

```text
Topic:   gridpythia/inverters/{device_id}/plan
```

ZeroPythia subscribes to this topic in AUTO mode and executes the schedule slot by slot.

Inverter controllers report SoC back on:

```text
Topic:   gridpythia/inverters/{device_id}/status
Payload: {"soc": 63.5, "mode": 2}
```

---

## Inverter modes

| Value | Name | Description |
| --- | --- | --- |
| `0` | `IDLE` | Battery passive; PV flows directly to load |
| `1` | `DISCHARGE` | Battery discharges; excess may feed into the grid |
| `2` | `DISCHARGE_ZERO_FEED_IN` | Battery discharges; zero export to grid |
| `3` | `AC_CHARGE` | Grid charges the battery |
| `4` | `AC_CHARGE_ZERO_FEED_IN` | Grid charges the battery; no export |

---

## Configuration reference

All settings live in a single `config.yaml`. The four top-level sections are:

### `prediction`

| Key | Default | Description |
| --- | --- | --- |
| `latitude` / `longitude` | — | Site coordinates (required) |
| `horizon` | `48` | Forecast window in hours |
| `dt_hours` | `0.25` | Time step (0.25 = 15 min) |

**`electricprice`** – electricity price provider:

| `provider` | Description |
| --- | --- |
| `EpexPredictor` | EPEX spot price via [epexpredictor.batzill.com](https://epexpredictor.batzill.com) (self-hostable) |
| `EnergyCharts` | Energy-Charts.info API |
| `Fixed` | Constant price (`charges_kwh`) |

**`pvforecast`** – solar generation forecast:

| Key | Description |
| --- | --- |
| `provider` | `OpenMeteo` (default) or `Akkudoktor` |
| `plane.peak_kw` | Panel peak power in kW |
| `plane.tilt` | Panel tilt angle in degrees |
| `plane.azimuth` | Panel azimuth (South = 180°) |

**`load`** – household load forecast:

| Key | Description |
| --- | --- |
| `provider` | `ProfileCSV` (default) – standard load profile from CSV |
| `adaptive.enabled` | Enable learned load profile (blended with the base profile) |

### `optimization`

```yaml
optimization:
  solver:
    provider: "highs"
    objective: "cost"          # cost | self_consumption
    solver_opts:
      time_limit: 30           # Maximum solver time in seconds
      mip_rel_gap: 0.01        # Relative optimality gap

  batteries:
    - device_id: "AB2000X"
      capacity_wh: 1920
      min_soc_percentage: 15
      max_soc_percentage: 99

  inverters:
    - device_id: "SF800Pro"
      battery_id: "AB2000X"
      has_pv: true
      max_ac_output_power_w: 800
      max_ac_charge_power_w: 1000
      zero_feed_in: true
```

### `server`

```yaml
server:
  bind_host: "127.0.0.1"
  bind_port: 8080
  timezone: "Europe/Berlin"
  inverter_status_max_age_s: 300   # Block optimization if status is stale

  scheduler:
    optimization_interval_minutes: 15   # Must be a divisor of 60

  mqtt:
    enabled: false
    broker: "mqtt://localhost:1883"
    client_id: "gridpythia"
    topic_prefix: "gridpythia"
```

### `update` (inside `server`)

```yaml
server:
  update:
    mode: "off"      # off | release | master
    branch: master
    remote: origin
```

See [Auto-Update](#auto-update) for details.

---

## Auto-Update

GridPythia can update itself automatically from Git. The updater fires once per day
(UTC) after each successful plan publish.

| Mode | Behaviour |
| --- | --- |
| `off` | No automatic updates (default) |
| `release` | Updates when a new semver release tag appears on the remote |
| `master` | Updates when the remote branch has new commits |

When an update is available the updater:

1. Pulls / checks out the new ref.
2. Runs `uv sync --no-dev` to synchronise the virtual environment.
3. Sends `SIGTERM` to itself so systemd restarts with the new code.

> **Note:** When running as a systemd service, `Restart=on-failure` ensures the
> process comes back up automatically after the self-restart.

---

## Running as a systemd Service

The repository ships ready-to-use install and uninstall scripts for Linux systems
running systemd (e.g. Raspberry Pi OS, Debian, Ubuntu).

### Prerequisites

- Python 3.13 or newer and [uv](https://docs.astral.sh/uv/) installed:
  ```bash
  # Official uv installer (Linux/macOS)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A `.venv` inside the project directory (the install script creates it automatically if missing):
  ```bash
  uv sync --no-dev
  ```
- `sudo` access on the target machine.
- Network connectivity (the service waits for `network-online.target` before starting).

### Install

```bash
# Minimal – use all defaults
sudo ./install.sh

# With explicit bind address and port
sudo ./install.sh --host 0.0.0.0 --port 8080

# With an external config file
sudo ./install.sh --config /etc/gridpythia/config.yaml
```

| Option | Default | Description |
| --- | --- | --- |
| `-H` / `--host` | `0.0.0.0` | Web UI bind address |
| `-p` / `--port` | `8080` | Web UI TCP port |
| `-c` / `--config` | `<install-dir>/config.yaml` | Path to config.yaml |

The script will:

1. Check for `uv` and install it system-wide (`/usr/local/bin`) if not found.
2. Create a `.venv` and run `uv sync --no-dev` if no `.venv` exists yet.
3. Create a system group `pythia` and a dedicated system user `gridpythia` (no login, no home directory).
4. Add the installing user to the `pythia` group so project files remain editable.
5. Set ownership to `gridpythia:pythia` with `setgid` directories (new files inherit the group automatically).
6. Install `/etc/systemd/system/gridpythia.service` and enable + start the service.

> **After installation**, log out and back in once so the `pythia` group membership
> takes effect in your shell.

### Uninstall

```bash
# Remove the service only (application files are kept)
sudo ./uninstall.sh

# Also remove the 'gridpythia' system user
# (the 'pythia' group is removed only when no other members remain)
sudo ./uninstall.sh --remove-user
```

### Useful commands

```bash
sudo systemctl status  gridpythia
sudo journalctl -u gridpythia -f
sudo systemctl restart gridpythia
sudo systemctl stop    gridpythia
sudo systemctl disable gridpythia
```

---

## REST API overview

Interactive docs are available at `http://<host>:<port>/api/docs` when the server is running.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/config` | UI bootstrap: batteries, inverters, horizon settings |
| `GET` | `/api/predictions/status` | Prediction cache age and TTL |
| `POST` | `/api/predictions/fetch` | Fetch all forecast channels; returns Plotly charts |
| `POST` | `/api/inverters/{device_id}/status` | Report inverter SoC + mode |
| `GET` | `/api/inverters/status` | All known inverter states |
| `GET` | `/api/optimize/status` | Optimization cache age and TTL |
| `GET` | `/api/optimize` | Get last cached optimization result |
| `POST` | `/api/optimize` | Run MILP optimizer; returns schedule + charts |

---

## Development & Testing

### Run tests

```bash
pytest
```

### Code quality

```bash
ruff check .
```

### Dev server with auto-reload

```bash
python main.py --reload
```

---

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

- License text: [LICENSE.md](LICENSE.md)

Commercial use is not permitted under this license.
For commercial licensing, contact the licensor.
