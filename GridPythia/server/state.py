"""Module-level singleton state shared across all HTTP requests.

Objects here are intentionally global because they hold expensive compiled
models (CVXPY) and internal provider caches (TimeBucketCache) that must
survive between individual requests.

Mutated only by :mod:`GridPythia.server.services` functions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from GridPythia.optimization.solver import LinearOptimizer
    from GridPythia.prediction.prediction import PredictionData, PredictionSetup

from GridPythia.coordination.inverter_coordinator import InverterCoordinator

# ── Config path ───────────────────────────────────────────────────────────
# Set once by create_app() before the first request arrives.
config_path: Path = Path("config.yaml")

# ── Provider singleton ────────────────────────────────────────────────────
providers: "PredictionSetup | None" = None
providers_config_mtime: float = 0.0

# ── Optimizer singleton ───────────────────────────────────────────────────
# The CVXPY model is compiled once per LinearOptimizer instance and reused
# across calls via _update_runtime_parameters().
optimizer: "LinearOptimizer | None" = None
optimizer_config_mtime: float = 0.0
_optimizer_lock: "asyncio.Lock | None" = None


def get_optimizer_lock() -> asyncio.Lock:
    """Return (creating lazily) the asyncio lock that serialises solver calls."""
    global _optimizer_lock  # noqa: PLW0603
    if _optimizer_lock is None:
        _optimizer_lock = asyncio.Lock()
    return _optimizer_lock


# ── Prediction data cache ─────────────────────────────────────────────────
# Avoids double-fetching between a /api/predictions/fetch and /api/optimize
# call issued by the same browser session within the TTL window.
pdata_cache: "PredictionData | None" = None
pdata_cache_ts: datetime | None = None
pdata_forecast_from: datetime | None = None  # last real EnergyCharts timestamp
PDATA_CACHE_TTL_S: float = 300.0  # 5 minutes

# ── Partial-fetch retry state ─────────────────────────────────────────────
# Maps provider name → next retry datetime when that provider failed on the last fetch.
# The retry background task reads + clears this dict.
failed_provider_retry_at: dict[str, datetime] = {}
_retry_task: "asyncio.Task | None" = None

# ── MQTT connection state ─────────────────────────────────────────────────
# Flipped to True by the MQTT gateway when the broker connection is established.
mqtt_connected: bool = False

# ── Solution cache ────────────────────────────────────────────────────────
# Avoids recomputing optimization when navigating back after closure.
# Invalidated on config changes or when stale.
solution_cache: dict | None = None
solution_cache_ts: datetime | None = None
SOLUTION_CACHE_TTL_S: float = 3600.0  # 1 hour

# ── Inverter coordinator ──────────────────────────────────────────────────
# Tracks real-time inverter states (SoC, mode).  max_age_s is updated from
# the ServerConfig when the config is first loaded.
coordinator: InverterCoordinator = InverterCoordinator()
