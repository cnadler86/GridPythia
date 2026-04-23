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
