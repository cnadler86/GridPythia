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

from fastapi import WebSocket

if TYPE_CHECKING:
    from GridPythia.optimization.solver import LinearOptimizer
    from GridPythia.prediction.prediction import PredictionSetup

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


# ── Partial-fetch retry state ─────────────────────────────────────────────
# Maps provider name → next retry datetime when that provider failed on the last fetch.
# The retry background task reads + clears this dict.
failed_provider_retry_at: dict[str, datetime] = {}
_retry_task: "asyncio.Task | None" = None

# ── MQTT connection state ─────────────────────────────────────────────────
# Flipped to True by the MQTT gateway when the broker connection is established.
mqtt_connected: bool = False

# The MqttGateway instance when running; None otherwise.
# Used by the optimization router to publish plans after each solve.
if TYPE_CHECKING:
    from GridPythia.server.mqtt_gateway import MqttGateway

mqtt_gateway: "MqttGateway | None" = None

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

# ── Appliance load forecasts ──────────────────────────────────────────────
# Maps appliance_id → list of raw forecast slots [{"time": ISO-str, "load_wh": float}].
# Updated by the MQTT gateway (retained topic) or via the HTTP appliance endpoint.
appliance_forecasts: dict[str, list[dict]] = {}

# ── Scheduler next-run info ───────────────────────────────────────────────
# Set by run_scheduler() each cycle so WS clients can be hydrated on connect.
# Keys: dispatch_slot (ISO str), run_at (ISO str), lead_s (float).
scheduler_next_info: "dict | None" = None


class DashboardWebSocketHub:
    """Track dashboard websocket clients and broadcast JSON events."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, event: dict) -> None:
        async with self._lock:
            targets = list(self._clients)

        stale: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(event)
            except Exception:
                stale.append(ws)

        if not stale:
            return

        async with self._lock:
            for ws in stale:
                self._clients.discard(ws)


ws_hub = DashboardWebSocketHub()
