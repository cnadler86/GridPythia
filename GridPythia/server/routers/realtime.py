"""WebSocket endpoint for live dashboard updates."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import GridPythia.server.state as state
from GridPythia.server import services
from GridPythia.server.models import InverterStatusResponse
from GridPythia.simulation.devices import InverterMode

router = APIRouter(tags=["realtime"])

_MODE_NAMES: dict[int, str] = {m.value: m.name for m in InverterMode}


@router.websocket("/ws")
async def dashboard_ws(websocket: WebSocket) -> None:
    """Keep a websocket connection open and stream server-side events."""
    await state.ws_hub.connect(websocket)
    try:
        await websocket.send_json({"type": "hello", "payload": {"connected": True}})

        # ── Hydrate with cached optimization solution ─────────────────────
        cached_solution = services.get_cached_solution()
        if cached_solution is not None:
            await websocket.send_json(
                {
                    "type": "optimization_updated",
                    "payload": cached_solution,
                }
            )

        # ── Hydrate with scheduler timing ─────────────────────────────────
        if state.scheduler_next_info is not None:
            await websocket.send_json(
                {"type": "scheduler_status", "payload": state.scheduler_next_info}
            )

        # ── Hydrate with all known inverter states ────────────────────────
        try:
            cfg, _ = services.load_config()
            max_age = cfg.server.inverter_status_max_age_s
            inv_states = [
                InverterStatusResponse(
                    device_id=s.device_id,
                    soc=s.soc,
                    mode=s.mode.value,
                    mode_name=s.mode.name,
                    reported_at=s.reported_at.isoformat(),
                    age_s=round(s.age_s(), 1),
                    is_fresh=s.is_fresh(max_age),
                ).model_dump()
                for s in state.coordinator.snapshot().values()
            ]
            await websocket.send_json({"type": "inverter_status_all", "payload": inv_states})
            # ── Hydrate with MQTT status ──────────────────────────────────
            await websocket.send_json(
                {
                    "type": "mqtt_status",
                    "payload": {
                        "enabled": cfg.server.mqtt.enabled,
                        "connected": state.mqtt_connected,
                    },
                }
            )
        except Exception:
            pass  # non-critical – client can fall back to polling

        while True:
            # We currently ignore incoming frames and only use this loop
            # to detect disconnects from the client side.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await state.ws_hub.disconnect(websocket)
