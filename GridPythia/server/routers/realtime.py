"""WebSocket endpoint for live dashboard updates."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import GridPythia.server.state as state
from GridPythia.server import services

router = APIRouter(tags=["realtime"])


@router.websocket("/ws")
async def dashboard_ws(websocket: WebSocket) -> None:
    """Keep a websocket connection open and stream server-side events."""
    await state.ws_hub.connect(websocket)
    try:
        await websocket.send_json({"type": "hello", "payload": {"connected": True}})

        # Hydrate newly connected clients from server-side caches.
        cached_pdata = services.get_cached_pdata_any_age()
        if cached_pdata is not None:
            pdata, forecast_from = cached_pdata
            age_s = services.get_cached_pdata_age_s()
            await websocket.send_json(
                {
                    "type": "predictions_updated",
                    "payload": {
                        "charts": services.make_prediction_figures(pdata, forecast_from),
                        "from_cache": True,
                        "stale_cache": (age_s is not None and age_s >= state.PDATA_CACHE_TTL_S),
                        "errors": {},
                    },
                }
            )

        cached_solution = services.get_cached_solution()
        if cached_solution is not None:
            await websocket.send_json(
                {
                    "type": "optimization_updated",
                    "payload": cached_solution,
                }
            )

        while True:
            # We currently ignore incoming frames and only use this loop
            # to detect disconnects from the client side.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await state.ws_hub.disconnect(websocket)
