"""WebSocket endpoint for live dashboard updates."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import GridPythia.server.state as state

router = APIRouter(tags=["realtime"])


@router.websocket("/ws")
async def dashboard_ws(websocket: WebSocket) -> None:
    """Keep a websocket connection open and stream server-side events."""
    await state.ws_hub.connect(websocket)
    try:
        await websocket.send_json({"type": "hello", "payload": {"connected": True}})
        while True:
            # We currently ignore incoming frames and only use this loop
            # to detect disconnects from the client side.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await state.ws_hub.disconnect(websocket)
