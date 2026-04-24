"""MQTT gateway for receiving inverter status reports.

Subscribes to ``{topic_prefix}/inverters/{device_id}/status`` and forwards
each message to the :class:`~GridPythia.coordination.InverterCoordinator`.

**Expected payload** (JSON):

.. code-block:: json

    {"soc": 63.5, "mode": 0}

``soc`` is battery state-of-charge in %, ``mode`` is the InverterMode integer
(0 = IDLE, 1 = DISCHARGE, 2 = DISCHARGE_ZFI, 3 = AC_CHARGE,
4 = AC_CHARGE_ZFI).  ``mode`` is optional and defaults to 0 (IDLE).

**Topic convention**::

    gridpythia/inverters/SF800Pro/status

The gateway runs paho-mqtt's own network thread via ``loop_start()``/``loop_stop()``.
This avoids any asyncio event-loop incompatibility (paho-mqtt's ``add_reader``/
``add_writer`` calls are not supported by the ProactorEventLoop on Windows).
The on_message callback updates the coordinator directly; dict writes in CPython
are GIL-protected and are safe to call from a non-async thread.
"""

from __future__ import annotations

import json
import re
from threading import Event
from urllib.parse import urlparse

import paho.mqtt.client as mqtt
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.config.server import MqttConfig

logger = get_logger(__name__)

# Matches: {prefix}/inverters/{device_id}/status
_TOPIC_RE = re.compile(r"^(.+)/inverters/([^/]+)/status$")


def _parse_broker(broker_url: str) -> tuple[str, int]:
    """Return (hostname, port) from a broker URL like ``mqtt://localhost:1883``."""
    parsed = urlparse(broker_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 1883
    return host, port


class MqttGateway:
    """Thin wrapper around a paho-mqtt client running in its own thread."""

    def __init__(self, cfg: MqttConfig) -> None:
        self._cfg = cfg
        self._stop = Event()
        host, port = _parse_broker(cfg.broker)
        self._host = host
        self._port = port
        self._subscribe_topic = f"{cfg.topic_prefix}/inverters/+/status"

        self._client = mqtt.Client(
            client_id=cfg.client_id,
            clean_session=True,
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        if cfg.username:
            self._client.username_pw_set(cfg.username, cfg.password or None)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Automatic reconnect: wait 1 s before first try, max 30 s
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

    # ── paho callbacks (run in paho's network thread) ─────────────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:  # type: ignore[override]
        if reason_code == 0:
            state.mqtt_connected = True
            logger.info("mqtt_connected", broker=self._cfg.broker, topic=self._subscribe_topic)
            client.subscribe(self._subscribe_topic, qos=0)
        else:
            state.mqtt_connected = False
            logger.warning("mqtt_connect_refused", reason=str(reason_code))

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties) -> None:  # type: ignore[override]
        state.mqtt_connected = False
        if reason_code != 0:
            logger.warning("mqtt_disconnected_unexpected", reason=str(reason_code))
        else:
            logger.info("mqtt_disconnected_clean")

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        topic_str = msg.topic
        m = _TOPIC_RE.match(topic_str)
        if not m or m.group(1) != self._cfg.topic_prefix:
            return

        device_id = m.group(2)

        try:
            payload = json.loads(msg.payload)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("mqtt_bad_payload", topic=topic_str, error=str(exc))
            return

        if "soc" not in payload:
            logger.warning("mqtt_missing_soc", topic=topic_str, payload=payload)
            return

        try:
            soc = float(payload["soc"])
            mode = int(payload.get("mode", 0))
        except (TypeError, ValueError) as exc:
            logger.warning("mqtt_invalid_values", topic=topic_str, error=str(exc))
            return

        try:
            state.coordinator.update_status(device_id, soc=soc, mode=mode)
            logger.info("mqtt_inverter_status", device_id=device_id, soc=soc, mode=mode)
        except ValueError as exc:
            logger.warning("mqtt_coordinator_error", device_id=device_id, error=str(exc))

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect to the broker and start paho's background network thread."""
        try:
            self._client.connect_async(self._host, self._port, keepalive=60)
            self._client.loop_start()
            logger.info("mqtt_gateway_starting", host=self._host, port=self._port)
        except Exception as exc:  # noqa: BLE001
            logger.error("mqtt_gateway_start_failed", error=str(exc))

    def stop(self) -> None:
        """Disconnect and stop the background network thread."""
        self._stop.set()
        self._client.disconnect()
        self._client.loop_stop()
        state.mqtt_connected = False
        logger.info("mqtt_gateway_stopped")


# ── Async adapter used from the FastAPI lifespan ──────────────────────────

import asyncio


async def run_gateway(cfg: MqttConfig) -> None:
    """Start the MQTT gateway and keep it alive until cancelled.

    Designed to run as a background asyncio task; the actual MQTT I/O
    runs in paho's own thread so there is no event-loop conflict.
    """
    gw = MqttGateway(cfg)
    gw.start()
    try:
        # Yield control back to the event loop indefinitely.
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass
    finally:
        gw.stop()
