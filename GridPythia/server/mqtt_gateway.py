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
from datetime import datetime, timezone
from math import floor
from threading import Event
from urllib.parse import urlparse

import paho.mqtt.client as mqtt
from structlog import get_logger

import GridPythia.server.state as state
from GridPythia.config.server import MqttConfig

logger = get_logger(__name__)

# Matches: {prefix}/inverters/{device_id}/status
_TOPIC_RE = re.compile(r"^(.+)/inverters/([^/]+)/status$")

# Matches: {prefix}/appliance_load/forecast/{appliance_id}
_APPLIANCE_TOPIC_RE = re.compile(r"^(.+)/appliance_load/forecast/([^/]+)$")


def _parse_step_timestamp(step: dict) -> datetime | None:
    """Parse a plan-step timestamp, returning ``None`` for malformed entries."""
    raw_ts = step.get("timestamp")
    if not isinstance(raw_ts, str):
        return None
    try:
        return datetime.fromisoformat(raw_ts)
    except ValueError:
        return None


def _current_slot_start(published_at: datetime, dt_hours: float) -> datetime:
    """Return the start timestamp of the slot containing *published_at*."""
    step_seconds = max(1.0, float(dt_hours) * 3600.0)
    slot_epoch = floor(published_at.timestamp() / step_seconds) * step_seconds
    return datetime.fromtimestamp(slot_epoch, tz=published_at.tzinfo or timezone.utc)


def _stitch_current_slot_from_previous_plan(
    steps: list[dict],
    previous_steps: list[dict],
    *,
    published_at: datetime,
    dt_hours: float,
) -> list[dict]:
    """Prepend the active slot from the previous published plan when needed.

    If a newly solved plan starts at the next slot boundary because the solve ran
    shortly before dispatch, downstream consumers still need the currently active
    slot from the previously retained plan until the boundary is actually reached.
    """
    stitched_steps = [dict(step) for step in steps]
    if not stitched_steps or not previous_steps:
        return stitched_steps

    first_step_ts = _parse_step_timestamp(stitched_steps[0])
    if first_step_ts is None or published_at >= first_step_ts:
        return stitched_steps

    current_slot = _current_slot_start(published_at, dt_hours)
    if first_step_ts <= current_slot:
        return stitched_steps

    for prev_step in reversed(previous_steps):
        prev_step_ts = _parse_step_timestamp(prev_step)
        if prev_step_ts is None:
            continue
        if prev_step_ts == current_slot:
            if _parse_step_timestamp(stitched_steps[0]) == current_slot:
                return stitched_steps
            return [dict(prev_step), *stitched_steps]

    return stitched_steps


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
        self._last_published_steps_by_device: dict[str, list[dict]] = {}
        host, port = _parse_broker(cfg.broker)
        self._host = host
        self._port = port
        self._subscribe_topic = f"{cfg.topic_prefix}/inverters/+/status"
        self._appliance_topic = f"{cfg.topic_prefix}/appliance_load/forecast/+"

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
            client.subscribe(self._appliance_topic, qos=0)
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

        # ─ Appliance load forecast ──────────────────────────────────────────
        ma = _APPLIANCE_TOPIC_RE.match(topic_str)
        if ma and ma.group(1) == self._cfg.topic_prefix:
            appliance_id = ma.group(2)
            try:
                payload = json.loads(msg.payload) if msg.payload else []
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("mqtt_appliance_bad_payload", topic=topic_str, error=str(exc))
                return
            if not isinstance(payload, list):
                logger.warning("mqtt_appliance_not_list", topic=topic_str)
                return
            if payload:
                state.appliance_forecasts[appliance_id] = payload
                logger.info(
                    "mqtt_appliance_forecast_updated",
                    appliance_id=appliance_id,
                    slots=len(payload),
                )
            else:
                # Empty payload = clear the retained forecast
                state.appliance_forecasts.pop(appliance_id, None)
                logger.info("mqtt_appliance_forecast_cleared", appliance_id=appliance_id)
            return

        # ─ Inverter status ───────────────────────────────────────────────
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

    # ── Plan publishing ────────────────────────────────────────────────

    def publish_plans(
        self,
        inverter_plans: list[dict],
        *,
        dt_hours: float = 0.25,
    ) -> None:
        """Publish optimiser plan(s) via MQTT.

        Each element of *inverter_plans* must be a dict with at least
        ``device_id`` and ``steps`` keys (as returned by the API response).

        Topic:   ``{prefix}/inverters/{device_id}/plan``
        Payload: ``{device_id, published_at, dt_hours, steps: [...]}``,
                 where each step matches GridPythia's ``InverterPlanStep`` schema.

        The message is published with ``retain=True`` so that a newly
        connecting controller immediately receives the last known plan.
        """
        published_at = datetime.now(tz=timezone.utc).isoformat()
        published_at_dt = datetime.fromisoformat(published_at)
        for plan in inverter_plans:
            device_id = plan.get("device_id", "")
            if not device_id:
                continue
            raw_steps = plan.get("steps", [])
            effective_steps = _stitch_current_slot_from_previous_plan(
                raw_steps,
                self._last_published_steps_by_device.get(device_id, []),
                published_at=published_at_dt,
                dt_hours=dt_hours,
            )
            topic = f"{self._cfg.topic_prefix}/inverters/{device_id}/plan"
            payload = {
                "device_id": device_id,
                "published_at": published_at,
                "dt_hours": dt_hours,
                "steps": effective_steps,
            }
            try:
                self._client.publish(
                    topic,
                    json.dumps(payload),
                    qos=1,
                    retain=True,
                )
                logger.info(
                    "mqtt_plan_published",
                    device_id=device_id,
                    steps=len(payload["steps"]),
                    prepended_current_slot=(1 if len(payload["steps"]) > len(raw_steps) else 0),
                )
                self._last_published_steps_by_device[device_id] = [
                    dict(step) for step in payload["steps"]
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("mqtt_plan_publish_failed", device_id=device_id, error=str(exc))


# ── Async adapter used from the FastAPI lifespan ──────────────────────────

import asyncio


async def run_gateway(cfg: MqttConfig) -> None:
    """Start the MQTT gateway and keep it alive until cancelled.

    Designed to run as a background asyncio task; the actual MQTT I/O
    runs in paho's own thread so there is no event-loop conflict.

    The gateway instance is stored in ``state.mqtt_gateway`` so that other
    parts of the application (e.g. the optimization router) can call
    ``state.mqtt_gateway.publish_plans(...)`` to distribute schedules.
    """
    gw = MqttGateway(cfg)
    state.mqtt_gateway = gw
    gw.start()
    try:
        # Yield control back to the event loop indefinitely.
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass
    finally:
        gw.stop()
        state.mqtt_gateway = None
