"""Load learning service – business logic for adaptive load forecast."""

from __future__ import annotations

import asyncio
import json
from threading import Event
from urllib.parse import urlparse

import paho.mqtt.client as mqtt
from structlog import get_logger

from GridPythia.config import AppConfig
from GridPythia.prediction.load.adaptive import AdaptiveLoadProvider

logger = get_logger(__name__)


class LoadLearningService:
    """Manages the adaptive load provider, MQTT ingestion, and maintenance.

    This service is the single entry point for:
    * Ingesting power/energy measurements (via REST or MQTT).
    * Toggling vacation mode (runtime-only, not persisted).
    * Notifying the appliance tracker about appliance state changes.
    * Running periodic TSDB maintenance.
    * Querying learning statistics.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._adaptive_cfg = config.prediction.load.adaptive
        self._mqtt_cfg = config.server.mqtt
        self._provider: AdaptiveLoadProvider | None = None
        self._mqtt_client: mqtt.Client | None = None
        self._mqtt_stop = Event()
        self._maintenance_task: asyncio.Task | None = None
        # Runtime vacation mode (not from config)
        self._vacation_mode: bool = False

    @property
    def provider(self) -> AdaptiveLoadProvider | None:
        return self._provider

    def set_provider(self, provider: AdaptiveLoadProvider) -> None:
        """Inject the adaptive provider instance (set by server services)."""
        self._provider = provider

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start MQTT listener and maintenance loop."""
        if self._provider is None:
            self._resolve_provider()

        if self._mqtt_cfg.enabled and self._adaptive_cfg.mqtt_topic:
            self._start_mqtt()

        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(), name="tsdb-maintenance"
        )
        logger.info("load_learning_started")

    async def stop(self) -> None:
        """Stop MQTT and maintenance."""
        self._mqtt_stop.set()
        if self._provider is not None:
            self._provider.flush_accumulators(force_all=True)
        if self._mqtt_client is not None:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
            self._mqtt_client = None
        if self._maintenance_task is not None and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        logger.info("load_learning_stopped")

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest_power(self, watts: float, ts: float | None = None) -> None:
        """Ingest a power measurement (W)."""
        if self._provider is not None:
            self._provider.ingest_power(watts, ts)

    def ingest_energy(self, wh: float, duration_h: float, ts: float | None = None) -> None:
        """Ingest an energy measurement (Wh over duration)."""
        if self._provider is not None:
            self._provider.ingest_energy(wh, duration_h, ts)

    # ------------------------------------------------------------------
    # Vacation mode (runtime-only, not from config)
    # ------------------------------------------------------------------

    @property
    def vacation_mode(self) -> bool:
        if self._provider is not None:
            return self._provider.vacation_mode
        return self._vacation_mode

    @vacation_mode.setter
    def vacation_mode(self, active: bool) -> None:
        self._vacation_mode = active
        if self._provider is not None:
            self._provider.vacation_mode = active

    # ------------------------------------------------------------------
    # Appliance tracker notifications
    # ------------------------------------------------------------------

    def notify_appliance_active(self, appliance: str, ts: float | None = None) -> None:
        """Record that *appliance* started running."""
        if self._provider is not None:
            self._provider.appliance_tracker.notify_active(appliance, ts)

    def notify_appliance_inactive(
        self, appliance: str, avg_power_w: float = 0.0, ts: float | None = None
    ) -> None:
        """Record that *appliance* finished running."""
        if self._provider is not None:
            self._provider.appliance_tracker.notify_inactive(appliance, avg_power_w, ts)

    def notify_appliance_scheduled(self, appliance: str, scheduled_start_ts: float) -> None:
        """Record an announced scheduled start for *appliance*."""
        if self._provider is not None:
            self._provider.appliance_tracker.notify_scheduled(appliance, scheduled_start_ts)

    def update_active_forecast_appliances(self, appliance_ids: set[str]) -> None:
        """Tell the provider which appliances have explicit optimizer forecasts."""
        if self._provider is not None:
            self._provider.set_active_forecast_appliances(appliance_ids)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return learning statistics."""
        if self._provider is not None:
            return self._provider.get_stats()
        return {"status": "provider_not_initialized"}

    # ------------------------------------------------------------------
    # MQTT
    # ------------------------------------------------------------------

    def _start_mqtt(self) -> None:
        """Start MQTT client subscribing to the load measurement topic."""
        parsed = urlparse(self._mqtt_cfg.broker)
        host = parsed.hostname or "localhost"
        port = parsed.port or 1883

        client_id = f"{self._mqtt_cfg.client_id}-load-learning"
        self._mqtt_client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        if self._mqtt_cfg.username:
            self._mqtt_client.username_pw_set(self._mqtt_cfg.username, self._mqtt_cfg.password)

        self._mqtt_client.on_connect = self._on_connect
        self._mqtt_client.on_message = self._on_message

        try:
            self._mqtt_client.connect(host, port, keepalive=60)
            self._mqtt_client.loop_start()
            logger.info(
                "load_learning_mqtt_connected",
                broker=f"{host}:{port}",
                topic=self._adaptive_cfg.mqtt_topic,
            )
        except Exception as exc:
            logger.warning("load_learning_mqtt_connect_failed", error=str(exc))
            self._mqtt_client = None

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: object,
        rc: object,
        properties: object = None,
    ) -> None:
        topic = self._adaptive_cfg.mqtt_topic
        client.subscribe(topic)
        logger.debug("load_learning_mqtt_subscribed", topic=topic)

    def _on_message(self, client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
        """Process incoming MQTT power/energy measurement.

        Accepted payload formats:
        * Plain number: ``"350.5"``  → watts
        * JSON watts:   ``{"watts": 350.5}`` or ``{"value": 350.5}``
        * JSON energy:  ``{"wh": 87.6, "duration_h": 0.25, "ts": 1234567890}``
        """
        try:
            payload = msg.payload.decode("utf-8").strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = float(payload)

            if isinstance(data, (int, float)):
                self.ingest_power(float(data))
                return

            if not isinstance(data, dict):
                return

            ts = data.get("ts") or data.get("timestamp")
            if ts is not None:
                ts = float(ts)

            # Energy path
            if "wh" in data and "duration_h" in data:
                self.ingest_energy(float(data["wh"]), float(data["duration_h"]), ts)
                return

            # Power path
            watts = float(data.get("watts") or data.get("value") or data.get("power") or 0)
            self.ingest_power(watts, ts)

        except (ValueError, TypeError, KeyError) as exc:
            logger.debug(
                "load_learning_mqtt_parse_error", payload=msg.payload[:100], error=str(exc)
            )

    # ------------------------------------------------------------------
    # Maintenance loop
    # ------------------------------------------------------------------

    async def _maintenance_loop(self) -> None:
        """Periodically flush accumulators and run TSDB compaction / retention."""
        flush_interval = self._adaptive_cfg.flush_interval_s
        maintenance_interval = 3600
        ticks = 0
        while True:
            await asyncio.sleep(flush_interval)
            ticks += flush_interval
            if self._provider is not None:
                try:
                    self._provider.flush_accumulators()
                    if ticks >= maintenance_interval:
                        stats = self._provider.run_maintenance()
                        logger.debug("load_learning_maintenance", **stats)
                        ticks = 0
                except Exception as exc:
                    logger.warning("load_learning_maintenance_error", error=str(exc))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_provider(self) -> None:
        """Try to resolve the adaptive provider from server state."""
        try:
            import GridPythia.server.state as state

            if state.providers is not None and hasattr(state.providers, "load"):
                load_prov = state.providers.load
                if isinstance(load_prov, AdaptiveLoadProvider):
                    self._provider = load_prov
                    self._provider.vacation_mode = self._vacation_mode
                    logger.info("load_learning_provider_resolved")
        except Exception:
            logger.warning("load_learning_provider_resolve_failed", exc_info=True)
