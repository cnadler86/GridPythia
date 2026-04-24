"""Pydantic configuration models for server / runtime settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MqttConfig(BaseModel):
    """MQTT broker connection settings."""

    model_config = {"frozen": True}

    enabled: bool = False
    broker: str = "mqtt://localhost:1883"
    client_id: str = "gridpythia"
    username: str = ""
    password: str = ""
    topic_prefix: str = "gridpythia"


class SchedulerConfig(BaseModel):
    """Periodic task scheduler settings.

    ``optimization_interval_minutes`` must be a divisor of 60 so that
    optimization fires at full grid boundaries (e.g. 0:00, 0:15, 0:30, 0:45
    for 15 min). Supported values: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60.
    """

    model_config = {"frozen": True}

    optimization_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Run optimization every N minutes (must be divisor of 60)",
    )
    prediction_refresh_minutes: int = Field(
        default=30,
        ge=1,
        description="Refresh prediction data every N minutes",
    )


class ServerConfig(BaseModel):
    """Runtime server settings (not prediction or optimization parameters)."""

    model_config = {"frozen": True}

    bind_host: str = Field(
        default="127.0.0.1",
        description=(
            "Host/IP used by the web server bind. Use 0.0.0.0 to listen on all interfaces."
        ),
    )
    bind_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="TCP port used by the web server bind.",
    )

    inverter_status_max_age_s: float = Field(
        default=300.0,
        gt=0.0,
        description=(
            "Maximum age of an inverter status report (seconds) before it is "
            "considered stale. Optimization is blocked when any optimizable "
            "inverter has a stale or missing status."
        ),
    )
    mqtt: MqttConfig = Field(default_factory=MqttConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
