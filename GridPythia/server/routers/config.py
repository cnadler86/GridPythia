"""GET /api/config – return all UI bootstrap data."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import GridPythia.server.state as state
from GridPythia.server import services
from GridPythia.server.models import AppConfigResponse, BatteryInfo, InverterInfo

router = APIRouter(tags=["config"])


class MqttStatusResponse(BaseModel):
    enabled: bool
    connected: bool


@router.get("/mqtt/status", response_model=MqttStatusResponse)
async def get_mqtt_status() -> MqttStatusResponse:
    """Return current MQTT broker connection state."""
    try:
        cfg, _ = services.load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc
    return MqttStatusResponse(enabled=cfg.server.mqtt.enabled, connected=state.mqtt_connected)


@router.get("/config", response_model=AppConfigResponse)
async def get_app_config() -> AppConfigResponse:
    """Return all UI configuration needed to render the frontend.

    Called once on page load so the browser can build battery SoC inputs,
    inverter info badges and tab navigation dynamically.
    """
    try:
        cfg, raw_yaml = services.load_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config error: {exc}") from exc

    batteries = [
        BatteryInfo(
            device_id=b.device_id,
            min_soc_percentage=b.min_soc_percentage,
            max_soc_percentage=b.max_soc_percentage,
            initial_soc_percentage=b.initial_soc_percentage,
            capacity_wh=float(b.capacity_wh),
        )
        for b in cfg.optimization.batteries
    ]
    inverters = [
        InverterInfo(
            device_id=inv.device_id,
            has_pv=inv.has_pv,
            battery_id=inv.battery_id or None,
            max_ac_output_power_w=float(inv.max_ac_output_power_w),
            max_ac_charge_power_w=float(inv.max_ac_charge_power_w),
            zero_feed_in=inv.zero_feed_in,
        )
        for inv in cfg.optimization.inverters
    ]
    return AppConfigResponse(
        batteries=batteries,
        inverters=inverters,
        has_weather="weather" in raw_yaml.get("prediction", {}),
        horizon_h=float(cfg.prediction.horizon),
        dt_min=int(cfg.prediction.dt_hours * 60),
        objective=cfg.optimization.solver.objective,
        optimization_interval_min=cfg.server.scheduler.optimization_interval_minutes,
        inverter_status_max_age_s=cfg.server.inverter_status_max_age_s,
        mqtt_enabled=cfg.server.mqtt.enabled,
    )
