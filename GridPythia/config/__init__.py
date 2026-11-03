"""Global application configuration model and YAML parsing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from GridPythia.config.optimization import OptimizationConfig
from GridPythia.config.prediction import PredictionConfig
from GridPythia.config.server import ServerConfig


def config_log_summary(
    config: "AppConfig",
    config_path: str | Path,
    raw_yaml: dict[str, Any],
) -> dict[str, Any]:
    """Return a useful configuration summary that excludes credentials and secrets."""
    prediction = config.prediction
    optimization = config.optimization
    server = config.server
    load_path = ""
    if prediction.load.path:
        profile_path = Path(prediction.load.path)
        if not profile_path.is_absolute():
            profile_path = Path(config_path).parent / profile_path
        load_path = str(profile_path)

    return {
        "prediction": {
            "location": {
                "latitude": prediction.latitude,
                "longitude": prediction.longitude,
            },
            "horizon_hours": prediction.horizon,
            "time_step_minutes": prediction.dt_hours * 60,
            "electricprice": {
                "provider": prediction.electricprice.provider,
                "charges_kwh": prediction.electricprice.charges_kwh,
                "vat_rate": prediction.electricprice.vat_rate,
                "region": (
                    prediction.electricprice.epexpredictor.region
                    if prediction.electricprice.provider == "EpexPredictor"
                    else None
                ),
                "bidding_zone": (
                    prediction.electricprice.energycharts.bidding_zone
                    if prediction.electricprice.provider == "EnergyCharts"
                    else None
                ),
            },
            "feedintariff": {
                "provider": prediction.feedintariff.provider,
                "tariff_eur_kwh": prediction.feedintariff.tariff_kwh,
            },
            "load": {
                "provider": prediction.load.provider,
                "profile_path": str(load_path),
                "country": prediction.load.country,
                "subdivision": prediction.load.subdivision,
                "vacation_percentile": prediction.load.vacation_percentile,
            },
            "pvforecast": {
                "provider": prediction.pvforecast.provider,
                "inverter_id": prediction.pvforecast.plane.inverter_id,
                "peak_kw": prediction.pvforecast.plane.peak_kw,
                "tilt": prediction.pvforecast.plane.tilt,
                "azimuth": prediction.pvforecast.plane.azimuth,
                "loss_pct": prediction.pvforecast.plane.loss_pct,
                "weather_model": prediction.pvforecast.openmeteo.weather_model or None,
            },
            "weather": {
                "enabled": "weather" in raw_yaml.get("prediction", {}),
                "provider": prediction.weather.provider,
            },
        },
        "optimization": {
            "solver": optimization.solver.provider,
            "objective": optimization.solver.objective,
            "batteries": [
                {
                    "device_id": battery.device_id,
                    "capacity_wh": battery.capacity_wh,
                    "initial_soc_pct": battery.initial_soc_percentage,
                    "min_soc_pct": battery.min_soc_percentage,
                    "max_soc_pct": battery.max_soc_percentage,
                    "max_charge_power_w": battery.max_charge_power_w,
                    "max_discharge_power_w": battery.max_discharge_power_w,
                }
                for battery in optimization.batteries
            ],
            "inverters": [
                {
                    "device_id": inverter.device_id,
                    "battery_id": inverter.battery_id,
                    "has_pv": inverter.has_pv,
                    "max_ac_output_power_w": inverter.max_ac_output_power_w,
                    "max_ac_charge_power_w": inverter.max_ac_charge_power_w,
                    "zero_feed_in": inverter.zero_feed_in,
                }
                for inverter in optimization.inverters
            ],
        },
        "server": {
            "timezone": server.timezone,
            "bind_host": server.bind_host,
            "bind_port": server.bind_port,
            "mqtt_enabled": server.mqtt.enabled,
            "scheduler_interval_minutes": (
                server.scheduler.optimization_interval_minutes
            ),
            "update_mode": server.update.mode.value,
        },
    }


class AppConfig(BaseModel):
    """Root config model mirroring the config.yaml structure."""

    model_config = {"frozen": True}

    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        """Build an AppConfig from a mapping payload."""
        return cls.model_validate(payload)

    @classmethod
    def from_yaml(cls, text: str) -> "AppConfig":
        """Build an AppConfig from YAML text."""
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Root YAML node must be a mapping")
        return cls.from_dict(loaded)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "AppConfig":
        """Build an AppConfig from a YAML file path."""
        cfg_path = Path(path).expanduser()
        with cfg_path.open("r", encoding="utf-8") as fh:
            return cls.from_yaml(fh.read())
