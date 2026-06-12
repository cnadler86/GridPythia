"""Construct a :class:`PredictionSetup` from an :class:`AppConfig`.

Single source of truth for mapping the parsed YAML config onto provider
instances via the provider registry.  Used by the server services layer and
by :class:`~GridPythia.services.application.GridPythiaService` so both stay
in sync when providers or config options are added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from GridPythia.config import AppConfig
from GridPythia.prediction.prediction import PredictionSetup
from GridPythia.prediction.registry import provider_registry


def build_prediction_setup(
    cfg: AppConfig,
    raw_yaml: dict[str, Any],
    config_dir: Path,
    *,
    fresh_instances: bool = False,
) -> PredictionSetup:
    """Instantiate all prediction providers from *AppConfig*.

    Args:
        cfg: Parsed app config.
        raw_yaml: Raw YAML config dictionary (used to detect optional sections).
        config_dir: Directory of the config file; relative paths (e.g. the
            load profile CSV) are resolved against it.
        fresh_instances: Force newly constructed provider instances and bypass
            the registry singleton cache.
    """
    pred_cfg = cfg.prediction

    # Electric price
    ep_cfg = {
        "bidding_zone": pred_cfg.electricprice.energycharts.bidding_zone,
        "charges_kwh": pred_cfg.electricprice.charges_kwh,
        "vat_rate": pred_cfg.electricprice.vat_rate,
        "region": pred_cfg.electricprice.epexpredictor.region,
        "base_url": pred_cfg.electricprice.epexpredictor.base_url,
    }
    electricprice = provider_registry.create_electricprice(
        pred_cfg.electricprice.provider,
        ep_cfg,
        fresh=fresh_instances,
    )

    feedintariff = provider_registry.create_feedintariff(
        pred_cfg.feedintariff.provider,
        {"tariff_kwh": pred_cfg.feedintariff.tariff_kwh},
        fresh=fresh_instances,
    )

    raw_load_path = Path(pred_cfg.load.path)
    load_path = raw_load_path if raw_load_path.is_absolute() else (config_dir / raw_load_path)
    load_provider = provider_registry.create_load(
        pred_cfg.load.provider,
        {
            "path": str(load_path),
            "country": pred_cfg.load.country or None,
            "subdivision": pred_cfg.load.subdivision or None,
            "vacation_percentile": pred_cfg.load.vacation_percentile,
        },
        fresh=fresh_instances,
    )

    plane_cfg = pred_cfg.pvforecast.plane
    om_cfg = pred_cfg.pvforecast.openmeteo
    pv_provider = provider_registry.create_pvforecast(
        pred_cfg.pvforecast.provider,
        {
            "latitude": pred_cfg.latitude,
            "longitude": pred_cfg.longitude,
            "plane": {
                "peak_kw": plane_cfg.peak_kw,
                "tilt": plane_cfg.tilt,
                "azimuth": plane_cfg.azimuth,
                "userhorizon": list(plane_cfg.userhorizon),
                "loss_pct": plane_cfg.loss_pct,
                "inverter_id": plane_cfg.inverter_id,
            },
            "openmeteo": {
                "api_key": om_cfg.api_key or None,
                "weather_model": om_cfg.weather_model or None,
                "damping_morning": om_cfg.damping_morning,
                "damping_evening": om_cfg.damping_evening,
                "partial_shading": om_cfg.partial_shading,
            },
        },
        fresh=fresh_instances,
    )

    weather_provider = None
    if "weather" in raw_yaml.get("prediction", {}):
        weather_provider = provider_registry.create_weather(
            pred_cfg.weather.provider,
            {"latitude": pred_cfg.latitude, "longitude": pred_cfg.longitude},
            fresh=fresh_instances,
        )

    return PredictionSetup(
        electricprice=electricprice,
        feedintariff=feedintariff,
        load=load_provider,
        pv={plane_cfg.inverter_id: pv_provider},
        weather=weather_provider,
    )
