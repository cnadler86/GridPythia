"""Pydantic configuration models for prediction-related settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EnergyChartsConfigModel(BaseModel):
    """EnergyCharts-specific configuration."""

    bidding_zone: str = "DE-LU"


class EpexPredictorConfigModel(BaseModel):
    """EpexPredictor-specific configuration."""

    region: str = "DE"
    base_url: str = "https://epexpredictor.batzill.com"


class ElectricPriceConfig(BaseModel):
    """Configuration for electricity price providers."""

    provider: Literal["EnergyCharts", "Fixed", "EpexPredictor"] = "EnergyCharts"
    charges_kwh: float = Field(default=0.1528, ge=0.0)
    vat_rate: float = Field(default=0.19, ge=0.0)
    cache_ttl_hours: float | None = Field(
        default=1.0,
        ge=0.0,
        description="Provider cache TTL in hours. None = always fetch fresh.",
    )
    energycharts: EnergyChartsConfigModel = Field(default_factory=EnergyChartsConfigModel)
    epexpredictor: EpexPredictorConfigModel = Field(default_factory=EpexPredictorConfigModel)


class FeedInTariffConfig(BaseModel):
    """Configuration for feed-in tariff providers."""

    provider: Literal["Fixed"] = "Fixed"
    tariff_kwh: float = Field(default=0.0, ge=0.0)
    cache_ttl_hours: float | None = Field(
        default=24.0,
        ge=0.0,
        description="Provider cache TTL in hours. None = always fetch fresh.",
    )


class LoadConfigModel(BaseModel):
    """Configuration for household load profile providers."""

    provider: Literal["ProfileCSV"] = "ProfileCSV"
    path: str = ""
    country: str = "DE"
    subdivision: str = "BW"
    cache_ttl_hours: float | None = Field(
        default=24.0,
        ge=0.0,
        description="Provider cache TTL in hours. None = always fetch fresh.",
    )


class PVPlaneConfigModel(BaseModel):
    """PV plane geometry and static loss settings."""

    inverter_id: str = "inverter1"
    peak_kw: float = Field(default=0.41, gt=0.0)
    tilt: float = 75.0
    azimuth: float = 218.0
    loss_pct: float = 4.0
    userhorizon: list[float] = Field(default_factory=list)


class PVOpenMeteoConfigModel(BaseModel):
    """OpenMeteo provider tuning for PV forecast."""

    damping_morning: float = 2.0
    damping_evening: float = 0.2
    partial_shading: bool = False
    api_key: str = ""
    weather_model: str = ""


class PVForecastConfigModel(BaseModel):
    """Configuration for PV forecast providers."""

    provider: Literal["OpenMeteo", "Akkudoktor"] = "OpenMeteo"
    plane: PVPlaneConfigModel = Field(default_factory=PVPlaneConfigModel)
    openmeteo: PVOpenMeteoConfigModel = Field(default_factory=PVOpenMeteoConfigModel)
    cache_ttl_hours: float | None = Field(
        default=1.0,
        ge=0.0,
        description="Provider cache TTL in hours. None = always fetch fresh.",
    )


class WeatherConfigModel(BaseModel):
    """Configuration for weather providers."""

    provider: Literal["OpenMeteo", "BrightSky"] = "OpenMeteo"
    cache_ttl_hours: float | None = Field(
        default=1.0,
        ge=0.0,
        description="Provider cache TTL in hours. None = always fetch fresh.",
    )


class PredictionConfig(BaseModel):
    """Top-level prediction section from config.yaml."""

    latitude: float = 47.99545
    longitude: float = 7.83355
    horizon: float = Field(default=48.0, gt=0.0, description="Prediction horizon in hours")
    dt_hours: float = Field(default=0.25, gt=0.0, description="Time step duration in hours")
    electricprice: ElectricPriceConfig = Field(default_factory=ElectricPriceConfig)
    feedintariff: FeedInTariffConfig = Field(default_factory=FeedInTariffConfig)
    load: LoadConfigModel = Field(default_factory=LoadConfigModel)
    pvforecast: PVForecastConfigModel = Field(default_factory=PVForecastConfigModel)
    weather: WeatherConfigModel = Field(default_factory=WeatherConfigModel)
