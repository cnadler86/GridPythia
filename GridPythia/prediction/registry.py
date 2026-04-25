"""Provider registry for decoupled provider construction.

This module implements a registry pattern that decouples provider construction
from the consumer code (e.g., server/services.py). Providers register themselves
with a factory function, and consumers request providers by name.

Example:
-------
>>> from GridPythia.prediction.registry import provider_registry
>>> electricprice = provider_registry.create_electricprice("EnergyCharts", config)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from structlog import get_logger

from GridPythia.prediction.electricprice.provider import ElecPriceProvider
from GridPythia.prediction.feedintariff.provider import FeedInTariffProvider
from GridPythia.prediction.load.provider import LoadProvider
from GridPythia.prediction.pvforecast.provider import PVForecastProvider
from GridPythia.prediction.weather.provider import WeatherProvider

logger = get_logger(__name__)

T = TypeVar("T", bound="ProviderProtocol")


class ProviderProtocol(Protocol):
    """Minimal protocol for prediction providers."""

    @property
    def provider_id(self) -> str: ...


@dataclass
class ProviderRegistry(Generic[T]):
    """Registry for provider factory functions.

    Each provider type (electricprice, load, etc.) has its own registry instance.
    Providers are registered with a name and a factory function that receives
    a config dict and returns the provider instance.
    """

    _factories: dict[str, Callable[[Mapping[str, Any]], T]] = field(default_factory=dict)
    _category: str = "provider"

    def register(
        self,
        name: str,
        factory: Callable[[Mapping[str, Any]], T],
    ) -> None:
        """Register a provider factory under *name*.

        Args:
            name: Provider name (case-sensitive, e.g., "EnergyCharts").
            factory: Callable that takes a config dict and returns a provider.
        """
        if name in self._factories:
            logger.warning(
                f"{self._category}_registry_overwrite",
                name=name,
            )
        self._factories[name] = factory
        logger.debug(f"{self._category}_registered", name=name)

    def create(self, name: str, config: Mapping[str, Any]) -> T:
        """Create a provider instance by name.

        Args:
            name: Registered provider name.
            config: Configuration dict passed to the factory.

        Returns:
            Configured provider instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._factories:
            available = list(self._factories.keys())
            raise KeyError(f"Unknown {self._category} provider '{name}'. Available: {available}")
        return self._factories[name](config)

    def available(self) -> list[str]:
        """Return list of registered provider names."""
        return list(self._factories.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a provider name is registered."""
        return name in self._factories


# ── Registry instances for each provider type ─────────────────────────────

electricprice_registry: ProviderRegistry[ElecPriceProvider] = ProviderRegistry(
    _category="electricprice"
)
feedintariff_registry: ProviderRegistry[FeedInTariffProvider] = ProviderRegistry(
    _category="feedintariff"
)
load_registry: ProviderRegistry[LoadProvider] = ProviderRegistry(_category="load")
pvforecast_registry: ProviderRegistry[PVForecastProvider] = ProviderRegistry(_category="pvforecast")
weather_registry: ProviderRegistry[WeatherProvider] = ProviderRegistry(_category="weather")


# ── Registration of built-in providers ────────────────────────────────────


def _register_builtin_providers() -> None:
    """Register all built-in providers.

    Called once at module import time. Each provider is registered with
    a factory function that extracts relevant config keys.
    """
    # --- Electric Price ---
    from GridPythia.prediction.electricprice.energycharts import (
        ElecPriceEnergyCharts,
        EnergyChartsConfig,
    )
    from GridPythia.prediction.electricprice.epexpredictor import (
        ElecPriceEpexPredictor,
        EpexPredictorConfig,
    )
    from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
    from GridPythia.prediction.electricprice.provider import ElecPriceFallbackChain

    def _energycharts_factory(cfg: Mapping[str, Any]) -> ElecPriceEnergyCharts:
        return ElecPriceEnergyCharts(
            EnergyChartsConfig(
                bidding_zone=cfg.get("bidding_zone", "DE-LU"),
                charges_kwh=cfg.get("charges_kwh", 0.0),
                vat_rate=cfg.get("vat_rate", 0.19),
            )
        )

    def _fixed_price_factory(cfg: Mapping[str, Any]) -> ElecPriceFixed:
        return ElecPriceFixed(
            price_kwh=cfg.get("price_kwh", cfg.get("charges_kwh", 0.30)),
            charges_kwh=cfg.get("charges_kwh", 0.0),
            vat_rate=cfg.get("vat_rate", 0.19),
        )

    def _epexpredictor_factory(cfg: Mapping[str, Any]) -> ElecPriceFallbackChain:
        """Build an EpexPredictor with EnergyCharts as automatic fallback."""
        primary = ElecPriceEpexPredictor(
            EpexPredictorConfig(
                region=cfg.get("region", "DE"),
                charges_kwh=cfg.get("charges_kwh", 0.0),
                vat_rate=cfg.get("vat_rate", 0.19),
                base_url=cfg.get("base_url", "https://epexpredictor.batzill.com"),
            )
        )
        fallback = ElecPriceEnergyCharts(
            EnergyChartsConfig(
                bidding_zone=cfg.get("bidding_zone", "DE-LU"),
                charges_kwh=cfg.get("charges_kwh", 0.0),
                vat_rate=cfg.get("vat_rate", 0.19),
            )
        )
        return ElecPriceFallbackChain(primary=primary, fallback=fallback)

    electricprice_registry.register("EnergyCharts", _energycharts_factory)
    electricprice_registry.register("Fixed", _fixed_price_factory)
    electricprice_registry.register("EpexPredictor", _epexpredictor_factory)

    # --- Feed-In Tariff ---
    from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed

    def _feedin_fixed_factory(cfg: Mapping[str, Any]) -> FeedInTariffFixed:
        return FeedInTariffFixed(tariff_kwh=cfg.get("tariff_kwh", 0.0))

    feedintariff_registry.register("Fixed", _feedin_fixed_factory)

    # --- Load ---
    from GridPythia.prediction.load.config import LoadProfileConfig
    from GridPythia.prediction.load.provider import load_provider_from_config

    def _profilecsv_factory(cfg: Mapping[str, Any]) -> LoadProvider:
        from pathlib import Path

        return load_provider_from_config(
            LoadProfileConfig(
                path=Path(cfg.get("path", "")),
                country=cfg.get("country"),
                subdivision=cfg.get("subdivision"),
            )
        )

    load_registry.register("ProfileCSV", _profilecsv_factory)

    # --- PV Forecast ---
    from GridPythia.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor
    from GridPythia.prediction.pvforecast.openmeteo import PVForecastOpenMeteo
    from GridPythia.prediction.pvforecast.provider import PVPlaneConfig

    def _pv_openmeteo_factory(cfg: Mapping[str, Any]) -> PVForecastOpenMeteo:
        plane_cfg = cfg.get("plane", {})
        om_cfg = cfg.get("openmeteo", {})
        plane = PVPlaneConfig(
            peak_kw=plane_cfg.get("peak_kw", 1.0),
            tilt=plane_cfg.get("tilt", 30.0),
            azimuth=plane_cfg.get("azimuth", 180.0),
            userhorizon=tuple(plane_cfg.get("userhorizon", [])) or None,
            loss_pct=plane_cfg.get("loss_pct", 2.0),
            damping_morning=om_cfg.get("damping_morning", 0.0),
            damping_evening=om_cfg.get("damping_evening", 0.0),
            partial_shading=om_cfg.get("partial_shading", False),
            inverter_id=plane_cfg.get("inverter_id", "inverter1"),
        )
        return PVForecastOpenMeteo(
            planes=[plane],
            latitude=cfg.get("latitude", 0.0),
            longitude=cfg.get("longitude", 0.0),
            api_key=om_cfg.get("api_key") or None,
            weather_model=om_cfg.get("weather_model") or None,
        )

    def _pv_akkudoktor_factory(cfg: Mapping[str, Any]) -> PVForecastAkkudoktor:
        plane_cfg = cfg.get("plane", {})
        plane = PVPlaneConfig(
            peak_kw=plane_cfg.get("peak_kw", 1.0),
            tilt=plane_cfg.get("tilt", 30.0),
            azimuth=plane_cfg.get("azimuth", 180.0),
            userhorizon=tuple(plane_cfg.get("userhorizon", [])) or None,
            loss_pct=plane_cfg.get("loss_pct", 2.0),
            inverter_id=plane_cfg.get("inverter_id", "inverter1"),
        )
        return PVForecastAkkudoktor(
            planes=[plane],
            latitude=cfg.get("latitude", 0.0),
            longitude=cfg.get("longitude", 0.0),
        )

    pvforecast_registry.register("OpenMeteo", _pv_openmeteo_factory)
    pvforecast_registry.register("Akkudoktor", _pv_akkudoktor_factory)

    # --- Weather ---
    from GridPythia.prediction.weather.brightsky import WeatherBrightSky
    from GridPythia.prediction.weather.openmeteo import WeatherOpenMeteo

    def _weather_openmeteo_factory(cfg: Mapping[str, Any]) -> WeatherOpenMeteo:
        return WeatherOpenMeteo(
            latitude=cfg.get("latitude", 0.0),
            longitude=cfg.get("longitude", 0.0),
        )

    def _weather_brightsky_factory(cfg: Mapping[str, Any]) -> WeatherBrightSky:
        return WeatherBrightSky(
            latitude=cfg.get("latitude", 0.0),
            longitude=cfg.get("longitude", 0.0),
        )

    weather_registry.register("OpenMeteo", _weather_openmeteo_factory)
    weather_registry.register("BrightSky", _weather_brightsky_factory)


# Auto-register on module load
_register_builtin_providers()


# ── Convenience class combining all registries ────────────────────────────


class PredictionProviderRegistry:
    """Unified access to all provider registries.

    Example:
    -------
    >>> from GridPythia.prediction.registry import provider_registry
    >>> ep = provider_registry.create_electricprice("EnergyCharts", {...})
    """

    electricprice = electricprice_registry
    feedintariff = feedintariff_registry
    load = load_registry
    pvforecast = pvforecast_registry
    weather = weather_registry

    def create_electricprice(self, name: str, config: Mapping[str, Any]) -> ElecPriceProvider:
        return self.electricprice.create(name, config)

    def create_feedintariff(self, name: str, config: Mapping[str, Any]) -> FeedInTariffProvider:
        return self.feedintariff.create(name, config)

    def create_load(self, name: str, config: Mapping[str, Any]) -> LoadProvider:
        return self.load.create(name, config)

    def create_pvforecast(self, name: str, config: Mapping[str, Any]) -> PVForecastProvider:
        return self.pvforecast.create(name, config)

    def create_weather(self, name: str, config: Mapping[str, Any]) -> WeatherProvider:
        return self.weather.create(name, config)


# Singleton instance
provider_registry = PredictionProviderRegistry()
