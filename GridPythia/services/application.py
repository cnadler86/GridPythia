"""Application service encapsulating state and business logic.

This module provides `GridPythiaService` - a dependency-injectable service
that replaces the global state pattern in `server/state.py`. It manages:

- Configuration loading and hot-reload detection
- Provider singleton lifecycle
- Optimizer instance (CVXPY model compilation)
- Prediction data cache

For MQTT integration, this service can be instantiated per-connection or
shared across the application with proper locking.

Example:
-------
>>> from GridPythia.services.application import GridPythiaService
>>> svc = GridPythiaService.from_config_path(Path("config.yaml"))
>>> pdata = await svc.fetch_predictions()
>>> solution = await svc.optimize(pdata)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from structlog import get_logger

from GridPythia.config import AppConfig
from GridPythia.optimization.solution import LinearSolution, OptimizationObjective
from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.prediction.prediction import Prediction, PredictionData, PredictionSetup
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase

logger = get_logger(__name__)


@dataclass
class PredictionCache:
    """Cache for prediction data with TTL expiration."""

    data: PredictionData | None = None
    timestamp: datetime | None = None
    forecast_from: datetime | None = None
    ttl_seconds: float = 300.0

    def is_valid(self) -> bool:
        """Check if cached data is still within TTL."""
        if self.data is None or self.timestamp is None:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age < self.ttl_seconds

    def get(self) -> tuple[PredictionData, datetime | None] | None:
        """Return cached data if valid, else None."""
        if self.is_valid() and self.data is not None:
            return self.data, self.forecast_from
        return None

    def set(self, data: PredictionData, forecast_from: datetime | None = None) -> None:
        """Update cache with new prediction data."""
        self.data = data
        self.timestamp = datetime.now()
        self.forecast_from = forecast_from

    def invalidate(self) -> None:
        """Clear the cache."""
        self.data = None
        self.timestamp = None
        self.forecast_from = None


@dataclass
class GridPythiaService:
    """Central application service managing state and business logic.

    This class encapsulates all state that was previously global in
    `server/state.py`. It provides thread-safe access to:

    - Configuration (with hot-reload detection)
    - Prediction providers (singleton per config version)
    - Optimizer instance (compiled CVXPY model)
    - Prediction cache

    For web servers, create one instance and share it via dependency injection.
    For MQTT, the same instance can be shared with proper locking.

    Attributes:
        config_path: Path to the YAML configuration file.
        config: Parsed AppConfig (read-only after load).
        prediction_cache: TTL-based cache for prediction data.
    """

    config_path: Path
    config: AppConfig = field(init=False)
    raw_yaml: dict[str, Any] = field(default_factory=dict, init=False)
    prediction_cache: PredictionCache = field(default_factory=PredictionCache)

    _providers: PredictionSetup | None = field(default=None, init=False, repr=False)
    _optimizer: LinearOptimizer | None = field(default=None, init=False, repr=False)
    _inverters: list[InverterBase] = field(default_factory=list, init=False, repr=False)
    _config_mtime: float = field(default=0.0, init=False, repr=False)
    _config_loaded: bool = field(default=False, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Load configuration on initialization."""
        self._reload_config_if_changed()

    @classmethod
    def from_config_path(cls, path: Path | str) -> "GridPythiaService":
        """Create service instance from config file path."""
        return cls(config_path=Path(path))

    def _get_config_mtime(self) -> float:
        """Get config file modification time."""
        try:
            return self.config_path.stat().st_mtime
        except OSError:
            return 0.0

    def _reload_config_if_changed(self) -> bool:
        """Reload config if file has changed. Returns True if reloaded."""
        import yaml

        mtime = self._get_config_mtime()
        if mtime == self._config_mtime and self._config_loaded:
            return False

        self.raw_yaml = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.config = AppConfig.from_dict(self.raw_yaml)
        self._config_mtime = mtime
        self._config_loaded = True
        self._providers = None  # Invalidate cached providers
        self._optimizer = None  # Invalidate cached optimizer
        logger.info("config_reloaded", path=str(self.config_path))
        return True

    def _build_inverters(self) -> list[InverterBase]:
        """Build inverter instances from config."""
        batteries: dict[str, Battery] = {
            p.device_id: Battery(p) for p in self.config.optimization.batteries
        }
        inverters: list[InverterBase] = []
        for inv_params in self.config.optimization.inverters:
            bat = batteries.get(inv_params.battery_id) if inv_params.battery_id else None
            inverters.append(InverterBase(inv_params, battery=bat))
        return inverters

    @property
    def providers(self) -> PredictionSetup:
        """Get or create prediction providers (lazy singleton)."""
        self._reload_config_if_changed()
        if self._providers is None:
            self._providers = self._build_providers()
            logger.info("providers_created")
        return self._providers

    @property
    def optimizer(self) -> LinearOptimizer:
        """Get or create optimizer instance (lazy singleton)."""
        self._reload_config_if_changed()
        if self._optimizer is None:
            self._inverters = self._build_inverters()
            objective = (
                OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
                if self.config.optimization.solver.objective == "self_consumption"
                else OptimizationObjective.MINIMIZE_COST
            )
            self._optimizer = LinearOptimizer(
                inverters=self._inverters,
                objective=objective,
                solver_opts=dict(self.config.optimization.solver.solver_opts),
            )
            logger.info("optimizer_created")
        return self._optimizer

    @property
    def inverters(self) -> list[InverterBase]:
        """Get inverter instances (triggers optimizer creation if needed)."""
        _ = self.optimizer  # Ensure optimizer is created
        return self._inverters

    def _build_providers(self) -> PredictionSetup:
        """Build all prediction providers from config.

        Uses the registry pattern when available, falls back to direct
        construction for compatibility.
        """
        from GridPythia.prediction.electricprice.energycharts import (
            ElecPriceEnergyCharts,
            EnergyChartsConfig,
        )
        from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
        from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed
        from GridPythia.prediction.load.config import LoadProfileConfig
        from GridPythia.prediction.load.provider import load_provider_from_config
        from GridPythia.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor
        from GridPythia.prediction.pvforecast.openmeteo import PVForecastOpenMeteo
        from GridPythia.prediction.pvforecast.provider import PVPlaneConfig
        from GridPythia.prediction.weather.brightsky import WeatherBrightSky
        from GridPythia.prediction.weather.openmeteo import WeatherOpenMeteo

        pred_cfg = self.config.prediction

        # Electric price
        ep = pred_cfg.electricprice
        if ep.provider == "EnergyCharts":
            electricprice = ElecPriceEnergyCharts(
                EnergyChartsConfig(
                    bidding_zone=ep.energycharts.bidding_zone,
                    charges_kwh=ep.charges_kwh,
                    vat_rate=ep.vat_rate,
                )
            )
        else:
            electricprice = ElecPriceFixed(
                price_kwh=ep.charges_kwh,
                charges_kwh=ep.charges_kwh,
                vat_rate=ep.vat_rate,
            )

        feedintariff = FeedInTariffFixed(tariff_kwh=pred_cfg.feedintariff.tariff_kwh)

        # Load provider
        raw_load_path = Path(pred_cfg.load.path)
        load_path = (
            raw_load_path
            if raw_load_path.is_absolute()
            else (self.config_path.parent / raw_load_path)
        )

        from GridPythia.prediction.load.config import AdaptiveLoadConfig

        adaptive_cfg = AdaptiveLoadConfig(
            enabled=pred_cfg.load.adaptive.enabled,
            decay_days=pred_cfg.load.adaptive.decay_days,
            min_samples=pred_cfg.load.adaptive.min_samples,
            blend_factor=pred_cfg.load.adaptive.blend_factor,
            db_path=pred_cfg.load.adaptive.db_path,
            flush_interval_s=pred_cfg.load.adaptive.flush_interval_s,
            mqtt_topic=pred_cfg.load.adaptive.mqtt_topic,
        )

        load_provider = load_provider_from_config(
            LoadProfileConfig(
                path=load_path,
                country=pred_cfg.load.country or None,
                subdivision=pred_cfg.load.subdivision or None,
                adaptive=adaptive_cfg,
            )
        )

        # PV provider
        plane_cfg = pred_cfg.pvforecast.plane
        om_cfg = pred_cfg.pvforecast.openmeteo
        plane = PVPlaneConfig(
            peak_kw=plane_cfg.peak_kw,
            tilt=plane_cfg.tilt,
            azimuth=plane_cfg.azimuth,
            userhorizon=tuple(plane_cfg.userhorizon) if plane_cfg.userhorizon else None,
            loss_pct=plane_cfg.loss_pct,
            damping_morning=om_cfg.damping_morning,
            damping_evening=om_cfg.damping_evening,
            partial_shading=om_cfg.partial_shading,
            inverter_id=plane_cfg.inverter_id,
        )
        if pred_cfg.pvforecast.provider == "OpenMeteo":
            pv_provider = PVForecastOpenMeteo(
                planes=[plane],
                latitude=pred_cfg.latitude,
                longitude=pred_cfg.longitude,
                api_key=om_cfg.api_key or None,
                weather_model=om_cfg.weather_model or None,
            )
        else:
            pv_provider = PVForecastAkkudoktor(
                planes=[plane],
                latitude=pred_cfg.latitude,
                longitude=pred_cfg.longitude,
            )

        # Weather provider (optional)
        weather_provider = None
        if "weather" in self.raw_yaml.get("prediction", {}):
            w_cfg = pred_cfg.weather
            if w_cfg.provider == "BrightSky":
                weather_provider = WeatherBrightSky(
                    latitude=pred_cfg.latitude, longitude=pred_cfg.longitude
                )
            else:
                weather_provider = WeatherOpenMeteo(
                    latitude=pred_cfg.latitude, longitude=pred_cfg.longitude
                )

        return PredictionSetup(
            electricprice=electricprice,
            feedintariff=feedintariff,
            load=load_provider,
            pv={plane.inverter_id: pv_provider},
            weather=weather_provider,
        )

    async def fetch_predictions(
        self,
        start: datetime | None = None,
        hours: float | None = None,
        dt_hours: float | None = None,
        *,
        use_cache: bool = True,
    ) -> PredictionData:
        """Fetch prediction data, using cache if available.

        Args:
            start: Forecast start time (default: now in local timezone).
            hours: Horizon in hours (default: from config).
            dt_hours: Time step in hours (default: from config).
            use_cache: Whether to use cached data if available.

        Returns:
            PredictionData with aligned timestamps.
        """
        if use_cache:
            cached = self.prediction_cache.get()
            if cached is not None:
                logger.info("predictions_from_cache")
                return cached[0]

        pred = Prediction(self.providers)
        pdata = await pred.fetch(
            start=start,
            hours=hours or float(self.config.prediction.horizon),
            dt_hours=dt_hours or float(self.config.prediction.dt_hours),
        )

        # Extract forecast_from from EnergyCharts provider if applicable
        from GridPythia.prediction.electricprice.energycharts import ElecPriceEnergyCharts

        forecast_from = None
        if isinstance(self.providers.electricprice, ElecPriceEnergyCharts):
            forecast_from = self.providers.electricprice.last_real_ts

        self.prediction_cache.set(pdata, forecast_from)
        return pdata

    async def optimize(
        self,
        prediction: PredictionData | None = None,
        *,
        soc_wh: dict[str, float] | None = None,
        initial_modes: dict[str, InverterMode | int] | None = None,
        objective: OptimizationObjective | None = None,
        solver_opts: dict[str, Any] | None = None,
        validate_with_simulation: bool = False,
    ) -> LinearSolution:
        """Run the optimizer on prediction data.

        Args:
            prediction: Input data (fetched automatically if None).
            soc_wh: Per-inverter battery SoC overrides in Wh.
            initial_modes: Per-inverter initial mode at horizon start.
            objective: Override optimizer objective for this call.
            solver_opts: Additional HiGHS options.
            validate_with_simulation: Run simulation parity check.

        Returns:
            LinearSolution with inverter plans.
        """
        if prediction is None:
            prediction = await self.fetch_predictions()

        async with self._lock:
            solution = await asyncio.to_thread(
                lambda: self.optimizer.solve(
                    prediction,
                    soc=soc_wh,
                    initial_modes=initial_modes,
                    objective=objective,
                    solver_opts=solver_opts,
                    validate_with_simulation=validate_with_simulation,
                )
            )
        return solution

    def invalidate_cache(self) -> None:
        """Clear prediction cache (e.g., after config change)."""
        self.prediction_cache.invalidate()
        logger.info("cache_invalidated")

    def reload_config(self) -> bool:
        """Force config reload. Returns True if config changed."""
        self._config_mtime = 0.0  # Force reload
        reloaded = self._reload_config_if_changed()
        if reloaded:
            self.invalidate_cache()
        return reloaded
