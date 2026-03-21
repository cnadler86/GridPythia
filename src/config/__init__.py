"""Simple configuration for HEMS2 optimization."""

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class GeneticConfig:
    """Genetic algorithm configuration."""

    individuals: int = 300
    generations: int = 400
    seed: Optional[int] = None
    penalties: dict[str, Union[float, int, str]] = field(
        default_factory=lambda: {"ev_soc_miss": 10, "ac_charge_break_even": 1.0}
    )


@dataclass
class OptimizationConfig:
    """Optimization configuration."""

    horizon_hours: int = 24
    interval: int = 3600
    genetic: GeneticConfig = field(default_factory=GeneticConfig)


@dataclass
class PredictionConfig:
    """Prediction configuration."""

    hours: int = 48
    historic_hours: int = 24
    dt_hours: float = 1.0
    latitude: float = 52.52
    longitude: float = 13.405
    timezone: str = "Europe/Berlin"


@dataclass
class HEMSConfig:
    """Top-level configuration for HEMS2."""

    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
