"""PV forecast provider interface and plane configuration."""

from abc import abstractmethod
from dataclasses import dataclass, field

import polars as pl

from src.prediction.base import PredictionProvider


@dataclass
class PVPlaneConfig:
    """Physical configuration of one PV array / plane.

    Azimuth convention (consistent with EOS / PVGIS):
        north=0°, east=90°, south=180°, west=270°

    Args:
        peak_kw:         Nominal peak power in kW.
        tilt:            Tilt angle from horizontal (0–90°).
        azimuth:         Surface azimuth in degrees (north=0, south=180).
        userhorizon:     Horizon elevation in degrees at equally-spaced azimuth
                         steps clockwise from north.  ``None`` = flat horizon.
                         Converted to the library's ``horizon_map`` format when
                         used with the Open-Meteo provider.
        loss_pct:        System losses in percent.  Default 2 %.
        damping_morning: Morning shading damping factor (0–1).  Open-Meteo only.
        damping_evening: Evening shading damping factor (0–1).  Open-Meteo only.
        partial_shading: Enable partial-shading model.  Open-Meteo only.
    """

    peak_kw: float
    tilt: float
    azimuth: float
    userhorizon: list[float] | None = field(default=None)
    loss_pct: float = 2.0
    damping_morning: float = 0.0
    damping_evening: float = 0.0
    partial_shading: bool = False


class PVForecastProvider(PredictionProvider):
    """Returns PV power output in W per time step."""

    @abstractmethod
    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        """Return Float32 Series of watts, same length as *timestamps*."""
        ...
