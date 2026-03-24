"""PV forecast provider interface and plane configuration."""

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Sequence

import polars as pl

from src.prediction.base import PredictionProvider


@dataclass(frozen=True)
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
                         used with the Open-Meteo provider. Stored internally as
                         an immutable tuple.
        loss_pct:        System losses in percent.  Default 2 %.
        damping_morning: Morning shading damping factor (0–1).  Open-Meteo only.
        damping_evening: Evening shading damping factor (0–1).  Open-Meteo only.
        partial_shading: Enable partial-shading model.  Open-Meteo only.
        inverter:        Inverter model name. Default is "default".
    """

    peak_kw: float
    tilt: float
    azimuth: float
    userhorizon: Sequence[float] | None = field(default=None)
    loss_pct: float = 2.0
    damping_morning: float = 0.0
    damping_evening: float = 0.0
    partial_shading: bool = False
    inverter: str = "default"

    def __post_init__(self) -> None:
        if self.userhorizon is not None and not isinstance(self.userhorizon, tuple):
            object.__setattr__(self, "userhorizon", tuple(self.userhorizon))


class PVForecastProvider(PredictionProvider):
    """Returns PV power output in W per time step."""

    @staticmethod
    def _sum_series_by_key(series_by_key: Mapping[str, pl.Series]) -> pl.Series:
        """Sum a mapping of aligned power series into one total series."""
        if not series_by_key:
            return pl.Series([], dtype=pl.Float32)

        total = [0.0] * len(next(iter(series_by_key.values())))
        for series in series_by_key.values():
            for idx, value in enumerate(series):
                total[idx] += float(value)
        return pl.Series(total, dtype=pl.Float32)

    async def fetch_by_inverter(self, timestamps: pl.Series) -> dict[str, pl.Series]:
        """Return watt series keyed by inverter name.

        Providers without inverter-specific information fall back to a single
        synthetic inverter named ``default``.
        """
        return {"default": await self.fetch(timestamps)}

    @abstractmethod
    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        """Return Float32 Series of watts, same length as *timestamps*."""
        ...
