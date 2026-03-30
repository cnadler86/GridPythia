"""PV forecast provider interface and plane configuration."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Optional, Sequence, Tuple

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from GridPythia.prediction.base import PredictionProvider


class PVPlaneConfig(BaseModel):
    """Physical configuration of one PV array / plane.

    Azimuth convention (consistent with EOS / PVGIS):
        north=0°, east=90°, south=180°, west=270°

    Args:
        peak_kw:         Nominal peak power in kW.
        tilt:            Tilt angle from horizontal (0-90°).
        azimuth:         Surface azimuth in degrees (north=0, south=180).
        userhorizon:     Horizon elevation in degrees at equally-spaced azimuth
                         steps clockwise from north.  ``None`` = flat horizon.
                         Converted to the library's ``horizon_map`` format when
                         used with the Open-Meteo provider. Stored internally as
                         an immutable tuple.
        loss_pct:        System losses in percent.
        damping_morning: Morning shading damping factor.  Open-Meteo only.
        damping_evening: Evening shading damping factor.  Open-Meteo only.
        partial_shading: Enable partial-shading model.  Open-Meteo only.
        inverter:        Inverter model name. Default is "inverter1".
    """

    peak_kw: float = Field(..., gt=0.0)
    tilt: float = Field(..., ge=0.0, le=90.0)
    azimuth: float = Field(..., ge=0.0, le=360.0)
    userhorizon: Optional[Tuple[float, ...]] = None
    loss_pct: float = Field(2.0, ge=0.0, lt=100.0)
    damping_morning: float = 0.0
    damping_evening: float = 0.0
    partial_shading: bool = False
    inverter: str = "inverter1"

    model_config = ConfigDict(frozen=True)

    @field_validator("userhorizon", mode="before")
    def _to_tuple(cls, v: Sequence[float] | None):
        return None if v is None else tuple(v)


class PVForecastProvider(PredictionProvider):
    """Returns PV energy output in Wh per time step."""

    @staticmethod
    def _sum_series_by_key(series_by_key: Mapping[str, pl.Series]) -> pl.Series:
        """Sum a mapping of aligned energy series into one total series."""
        if not series_by_key:
            return pl.Series([], dtype=pl.Float32)

        total = [0.0] * len(next(iter(series_by_key.values())))
        for series in series_by_key.values():
            for idx, value in enumerate(series):
                total[idx] += float(value)
        return pl.Series(total, dtype=pl.Float32)

    async def fetch_by_inverter(self, timestamps: pl.Series) -> dict[str, pl.Series]:
        """Return Wh series keyed by inverter name.

        Providers without inverter-specific information fall back to a single
        synthetic inverter named ``inverter1``.
        """
        return {"inverter1": await self.fetch(timestamps)}

    @abstractmethod
    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        """Return Float32 Series of Wh, same length as *timestamps*."""
        ...
