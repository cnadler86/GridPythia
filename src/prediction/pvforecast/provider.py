"""PV forecast provider interface and plane configuration."""

from abc import abstractmethod
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.prediction.base import PredictionProvider


@dataclass
class PVPlaneConfig:
    """Physical configuration of one PV array / plane.

    Azimuth convention (consistent with EOS / PVGIS):
        north=0°, east=90°, south=180°, west=270°

    Args:
        peak_kw:       Nominal peak power in kW.
        tilt:          Tilt angle from horizontal (0–90°).  Default 30°.
        azimuth:       Surface azimuth in degrees (north=0, south=180).  Default 180°.
        inverter_pac_w: AC power rating of the inverter in watts.  ``None`` = no clipping.
        userhorizon:   Elevation of horizon in degrees at equally-spaced azimuth
                       steps clockwise from north.  ``None`` = flat horizon.
        loss_pct:      System losses in percent (cables, soiling, …).  Default 14 %.
    """

    peak_kw: float
    tilt: float = 30.0
    azimuth: float = 180.0
    inverter_pac_w: Optional[int] = None
    userhorizon: Optional[list[float]] = field(default=None)
    loss_pct: float = 14.0


class PVForecastProvider(PredictionProvider):
    """Returns PV power output in W per time step."""

    @abstractmethod
    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        """Return ``array('f', ...)`` of watts with ``n_steps`` entries."""
        ...
