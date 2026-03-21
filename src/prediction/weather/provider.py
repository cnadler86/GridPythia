"""Weather provider interface and data container."""

from abc import abstractmethod
from array import array
from dataclasses import dataclass
from datetime import datetime

from src.prediction.base import PredictionProvider


@dataclass
class WeatherData:
    """Multi-channel weather arrays, all of the same length (``n_steps``)."""

    temperature_c: array
    cloud_cover_pct: array
    wind_speed_kmh: array | None = None
    humidity_pct: array | None = None
    precipitation_mm: array | None = None
    pressure_hpa: array | None = None
    ghi_wm2: array | None = None
    dni_wm2: array | None = None
    dhi_wm2: array | None = None


class WeatherProvider(PredictionProvider):
    """Returns multi-channel weather data for a time window."""

    @abstractmethod
    def fetch(
        self, start: datetime, end: datetime, dt_hours: float = 1.0
    ) -> WeatherData:
        """Return a :class:`WeatherData` instance."""
        ...
