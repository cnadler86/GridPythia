"""Open-Meteo weather provider (free, no API key required)."""

import logging
from array import array
from datetime import datetime

from src.prediction.base import n_steps
from src.prediction.weather.provider import WeatherData, WeatherProvider

logger = logging.getLogger(__name__)

_HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
    "pressure_msl",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]


class WeatherOpenMeteo(WeatherProvider):
    """Fetch hourly weather from `api.open-meteo.com <https://open-meteo.com/>`_."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        timezone_str: str = "UTC",
    ) -> None:
        self._lat = latitude
        self._lon = longitude
        self._tz = timezone_str

    @property
    def provider_id(self) -> str:
        return "OpenMeteo"

    def fetch(
        self, start: datetime, end: datetime, dt_hours: float = 1.0
    ) -> WeatherData:
        import requests

        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)
        forecast_days = max(1, int(hours / 24) + 1)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self._lat,
            "longitude": self._lon,
            "hourly": ",".join(_HOURLY_FIELDS),
            "timezone": self._tz,
            "forecast_days": min(forecast_days, 16),
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})

        def _extract(key: str, scale: float = 1.0) -> array:
            raw = data.get(key, [])
            vals = [float(v) * scale if v is not None else 0.0 for v in raw[:steps]]
            if len(vals) < steps:
                vals.extend([0.0] * (steps - len(vals)))
            return array("f", vals)

        return WeatherData(
            temperature_c=_extract("temperature_2m"),
            humidity_pct=_extract("relative_humidity_2m"),
            cloud_cover_pct=_extract("cloud_cover"),
            wind_speed_kmh=_extract("wind_speed_10m"),
            precipitation_mm=_extract("precipitation"),
            pressure_hpa=_extract("pressure_msl"),
            ghi_wm2=_extract("shortwave_radiation"),
            dni_wm2=_extract("direct_radiation"),
            dhi_wm2=_extract("diffuse_radiation"),
        )
