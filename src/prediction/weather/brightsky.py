"""BrightSky weather provider (free, no API key, good for Germany)."""

import logging
from datetime import datetime

from src.prediction.base import make_array, n_steps
from src.prediction.weather.provider import WeatherData, WeatherProvider

logger = logging.getLogger(__name__)


class WeatherBrightSky(WeatherProvider):
    """Fetch hourly weather from `api.brightsky.dev <https://brightsky.dev/>`_.

    BrightSky is a free, open JSON API for DWD open weather data,
    particularly useful for locations in Germany.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        timezone_str: str = "Europe/Berlin",
    ) -> None:
        self._lat = latitude
        self._lon = longitude
        self._tz = timezone_str

    @property
    def provider_id(self) -> str:
        return "BrightSky"

    def fetch(
        self, start: datetime, end: datetime, dt_hours: float = 1.0
    ) -> WeatherData:
        import requests

        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)

        url = "https://api.brightsky.dev/weather"
        params = {
            "lat": self._lat,
            "lon": self._lon,
            "date": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_date": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "tz": self._tz,
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("weather", [])

        temperature = make_array(size=steps)
        cloud_cover = make_array(size=steps)
        wind_speed = make_array(size=steps)
        humidity = make_array(size=steps)
        precipitation = make_array(size=steps)
        pressure = make_array(size=steps)
        ghi = make_array(size=steps)

        for i, rec in enumerate(records[:steps]):
            temperature[i] = float(rec.get("temperature", 0) or 0)
            cloud_cover[i] = float(rec.get("cloud_cover", 0) or 0)
            wind_speed[i] = float(rec.get("wind_speed", 0) or 0)
            humidity[i] = float(rec.get("relative_humidity", 0) or 0)
            precipitation[i] = float(rec.get("precipitation", 0) or 0)
            pressure[i] = float(rec.get("pressure_msl", 0) or 0)
            # BrightSky 'solar' is in kJ/m²; convert to W/m² (÷ 3.6 for hourly)
            solar_kj = float(rec.get("solar", 0) or 0)
            ghi[i] = solar_kj / 3.6

        return WeatherData(
            temperature_c=temperature,
            cloud_cover_pct=cloud_cover,
            wind_speed_kmh=wind_speed,
            humidity_pct=humidity,
            precipitation_mm=precipitation,
            pressure_hpa=pressure,
            ghi_wm2=ghi,
        )
