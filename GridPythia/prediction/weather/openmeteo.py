"""Open-Meteo weather provider (free, no API key required)."""

from datetime import datetime, timezone

import aiohttp
import numpy as np
from structlog import get_logger

from GridPythia.prediction.base import resample_to_timestamps
from GridPythia.prediction.weather.provider import WeatherProvider

logger = get_logger(__name__)

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
    """Fetch hourly weather from ``api.open-meteo.com``."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
    ) -> None:
        self._lat = latitude
        self._lon = longitude

    @property
    def provider_id(self) -> str:
        return "OpenMeteo"

    async def fetch(self, timestamps: list) -> dict[str, np.ndarray]:
        ts_list = timestamps
        start = ts_list[0]
        end = ts_list[-1]

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        forecast_days = max(1, int((end_utc - start_utc).total_seconds() / 86400) + 1)

        logger.debug(
            "openmeteo_weather_request",
            lat=self._lat,
            lon=self._lon,
            forecast_days=min(forecast_days, 16),
        )

        params = {
            "latitude": self._lat,
            "longitude": self._lon,
            "hourly": ",".join(_HOURLY_FIELDS),
            "timezone": "UTC",
            "forecast_days": min(forecast_days, 16),
            "timeformat": "unixtime",
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("https://api.open-meteo.com/v1/forecast", params=params) as resp:
                resp.raise_for_status()
                body = await resp.json(content_type=None)

        hourly = body.get("hourly", {})
        time_arr: list[int] = hourly.get("time", [])

        n_hourly = max(1, round((end_utc - start_utc).total_seconds() / 3600) + 2)

        def _build(key: str) -> list[float]:
            raw = hourly.get(key, [])
            out = [0.0] * n_hourly
            for ts_unix, val in zip(time_arr, raw, strict=False):
                dt_utc = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
                offset_h = (dt_utc - start_utc).total_seconds() / 3600.0
                idx = round(offset_h)
                if 0 <= idx < n_hourly and val is not None:
                    out[idx] = float(val)
            return out

        return {
            "temperature_c": resample_to_timestamps(
                _build("temperature_2m"), 1.0, timestamps, pad_value=0.0
            ),
            "humidity_pct": resample_to_timestamps(
                _build("relative_humidity_2m"), 1.0, timestamps, pad_value=0.0
            ),
            "cloud_cover_pct": resample_to_timestamps(
                _build("cloud_cover"), 1.0, timestamps, pad_value=0.0
            ),
            "wind_speed_kmh": resample_to_timestamps(
                _build("wind_speed_10m"), 1.0, timestamps, pad_value=0.0
            ),
            "precipitation_mm": resample_to_timestamps(
                _build("precipitation"), 1.0, timestamps, pad_value=0.0
            ),
            "pressure_hpa": resample_to_timestamps(
                _build("pressure_msl"), 1.0, timestamps, pad_value=0.0
            ),
            "ghi_wm2": resample_to_timestamps(
                _build("shortwave_radiation"), 1.0, timestamps, pad_value=0.0
            ),
            "dni_wm2": resample_to_timestamps(
                _build("direct_radiation"), 1.0, timestamps, pad_value=0.0
            ),
            "dhi_wm2": resample_to_timestamps(
                _build("diffuse_radiation"), 1.0, timestamps, pad_value=0.0
            ),
        }
