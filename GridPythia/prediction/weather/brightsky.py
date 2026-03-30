"""BrightSky weather provider (free, no API key, good for Germany)."""

import logging
from datetime import datetime, timezone

import aiohttp
import polars as pl

from GridPythia.prediction.base import resample_to_timestamps
from GridPythia.prediction.weather.provider import WeatherProvider

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
    ) -> None:
        self._lat = latitude
        self._lon = longitude

    @property
    def provider_id(self) -> str:
        return "BrightSky"

    async def fetch(self, timestamps: pl.Series) -> pl.DataFrame:
        ts_list: list[datetime] = timestamps.to_list()
        start = ts_list[0]
        end = ts_list[-1]

        def _fmt(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%dT%H:%M:%S")

        url = "https://api.brightsky.dev/weather"
        params = {
            "lat": self._lat,
            "lon": self._lon,
            "date": _fmt(start),
            "last_date": _fmt(end),
            "tz": "UTC",
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                body = await resp.json(content_type=None)

        records = body.get("weather", [])

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        start_utc = _to_utc(start)
        n_hourly = max(1, round((_to_utc(end) - start_utc).total_seconds() / 3600) + 2)

        tmp = [0.0] * n_hourly
        cld = [0.0] * n_hourly
        wnd = [0.0] * n_hourly
        hum = [0.0] * n_hourly
        pcp = [0.0] * n_hourly
        prs = [0.0] * n_hourly
        ghi = [0.0] * n_hourly

        for rec in records:
            ts_str: str = rec.get("timestamp", "")
            if not ts_str:
                continue
            try:
                rec_dt = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            offset_h = (_to_utc(rec_dt) - start_utc).total_seconds() / 3600.0
            idx = round(offset_h)
            if 0 <= idx < n_hourly:
                tmp[idx] = float(rec.get("temperature", 0) or 0)
                cld[idx] = float(rec.get("cloud_cover", 0) or 0)
                wnd[idx] = float(rec.get("wind_speed", 0) or 0)
                hum[idx] = float(rec.get("relative_humidity", 0) or 0)
                pcp[idx] = float(rec.get("precipitation", 0) or 0)
                prs[idx] = float(rec.get("pressure_msl", 0) or 0)
                # BrightSky 'solar' is in kJ/m²; convert to W/m² (÷ 3.6 for hourly avg)
                ghi[idx] = float(rec.get("solar", 0) or 0) / 3.6

        return pl.DataFrame(
            {
                "temperature_c": resample_to_timestamps(tmp, 1.0, timestamps, pad_value=0.0),
                "cloud_cover_pct": resample_to_timestamps(cld, 1.0, timestamps, pad_value=0.0),
                "wind_speed_kmh": resample_to_timestamps(wnd, 1.0, timestamps, pad_value=0.0),
                "humidity_pct": resample_to_timestamps(hum, 1.0, timestamps, pad_value=0.0),
                "precipitation_mm": resample_to_timestamps(pcp, 1.0, timestamps, pad_value=0.0),
                "pressure_hpa": resample_to_timestamps(prs, 1.0, timestamps, pad_value=0.0),
                "ghi_wm2": resample_to_timestamps(ghi, 1.0, timestamps, pad_value=0.0),
            }
        )
