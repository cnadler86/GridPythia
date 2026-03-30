"""Weather data import from pre-built dictionaries."""

import polars as pl

from GridPythia.prediction.base import resample_to_timestamps
from GridPythia.prediction.weather.provider import WeatherProvider

_REQUIRED = ("temperature_c", "cloud_cover_pct")
_OPTIONAL = (
    "wind_speed_kmh",
    "humidity_pct",
    "precipitation_mm",
    "pressure_hpa",
    "ghi_wm2",
    "dni_wm2",
    "dhi_wm2",
)


class WeatherImport(WeatherProvider):
    """Provide weather data from pre-built channel dictionaries.

    *data* maps channel names to lists of floats.  ``temperature_c`` and
    ``cloud_cover_pct`` are required; all others are optional.
    """

    def __init__(
        self,
        data: dict[str, list[float]],
        source_dt_hours: float = 1.0,
    ) -> None:
        self._data: dict[str, list[float]] = {k: list(v) for k, v in data.items()}
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "WeatherImport"

    async def fetch(self, timestamps: pl.Series) -> pl.DataFrame:
        cols: dict[str, pl.Series] = {}
        zeros = pl.Series([0.0] * len(timestamps), dtype=pl.Float32)

        for key in _REQUIRED:
            raw = self._data.get(key)
            if raw is not None:
                cols[key] = resample_to_timestamps(raw, self._source_dt, timestamps, pad_value=0.0)
            else:
                cols[key] = zeros

        for key in _OPTIONAL:
            raw = self._data.get(key)
            if raw is not None:
                cols[key] = resample_to_timestamps(raw, self._source_dt, timestamps, pad_value=0.0)

        return pl.DataFrame(cols)
