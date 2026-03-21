"""Weather data import from dictionaries."""

from array import array
from datetime import datetime

from src.prediction.base import make_array, n_steps, resample
from src.prediction.weather.provider import WeatherData, WeatherProvider


class WeatherImport(WeatherProvider):
    """Provide weather data from pre-built channel dictionaries.

    *data* maps channel names (``"temperature_c"``, ``"cloud_cover_pct"``, …)
    to lists of floats.  At minimum ``temperature_c`` and ``cloud_cover_pct``
    are required.
    """

    def __init__(
        self,
        data: dict[str, list[float]],
        source_dt_hours: float = 1.0,
    ) -> None:
        self._data = {k: array("f", v) for k, v in data.items()}
        self._source_dt = source_dt_hours

    @property
    def provider_id(self) -> str:
        return "WeatherImport"

    def _channel(self, key: str, steps: int, dt_hours: float) -> array | None:
        raw = self._data.get(key)
        if raw is None:
            return None
        data = raw
        if abs(self._source_dt - dt_hours) > 1e-9:
            data = resample(data, self._source_dt, dt_hours)
        result = make_array(size=steps)
        for i in range(min(steps, len(data))):
            result[i] = data[i]
        return result

    def fetch(
        self, start: datetime, end: datetime, dt_hours: float = 1.0
    ) -> WeatherData:
        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)

        return WeatherData(
            temperature_c=self._channel("temperature_c", steps, dt_hours)
            or make_array(size=steps),
            cloud_cover_pct=self._channel("cloud_cover_pct", steps, dt_hours)
            or make_array(size=steps),
            wind_speed_kmh=self._channel("wind_speed_kmh", steps, dt_hours),
            humidity_pct=self._channel("humidity_pct", steps, dt_hours),
            precipitation_mm=self._channel("precipitation_mm", steps, dt_hours),
            pressure_hpa=self._channel("pressure_hpa", steps, dt_hours),
            ghi_wm2=self._channel("ghi_wm2", steps, dt_hours),
            dni_wm2=self._channel("dni_wm2", steps, dt_hours),
            dhi_wm2=self._channel("dhi_wm2", steps, dt_hours),
        )
