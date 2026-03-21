"""Forecast.Solar PV forecast provider.

API documentation: https://doc.forecast.solar/doku.php?id=api

Free public endpoint (no API key):
    ``GET https://api.forecast.solar/estimate/:lat/:lon/:dec/:az/:kwp``

Rate-limits on the free tier: ≤ 12 req/h per IP.

Azimuth convention for this API:
    south=0°, west=+90°, east=−90°, i.e. ``fs_az = plane.azimuth - 180``
    (same formula as Akkudoktor but the accepted range here is −180..180 exactly).
"""

import logging
from array import array
from datetime import datetime

from src.prediction.base import make_array, n_steps, resample
from src.prediction.pvforecast.provider import PVForecastProvider, PVPlaneConfig

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.forecast.solar/estimate"


class PVForecastSolar(PVForecastProvider):
    """Fetch PV forecast from ``api.forecast.solar``.

    Supports multiple planes; AC power values are **summed** across planes.

    For each plane one HTTP request is made (the API only accepts a single
    plane per call).  Results are combined before resampling.

    Args:
        planes:        One or more :class:`~src.prediction.pvforecast.provider.PVPlaneConfig`.
        latitude:      Location latitude in decimal degrees (−90..90).
        longitude:     Location longitude in decimal degrees (−180..180).
        api_key:       Optional personal API key for higher rate-limits / pro endpoints.
        timezone_str:  IANA timezone string (e.g. ``"Europe/Berlin"``).
                       Passed as a query parameter for the time zone of the returned
                       timestamps.
    """

    def __init__(
        self,
        planes: list[PVPlaneConfig],
        latitude: float,
        longitude: float,
        api_key: str | None = None,
        timezone_str: str = "Europe/Berlin",
    ) -> None:
        if not planes:
            raise ValueError("At least one PVPlaneConfig is required.")
        self._planes = planes
        self._lat = latitude
        self._lon = longitude
        self._api_key = api_key
        self._tz = timezone_str

    @property
    def provider_id(self) -> str:
        return "PVForecastSolar"

    # ── URL builder ──────────────────────────────────────────────────────

    def _build_url(self, plane: PVPlaneConfig) -> str:
        """Build the estimate endpoint URL for a single plane.

        Path: ``/estimate[/:apikey]/:lat/:lon/:dec/:az/:kwp``
        """
        # Azimuth: HEMS2 south=180 → forecast.solar south=0
        fs_az = plane.azimuth - 180.0  # range: -180..180
        dec = plane.tilt
        kwp = plane.peak_kw

        key_segment = f"/{self._api_key}" if self._api_key else ""
        return (
            f"{_BASE_URL}{key_segment}"
            f"/{self._lat:.6f}/{self._lon:.6f}"
            f"/{dec:.1f}/{fs_az:.1f}/{kwp:.3f}"
        )

    def _query_params(self) -> dict[str, str]:
        params: dict[str, str] = {"time": "utc"}
        if self._tz:
            params["time"] = self._tz
        return params

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _request_plane(self, plane: PVPlaneConfig) -> dict[str, float]:
        """Return ``{ISO-datetime-str: watts}`` for a single plane."""
        import requests

        url = self._build_url(plane)
        logger.debug("forecast.solar request: %s", url)
        resp = requests.get(url, params=self._query_params(), timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # The API returns {"result": {"watts": {"2025-06-15 06:00:00": 123, …}, …}}
        return data.get("result", {}).get("watts", {})

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dt(dt_str: str, tz_str: str) -> datetime:
        """Parse the datetime strings returned by the API."""
        from zoneinfo import ZoneInfo

        # Strings are in the requested timezone when time≠utc, e.g. "2025-06-15 06:00:00"
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.fromisoformat(dt_str)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(tz_str))
        return dt

    # ── public API ───────────────────────────────────────────────────────

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        hours = (end - start).total_seconds() / 3600
        hourly_steps = n_steps(hours, 1.0)
        hourly: array = make_array(size=hourly_steps)

        for plane in self._planes:
            watts_by_dt = self._request_plane(plane)
            for dt_str, power_w in watts_by_dt.items():
                dt = self._parse_dt(dt_str, self._tz)
                offset_h = (dt - start).total_seconds() / 3600.0
                idx = round(offset_h)
                if 0 <= idx < hourly_steps:
                    hourly[idx] = hourly[idx] + max(0.0, float(power_w))

        if abs(dt_hours - 1.0) < 1e-9:
            return hourly

        steps = n_steps(hours, dt_hours)
        result = resample(hourly, 1.0, dt_hours)
        # Trim / zero-pad to exact count
        if len(result) >= steps:
            return array("f", result[:steps])
        return array("f", list(result) + [0.0] * (steps - len(result)))
