"""Akkudoktor PV forecast provider.

API documentation: https://akkudoktor.net
Endpoint: https://api.akkudoktor.net/forecast
"""

import logging
from array import array
from dataclasses import dataclass
from datetime import datetime

from src.prediction.base import make_array, n_steps, resample
from src.prediction.pvforecast.provider import PVForecastProvider, PVPlaneConfig

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.akkudoktor.net/forecast"


@dataclass
class _ForecastValue:
    dt: datetime
    dc_power_w: float
    ac_power_w: float


class PVForecastAkkudoktor(PVForecastProvider):
    """Fetch PV forecast from ``api.akkudoktor.net``.

    Supports multiple planes; powers are **summed** across all planes.

    Azimuth convention:
        HEMS2 / EOS uses PVGIS convention (south=180°).
        Akkudoktor uses south=0°, so the conversion is::

            akkudoktor_az = plane.azimuth - 180

    Args:
        planes:        One or more :class:`~src.prediction.pvforecast.provider.PVPlaneConfig`.
        latitude:      Location latitude in decimal degrees.
        longitude:     Location longitude in decimal degrees.
        timezone_str:  IANA timezone string (e.g. ``"Europe/Berlin"``).
    """

    def __init__(
        self,
        planes: list[PVPlaneConfig],
        latitude: float,
        longitude: float,
        timezone_str: str = "Europe/Berlin",
    ) -> None:
        if not planes:
            raise ValueError("At least one PVPlaneConfig is required.")
        self._planes = planes
        self._lat = latitude
        self._lon = longitude
        self._tz = timezone_str

    @property
    def provider_id(self) -> str:
        return "PVForecastAkkudoktor"

    # ── URL builder ──────────────────────────────────────────────────────

    def _build_url(self) -> str:
        params: list[str] = [
            f"lat={self._lat}",
            f"lon={self._lon}",
        ]
        for plane in self._planes:
            params.append(f"power={int(plane.peak_kw * 1000)}")
            # Azimuth conversion: HEMS2 south=180° → Akkudoktor south=0°
            ak_az = int(plane.azimuth) - 180
            params.append(f"azimuth={ak_az}")
            params.append(f"tilt={int(plane.tilt)}")
            pac = plane.inverter_pac_w if plane.inverter_pac_w is not None else 25000
            params.append(f"powerInverter={pac}")
            horizon = plane.userhorizon or [0, 0]
            params.append("horizont=" + ",".join(str(round(h)) for h in horizon))

        params.extend(
            [
                "past_days=5",
                "cellCoEff=-0.36",
                "inverterEfficiency=0.8",
                "albedo=0.25",
                f"timezone={self._tz}",
                "hourly=relativehumidity_2m%2Cwindspeed_10m",
            ]
        )
        return f"{_BASE_URL}?{'&'.join(params)}"

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _request(self) -> list[_ForecastValue]:
        """Call the Akkudoktor API and return summed per-hour values."""
        import requests

        url = self._build_url()
        logger.debug("Akkudoktor request: %s", url)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # ``values`` is a list-of-lists: one inner list per plane, each entry is a dict
        raw_values: list[list[dict]] = data.get("values", [])
        if not raw_values:
            raise ValueError("Akkudoktor API returned no 'values'.")

        # Transpose to (timestep, planes); sum power across planes
        results: list[_ForecastValue] = []
        for per_plane in zip(*raw_values):
            # All planes share the same datetime string
            dt_str: str = per_plane[0]["datetime"]
            # Parse ISO-8601 datetime returned by Akkudoktor
            dt = datetime.fromisoformat(dt_str)
            if dt.tzinfo is None:
                from zoneinfo import ZoneInfo

                dt = dt.replace(tzinfo=ZoneInfo(self._tz))

            dc = sum(float(v.get("dcPower", 0) or 0) for v in per_plane)
            ac = sum(float(v.get("power", 0) or 0) for v in per_plane)
            results.append(_ForecastValue(dt=dt, dc_power_w=dc, ac_power_w=ac))

        return results

    # ── public API ───────────────────────────────────────────────────────

    def fetch(self, start: datetime, end: datetime, dt_hours: float = 1.0) -> array:
        hours = (end - start).total_seconds() / 3600
        steps = n_steps(hours, dt_hours)

        values = self._request()

        # Build hourly lookup by index
        hourly: array = make_array(size=n_steps(hours, 1.0))
        for fv in values:
            offset_h = (fv.dt - start).total_seconds() / 3600.0
            idx = round(offset_h)
            if 0 <= idx < len(hourly):
                hourly[idx] = max(0.0, fv.ac_power_w)

        if abs(dt_hours - 1.0) < 1e-9:
            return hourly

        return resample(hourly, 1.0, dt_hours)
