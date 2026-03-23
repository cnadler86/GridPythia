"""Open-Meteo PV forecast provider via open-meteo-solar-forecast library.

Uses ``OpenMeteoSolarForecast`` which calls the Open-Meteo API internally and
delivers 15-minute-resolution AC power estimates that are averaged into hourly
buckets before resampling to the requested timestamps.

Azimuth convention:
    HEMS2 / PVGIS: north=0°, south=180°.
    Open-Meteo-Solar-Forecast (and Open-Meteo API): south=0°.
    Conversion: ``om_az = plane.azimuth − 180``.

Caching:
    API responses are cached per date-range pair for
    :attr:`PVForecastOpenMeteo._TTL_S` seconds (default 3600 = 1 h).
"""

import logging
from datetime import date, datetime, timezone
from typing import ClassVar

import polars as pl
from open_meteo_solar_forecast import OpenMeteoSolarForecast

from src.prediction.base import resample_to_timestamps
from src.prediction.pvforecast.provider import PVForecastProvider, PVPlaneConfig

logger = logging.getLogger(__name__)


def _userhorizon_to_map(
    userhorizon: list[float] | None,
) -> tuple[tuple[float, float], ...]:
    """Convert equally-spaced horizon elevations to ``(azimuth, elevation)`` pairs.

    The library's *horizon_map* expects a sequence of ``(azimuth_deg, elevation_deg)``
    tuples.  When *userhorizon* is ``None`` the library's default flat-horizon map is
    returned (two sentinel points at 20° elevation).
    """
    if not userhorizon:
        return ((0.0, 20.0), (360.0, 20.0))
    n = len(userhorizon)
    return tuple((i * 360.0 / n, elev) for i, elev in enumerate(userhorizon))


class PVForecastOpenMeteo(PVForecastProvider):
    """PV forecast via the ``open-meteo-solar-forecast`` library.

    The library queries the Open-Meteo API at 15-minute resolution.  For each
    plane a separate request is issued; the hourly averages are summed and then
    resampled to the target ``timestamps``.

    ``forecast_days`` and ``past_days`` are derived automatically from the
    ``timestamps`` argument passed to :meth:`fetch`.  Results are cached for
    :attr:`_TTL_S` seconds so repeated calls within that window skip the network
    request.

    Args:
        planes:        One or more :class:`~src.prediction.pvforecast.provider.PVPlaneConfig`.
        latitude:      Location latitude in decimal degrees.
        longitude:     Location longitude in decimal degrees.
        timezone_str:  IANA timezone string (unused in API calls but kept for
                       consistency with other providers).
        ac_kwp:        AC inverter output capacity in kW.  ``None`` = unlimited.
        api_key:       Optional Open-Meteo API key for commercial endpoints.
        base_url:      Override the default Open-Meteo API base URL.
        weather_model: Open-Meteo weather model identifier
                       (e.g. ``"best_match"``, ``"ecmwf_ifs04"``).  ``None`` = API default.
    """

    _TTL_S: ClassVar[int] = 3600  # 1-hour TTL for cached API responses

    def __init__(
        self,
        planes: list[PVPlaneConfig],
        latitude: float,
        longitude: float,
        timezone_str: str = "UTC",
        ac_kwp: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        weather_model: str | None = None,
    ) -> None:
        if not planes:
            raise ValueError("At least one PVPlaneConfig is required.")
        self._planes = planes
        self._lat = latitude
        self._lon = longitude
        self._tz = timezone_str
        self._ac_kwp = ac_kwp
        self._api_key = api_key
        self._base_url = base_url
        self._weather_model = weather_model
        # Cache: key=(start_date, end_date) → (mono_time, [plane0_dict, plane1_dict, ...])
        self._cache: dict[tuple[date, date], tuple[float, list[dict[datetime, float]]]] = {}

    @property
    def provider_id(self) -> str:
        return "PVForecastOpenMeteo"

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _derive_days(start_utc: datetime, end_utc: datetime) -> tuple[int, int]:
        """Return ``(past_days, forecast_days)`` spanning *start_utc* → *end_utc*.

        *past_days* is always 1 so the API covers the full first day.
        *forecast_days* is derived from the span between *start_utc* and *end_utc*.
        The first timestamp in the requested series acts as the time reference;
        no wall-clock ``datetime.now()`` is used.
        """
        past_days = 1
        forecast_days = max(1, (end_utc.date() - start_utc.date()).days + 2)
        return past_days, forecast_days

    async def _fetch_plane(
        self,
        plane: PVPlaneConfig,
        forecast_days: int,
        past_days: int,
    ) -> dict[datetime, float]:
        """Fetch 15-min data for *plane* and return a ``{slot_utc: watts}`` dict.

        Keys are UTC datetimes floored to the nearest 15-minute boundary,
        preserving the full 15-minute resolution of the Open-Meteo API.
        """
        om_az = plane.azimuth - 180.0
        logger.debug(
            "OpenMeteo request: lat=%s lon=%s az=%s tilt=%s kwp=%s forecast_days=%s past_days=%s",
            self._lat,
            self._lon,
            om_az,
            plane.tilt,
            plane.peak_kw,
            forecast_days,
            past_days,
        )
        async with OpenMeteoSolarForecast(
            azimuth=om_az,
            declination=plane.tilt,
            dc_kwp=plane.peak_kw,
            latitude=self._lat,
            longitude=self._lon,
            efficiency_factor=1.0 - plane.loss_pct / 100.0,
            forecast_days=forecast_days,
            past_days=past_days,
            damping_morning=plane.damping_morning,
            damping_evening=plane.damping_evening,
            partial_shading=plane.partial_shading,
            use_horizon=plane.userhorizon is not None,
            horizon_map=_userhorizon_to_map(plane.userhorizon),
            ac_kwp=self._ac_kwp,
            api_key=self._api_key,
            **(({"base_url": self._base_url}) if self._base_url else {}),
            **(({"weather_model": self._weather_model}) if self._weather_model else {}),
        ) as forecaster:
            estimate = await forecaster.estimate()

        # Keep full 15-min resolution; floor each timestamp to its 15-min slot
        result: dict[datetime, float] = {}
        for ts, w in estimate.watts.items():
            ts_utc = ts.astimezone(timezone.utc)
            slot = ts_utc.replace(
                minute=(ts_utc.minute // 15) * 15,
                second=0,
                microsecond=0,
            )
            result[slot] = float(w)
        return result

    # ── public API ────────────────────────────────────────────────────

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        ts_list: list[datetime] = timestamps.to_list()

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        start_utc = _to_utc(ts_list[0])
        end_utc = _to_utc(ts_list[-1])

        past_days, forecast_days = self._derive_days(start_utc, end_utc)
        cache_key = (start_utc.date(), end_utc.date())

        first_time = start_utc.timestamp()
        cached = self._cache.get(cache_key)
        if cached is None or (first_time - cached[0]) >= self._TTL_S:
            data: list[dict[datetime, float]] = [
                await self._fetch_plane(p, forecast_days, past_days) for p in self._planes
            ]
            self._cache[cache_key] = (first_time, data)
        else:
            logger.debug("OpenMeteo: using cached response (age=%.0fs)", first_time - cached[0])
            data = cached[1]

        # Build 15-min array aligned to start_utc, summing all planes
        n_slots = max(1, round((end_utc - start_utc).total_seconds() / 900) + 4)
        quarterly = [0.0] * n_slots
        for plane_data in data:
            for slot_utc, watts in plane_data.items():
                idx = round((slot_utc - start_utc).total_seconds() / 900)
                if 0 <= idx < n_slots:
                    quarterly[idx] += max(0.0, watts)

        return resample_to_timestamps(quarterly, 0.25, timestamps, pad_value=0.0)
