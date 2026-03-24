"""Open-Meteo PV forecast provider via open-meteo-solar-forecast library.

Uses ``OpenMeteoSolarForecast`` which calls the Open-Meteo API internally and
delivers 15-minute-resolution AC power estimates that are averaged into hourly
buckets before resampling to the requested timestamps.

Azimuth convention:
    HEMS2 / PVGIS: north=0°, south=180°.
    Open-Meteo-Solar-Forecast (and Open-Meteo API): south=0°.
    Conversion: ``om_az = plane.azimuth − 180``.

Caching:
    Responses are cached per plane for
    :attr:`PVForecastOpenMeteo._TTL_S` seconds (default 3600 = 1 h).
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from time import monotonic
from typing import ClassVar, Sequence

import polars as pl
from open_meteo_solar_forecast import OpenMeteoSolarForecast

from src.prediction.base import resample_to_timestamps
from src.prediction.pvforecast.provider import PVForecastProvider, PVPlaneConfig

logger = logging.getLogger(__name__)


def _userhorizon_to_map(
    userhorizon: Sequence[float] | None,
) -> tuple[tuple[float, float], ...]:
    """Convert equally-spaced horizon elevations to ``(azimuth, elevation)`` pairs.

    The library's *horizon_map* expects a sequence of ``(azimuth_deg, elevation_deg)``
    tuples.  When *userhorizon* is ``None`` the library's default flat-horizon map is
    returned (two sentinel points at 0° elevation).
    """
    if not userhorizon:
        return ((0.0, 0.0), (360.0, 0.0))
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
        api_key:       Optional Open-Meteo API key for commercial endpoints.
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
        api_key: str | None = None,
        weather_model: str | None = None,
    ) -> None:
        if not planes:
            raise ValueError("At least one PVPlaneConfig is required.")
        self._planes = planes
        self._lat = latitude
        self._lon = longitude
        self._tz = timezone_str
        self._api_key = api_key
        self._weather_model = weather_model
        # Per-plane cache: plane -> (forecast_days, past_days, fetched_at_mono, data)
        self._cache: dict[
            PVPlaneConfig,
            tuple[int, int, float, dict[datetime, float]],
        ] = {}

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

        Keys are the UTC-converted 15-minute timestamps returned by the
        Open-Meteo API.
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
            api_key=self._api_key,
            weather_model=self._weather_model if self._weather_model else None,
        ) as forecaster:
            estimate = await forecaster.estimate()

        # The API already returns 15-minute timestamps; only normalize to UTC.
        result: dict[datetime, float] = {}
        logger.debug(
            "OpenMeteo response: %d estimates for plane with az=%s tilt=%s.",
            len(estimate.watts),
            om_az,
            plane.tilt,
        )
        for ts, w in estimate.watts.items():
            result[ts.astimezone(timezone.utc)] = float(w)
        return result

    # ── public API ────────────────────────────────────────────────────

    async def fetch_by_inverter(self, timestamps: pl.Series) -> dict[str, pl.Series]:
        ts_list: list[datetime] = timestamps.to_list()

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        start_utc = _to_utc(ts_list[0])
        end_utc = _to_utc(ts_list[-1])

        past_days, forecast_days = self._derive_days(start_utc, end_utc)
        now_mono = monotonic()
        plane_data_by_inverter: dict[str, list[dict[datetime, float]]] = defaultdict(list)
        for plane in self._planes:
            cached = self._cache.get(plane)
            if (
                cached is None
                or cached[0] != forecast_days
                or cached[1] != past_days
                or (now_mono - cached[2]) >= self._TTL_S
            ):
                plane_data = await self._fetch_plane(plane, forecast_days, past_days)
                self._cache[plane] = (forecast_days, past_days, now_mono, plane_data)
            else:
                logger.debug(
                    "OpenMeteo: using cached response for plane az=%s tilt=%s (age=%.0fs)",
                    plane.azimuth - 180.0,
                    plane.tilt,
                    now_mono - cached[2],
                )
                plane_data = cached[3]
            plane_data_by_inverter[plane.inverter].append(plane_data)

        n_slots = max(1, round((end_utc - start_utc).total_seconds() / 900) + 4)
        result: dict[str, pl.Series] = {}
        for inverter, plane_data_list in plane_data_by_inverter.items():
            quarterly = [0.0] * n_slots
            for plane_data in plane_data_list:
                for slot_utc, watts in plane_data.items():
                    idx = round((slot_utc - start_utc).total_seconds() / 900)
                    if 0 <= idx < n_slots:
                        quarterly[idx] += max(0.0, watts)
            result[inverter] = resample_to_timestamps(quarterly, 0.25, timestamps, pad_value=0.0)

        return result

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        return self._sum_series_by_key(await self.fetch_by_inverter(timestamps))
