"""Akkudoktor PV forecast provider.

API documentation: https://akkudoktor.net
Endpoint: https://api.akkudoktor.net/forecast
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import polars as pl
from structlog import get_logger

from GridPythia.prediction.base import resample_to_timestamps
from GridPythia.prediction.pvforecast.provider import PVForecastProvider, PVPlaneConfig

logger = get_logger(__name__)

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
        planes:        One or more :class:`~GridPythia.prediction.pvforecast.provider.PVPlaneConfig`.
        latitude:      Location latitude in decimal degrees.
        longitude:     Location longitude in decimal degrees.
    """

    def __init__(
        self,
        planes: list[PVPlaneConfig],
        latitude: float,
        longitude: float,
    ) -> None:
        if not planes:
            raise ValueError("At least one PVPlaneConfig is required.")
        self._planes = planes
        self._lat = latitude
        self._lon = longitude

    @property
    def provider_id(self) -> str:
        return "PVForecastAkkudoktor"

    @staticmethod
    def _target_dt_hours(timestamps: pl.Series) -> float:
        ts_list: list[datetime] = timestamps.to_list()
        if len(ts_list) >= 2:
            return (ts_list[1] - ts_list[0]).total_seconds() / 3600.0
        return 1.0

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
            horizon = plane.userhorizon or [0, 0]
            params.append("horizont=" + ",".join(str(round(h)) for h in horizon))

        params.extend(
            [
                "past_days=5",
                "cellCoEff=-0.36",
                "inverterEfficiency=0.8",
                "albedo=0.25",
                "timezone=UTC",
                "hourly=relativehumidity_2m%2Cwindspeed_10m",
            ]
        )
        return f"{_BASE_URL}?{'&'.join(params)}"

    # ── HTTP ─────────────────────────────────────────────────────────────

    async def _request_raw(self) -> list[list[dict]]:
        """Call the Akkudoktor API and return raw per-plane hourly values."""
        url = self._build_url()
        logger.debug("akkudoktor_request", url=url)

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

        raw_values: list[list[dict]] = data.get("values", [])
        if not raw_values:
            raise ValueError("Akkudoktor API returned no 'values'.")
        logger.debug("akkudoktor_response_received", planes=len(raw_values))
        return raw_values

    def _parse_plane_values(self, raw_plane: list[dict]) -> list[_ForecastValue]:
        values: list[_ForecastValue] = []
        for entry in raw_plane:
            dt = datetime.fromisoformat(entry["datetime"])
            if dt.tzinfo is None:
                from zoneinfo import ZoneInfo

                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            values.append(
                _ForecastValue(
                    dt=dt,
                    dc_power_w=float(entry.get("dcPower", 0) or 0),
                    ac_power_w=float(entry.get("power", 0) or 0),
                )
            )
        return values

    async def _request(self) -> list[_ForecastValue]:
        """Call the Akkudoktor API and return summed per-hour values."""
        raw_values = await self._request_raw()

        results: list[_ForecastValue] = []
        for per_plane in zip(*raw_values, strict=False):
            dt_str: str = per_plane[0]["datetime"]
            dt = datetime.fromisoformat(dt_str)
            if dt.tzinfo is None:
                from zoneinfo import ZoneInfo

                dt = dt.replace(tzinfo=ZoneInfo("UTC"))

            dc = sum(float(v.get("dcPower", 0) or 0) for v in per_plane)
            ac = sum(float(v.get("power", 0) or 0) for v in per_plane)
            results.append(_ForecastValue(dt=dt, dc_power_w=dc, ac_power_w=ac))

        return results

    async def fetch_by_inverter(self, timestamps: pl.Series) -> dict[str, pl.Series]:
        ts_list: list[datetime] = timestamps.to_list()
        start = ts_list[0]
        end = ts_list[-1]
        target_dt_hours = self._target_dt_hours(timestamps)

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        start_utc = _to_utc(start)
        n_hourly = max(1, round((_to_utc(end) - start_utc).total_seconds() / 3600) + 2)
        hourly_by_inverter: dict[str, list[float]] = defaultdict(lambda: [0.0] * n_hourly)

        raw_values = await self._request_raw()
        for plane, raw_plane in zip(self._planes, raw_values, strict=False):
            hourly = hourly_by_inverter[plane.inverter]
            for fv in self._parse_plane_values(raw_plane):
                fv_utc = _to_utc(fv.dt)
                offset_h = (fv_utc - start_utc).total_seconds() / 3600.0
                idx = round(offset_h)
                if 0 <= idx < n_hourly:
                    hourly[idx] += max(0.0, fv.ac_power_w)

        return {
            inverter: resample_to_timestamps(hourly, 1.0, timestamps, pad_value=0.0)
            * target_dt_hours
            for inverter, hourly in hourly_by_inverter.items()
        }

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        ts_list: list[datetime] = timestamps.to_list()
        start = ts_list[0]
        end = ts_list[-1]
        target_dt_hours = self._target_dt_hours(timestamps)

        def _to_utc(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        start_utc = _to_utc(start)
        n_hourly = max(1, round((_to_utc(end) - start_utc).total_seconds() / 3600) + 2)
        hourly = [0.0] * n_hourly

        values = await self._request()
        for fv in values:
            fv_utc = _to_utc(fv.dt)
            offset_h = (fv_utc - start_utc).total_seconds() / 3600.0
            idx = round(offset_h)
            if 0 <= idx < n_hourly:
                hourly[idx] = max(0.0, fv.ac_power_w)

        return resample_to_timestamps(hourly, 1.0, timestamps, pad_value=0.0) * target_dt_hours
