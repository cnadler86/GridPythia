"""Tests for PV forecast providers."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest

from src.prediction.base import make_timestamps
from src.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor, _ForecastValue
from src.prediction.pvforecast.import_ import PVForecastImport
from src.prediction.pvforecast.openmeteo import PVForecastOpenMeteo
from src.prediction.pvforecast.provider import PVPlaneConfig

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)
_PLANE = PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0)


def _ts(hours: float = 24, dt: float = 1.0) -> pl.Series:
    return make_timestamps(START, hours, dt)


# ── PVPlaneConfig ─────────────────────────────────────────────────────────────


class TestPVPlaneConfig:
    def test_defaults(self):
        p = PVPlaneConfig(peak_kw=3.0, tilt=30.0, azimuth=180.0)
        assert p.loss_pct == 2.0
        assert p.userhorizon is None
        assert p.inverter.startswith("default")

    def test_custom(self):
        p = PVPlaneConfig(peak_kw=4.0, tilt=25.0, azimuth=90.0, userhorizon=[10, 20, 30])
        assert p.peak_kw == 4.0
        assert p.azimuth == 90.0
        assert p.userhorizon == (10, 20, 30)

    def test_userhorizon_is_normalized_for_hashing(self):
        p1 = PVPlaneConfig(
            peak_kw=4.0, tilt=25.0, azimuth=90.0, userhorizon=[10, 20, 30], inverter="inv-a"
        )
        p2 = PVPlaneConfig(
            peak_kw=4.0, tilt=25.0, azimuth=90.0, userhorizon=(10, 20, 30), inverter="inv-a"
        )
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_can_be_used_as_dict_key(self):
        p1 = PVPlaneConfig(
            peak_kw=4.0, tilt=25.0, azimuth=90.0, userhorizon=[10, 20, 30], inverter="inv-a"
        )
        p2 = PVPlaneConfig(
            peak_kw=4.0, tilt=25.0, azimuth=90.0, userhorizon=(10, 20, 30), inverter="inv-a"
        )
        cache = {p1: "cached"}
        assert cache[p2] == "cached"


# ── PVForecastImport ──────────────────────────────────────────────────────────


class TestPVForecastImport:
    async def test_full_day(self):
        power = [0.0] * 6 + [500.0] * 12 + [0.0] * 6
        provider = PVForecastImport(power_w=power)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[0] == pytest.approx(0.0)
        assert result[6] == pytest.approx(500.0)

    async def test_shorter_pads_zero(self):
        power = [1000.0] * 12
        provider = PVForecastImport(power_w=power)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[11] == pytest.approx(1000.0)
        assert result[12] == pytest.approx(0.0)

    async def test_quarter_hour(self):
        power = [0.0, 100.0]
        ts = make_timestamps(START, hours=2, dt_hours=0.25)
        provider = PVForecastImport(power_w=power, source_dt_hours=1.0)
        result = await provider.fetch(ts)
        assert len(result) == 8

    async def test_provider_id(self):
        assert PVForecastImport(power_w=[]).provider_id == "PVForecastImport"

    async def test_returns_polars_float32(self):
        result = await PVForecastImport(power_w=[0.0] * 24).fetch(_ts())
        assert isinstance(result, pl.Series)
        assert result.dtype == pl.Float32

    async def test_fetch_by_inverter_uses_default(self):
        provider = PVForecastImport(power_w=[0.0] * 24)
        result = await provider.fetch_by_inverter(_ts())
        assert set(result) == {"default"}
        assert result["default"].dtype == pl.Float32


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_akkudoktor_response(start: datetime, n_hours: int = 48) -> list[_ForecastValue]:
    from datetime import timedelta

    ac_profile = ([0.0] * 6 + [1000.0] * 12 + [0.0] * 6) * (n_hours // 24 + 1)
    return [
        _ForecastValue(
            dt=start + timedelta(hours=i),
            dc_power_w=ac_profile[i] * 1.04,
            ac_power_w=ac_profile[i],
        )
        for i in range(n_hours)
    ]


def _make_forecastsolar_response(start: datetime, ac_w: float = 2000.0) -> dict[str, float]:
    from datetime import timedelta

    return {
        (start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"): (
            ac_w if 7 <= i < 19 else 0.0
        )
        for i in range(24)
    }


# ── PVForecastAkkudoktor ──────────────────────────────────────────────────────


class TestPVForecastAkkudoktor:
    def _make_provider(self) -> PVForecastAkkudoktor:
        planes = [
            PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0),
            PVPlaneConfig(peak_kw=3.0, tilt=20.0, azimuth=90.0),
        ]
        return PVForecastAkkudoktor(planes=planes, latitude=52.52, longitude=13.405)

    def test_provider_id(self):
        assert self._make_provider().provider_id == "PVForecastAkkudoktor"

    def test_url_contains_lat_lon(self):
        url = self._make_provider()._build_url()
        assert "lat=52.52" in url
        assert "lon=13.405" in url

    def test_url_azimuth_south(self):
        p = PVForecastAkkudoktor(
            planes=[PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0)],
            latitude=52.52, longitude=13.405,
        )
        assert "azimuth=0" in p._build_url()

    def test_url_azimuth_east(self):
        p = PVForecastAkkudoktor(
            planes=[PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=90.0)],
            latitude=52.52, longitude=13.405,
        )
        assert "azimuth=-90" in p._build_url()

    def test_requires_at_least_one_plane(self):
        with pytest.raises(ValueError):
            PVForecastAkkudoktor(planes=[], latitude=52.0, longitude=13.0)

    @patch.object(PVForecastAkkudoktor, "_request", new_callable=AsyncMock)
    async def test_fetch_hourly(self, mock_req):
        mock_req.return_value = _make_akkudoktor_response(START, n_hours=48)
        provider = self._make_provider()
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[0] == pytest.approx(0.0)
        assert result[10] > 0.0

    @patch.object(PVForecastAkkudoktor, "_request", new_callable=AsyncMock)
    async def test_fetch_quarter_hour(self, mock_req):
        mock_req.return_value = _make_akkudoktor_response(START, n_hours=48)
        result = await self._make_provider().fetch(_ts(dt=0.25))
        assert len(result) == 96

    @patch.object(PVForecastAkkudoktor, "_request", new_callable=AsyncMock)
    async def test_no_negative_power(self, mock_req):
        values = _make_akkudoktor_response(START, n_hours=24)
        values[10] = _ForecastValue(dt=values[10].dt, dc_power_w=-500.0, ac_power_w=-500.0)
        mock_req.return_value = values
        provider = PVForecastAkkudoktor(planes=[_PLANE], latitude=52.52, longitude=13.405)
        result = await provider.fetch(_ts())
        assert all(v >= 0.0 for v in result.to_list())

    @patch.object(PVForecastAkkudoktor, "_request_raw", new_callable=AsyncMock)
    async def test_fetch_by_inverter_groups_planes(self, mock_req_raw):
        mock_req_raw.return_value = [
            [
                {"datetime": "2025-06-15T10:00:00+00:00", "dcPower": 520.0, "power": 500.0},
                {"datetime": "2025-06-15T11:00:00+00:00", "dcPower": 520.0, "power": 500.0},
            ],
            [
                {"datetime": "2025-06-15T10:00:00+00:00", "dcPower": 310.0, "power": 300.0},
                {"datetime": "2025-06-15T11:00:00+00:00", "dcPower": 310.0, "power": 300.0},
            ],
            [
                {"datetime": "2025-06-15T10:00:00+00:00", "dcPower": 210.0, "power": 200.0},
                {"datetime": "2025-06-15T11:00:00+00:00", "dcPower": 210.0, "power": 200.0},
            ],
        ]
        provider = PVForecastAkkudoktor(
            planes=[
                PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0, inverter="inv-a"),
                PVPlaneConfig(peak_kw=3.0, tilt=20.0, azimuth=90.0, inverter="inv-a"),
                PVPlaneConfig(peak_kw=2.0, tilt=20.0, azimuth=270.0, inverter="inv-b"),
            ],
            latitude=52.52,
            longitude=13.405,
        )

        result = await provider.fetch_by_inverter(_ts())

        assert set(result) == {"inv-a", "inv-b"}
        assert result["inv-a"][10] == pytest.approx(800.0)
        assert result["inv-b"][10] == pytest.approx(200.0)


# ── PVForecastOpenMeteo ───────────────────────────────────────────────────────


class TestPVForecastOpenMeteo:
    def _make_provider(self) -> PVForecastOpenMeteo:
        return PVForecastOpenMeteo(
            planes=[_PLANE], latitude=52.52, longitude=13.405, timezone_str="UTC"
        )

    def test_provider_id(self):
        assert self._make_provider().provider_id == "PVForecastOpenMeteo"

    def test_requires_at_least_one_plane(self):
        with pytest.raises(ValueError):
            PVForecastOpenMeteo(planes=[], latitude=52.0, longitude=13.0)

    def test_azimuth_conversion(self):
        from datetime import timedelta, timezone as tz_mod
        import asyncio

        # om_az = plane.azimuth - 180; south=180 -> 0, east=90 -> -90
        p = PVForecastOpenMeteo(
            planes=[PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0)],
            latitude=52.52, longitude=13.405,
        )
        assert p._planes[0].azimuth - 180.0 == pytest.approx(0.0)

    @patch.object(PVForecastOpenMeteo, "_fetch_plane", new_callable=AsyncMock)
    async def test_fetch_single_plane(self, mock_fetch):
        from datetime import timedelta
        # 15-min keyed dict: 0 W until slot 24 (hour 6), 500 W until slot 72 (hour 18)
        plane_data = {
            START + timedelta(minutes=15 * i): 500.0 if 24 <= i < 72 else 0.0
            for i in range(104)
        }
        mock_fetch.return_value = plane_data
        result = await self._make_provider().fetch(_ts())
        assert len(result) == 24
        assert result[6] == pytest.approx(500.0)
        assert result[0] == pytest.approx(0.0)

    @patch.object(PVForecastOpenMeteo, "_fetch_plane", new_callable=AsyncMock)
    async def test_fetch_quarter_hour(self, mock_fetch):
        from datetime import timedelta
        plane_data = {START + timedelta(minutes=15 * i): 100.0 for i in range(104)}
        mock_fetch.return_value = plane_data
        result = await self._make_provider().fetch(_ts(dt=0.25))
        assert len(result) == 96

    @patch.object(PVForecastOpenMeteo, "_fetch_plane", new_callable=AsyncMock)
    async def test_planes_summed(self, mock_fetch):
        """Two planes: each returns 1000 W -> total 2000 W."""
        from datetime import timedelta
        plane_data = {
            START + timedelta(minutes=15 * i): 1000.0 if 24 <= i < 72 else 0.0
            for i in range(104)
        }
        mock_fetch.return_value = plane_data
        p = PVForecastOpenMeteo(
            planes=[_PLANE, PVPlaneConfig(peak_kw=3.0, tilt=20.0, azimuth=90.0)],
            latitude=52.52, longitude=13.405,
        )
        result = await p.fetch(_ts())
        # Both planes return the same mock data -> sum = 2000
        assert result[10] == pytest.approx(2000.0)

    @patch.object(PVForecastOpenMeteo, "_fetch_plane", new_callable=AsyncMock)
    async def test_reuses_cache_for_repeated_fetch(self, mock_fetch):
        from datetime import timedelta

        plane_data = {
            START + timedelta(minutes=15 * i): 500.0 if 24 <= i < 72 else 0.0
            for i in range(104)
        }
        mock_fetch.return_value = plane_data

        provider = self._make_provider()
        await provider.fetch(_ts())
        await provider.fetch(_ts())

        assert mock_fetch.await_count == 1

    @patch.object(PVForecastOpenMeteo, "_fetch_plane", new_callable=AsyncMock)
    async def test_reuses_cache_for_equal_plane_configs(self, mock_fetch):
        from datetime import timedelta

        plane_data = {
            START + timedelta(minutes=15 * i): 500.0 if 24 <= i < 72 else 0.0
            for i in range(104)
        }
        mock_fetch.return_value = plane_data

        plane1 = PVPlaneConfig(
            peak_kw=5.0, tilt=30.0, azimuth=180.0, userhorizon=[0.0, 5.0], inverter="inv-a"
        )
        plane2 = PVPlaneConfig(
            peak_kw=5.0, tilt=30.0, azimuth=180.0, userhorizon=(0.0, 5.0), inverter="inv-a"
        )
        provider = PVForecastOpenMeteo(
            planes=[plane1, plane2], latitude=52.52, longitude=13.405, timezone_str="UTC"
        )

        result = await provider.fetch(_ts())

        assert result[6] == pytest.approx(1000.0)
        assert mock_fetch.await_count == 1

    @patch.object(PVForecastOpenMeteo, "_fetch_plane", new_callable=AsyncMock)
    async def test_invalidates_cache_when_forecast_window_changes(self, mock_fetch):
        from datetime import timedelta

        plane_data = {
            START + timedelta(minutes=15 * i): 500.0 if 24 <= i < 72 else 0.0
            for i in range(200)
        }
        mock_fetch.return_value = plane_data

        provider = self._make_provider()
        await provider.fetch(_ts(hours=24))
        await provider.fetch(_ts(hours=72))

        assert mock_fetch.await_count == 2

    @patch.object(PVForecastOpenMeteo, "_fetch_plane", new_callable=AsyncMock)
    async def test_fetch_by_inverter_groups_planes(self, mock_fetch):
        from datetime import timedelta

        plane_data = {
            START + timedelta(minutes=15 * i): 100.0 if 24 <= i < 72 else 0.0
            for i in range(104)
        }
        mock_fetch.return_value = plane_data
        provider = PVForecastOpenMeteo(
            planes=[
                PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0, inverter="inv-a"),
                PVPlaneConfig(peak_kw=3.0, tilt=20.0, azimuth=90.0, inverter="inv-a"),
                PVPlaneConfig(peak_kw=2.0, tilt=20.0, azimuth=270.0, inverter="inv-b"),
            ],
            latitude=52.52,
            longitude=13.405,
            timezone_str="UTC",
        )

        result = await provider.fetch_by_inverter(_ts())

        assert set(result) == {"inv-a", "inv-b"}
        assert result["inv-a"][10] == pytest.approx(200.0)
        assert result["inv-b"][10] == pytest.approx(100.0)
