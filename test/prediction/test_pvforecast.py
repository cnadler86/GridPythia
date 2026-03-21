"""Tests for PV forecast providers."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor
from src.prediction.pvforecast.forecastsolar import PVForecastSolar
from src.prediction.pvforecast.import_ import PVForecastImport
from src.prediction.pvforecast.provider import PVPlaneConfig

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)
END_24H = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)

_PLANE = PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0, inverter_pac_w=5000)


# ── PVPlaneConfig ────────────────────────────────────────────────────────


class TestPVPlaneConfig:
    def test_defaults(self):
        p = PVPlaneConfig(peak_kw=3.0)
        assert p.tilt == 30.0
        assert p.azimuth == 180.0
        assert p.inverter_pac_w is None
        assert p.loss_pct == 14.0
        assert p.userhorizon is None

    def test_custom(self):
        p = PVPlaneConfig(
            peak_kw=4.0,
            tilt=25.0,
            azimuth=90.0,
            inverter_pac_w=4000,
            userhorizon=[10, 20, 30],
        )
        assert p.peak_kw == 4.0
        assert p.azimuth == 90.0
        assert p.userhorizon == [10, 20, 30]


# ── PVForecastImport ─────────────────────────────────────────────────────


class TestPVForecastImport:
    def test_full_day(self):
        power = (
            [0.0] * 6
            + [500 * (i / 6) for i in range(6)]
            + [500 * (1 - i / 6) for i in range(6)]
            + [0.0] * 6
        )
        provider = PVForecastImport(power_w=power)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result) == 24
        assert result[0] == pytest.approx(0.0)
        assert result[12] == pytest.approx(500 * (1 - 0 / 6))

    def test_shorter_than_window_pads_with_zero(self):
        power = [1000.0] * 12
        provider = PVForecastImport(power_w=power)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result) == 24
        assert result[11] == pytest.approx(1000.0)
        assert result[12] == pytest.approx(0.0)  # zero-padded (night)

    def test_quarter_hour(self):
        power = [0.0, 100.0]  # 2h hourly
        end_2h = datetime(2025, 6, 15, 2, 0, tzinfo=timezone.utc)
        provider = PVForecastImport(power_w=power, source_dt_hours=1.0)
        result = provider.fetch(START, end_2h, dt_hours=0.25)
        assert len(result) == 8

    def test_provider_id(self):
        assert PVForecastImport(power_w=[]).provider_id == "PVForecastImport"


# ── Helpers ───────────────────────────────────────────────────────────────


def _akkudoktor_response(start: datetime, n_hours: int = 48) -> dict:
    """Build a minimal Akkudoktor-style JSON response with two planes."""
    from datetime import timedelta

    def _vals(ac_w: float):
        return [
            {
                "datetime": (start + timedelta(hours=i)).isoformat(),
                "dcPower": ac_w * 1.04,
                "power": ac_w,
                "sunTilt": 30.0,
                "sunAzimuth": 180.0,
                "temperature": 20.0,
                "relativehumidity_2m": 60.0,
                "windspeed_10m": 10.0,
            }
            for i in range(n_hours)
        ]

    ac_profile = [0.0] * 6 + [1000.0] * 12 + [0.0] * 6
    # Pad to n_hours
    ac_profile = (ac_profile * (n_hours // len(ac_profile) + 1))[:n_hours]
    plane_a = _vals(0)
    plane_b = _vals(0)
    for i, ac in enumerate(ac_profile):
        plane_a[i]["power"] = ac * 0.6
        plane_a[i]["dcPower"] = ac * 0.6 * 1.04
        plane_b[i]["power"] = ac * 0.4
        plane_b[i]["dcPower"] = ac * 0.4 * 1.04
    return {"values": [plane_a, plane_b]}


def _forecastsolar_response(start: datetime, ac_w: float = 2000.0) -> dict:
    """Build a minimal forecast.solar-style JSON response.

    Uses UTC timestamps so offset from ``START`` (which is also UTC) is
    unambiguous when parsed back in the provider.
    """
    from datetime import timedelta

    watts = {}
    for i in range(24):
        dt = start + timedelta(hours=i)
        # forecast.solar returns naive strings in the requested tz;
        # for the test we keep UTC and rely on the provider's ZoneInfo("UTC") path
        power = ac_w if 7 <= i < 19 else 0.0
        watts[dt.strftime("%Y-%m-%d %H:%M:%S")] = power
    return {"result": {"watts": watts, "watt_hours": {}}}


# ── PVForecastAkkudoktor ──────────────────────────────────────────────────


class TestPVForecastAkkudoktor:
    def _make_provider(self) -> PVForecastAkkudoktor:
        planes = [
            PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0, inverter_pac_w=5000),
            PVPlaneConfig(peak_kw=3.0, tilt=20.0, azimuth=90.0, inverter_pac_w=3000),
        ]
        return PVForecastAkkudoktor(planes=planes, latitude=52.52, longitude=13.405)

    def test_provider_id(self):
        assert self._make_provider().provider_id == "PVForecastAkkudoktor"

    def test_url_contains_lat_lon(self):
        p = self._make_provider()
        url = p._build_url()
        assert "lat=52.52" in url
        assert "lon=13.405" in url

    def test_url_azimuth_conversion(self):
        """south=180° must become 0° for Akkudoktor."""
        p = PVForecastAkkudoktor(
            planes=[PVPlaneConfig(peak_kw=5.0, azimuth=180.0)],
            latitude=52.52,
            longitude=13.405,
        )
        assert "azimuth=0" in p._build_url()

    def test_url_azimuth_east(self):
        """east=90° must become -90° for Akkudoktor."""
        p = PVForecastAkkudoktor(
            planes=[PVPlaneConfig(peak_kw=5.0, azimuth=90.0)],
            latitude=52.52,
            longitude=13.405,
        )
        assert "azimuth=-90" in p._build_url()

    def test_url_two_planes(self):
        p = self._make_provider()
        url = p._build_url()
        # Each plane adds `power=`, verify at least 2 occurrences
        assert url.count("power=") >= 2

    def test_requires_at_least_one_plane(self):
        with pytest.raises(ValueError):
            PVForecastAkkudoktor(planes=[], latitude=52.0, longitude=13.0)

    @patch("requests.get")
    def test_fetch_hourly(self, mock_get):
        response_data = _akkudoktor_response(START, n_hours=48)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = self._make_provider()
        result = provider.fetch(START, END_24H, dt_hours=1.0)

        assert len(result) == 24
        # Night hours should be 0
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        # Day hours (6–17) should be > 0
        assert result[10] > 0.0

    @patch("requests.get")
    def test_fetch_quarter_hour(self, mock_get):
        response_data = _akkudoktor_response(START, n_hours=48)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = self._make_provider()
        result = provider.fetch(START, END_24H, dt_hours=0.25)
        assert len(result) == 96

    @patch("requests.get")
    def test_planes_summed(self, mock_get):
        """Both planes' ac power must be summed."""
        response_data = _akkudoktor_response(START, n_hours=24)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = self._make_provider()
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        # hour 10: both planes contribute 1000 * 0.6 + 1000 * 0.4 = 1000 W total
        assert result[10] == pytest.approx(1000.0, abs=1.0)

    @patch("requests.get")
    def test_no_negative_power(self, mock_get):
        """Values must be clamped to ≥ 0."""
        data = _akkudoktor_response(START, n_hours=24)
        # Inject a negative value in one plane
        data["values"][0][10]["power"] = -500.0
        data["values"][1][10]["power"] = -200.0
        mock_resp = MagicMock()
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = PVForecastAkkudoktor(
            planes=[_PLANE], latitude=52.52, longitude=13.405
        )
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        for v in result:
            assert v >= 0.0


# ── PVForecastSolar ───────────────────────────────────────────────────────


class TestPVForecastSolar:
    def _make_provider(self) -> PVForecastSolar:
        # Use UTC so that test datetime strings map 1:1 to array indices
        return PVForecastSolar(
            planes=[_PLANE],
            latitude=52.52,
            longitude=13.405,
            timezone_str="UTC",
        )

    def test_provider_id(self):
        assert self._make_provider().provider_id == "PVForecastSolar"

    def test_url_south_azimuth(self):
        """south=180° must become 0.0 for forecast.solar."""
        p = PVForecastSolar(
            planes=[PVPlaneConfig(peak_kw=5.0, tilt=30.0, azimuth=180.0)],
            latitude=52.52,
            longitude=13.405,
        )
        url = p._build_url(_PLANE)
        # URL path: /lat/lon/dec/az/kwp — az should be 0.0
        assert "/0.0/" in url

    def test_url_east_azimuth(self):
        """east=90° must become -90.0 for forecast.solar."""
        plane = PVPlaneConfig(peak_kw=3.0, tilt=20.0, azimuth=90.0)
        p = PVForecastSolar(planes=[plane], latitude=52.52, longitude=13.405)
        url = p._build_url(plane)
        assert "/-90.0/" in url

    def test_url_contains_lat_lon_kwp(self):
        plane = PVPlaneConfig(peak_kw=5.5, tilt=30.0, azimuth=180.0)
        p = PVForecastSolar(planes=[plane], latitude=48.1, longitude=11.6)
        url = p._build_url(plane)
        assert "48.1" in url
        assert "11.6" in url
        assert "5.500" in url

    def test_url_with_api_key(self):
        p = PVForecastSolar(
            planes=[_PLANE], latitude=52.52, longitude=13.405, api_key="MYKEY"
        )
        url = p._build_url(_PLANE)
        assert "/MYKEY/" in url

    def test_requires_at_least_one_plane(self):
        with pytest.raises(ValueError):
            PVForecastSolar(planes=[], latitude=52.0, longitude=13.0)

    @patch("requests.get")
    def test_fetch_hourly(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _forecastsolar_response(START, ac_w=2000.0)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = self._make_provider()
        result = provider.fetch(START, END_24H, dt_hours=1.0)

        assert len(result) == 24
        assert result[0] == pytest.approx(0.0)  # midnight
        assert result[6] == pytest.approx(0.0)  # 6:00 → not in 7–18
        assert result[10] == pytest.approx(2000.0)
        assert result[19] == pytest.approx(0.0)  # after sunset

    @patch("requests.get")
    def test_fetch_quarter_hour(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _forecastsolar_response(START, ac_w=1500.0)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = self._make_provider()
        result = provider.fetch(START, END_24H, dt_hours=0.25)
        assert len(result) == 96

    @patch("requests.get")
    def test_multi_plane_summed(self, mock_get):
        """Two planes: each returns 1000 W during the day → total 2000 W."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _forecastsolar_response(START, ac_w=1000.0)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        p = PVForecastSolar(
            planes=[
                PVPlaneConfig(peak_kw=3.0, azimuth=180.0),
                PVPlaneConfig(peak_kw=2.0, azimuth=90.0),
            ],
            latitude=52.52,
            longitude=13.405,
            timezone_str="UTC",
        )
        result = p.fetch(START, END_24H, dt_hours=1.0)
        # Two calls, each returning 1000 W at hour 10
        assert result[10] == pytest.approx(2000.0)
