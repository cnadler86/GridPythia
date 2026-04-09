from __future__ import annotations

from datetime import datetime, timezone

from GridPythia.prediction.weather.brightsky import WeatherBrightSky
from GridPythia.prediction.weather.openmeteo import WeatherOpenMeteo

from typing import Any

class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self, content_type=None) -> dict:
        return self._body


class _FakeSession:
    def __init__(self, body: dict, capture: dict[str, object]) -> None:
        self._body = body
        self._capture = capture

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, params: dict[str, object]) -> _FakeResponse:
        self._capture["url"] = url
        self._capture["params"] = params
        return _FakeResponse(self._body)


async def test_brightsky_maps_channels_and_converts_solar(monkeypatch) -> None:
    capture: dict[str, Any] = {}
    body = {
        "weather": [
            {
                "timestamp": "2025-06-15T00:00:00+00:00",
                "temperature": 20.0,
                "cloud_cover": 40.0,
                "wind_speed": 12.0,
                "relative_humidity": 55.0,
                "precipitation": 1.5,
                "pressure_msl": 1012.0,
                "solar": 360.0,
            },
            {"timestamp": "invalid"},
            {
                "timestamp": "2025-06-15T01:00:00+00:00",
                "temperature": 22.0,
                "cloud_cover": 20.0,
                "wind_speed": 10.0,
                "relative_humidity": 50.0,
                "precipitation": 0.0,
                "pressure_msl": 1010.0,
                "solar": 720.0,
            },
        ]
    }

    monkeypatch.setattr(
        "GridPythia.prediction.weather.brightsky.aiohttp.ClientSession",
        lambda timeout: _FakeSession(body, capture),
    )

    provider = WeatherBrightSky(latitude=48.0, longitude=8.0)
    timestamps = [
        datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 6, 15, 1, 0, tzinfo=timezone.utc),
    ]
    result = await provider.fetch(timestamps)

    assert provider.provider_id == "BrightSky"
    assert capture["url"] == "https://api.brightsky.dev/weather"
    assert isinstance(capture["params"], dict)
    assert capture["params"]["tz"] == "UTC"
    assert result["temperature_c"].tolist() == [20.0, 22.0]
    assert result["cloud_cover_pct"].tolist() == [40.0, 20.0]
    assert result["ghi_wm2"].tolist() == [100.0, 200.0]
    assert result["pressure_hpa"].tolist() == [1012.0, 1010.0]


async def test_openmeteo_maps_hourly_fields(monkeypatch) -> None:
    capture: dict[str, Any] = {}
    start = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 6, 15, 1, 0, tzinfo=timezone.utc)
    body = {
        "hourly": {
            "time": [int(start.timestamp()), int(end.timestamp())],
            "temperature_2m": [19.0, 21.0],
            "relative_humidity_2m": [60.0, 50.0],
            "cloud_cover": [70.0, 30.0],
            "wind_speed_10m": [11.0, 9.0],
            "precipitation": [0.2, 0.0],
            "pressure_msl": [1009.0, 1008.0],
            "shortwave_radiation": [120.0, 240.0],
            "direct_radiation": [80.0, 160.0],
            "diffuse_radiation": [40.0, 80.0],
        }
    }

    monkeypatch.setattr(
        "GridPythia.prediction.weather.openmeteo.aiohttp.ClientSession",
        lambda timeout: _FakeSession(body, capture),
    )

    provider = WeatherOpenMeteo(latitude=48.0, longitude=8.0)
    result = await provider.fetch([start, end])

    assert provider.provider_id == "OpenMeteo"
    assert capture["url"] == "https://api.open-meteo.com/v1/forecast"
    assert isinstance(capture["params"], dict)
    assert capture["params"]["timezone"] == "UTC"
    assert capture["params"]["forecast_days"] == 1
    assert result["temperature_c"].tolist() == [19.0, 21.0]
    assert result["humidity_pct"].tolist() == [60.0, 50.0]
    assert result["ghi_wm2"].tolist() == [120.0, 240.0]
    assert result["dni_wm2"].tolist() == [80.0, 160.0]
    assert result["dhi_wm2"].tolist() == [40.0, 80.0]