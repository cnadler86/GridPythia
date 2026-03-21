"""Tests for the unified Prediction orchestrator."""

from datetime import datetime, timezone

import pytest

from src.prediction.electricprice.fixed import ElecPriceFixed
from src.prediction.feedintariff.fixed import FeedInTariffFixed
from src.prediction.load.fixed import LoadFixed
from src.prediction.prediction import Prediction, PredictionData, PredictionSetup
from src.prediction.pvforecast.import_ import PVForecastImport
from src.prediction.weather.import_ import WeatherImport

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


class TestPrediction:
    def _make_prediction(self) -> Prediction:
        pv_profile = [0.0] * 6 + [1000.0] * 12 + [0.0] * 6
        weather_data = {
            "temperature_c": [20.0] * 24,
            "cloud_cover_pct": [40.0] * 24,
        }
        setup = PredictionSetup(
            electricprice=ElecPriceFixed(price_kwh=0.30),
            feedintariff=FeedInTariffFixed(tariff_kwh=0.082),
            load=LoadFixed(power_w=500.0),
            pv={"roof": PVForecastImport(power_w=pv_profile)},
            weather=WeatherImport(data=weather_data),
        )
        return Prediction(setup)

    def test_fetch_hourly(self):
        pred = self._make_prediction()
        data = pred.fetch(start=START, hours=24, dt_hours=1.0)
        assert isinstance(data, PredictionData)
        assert data.steps == 24
        assert data.dt_hours == 1.0
        assert len(data.electricprice_per_wh) == 24
        assert len(data.feedintariff_per_wh) == 24
        assert len(data.load_power_w) == 24
        assert "roof" in data.pv_power_w
        assert len(data.pv_power_w["roof"]) == 24
        assert data.weather is not None
        assert len(data.weather.temperature_c) == 24

    def test_fetch_quarter_hour(self):
        pred = self._make_prediction()
        data = pred.fetch(start=START, hours=24, dt_hours=0.25)
        assert data.steps == 96
        assert len(data.electricprice_per_wh) == 96
        assert len(data.load_power_w) == 96

    def test_values_correct(self):
        pred = self._make_prediction()
        data = pred.fetch(start=START, hours=24, dt_hours=1.0)
        # Price: 0.30 EUR/kWh = 0.0003 EUR/Wh
        assert data.electricprice_per_wh[0] == pytest.approx(0.0003)
        # Tariff: 0.082 EUR/kWh = 0.000082 EUR/Wh
        assert data.feedintariff_per_wh[0] == pytest.approx(0.000082)
        # Load: 500 W
        assert data.load_power_w[0] == pytest.approx(500.0)
        # PV: 0 at night, 1000 during day
        assert data.pv_power_w["roof"][0] == pytest.approx(0.0)
        assert data.pv_power_w["roof"][10] == pytest.approx(1000.0)

    def test_multiple_pv_plants(self):
        setup = PredictionSetup(
            pv={
                "east": PVForecastImport(power_w=[500.0] * 24),
                "west": PVForecastImport(power_w=[300.0] * 24),
            },
        )
        pred = Prediction(setup)
        data = pred.fetch(start=START, hours=24)
        assert "east" in data.pv_power_w
        assert "west" in data.pv_power_w
        assert data.pv_power_w["east"][0] == pytest.approx(500.0)
        assert data.pv_power_w["west"][0] == pytest.approx(300.0)

    def test_no_providers_gives_zeros(self):
        pred = Prediction(PredictionSetup())
        data = pred.fetch(start=START, hours=24)
        assert data.steps == 24
        assert all(v == 0.0 for v in data.electricprice_per_wh)
        assert all(v == 0.0 for v in data.load_power_w)
        assert len(data.pv_power_w) == 0
        assert data.weather is None

    def test_48_hours(self):
        pred = self._make_prediction()
        data = pred.fetch(start=START, hours=48, dt_hours=1.0)
        assert data.steps == 48
        assert len(data.electricprice_per_wh) == 48
