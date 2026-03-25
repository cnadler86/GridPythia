"""Tests for the unified Prediction orchestrator."""

from datetime import datetime, timezone

import polars as pl
import pytest

from src.prediction.base import make_timestamps
from src.prediction.electricprice.fixed import ElecPriceFixed
from src.prediction.feedintariff.fixed import FeedInTariffFixed
from src.prediction.load.fixed import LoadFixed
from src.prediction.prediction import Prediction, PredictionData, PredictionSetup
from src.prediction.pvforecast.import_ import PVForecastImport
from src.prediction.pvforecast.provider import PVForecastProvider
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

    async def test_fetch_hourly(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24, dt_hours=1.0)
        assert isinstance(data, PredictionData)
        assert data.steps == 24
        assert data.dt_hours == 1.0
        assert len(data["electricprice_eur_wh"]) == 24
        assert len(data["feedintariff_eur_wh"]) == 24
        assert len(data["load_w"]) == 24
        assert "pv_roof_default_w" in data.df.columns
        assert len(data["pv_roof_default_w"]) == 24
        assert "weather_temperature_c" in data.df.columns

    async def test_fetch_quarter_hour(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24, dt_hours=0.25)
        assert data.steps == 96
        assert len(data["electricprice_eur_wh"]) == 96
        assert len(data["load_w"]) == 96

    async def test_values_correct(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24, dt_hours=1.0)
        assert data["electricprice_eur_wh"][0] == pytest.approx(0.0003)
        assert data["feedintariff_eur_wh"][0] == pytest.approx(0.000082)
        assert data["load_w"][0] == pytest.approx(500.0)
        assert data["pv_roof_default_w"][0] == pytest.approx(0.0)
        assert data["pv_roof_default_w"][10] == pytest.approx(1000.0)

    async def test_multiple_pv_plants(self):
        setup = PredictionSetup(
            pv={
                "east": PVForecastImport(power_w=[500.0] * 24),
                "west": PVForecastImport(power_w=[300.0] * 24),
            },
        )
        pred = Prediction(setup)
        data = await pred.fetch(start=START, hours=24)
        assert "pv_east_default_w" in data.df.columns
        assert "pv_west_default_w" in data.df.columns
        assert data["pv_east_default_w"][0] == pytest.approx(500.0)
        assert data["pv_west_default_w"][0] == pytest.approx(300.0)

    async def test_no_providers_gives_zeros(self):
        pred = Prediction(PredictionSetup())
        data = await pred.fetch(start=START, hours=24)
        assert data.steps == 24
        assert all(v == 0.0 for v in data["electricprice_eur_wh"].to_list())
        assert all(v == 0.0 for v in data["load_w"].to_list())
        assert data.pv_names == []
        assert "weather_temperature_c" not in data.df.columns

    async def test_48_hours(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=48, dt_hours=1.0)
        assert data.steps == 48
        assert len(data["electricprice_eur_wh"]) == 48

    async def test_pv_names_property(self):
        setup = PredictionSetup(
            pv={
                "north": PVForecastImport(power_w=[0.0] * 24),
                "south": PVForecastImport(power_w=[0.0] * 24),
            }
        )
        data = await Prediction(setup).fetch(start=START, hours=24)
        assert set(data.pv_names) == {"north_default", "south_default"}

    async def test_multiple_inverters_per_provider(self):
        class MultiInverterPV(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "MultiInverterPV"

            async def fetch(self, timestamps: pl.Series) -> pl.Series:
                raise AssertionError("Prediction should use fetch_by_inverter for PV providers.")

            async def fetch_by_inverter(self, timestamps: pl.Series) -> dict[str, pl.Series]:
                steps = len(timestamps)
                return {
                    "inv1": pl.Series([100.0] * steps, dtype=pl.Float32),
                    "inv2": pl.Series([200.0] * steps, dtype=pl.Float32),
                }

        data = await Prediction(PredictionSetup(pv={"roof": MultiInverterPV()})).fetch(
            start=START,
            hours=24,
        )

        assert "pv_roof_inv1_w" in data.df.columns
        assert "pv_roof_inv2_w" in data.df.columns
        assert data["pv_roof_inv1_w"][0] == pytest.approx(100.0)
        assert data["pv_roof_inv2_w"][0] == pytest.approx(200.0)
        assert set(data.pv_names) == {"roof_inv1", "roof_inv2"}

    async def test_timestamps_property(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        ts = data.timestamps
        assert len(ts) == 24
        assert ts[0] == START

    async def test_getitem_returns_series(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        assert isinstance(data["load_w"], pl.Series)

    async def test_weather_columns_prefixed(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        # All weather columns should appear with 'weather_' prefix in df
        weather_cols = [c for c in data.df.columns if c.startswith("weather_")]
        assert "weather_temperature_c" in weather_cols
        assert "weather_cloud_cover_pct" in weather_cols

    async def test_df_is_polars_dataframe(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        assert isinstance(data.df, pl.DataFrame)
