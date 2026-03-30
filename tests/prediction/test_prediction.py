"""Tests for the unified Prediction orchestrator."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
from GridPythia.prediction.electricprice.provider import ElecPriceProvider
from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed
from GridPythia.prediction.prediction import Prediction, PredictionData, PredictionSetup
from GridPythia.prediction.pvforecast.provider import PVForecastProvider
from GridPythia.prediction.weather.provider import WeatherProvider

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


class TestPrediction:
    def _make_prediction(self) -> Prediction:
        class DaylightPV(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "DaylightPV"

            async def fetch(self, timestamps: pl.Series) -> pl.Series:
                raise AssertionError("Use fetch_by_inverter")

            async def fetch_by_inverter(self, timestamps: pl.Series) -> dict[str, pl.Series]:
                ts_list = timestamps.to_list()
                values = [1000.0 if 6 <= ts.hour < 18 else 0.0 for ts in ts_list]
                return {"inverter1": pl.Series(values, dtype=pl.Float32)}

        class FixedWeather(WeatherProvider):
            @property
            def provider_id(self) -> str:
                return "FixedWeather"

            async def fetch(self, timestamps: pl.Series) -> pl.DataFrame:
                n = len(timestamps)
                return pl.DataFrame(
                    {
                        "temperature_c": pl.Series([20.0] * n, dtype=pl.Float32),
                        "cloud_cover_pct": pl.Series([40.0] * n, dtype=pl.Float32),
                    }
                )

        setup = PredictionSetup(
            electricprice=ElecPriceFixed(price_kwh=0.30),
            feedintariff=FeedInTariffFixed(tariff_kwh=0.082),
            pv={"roof": DaylightPV()},
            weather=FixedWeather(),
        )
        return Prediction(setup)

    async def test_fetch_hourly(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24, dt_hours=1.0)
        assert isinstance(data, PredictionData)
        assert data.steps == 24
        assert data.dt_hours == 1.0
        assert data.electricprice is not None
        assert data.feedintariff is not None
        assert len(data.electricprice) == 24
        assert len(data.feedintariff) == 24
        assert len(data["load_wh"]) == 24
        # PV column name now only uses inverter_id: pv_{inverter_id}_wh (energy in Wh)
        assert "pv_inverter1_wh" in data.df.columns
        assert len(data["pv_inverter1_wh"]) == 24
        assert "weather_temperature_c" in data.df.columns

    async def test_fetch_quarter_hour(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24, dt_hours=0.25)
        assert data.steps == 96
        assert len(data["electricprice_eur_wh"]) == 96
        assert len(data["load_wh"]) == 96

    async def test_values_correct(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24, dt_hours=1.0)
        assert data.electricprice is not None
        assert data.feedintariff is not None
        assert data.electricprice[0] == pytest.approx(0.0003)
        assert data.feedintariff[0] == pytest.approx(0.000082)
        # With dt_hours=1.0, energy is 0 Wh and 1000 Wh respectively
        assert data["pv_inverter1_wh"][0] == pytest.approx(0.0)
        assert data["pv_inverter1_wh"][10] == pytest.approx(1000.0)

    async def test_multiple_pv_plants(self):
        # Test multiple inverters from the same provider
        class MultiInverterPV(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "MultiInverterPV"

            async def fetch(self, timestamps: pl.Series) -> pl.Series:
                raise AssertionError("Use fetch_by_inverter")

            async def fetch_by_inverter(self, timestamps: pl.Series) -> dict[str, pl.Series]:
                steps = len(timestamps)
                return {
                    "east": pl.Series([500.0] * steps, dtype=pl.Float32),
                    "west": pl.Series([300.0] * steps, dtype=pl.Float32),
                }

        setup = PredictionSetup(
            pv={"roof": MultiInverterPV()},
        )
        pred = Prediction(setup)
        data = await pred.fetch(start=START, hours=24)
        # Column names now only use inverter ID, with Wh suffix
        assert "pv_east_wh" in data.df.columns
        assert "pv_west_wh" in data.df.columns
        # With dt_hours=1.0 (default), energy is 500 Wh and 300 Wh
        assert data["pv_east_wh"][0] == pytest.approx(500.0)
        assert data["pv_west_wh"][0] == pytest.approx(300.0)

    async def test_no_providers_gives_zeros(self):
        pred = Prediction(PredictionSetup())
        data = await pred.fetch(start=START, hours=24)
        assert data.steps == 24
        assert all(v == 0.0 for v in data["electricprice_eur_wh"].to_list())
        assert all(v == 0.0 for v in data["load_wh"].to_list())
        assert "weather_temperature_c" not in data.df.columns

    async def test_48_hours(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=48, dt_hours=1.0)
        assert data.steps == 48
        assert len(data["electricprice_eur_wh"]) == 48

    async def test_pv_names_property(self):
        # Test multiple inverters from different providers
        class MultiInvProvider(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "MultiInvProvider"

            def __init__(self, inverter_ids: list[str]):
                self.inverter_ids = inverter_ids

            async def fetch(self, timestamps: pl.Series) -> pl.Series:
                raise AssertionError("Use fetch_by_inverter")

            async def fetch_by_inverter(self, timestamps: pl.Series) -> dict[str, pl.Series]:
                steps = len(timestamps)
                return {
                    inv_id: pl.Series([100.0] * steps, dtype=pl.Float32)
                    for inv_id in self.inverter_ids
                }

        setup = PredictionSetup(
            pv={"roof": MultiInvProvider(["north", "south"])}
        )
        data = await Prediction(setup).fetch(start=START, hours=24)
        # With new structure, pv_names only contains inverter IDs
        assert set(data.pv_names) == {"north", "south"}

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

        # Column names now only use inverter_id with Wh suffix: pv_{inverter_id}_wh
        assert "pv_inv1_wh" in data.df.columns
        assert "pv_inv2_wh" in data.df.columns
        # With dt_hours=1.0 (default), energy is 100 Wh and 200 Wh
        assert data["pv_inv1_wh"][0] == pytest.approx(100.0)
        assert data["pv_inv2_wh"][0] == pytest.approx(200.0)
        # pv_names now only returns the inverter IDs
        assert set(data.pv_names) == {"inv1", "inv2"}

    async def test_timestamps_property(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        ts = data.timestamps
        assert len(ts) == 24
        assert ts[0] == START

    async def test_getitem_returns_series(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        assert isinstance(data["load_wh"], pl.Series)

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

    async def test_missing_pv_returns_empty_dict(self):
        """Verify get_pv_series returns None for missing inverter."""
        class SingleInverterPV(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "SingleInverterPV"

            async def fetch(self, timestamps: pl.Series) -> pl.Series:
                return pl.Series([100.0] * len(timestamps), dtype=pl.Float32)

        setup = PredictionSetup(
            electricprice=ElecPriceFixed(price_kwh=0.30),
            pv={"roof": SingleInverterPV()},
        )
        pred = Prediction(setup)
        data = await pred.fetch(start=START, hours=24)

        # Existing inverter should return Series
        assert data.get_pv_series("inverter1") is not None
        # Non-existing inverter should return None
        assert data.get_pv_series("nonexistent") is None

    async def test_timezone_is_converted_to_utc_for_providers(self):
        class CapturePrice(ElecPriceProvider):
            def __init__(self) -> None:
                self.seen_tzinfo = None

            @property
            def provider_id(self) -> str:
                return "CapturePrice"

            async def fetch(self, timestamps: pl.Series) -> pl.Series:
                first = timestamps.to_list()[0]
                self.seen_tzinfo = first.tzinfo
                return pl.Series([0.0] * len(timestamps), dtype=pl.Float32)

        provider = CapturePrice()
        pred = Prediction(PredictionSetup(electricprice=provider))
        start_local = datetime(2025, 6, 15, 2, 0, tzinfo=ZoneInfo("Europe/Berlin"))
        data = await pred.fetch(start=start_local, hours=2, dt_hours=1.0)

        assert str(provider.seen_tzinfo) == "UTC"
        assert str(data.timestamps.to_list()[0].tzinfo) in {"Europe/Berlin", "CEST", "CET"}
