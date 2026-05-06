"""Tests for the unified Prediction orchestrator."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
from GridPythia.prediction.electricprice.provider import ElecPriceProvider
from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed
from GridPythia.prediction.load.provider import LoadProvider
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

            async def fetch(self, timestamps: list) -> np.ndarray:
                raise AssertionError("Use fetch_by_inverter")

            async def fetch_by_inverter(self, timestamps: list) -> dict[str, np.ndarray]:
                values = [1000.0 if 6 <= ts.hour < 18 else 0.0 for ts in timestamps]
                return {"inverter1": np.array(values, dtype=np.float32)}

        class FixedWeather(WeatherProvider):
            @property
            def provider_id(self) -> str:
                return "FixedWeather"

            async def fetch(self, timestamps: list) -> dict[str, np.ndarray]:
                n = len(timestamps)
                return {
                    "temperature_c": np.full(n, 20.0, dtype=np.float32),
                    "cloud_cover_pct": np.full(n, 40.0, dtype=np.float32),
                }

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
        assert len(data.load_wh) == 24
        assert "inverter1" in data.pv_by_inverter
        assert len(data.pv_by_inverter["inverter1"]) == 24
        assert "temperature_c" in data.weather_by_channel

    async def test_fetch_quarter_hour(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24, dt_hours=0.25)
        assert data.steps == 96
        assert data.electricprice is not None
        assert len(data.electricprice) == 96
        assert len(data.load_wh) == 96

    async def test_values_correct(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24, dt_hours=1.0)
        assert data.electricprice is not None
        assert data.feedintariff is not None
        assert data.electricprice[0] == pytest.approx(0.0003)
        assert data.feedintariff[0] == pytest.approx(0.000082)
        # With dt_hours=1.0, energy is 0 Wh and 1000 Wh respectively
        assert data.pv_by_inverter["inverter1"][0] == pytest.approx(0.0)
        assert data.pv_by_inverter["inverter1"][10] == pytest.approx(1000.0)

    async def test_multiple_pv_plants(self):
        # Test multiple inverters from the same provider
        class MultiInverterPV(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "MultiInverterPV"

            async def fetch(self, timestamps: list) -> np.ndarray:
                raise AssertionError("Use fetch_by_inverter")

            async def fetch_by_inverter(self, timestamps: list) -> dict[str, np.ndarray]:
                steps = len(timestamps)
                return {
                    "east": np.full(steps, 500.0, dtype=np.float32),
                    "west": np.full(steps, 300.0, dtype=np.float32),
                }

        setup = PredictionSetup(
            pv={"roof": MultiInverterPV()},
        )
        pred = Prediction(setup)
        data = await pred.fetch(start=START, hours=24)
        assert set(data.pv_by_inverter) == {"east", "west"}
        # With dt_hours=1.0 (default), energy is 500 Wh and 300 Wh
        assert data.pv_by_inverter["east"][0] == pytest.approx(500.0)
        assert data.pv_by_inverter["west"][0] == pytest.approx(300.0)

    async def test_no_providers_gives_zeros(self):
        pred = Prediction(PredictionSetup())
        data = await pred.fetch(start=START, hours=24)
        assert data.steps == 24
        assert data.electricprice is not None
        assert np.all(data.electricprice == 0.0)
        assert np.all(data.load_wh == 0.0)
        assert not data.weather_by_channel

    async def test_48_hours(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=48, dt_hours=1.0)
        assert data.steps == 48
        assert data.electricprice is not None
        assert len(data.electricprice) == 48

    async def test_pv_by_inverter_keys(self):
        # Test multiple inverters from different providers
        class MultiInvProvider(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "MultiInvProvider"

            def __init__(self, inverter_ids: list[str]):
                self.inverter_ids = inverter_ids

            async def fetch(self, timestamps: list) -> np.ndarray:
                raise AssertionError("Use fetch_by_inverter")

            async def fetch_by_inverter(self, timestamps: list) -> dict[str, np.ndarray]:
                steps = len(timestamps)
                return {
                    inv_id: np.full(steps, 100.0, dtype=np.float32)
                    for inv_id in self.inverter_ids
                }

        setup = PredictionSetup(
            pv={"roof": MultiInvProvider(["north", "south"])}
        )
        data = await Prediction(setup).fetch(start=START, hours=24)
        assert set(data.pv_by_inverter) == {"north", "south"}

    async def test_multiple_inverters_per_provider(self):
        class MultiInverterPV(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "MultiInverterPV"

            async def fetch(self, timestamps: list) -> np.ndarray:
                raise AssertionError("Prediction should use fetch_by_inverter for PV providers.")

            async def fetch_by_inverter(self, timestamps: list) -> dict[str, np.ndarray]:
                steps = len(timestamps)
                return {
                    "inv1": np.full(steps, 100.0, dtype=np.float32),
                    "inv2": np.full(steps, 200.0, dtype=np.float32),
                }

        data = await Prediction(PredictionSetup(pv={"roof": MultiInverterPV()})).fetch(
            start=START,
            hours=24,
        )

        # With dt_hours=1.0 (default), energy is 100 Wh and 200 Wh
        assert data.pv_by_inverter["inv1"][0] == pytest.approx(100.0)
        assert data.pv_by_inverter["inv2"][0] == pytest.approx(200.0)
        assert set(data.pv_by_inverter) == {"inv1", "inv2"}

    async def test_timestamps_property(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        ts = data.timestamps
        assert len(ts) == 24
        assert ts[0] == START

    async def test_load_property_returns_ndarray(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        assert isinstance(data.load_wh, np.ndarray)

    async def test_weather_channels_are_exposed(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        assert set(data.weather_by_channel) == {"temperature_c", "cloud_cover_pct"}

    async def test_typed_prediction_api_exposes_weather_and_solver_view(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)

        assert set(data.weather_by_channel) == {"temperature_c", "cloud_cover_pct"}

        solver_view = data.to_solver_view()
        assert solver_view.steps == 24
        assert solver_view.load_wh.dtype == np.float64
        assert "inverter1" in solver_view.pv_by_inverter
        assert data.to_solver_view() is solver_view

    def test_predictiondata_supports_typed_constructor(self):
        timestamps = [START, START.replace(hour=1)]
        data = PredictionData(
            timestamps=timestamps,
            dt_hours=1.0,
            load_wh=np.array([100.0, 200.0], dtype=np.float32),
            electricprice_eur_wh=np.array([0.1, 0.2], dtype=np.float32),
            feedintariff_eur_wh=np.array([0.05, 0.05], dtype=np.float32),
            pv_by_inverter={"inv1": np.array([10.0, 20.0], dtype=np.float32)},
            weather_by_channel={"temperature_c": np.array([21.0, 22.0], dtype=np.float32)},
        )

        assert data.pv_by_inverter["inv1"][1] == pytest.approx(20.0)
        assert data.weather_by_channel["temperature_c"][0] == pytest.approx(21.0)
        assert data.electricprice is not None
        assert data.electricprice[1] == pytest.approx(0.2)

    async def test_weather_api_returns_dict(self):
        pred = self._make_prediction()
        data = await pred.fetch(start=START, hours=24)
        assert isinstance(data.weather_by_channel, dict)

    async def test_missing_pv_returns_empty_mapping(self):
        class SingleInverterPV(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "SingleInverterPV"

            async def fetch(self, timestamps: list) -> np.ndarray:
                return np.full(len(timestamps), 100.0, dtype=np.float32)

        setup = PredictionSetup(
            electricprice=ElecPriceFixed(price_kwh=0.30),
            pv={"roof": SingleInverterPV()},
        )
        pred = Prediction(setup)
        data = await pred.fetch(start=START, hours=24)

        assert "inverter1" in data.pv_by_inverter
        assert data.pv_by_inverter.get("nonexistent") is None

    async def test_timezone_is_converted_to_utc_for_providers(self):
        class CapturePrice(ElecPriceProvider):
            def __init__(self) -> None:
                self.seen_tzinfo = None

            @property
            def provider_id(self) -> str:
                return "CapturePrice"

            async def fetch(self, timestamps: list) -> np.ndarray:
                first = timestamps[0]
                self.seen_tzinfo = first.tzinfo
                return np.zeros(len(timestamps), dtype=np.float32)

        provider = CapturePrice()
        pred = Prediction(PredictionSetup(electricprice=provider))
        start_local = datetime(2025, 6, 15, 2, 0, tzinfo=ZoneInfo("Europe/Berlin"))
        data = await pred.fetch(start=start_local, hours=2, dt_hours=1.0)

        assert str(provider.seen_tzinfo) == "UTC"
        assert str(data.timestamps[0].tzinfo) in {"Europe/Berlin", "CEST", "CET"}

    async def test_timezone_contract_load_keeps_original_timezone(self):
        class CapturePrice(ElecPriceProvider):
            def __init__(self) -> None:
                self.seen_tzinfo = None

            @property
            def provider_id(self) -> str:
                return "CapturePrice"

            async def fetch(self, timestamps: list) -> np.ndarray:
                self.seen_tzinfo = timestamps[0].tzinfo
                return np.zeros(len(timestamps), dtype=np.float32)

        class CaptureLoad(LoadProvider):
            def __init__(self) -> None:
                super().__init__()
                self.seen_tzinfo = None

            @property
            def provider_id(self) -> str:
                return "CaptureLoad"

            def _get_day_profile_w(self, day_type):
                return [0.0] * 24, 1.0

            async def fetch(self, timestamps: list, *, use_vacation_profile: bool = False) -> np.ndarray:
                self.seen_tzinfo = timestamps[0].tzinfo
                return np.zeros(len(timestamps), dtype=np.float32)

        price = CapturePrice()
        load = CaptureLoad()
        pred = Prediction(PredictionSetup(electricprice=price, load=load))
        start_local = datetime(2025, 6, 15, 2, 0, tzinfo=ZoneInfo("Europe/Berlin"))
        await pred.fetch(start=start_local, hours=2, dt_hours=1.0)

        assert str(price.seen_tzinfo) == "UTC"
        assert str(load.seen_tzinfo) in {"Europe/Berlin", "CEST", "CET"}

    async def test_prediction_rejects_length_mismatch_from_provider(self):
        class BadPrice(ElecPriceProvider):
            @property
            def provider_id(self) -> str:
                return "BadPrice"

            async def fetch(self, timestamps: list) -> np.ndarray:
                return np.zeros(max(0, len(timestamps) - 1), dtype=np.float32)

        pred = Prediction(PredictionSetup(electricprice=BadPrice()))
        with pytest.raises(ValueError, match="length mismatch"):
            await pred.fetch(start=START, hours=4, dt_hours=1.0)

    async def test_prediction_rejects_non_finite_values(self):
        class BadWeather(WeatherProvider):
            @property
            def provider_id(self) -> str:
                return "BadWeather"

            async def fetch(self, timestamps: list) -> dict[str, np.ndarray]:
                n = len(timestamps)
                arr = np.zeros(n, dtype=np.float32)
                arr[0] = np.nan
                return {"temperature_c": arr}

        pred = Prediction(PredictionSetup(weather=BadWeather()))
        with pytest.raises(ValueError, match="non-finite"):
            await pred.fetch(start=START, hours=4, dt_hours=1.0)

    async def test_prediction_rejects_duplicate_pv_inverter_ids(self):
        class PVOne(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "PVOne"

            async def fetch(self, timestamps: list) -> np.ndarray:
                return np.zeros(len(timestamps), dtype=np.float32)

            async def fetch_by_inverter(self, timestamps: list) -> dict[str, np.ndarray]:
                return {"shared": np.full(len(timestamps), 10.0, dtype=np.float32)}

        class PVTwo(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "PVTwo"

            async def fetch(self, timestamps: list) -> np.ndarray:
                return np.zeros(len(timestamps), dtype=np.float32)

            async def fetch_by_inverter(self, timestamps: list) -> dict[str, np.ndarray]:
                return {"shared": np.full(len(timestamps), 20.0, dtype=np.float32)}

        pred = Prediction(PredictionSetup(pv={"a": PVOne(), "b": PVTwo()}))
        with pytest.raises(ValueError, match="Duplicate PV inverter id"):
            await pred.fetch(start=START, hours=4, dt_hours=1.0)

    async def test_fetch_unaligned_start_aligns_grid_and_returns_full_slots(self):
        class FlatLoad(LoadProvider):
            @property
            def provider_id(self) -> str:
                return "FlatLoad"

            def _get_day_profile_w(self, day_type):
                return [0.0] * 24, 1.0

            async def fetch(self, timestamps: list, *, use_vacation_profile: bool = False) -> np.ndarray:
                return np.full(len(timestamps), 120.0, dtype=np.float32)

        class FlatPV(PVForecastProvider):
            @property
            def provider_id(self) -> str:
                return "FlatPV"

            async def fetch(self, timestamps: list) -> np.ndarray:
                return np.full(len(timestamps), 80.0, dtype=np.float32)

            async def fetch_by_inverter(self, timestamps: list) -> dict[str, np.ndarray]:
                return {"inverter1": np.full(len(timestamps), 80.0, dtype=np.float32)}

        pred = Prediction(
            PredictionSetup(
                electricprice=ElecPriceFixed(price_kwh=0.30),
                load=FlatLoad(),
                pv={"roof": FlatPV()},
            )
        )
        start = datetime(2025, 6, 15, 4, 5, tzinfo=timezone.utc)
        data = await pred.fetch(start=start, hours=48, dt_hours=0.25)

        assert data.requested_start == start
        assert data.timestamps[0] == datetime(2025, 6, 15, 4, 0, tzinfo=timezone.utc)
        assert data.timestamps[1] == datetime(2025, 6, 15, 4, 15, tzinfo=timezone.utc)

        assert data.load_wh[0] == pytest.approx(120.0)
        assert data.load_wh[1] == pytest.approx(120.0)
        assert data.pv_by_inverter["inverter1"][0] == pytest.approx(80.0)
        assert data.pv_by_inverter["inverter1"][1] == pytest.approx(80.0)
        assert data.electricprice is not None
        assert data.electricprice[0] == pytest.approx(0.0003)


    async def test_fetch_default_start_is_timezone_aware(self):
        pred = Prediction(PredictionSetup())
        data = await pred.fetch(hours=1, dt_hours=1.0)

        assert data.requested_start is not None
        assert data.requested_start.tzinfo is not None
        assert data.timestamps[0].tzinfo is not None

    def test_predictiondata_to_dict_includes_requested_start(self):
        requested_start = START + timedelta(minutes=5)
        data = PredictionData(
            requested_start=requested_start,
            timestamps=[START, START.replace(hour=1)],
            dt_hours=1.0,
            load_wh=np.array([1.0, 2.0], dtype=np.float32),
        )

        payload = data.to_dict()
        assert payload["requested_start"] == requested_start.isoformat()

    # ── TZ enforcement ────────────────────────────────────────────────

    async def test_fetch_rejects_naive_start(self):
        """fetch() must raise ValueError when given a naive (TZ-unaware) datetime."""
        pred = self._make_prediction()
        naive_start = datetime(2025, 6, 15, 11, 0)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            await pred.fetch(start=naive_start, hours=4, dt_hours=0.25)

    async def test_fetch_partial_rejects_naive_start(self):
        pred = self._make_prediction()
        naive_start = datetime(2025, 6, 15, 11, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            await pred.fetch_partial(start=naive_start, hours=4, dt_hours=0.25)

    # ── Slot alignment ────────────────────────────────────────────────

    async def test_aligned_start_exact_hours_coverage(self):
        """Aligned start + integer hours: n == hours/dt, last slot covers end exactly."""
        pred = self._make_prediction()
        start = datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc)  # aligned
        data = await pred.fetch(start=start, hours=8, dt_hours=0.25)

        # 8h / 0.25h = 32 slots; last timestamp at 11:00 + 31*15min = 18:45
        assert data.steps == 32
        assert data.timestamps[0] == start
        expected_last = start + timedelta(hours=7, minutes=45)
        assert data.timestamps[-1] == expected_last

    async def test_unaligned_start_covers_full_range(self):
        """Unaligned start: floored fetch start + extra slot to cover the range."""
        pred = self._make_prediction()
        # 11:18 → floor to 11:15; end = 19:18 → last slot start = 19:15
        start = datetime(2025, 6, 15, 11, 18, tzinfo=timezone.utc)
        data = await pred.fetch(start=start, hours=8, dt_hours=0.25)

        # Fetch starts at 11:15 (floored), covers up to 19:15 slot [→ 19:30]
        assert data.timestamps[0] == datetime(2025, 6, 15, 11, 15, tzinfo=timezone.utc)
        # Last timestamp = 19:15 (slot [19:15, 19:30) covers requested end 19:18)
        assert data.timestamps[-1] == datetime(2025, 6, 15, 19, 15, tzinfo=timezone.utc)

    async def test_unaligned_start_partial_end_gets_extra_slot(self):
        """When end falls inside a slot, that slot's start IS the last timestamp."""
        pred = self._make_prediction()
        # start=11:00 (aligned), hours=8.1 → end=19:06
        # Old code (round) would give last=18:45; new code (int floor) gives last=19:00
        start = datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc)
        data = await pred.fetch(start=start, hours=8.1, dt_hours=0.25)

        # end = 19:06; last slot containing it starts at 19:00
        assert data.timestamps[-1] == datetime(2025, 6, 15, 19, 0, tzinfo=timezone.utc)
        # The slot [19:00, 19:15) covers 19:06 ✓

    async def test_fetch_start_idx_for_unaligned_is_one(self):
        """Provider timestamps[0] is before the requested start for unaligned inputs."""
        pred = self._make_prediction()
        start = datetime(2025, 6, 15, 11, 7, tzinfo=timezone.utc)
        data = await pred.fetch(start=start, hours=4, dt_hours=0.25)

        # timestamps[0] = 11:00 (floor), timestamps[1] = 11:15 = ceil(11:07)
        assert data.timestamps[0] == datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc)
        assert data.timestamps[1] == datetime(2025, 6, 15, 11, 15, tzinfo=timezone.utc)
