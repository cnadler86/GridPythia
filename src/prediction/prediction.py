"""Unified prediction orchestration."""

from array import array
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from src.prediction.base import make_array, n_steps
from src.prediction.electricprice.provider import ElecPriceProvider
from src.prediction.feedintariff.provider import FeedInTariffProvider
from src.prediction.load.provider import LoadProvider
from src.prediction.pvforecast.provider import PVForecastProvider
from src.prediction.weather.provider import WeatherData, WeatherProvider


@dataclass
class PredictionData:
    """All prediction data for a time window.

    Every array has *steps* entries.  Power values are in **watts**,
    prices / tariffs in **EUR / Wh**.
    """

    start: datetime
    dt_hours: float
    steps: int

    electricprice_per_wh: array
    feedintariff_per_wh: array
    load_power_w: array
    pv_power_w: dict[str, array]
    weather: Optional[WeatherData] = None


@dataclass
class PredictionSetup:
    """Wire providers before calling :pymethod:`Prediction.fetch`.

    All fields are optional — unprovided domains produce zero-filled arrays.
    *pv* is a dict mapping plant names to their forecast provider.
    """

    electricprice: Optional[ElecPriceProvider] = None
    feedintariff: Optional[FeedInTariffProvider] = None
    load: Optional[LoadProvider] = None
    pv: dict[str, PVForecastProvider] = field(default_factory=dict)
    weather: Optional[WeatherProvider] = None


class Prediction:
    """Configure once, then fetch aggregated predictions.

    Example::

        pred = Prediction(PredictionSetup(
            electricprice=ElecPriceFixed(price_kwh=0.30),
            feedintariff=FeedInTariffFixed(tariff_kwh=0.082),
            load=LoadFixed(power_w=500),
            pv={"roof": PVForecastImport(power_w=[0]*6 + [500]*12 + [0]*6)},
        ))
        data = pred.fetch(start=datetime.now(), hours=24, dt_hours=1.0)
    """

    def __init__(self, setup: PredictionSetup) -> None:
        self.setup = setup

    def fetch(
        self,
        start: datetime,
        hours: int | float,
        dt_hours: float = 1.0,
    ) -> PredictionData:
        """Fetch all prediction channels for the next *hours* from *start*."""
        end = start + timedelta(hours=hours)
        steps = n_steps(hours, dt_hours)

        eprice = (
            self.setup.electricprice.fetch(start, end, dt_hours)
            if self.setup.electricprice
            else make_array(size=steps)
        )
        ftariff = (
            self.setup.feedintariff.fetch(start, end, dt_hours)
            if self.setup.feedintariff
            else make_array(size=steps)
        )
        load = (
            self.setup.load.fetch(start, end, dt_hours)
            if self.setup.load
            else make_array(size=steps)
        )
        pv = {
            name: provider.fetch(start, end, dt_hours)
            for name, provider in self.setup.pv.items()
        }
        weather = (
            self.setup.weather.fetch(start, end, dt_hours)
            if self.setup.weather
            else None
        )

        return PredictionData(
            start=start,
            dt_hours=dt_hours,
            steps=steps,
            electricprice_per_wh=eprice,
            feedintariff_per_wh=ftariff,
            load_power_w=load,
            pv_power_w=pv,
            weather=weather,
        )
