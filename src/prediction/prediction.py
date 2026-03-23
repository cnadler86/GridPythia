"""Unified prediction orchestration."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from src.prediction.base import make_timestamps
from src.prediction.electricprice.provider import ElecPriceProvider
from src.prediction.feedintariff.provider import FeedInTariffProvider
from src.prediction.load.provider import LoadProvider
from src.prediction.pvforecast.provider import PVForecastProvider
from src.prediction.weather.provider import WeatherProvider


@dataclass
class PredictionData:
    """All prediction channels aligned on a shared time axis.

    The internal :attr:`df` has the following columns:

    * ``timestamp`` — ``pl.Datetime``
    * ``electricprice_eur_wh`` — ``pl.Float32``
    * ``feedintariff_eur_wh`` — ``pl.Float32``
    * ``load_w`` — ``pl.Float32``
    * ``pv_{name}_w`` — ``pl.Float32`` for each registered PV plant
    * ``weather_{channel}`` — ``pl.Float32`` for each weather channel delivered
      by the weather provider (e.g. ``weather_temperature_c``)

    Quick access: ``data["load_w"]`` returns the corresponding ``pl.Series``.
    """

    df: pl.DataFrame
    dt_hours: float

    def __getitem__(self, key: str) -> pl.Series:
        return self.df[key]

    @property
    def timestamps(self) -> pl.Series:
        return self.df["timestamp"]

    @property
    def steps(self) -> int:
        return len(self.df)

    @property
    def pv_names(self) -> list[str]:
        """Plant names extracted from ``pv_{name}_w`` columns."""
        return [
            c.removeprefix("pv_").removesuffix("_w")
            for c in self.df.columns
            if c.startswith("pv_") and c.endswith("_w")
        ]


@dataclass
class PredictionSetup:
    """Wire providers before calling :pymeth:`Prediction.fetch`.

    All fields are optional — omitted domains produce zero-filled columns.
    *pv* maps plant names to their forecast provider.
    """

    electricprice: ElecPriceProvider | None = None
    feedintariff: FeedInTariffProvider | None = None
    load: LoadProvider | None = None
    pv: dict[str, PVForecastProvider] = field(default_factory=dict)
    weather: WeatherProvider | None = None


class Prediction:
    """Configure providers once, then fetch all channels in one async call.

    Example::

        pred = Prediction(PredictionSetup(
            electricprice=ElecPriceFixed(price_kwh=0.30),
            feedintariff=FeedInTariffFixed(tariff_kwh=0.082),
            load=LoadFixed(power_w=500),
            pv={"roof": PVForecastImport(power_w=[0]*6 + [500]*12 + [0]*6)},
        ))
        data = await pred.fetch(start=datetime.now(), hours=24, dt_hours=1.0)
        data["load_w"]  # → pl.Series
    """

    def __init__(self, setup: PredictionSetup) -> None:
        self.setup = setup

    async def fetch(
        self,
        start: datetime,
        hours: int | float,
        dt_hours: float = 1.0,
    ) -> PredictionData:
        """Fetch all prediction channels in parallel for the next *hours* from *start*."""
        timestamps = make_timestamps(start, hours, dt_hours)
        n = len(timestamps)

        async def _zeros() -> pl.Series:
            return pl.Series([0.0] * n, dtype=pl.Float32)

        # Build all coroutines; run in parallel with asyncio.gather
        eprice_coro = (
            self.setup.electricprice.fetch(timestamps) if self.setup.electricprice else _zeros()
        )
        ftariff_coro = (
            self.setup.feedintariff.fetch(timestamps) if self.setup.feedintariff else _zeros()
        )
        load_coro = self.setup.load.fetch(timestamps) if self.setup.load else _zeros()
        weather_coro = self.setup.weather.fetch(timestamps) if self.setup.weather else None

        pv_names = list(self.setup.pv)
        pv_coros = [self.setup.pv[name].fetch(timestamps) for name in pv_names]

        all_coros = [eprice_coro, ftariff_coro, load_coro] + pv_coros
        if weather_coro is not None:
            all_coros.append(weather_coro)

        results = await asyncio.gather(*all_coros)

        # Unpack results
        eprice, ftariff, load_w, *rest = results
        if weather_coro is not None:
            pv_series = rest[: len(pv_names)]
            weather_df: pl.DataFrame | None = rest[len(pv_names)]
        else:
            pv_series = rest
            weather_df = None

        # Build the unified DataFrame
        data: dict[str, pl.Series] = {
            "timestamp": timestamps,
            "electricprice_eur_wh": eprice,
            "feedintariff_eur_wh": ftariff,
            "load_w": load_w,
        }
        for name, series in zip(pv_names, pv_series):
            data[f"pv_{name}_w"] = series

        if weather_df is not None:
            for col_name in weather_df.columns:
                data[f"weather_{col_name}"] = weather_df[col_name]

        df = pl.DataFrame(data)

        return PredictionData(df=df, dt_hours=dt_hours)
