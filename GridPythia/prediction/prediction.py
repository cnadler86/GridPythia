"""Unified prediction orchestration."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl
from structlog import get_logger

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.electricprice.provider import ElecPriceProvider
from GridPythia.prediction.feedintariff.provider import FeedInTariffProvider
from GridPythia.prediction.load.provider import LoadProvider
from GridPythia.prediction.pvforecast.provider import PVForecastProvider
from GridPythia.prediction.weather.provider import WeatherProvider

logger = get_logger(__name__)


@dataclass
class PredictionData:
    """All prediction channels aligned on a shared time axis.

    The internal :attr:`_df` has the following columns:

    * ``timestamp`` — ``pl.Datetime``
    * ``electricprice_eur_wh`` — ``pl.Float32`` (EUR/Wh)
    * ``feedintariff_eur_wh`` — ``pl.Float32`` (EUR/Wh)
    * ``load_wh`` — ``pl.Float32`` (Wh, energy per timestep)
    * ``pv_{inverter_id}_wh`` — ``pl.Float32`` (Wh, energy per timestep) for each registered inverter with PV
    * ``weather_{channel}`` — ``pl.Float32`` for each weather channel delivered
      by the weather provider (e.g. ``weather_temperature_c``)

    Quick access via properties: ``data.load_wh``, ``data.electricprice``, ``data.feedintariff``.
    For PV: ``data.get_pv_series(inverter_id)`` or ``data.pv_by_inverter``.
    """

    _df: pl.DataFrame
    dt_hours: float = 0.0

    def __getitem__(self, key: str) -> pl.Series:
        """Direct column access for internal use; prefer properties for public API."""
        return self._df[key]

    @property
    def df(self) -> pl.DataFrame:
        """Read-only access to internal DataFrame for iteration/inspection only.

        Prefer using typed properties (load_wh, electricprice, etc) for direct access.
        """
        return self._df

    @property
    def timestamps(self) -> pl.Series:
        return self._df["timestamp"]

    @property
    def steps(self) -> int:
        return len(self._df)

    @property
    def load_wh(self) -> pl.Series:
        """Load energy in Wh (integrated over dt_hours)."""
        return self._df["load_wh"]

    @property
    def electricprice(self) -> pl.Series | None:
        """Electricity price in EUR/Wh.

        Returns None if the column is not present in the prediction data.
        """
        try:
            return self._df["electricprice_eur_wh"]
        except pl.exceptions.ColumnNotFoundError:
            return None

    @property
    def feedintariff(self) -> pl.Series | None:
        """Feed-in tariff in EUR/Wh.

        Returns None if the column is not present in the prediction data.
        """
        try:
            return self._df["feedintariff_eur_wh"]
        except pl.exceptions.ColumnNotFoundError:
            return None

    @property
    def pv_by_inverter(self) -> dict[str, pl.Series]:
        """Return dict mapping inverter_id to corresponding PV Series (in Wh).

        Useful for looking up PV forecast by inverter device ID.
        Returns empty dict if no PV columns present.
        """
        result = {}
        for col in self._df.columns:
            if col.startswith("pv_") and col.endswith("_wh"):
                inverter_id = col[len("pv_") : -len("_wh")]
                result[inverter_id] = self._df[col]
        return result

    def get_pv_series(self, inverter_id: str) -> pl.Series | None:
        """Get PV forecast Series in Wh for a specific inverter, or None if not found.

        Args:
            inverter_id: The inverter identifier.

        Returns:
            pl.Series with PV energy in Wh, or None if the inverter is not in the prediction.
        """
        col_name = f"pv_{inverter_id}_wh"
        try:
            return self._df[col_name]
        except pl.exceptions.ColumnNotFoundError:
            return None

    @property
    def pv_names(self) -> list[str]:
        """PV inverter IDs extracted from ``pv_{inverter_id}_wh`` columns.

        Deprecated: use pv_by_inverter.keys() instead.
        """
        return list(self.pv_by_inverter.keys())


@dataclass
class PredictionSetup:
    """Wire providers before calling :pymeth:`Prediction.fetch`.

    All fields are optional — omitted domains produce zero-filled columns.
    *pv* maps plant-name prefixes to their forecast provider.
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
            pv={"roof": my_pv_provider},
        ))
        data = await pred.fetch(start=datetime.now(), hours=24, dt_hours=1.0)
        data.load_wh  # → pl.Series with energy in Wh
    """

    def __init__(self, setup: PredictionSetup) -> None:
        self.setup = setup

    async def fetch(
        self,
        start: datetime,
        hours: int | float,
        dt_hours: float = 1.0,
    ) -> PredictionData:
        """Fetch all prediction channels in parallel for the next *hours* from *start*.

        Providers are always called with UTC timestamps when *start* is timezone-aware.
        Returned timestamps keep the original requested timezone.
        """
        timestamps = make_timestamps(start, hours, dt_hours)
        provider_timestamps = timestamps
        if start.tzinfo is not None:
            tz_name: str = getattr(start.tzinfo, "key", None) or str(start.tzinfo)
            if tz_name not in ("UTC", "utc"):
                provider_timestamps = timestamps.dt.convert_time_zone("UTC")
        n = len(timestamps)

        providers = []
        if self.setup.electricprice:
            providers.append(self.setup.electricprice.provider_id)
        if self.setup.feedintariff:
            providers.append(self.setup.feedintariff.provider_id)
        if self.setup.load:
            providers.append(self.setup.load.provider_id)
        for name in self.setup.pv:
            providers.append(self.setup.pv[name].provider_id)
        if self.setup.weather:
            providers.append(self.setup.weather.provider_id)

        logger.info(
            "prediction_fetch_start",
            start=start.isoformat(),
            hours=hours,
            dt_hours=dt_hours,
            steps=n,
            providers=providers,
        )

        async def _zeros() -> pl.Series:
            return pl.Series([0.0] * n, dtype=pl.Float32)

        # Build all coroutines; run in parallel with asyncio.gather
        eprice_coro = (
            self.setup.electricprice.fetch(provider_timestamps)
            if self.setup.electricprice
            else _zeros()
        )
        ftariff_coro = (
            self.setup.feedintariff.fetch(provider_timestamps)
            if self.setup.feedintariff
            else _zeros()
        )
        load_coro = self.setup.load.fetch(provider_timestamps) if self.setup.load else _zeros()
        weather_coro = self.setup.weather.fetch(provider_timestamps) if self.setup.weather else None

        pv_names = list(self.setup.pv)
        pv_coros = [self.setup.pv[name].fetch_by_inverter(provider_timestamps) for name in pv_names]

        all_coros = [eprice_coro, ftariff_coro, load_coro] + pv_coros
        if weather_coro is not None:
            all_coros.append(weather_coro)

        results = await asyncio.gather(*all_coros)

        # Unpack results
        eprice, ftariff, load_wh, *rest = results
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
            "load_wh": load_wh,
        }

        # Add PV data: column format is pv_{inverter_id}_wh (energy)
        for _, series_by_inverter in zip(pv_names, pv_series, strict=False):
            for inverter, series in series_by_inverter.items():
                # Only use inverter as key, not the provider name.
                data[f"pv_{inverter}_wh"] = series

        if weather_df is not None:
            for col_name in weather_df.columns:
                data[f"weather_{col_name}"] = weather_df[col_name]

        df = pl.DataFrame(data)

        logger.info(
            "prediction_fetch_complete",
            steps=len(df),
            columns=df.columns,
        )

        return PredictionData(_df=df, dt_hours=dt_hours)
