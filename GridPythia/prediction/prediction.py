"""Unified prediction orchestration."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
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

    Channels:
    * ``electricprice_eur_wh`` — ``np.float32`` (EUR/Wh)
    * ``feedintariff_eur_wh`` — ``np.float32`` (EUR/Wh)
    * ``load_wh`` — ``np.float32`` (Wh, energy per timestep)
    * ``pv_{inverter_id}_wh`` — ``np.float32`` (Wh, energy per timestep) per inverter
    * ``weather_{channel}`` — ``np.float32`` for each weather channel

    Quick access via properties: ``data.load_wh``, ``data.electricprice``,
    ``data.feedintariff``.
    For PV: ``data.get_pv_series(inverter_id)`` or ``data.pv_by_inverter``.
    """

    _timestamps: list[datetime]
    _arrays: dict[str, np.ndarray]
    dt_hours: float = 0.0

    def __getitem__(self, key: str) -> np.ndarray:
        """Direct column access; prefer typed properties for public API."""
        return self._arrays[key]

    @property
    def columns(self) -> list[str]:
        """Names of all numeric data channels."""
        return list(self._arrays.keys())

    @property
    def timestamps(self) -> list[datetime]:
        return self._timestamps

    @property
    def steps(self) -> int:
        return len(self._timestamps)

    @property
    def load_wh(self) -> np.ndarray:
        """Load energy in Wh (integrated over dt_hours)."""
        return self._arrays["load_wh"]

    @property
    def electricprice(self) -> np.ndarray | None:
        """Electricity price in EUR/Wh, or None when not configured."""
        return self._arrays.get("electricprice_eur_wh")

    @property
    def feedintariff(self) -> np.ndarray | None:
        """Feed-in tariff in EUR/Wh, or None when not configured."""
        return self._arrays.get("feedintariff_eur_wh")

    @property
    def pv_by_inverter(self) -> dict[str, np.ndarray]:
        """Dict mapping inverter_id → PV energy array (Wh per step)."""
        return {
            k[len("pv_") : -len("_wh")]: v
            for k, v in self._arrays.items()
            if k.startswith("pv_") and k.endswith("_wh")
        }

    def get_pv_series(self, inverter_id: str) -> np.ndarray | None:
        """Return PV energy array (Wh) for *inverter_id*, or None if not found."""
        return self._arrays.get(f"pv_{inverter_id}_wh")

    @property
    def pv_names(self) -> list[str]:
        """PV inverter IDs (from ``pv_{inverter_id}_wh`` keys)."""
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
        data.load_wh  # → np.ndarray with energy in Wh
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
                from datetime import timezone as _utc_mod

                provider_timestamps = [ts.astimezone(_utc_mod.utc) for ts in timestamps]
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

        async def _zeros() -> np.ndarray:
            return np.zeros(n, dtype=np.float32)

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
            weather_dict: dict[str, np.ndarray] | None = rest[len(pv_names)]
        else:
            pv_series = rest
            weather_dict = None

        # Build the channel dict (numpy float32 arrays)
        arrays: dict[str, np.ndarray] = {
            "electricprice_eur_wh": eprice,
            "feedintariff_eur_wh": ftariff,
            "load_wh": load_wh,
        }

        # Add PV data: key format is pv_{inverter_id}_wh (energy in Wh)
        for _, series_by_inverter in zip(pv_names, pv_series, strict=False):
            for inverter, arr in series_by_inverter.items():
                arrays[f"pv_{inverter}_wh"] = arr

        if weather_dict is not None:
            for ch_name, arr in weather_dict.items():
                arrays[f"weather_{ch_name}"] = arr

        logger.info(
            "prediction_fetch_complete",
            steps=n,
            columns=list(arrays.keys()),
        )

        return PredictionData(_timestamps=timestamps, _arrays=arrays, dt_hours=dt_hours)
