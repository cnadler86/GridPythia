"""Unified prediction orchestration."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

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

    def __post_init__(self) -> None:
        if self.dt_hours <= 0:
            raise ValueError(f"dt_hours must be > 0, got {self.dt_hours}")

        n = len(self._timestamps)
        for key, arr in list(self._arrays.items()):
            arr_np = np.asarray(arr, dtype=np.float32)
            if arr_np.ndim != 1:
                raise ValueError(f"Prediction array '{key}' must be 1D, got shape {arr_np.shape}")
            if len(arr_np) != n:
                raise ValueError(
                    f"Prediction array '{key}' length mismatch: expected {n}, got {len(arr_np)}"
                )
            if not np.all(np.isfinite(arr_np)):
                raise ValueError(f"Prediction array '{key}' contains non-finite values")
            self._arrays[key] = arr_np

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

    def to_dict(self) -> dict:
        """Serialize PredictionData to a JSON-serializable dict.

        Timestamps are ISO-formatted strings; numeric arrays are converted
        to native Python lists of floats.
        """
        return {
            "timestamps": [ts.isoformat() for ts in self._timestamps],
            "dt_hours": float(self.dt_hours),
            **{k: np.asarray(v, dtype=float).tolist() for k, v in self._arrays.items()},
        }


@dataclass(frozen=True)
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

    @staticmethod
    def _to_utc_timestamps(timestamps: list[datetime]) -> list[datetime]:
        """Normalize timestamps to UTC for internet-backed provider calls."""
        return [
            ts.astimezone(timezone.utc)
            if ts.tzinfo is not None
            else ts.replace(tzinfo=timezone.utc)
            for ts in timestamps
        ]

    @staticmethod
    def _validate_series(name: str, values: np.ndarray | list[float], n: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Prediction provider result '{name}' must be 1D, got {arr.shape}")
        if len(arr) != n:
            raise ValueError(
                f"Prediction provider result '{name}' length mismatch: expected {n}, got {len(arr)}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Prediction provider result '{name}' contains non-finite values")
        return arr

    async def fetch(
        self,
        start: datetime,
        hours: int | float,
        dt_hours: float = 1.0,
    ) -> PredictionData:
        """Fetch all prediction channels in parallel for the next *hours* from *start*.

        Timezone contract:
        - Internet-backed providers (electricity price, feed-in tariff, PV, weather)
          are always called with UTC timestamps.
        - Load providers are called with the original requested timestamps
          (load profiles are timezone-invariant by design).

        Returned timestamps keep the original requested timezone representation.
        """
        timestamps = make_timestamps(start, hours, dt_hours)
        internet_provider_timestamps = self._to_utc_timestamps(timestamps)
        load_provider_timestamps = timestamps
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

        eprice_coro = (
            self.setup.electricprice.fetch(internet_provider_timestamps)
            if self.setup.electricprice
            else _zeros()
        )
        ftariff_coro = (
            self.setup.feedintariff.fetch(internet_provider_timestamps)
            if self.setup.feedintariff
            else _zeros()
        )
        load_coro = self.setup.load.fetch(load_provider_timestamps) if self.setup.load else _zeros()
        weather_coro = (
            self.setup.weather.fetch(internet_provider_timestamps) if self.setup.weather else None
        )

        pv_names = list(self.setup.pv)
        pv_coros = [
            self.setup.pv[name].fetch_by_inverter(internet_provider_timestamps) for name in pv_names
        ]

        all_coros = [eprice_coro, ftariff_coro, load_coro] + pv_coros
        if weather_coro is not None:
            all_coros.append(weather_coro)

        results = await asyncio.gather(*all_coros)

        eprice, ftariff, load_wh, *rest = results
        eprice = self._validate_series("electricprice_eur_wh", eprice, n)
        ftariff = self._validate_series("feedintariff_eur_wh", ftariff, n)
        load_wh = self._validate_series("load_wh", load_wh, n)

        if weather_coro is not None:
            pv_series = rest[: len(pv_names)]
            weather_raw = rest[len(pv_names)]
            if not isinstance(weather_raw, Mapping):
                raise ValueError("Weather provider must return a mapping of channel -> 1D array")
            weather_dict: dict[str, np.ndarray] | None = {
                str(ch_name): self._validate_series(f"weather_{ch_name}", arr, n)
                for ch_name, arr in weather_raw.items()
            }
        else:
            pv_series = rest
            weather_dict = None

        arrays: dict[str, np.ndarray] = {
            "electricprice_eur_wh": eprice,
            "feedintariff_eur_wh": ftariff,
            "load_wh": load_wh,
        }

        for provider_name, series_by_inverter in zip(pv_names, pv_series, strict=False):
            if not isinstance(series_by_inverter, Mapping):
                raise ValueError(
                    f"PV provider '{provider_name}' must return a mapping of inverter_id -> 1D array"
                )
            for inverter, arr in series_by_inverter.items():
                if not isinstance(inverter, str) or not inverter:
                    raise ValueError(
                        f"PV provider '{provider_name}' returned invalid inverter id: {inverter!r}"
                    )
                key = f"pv_{inverter}_wh"
                if key in arrays:
                    raise ValueError(
                        f"Duplicate PV inverter id '{inverter}' returned by multiple providers"
                    )
                arrays[key] = self._validate_series(key, arr, n)

        if weather_dict is not None:
            for ch_name, arr in weather_dict.items():
                arrays[f"weather_{ch_name}"] = arr

        logger.info(
            "prediction_fetch_complete",
            steps=n,
            columns=list(arrays.keys()),
        )

        return PredictionData(_timestamps=timestamps, _arrays=arrays, dt_hours=dt_hours)
