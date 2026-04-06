"""Unified prediction orchestration and consumer-facing prediction contract."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from structlog import get_logger

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.electricprice.provider import ElecPriceProvider
from GridPythia.prediction.feedintariff.provider import FeedInTariffProvider
from GridPythia.prediction.load.provider import LoadProvider
from GridPythia.prediction.pvforecast.provider import PVForecastProvider
from GridPythia.prediction.weather.provider import WeatherProvider

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PredictionSolverView:
    """Dense numeric prediction view for solver-style consumers."""

    dt_hours: float
    load_wh: np.ndarray
    electricprice_eur_wh: np.ndarray
    feedintariff_eur_wh: np.ndarray
    pv_by_inverter: dict[str, np.ndarray]

    @property
    def steps(self) -> int:
        return int(self.load_wh.shape[0])


class PredictionData:
    """Aligned prediction channels with an explicit typed consumer API."""

    __slots__ = (
        "_timestamps",
        "dt_hours",
        "_load_wh",
        "_electricprice_eur_wh",
        "_feedintariff_eur_wh",
        "_pv_by_inverter",
        "_weather_by_channel",
        "_solver_view_cache",
    )

    def __init__(
        self,
        *,
        timestamps: Sequence[datetime] | None = None,
        dt_hours: float,
        load_wh: Sequence[float] | np.ndarray | None = None,
        electricprice_eur_wh: Sequence[float] | np.ndarray | None = None,
        feedintariff_eur_wh: Sequence[float] | np.ndarray | None = None,
        pv_by_inverter: Mapping[str, Sequence[float] | np.ndarray] | None = None,
        weather_by_channel: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    ) -> None:
        if dt_hours <= 0:
            raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
        if timestamps is None or load_wh is None:
            raise TypeError("PredictionData requires timestamps and load_wh")
        timestamps_tuple = tuple(timestamps)

        steps = len(timestamps_tuple)
        if steps == 0:
            raise ValueError("PredictionData requires at least one timestamp")

        self._timestamps = timestamps_tuple
        self.dt_hours = float(dt_hours)
        self._load_wh = self._coerce_series("load_wh", load_wh, steps)
        self._electricprice_eur_wh = self._coerce_optional_series(
            "electricprice_eur_wh", electricprice_eur_wh, steps
        )
        self._feedintariff_eur_wh = self._coerce_optional_series(
            "feedintariff_eur_wh", feedintariff_eur_wh, steps
        )
        self._pv_by_inverter = self._coerce_named_series_map(
            prefix="pv",
            data=pv_by_inverter,
            steps=steps,
        )
        self._weather_by_channel = self._coerce_named_series_map(
            prefix="weather",
            data=weather_by_channel,
            steps=steps,
        )
        self._solver_view_cache: dict[str, PredictionSolverView] = {}

    @staticmethod
    def _coerce_series(
        name: str,
        values: Sequence[float] | np.ndarray | None,
        expected_len: int,
    ) -> np.ndarray:
        if values is None:
            raise ValueError(f"Prediction array '{name}' is required")
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Prediction array '{name}' must be 1D, got shape {arr.shape}")
        if len(arr) != expected_len:
            raise ValueError(
                f"Prediction array '{name}' length mismatch: expected {expected_len}, got {len(arr)}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Prediction array '{name}' contains non-finite values")
        return np.ascontiguousarray(arr)

    @classmethod
    def _coerce_optional_series(
        cls,
        name: str,
        values: Sequence[float] | np.ndarray | None,
        expected_len: int,
    ) -> np.ndarray | None:
        if values is None:
            return None
        return cls._coerce_series(name, values, expected_len)

    @classmethod
    def _coerce_named_series_map(
        cls,
        *,
        prefix: str,
        data: Mapping[str, Sequence[float] | np.ndarray] | None,
        steps: int,
    ) -> dict[str, np.ndarray]:
        if not data:
            return {}

        out: dict[str, np.ndarray] = {}
        for name, values in data.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Prediction {prefix} series names must be non-empty strings")
            out[name] = cls._coerce_series(f"{prefix}_{name}", values, steps)
        return out

    @property
    def timestamps(self) -> list[datetime]:
        return list(self._timestamps)

    @property
    def steps(self) -> int:
        return len(self._timestamps)

    @property
    def load_wh(self) -> np.ndarray:
        return self._load_wh

    @property
    def electricprice(self) -> np.ndarray | None:
        return self._electricprice_eur_wh

    @property
    def feedintariff(self) -> np.ndarray | None:
        return self._feedintariff_eur_wh

    @property
    def pv_by_inverter(self) -> dict[str, np.ndarray]:
        return self._pv_by_inverter

    @property
    def weather_by_channel(self) -> dict[str, np.ndarray]:
        return self._weather_by_channel

    def to_solver_view(self, dtype: Any = np.float64) -> PredictionSolverView:
        """Return a cached dense numeric view optimized for repeated solver access."""
        dtype_obj = np.dtype(dtype)
        cache_key = dtype_obj.str
        cached = self._solver_view_cache.get(cache_key)
        if cached is not None:
            return cached

        zeros = np.zeros(self.steps, dtype=dtype_obj)
        solver_view = PredictionSolverView(
            dt_hours=self.dt_hours,
            load_wh=np.asarray(self._load_wh, dtype=dtype_obj),
            electricprice_eur_wh=(
                np.asarray(self._electricprice_eur_wh, dtype=dtype_obj)
                if self._electricprice_eur_wh is not None
                else zeros.copy()
            ),
            feedintariff_eur_wh=(
                np.asarray(self._feedintariff_eur_wh, dtype=dtype_obj)
                if self._feedintariff_eur_wh is not None
                else zeros.copy()
            ),
            pv_by_inverter={
                inverter_id: np.asarray(arr, dtype=dtype_obj)
                for inverter_id, arr in self._pv_by_inverter.items()
            },
        )
        self._solver_view_cache[cache_key] = solver_view
        return solver_view

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "timestamps": [ts.isoformat() for ts in self._timestamps],
            "dt_hours": float(self.dt_hours),
        }
        data["load_wh"] = np.asarray(self._load_wh, dtype=float).tolist()
        data["electricprice_eur_wh"] = (
            np.asarray(self._electricprice_eur_wh, dtype=float).tolist()
            if self._electricprice_eur_wh is not None
            else None
        )
        data["feedintariff_eur_wh"] = (
            np.asarray(self._feedintariff_eur_wh, dtype=float).tolist()
            if self._feedintariff_eur_wh is not None
            else None
        )
        for inverter_id, arr in self._pv_by_inverter.items():
            data[f"pv_{inverter_id}_wh"] = np.asarray(arr, dtype=float).tolist()
        for channel, arr in self._weather_by_channel.items():
            data[f"weather_{channel}"] = np.asarray(arr, dtype=float).tolist()
        return data


@dataclass(frozen=True)
class PredictionSetup:
    """Wire providers before calling :pymeth:`Prediction.fetch`.

    All fields are optional — omitted numeric core domains become zero arrays.
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

        pv_arrays: dict[str, np.ndarray] = {}

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
                if inverter in pv_arrays:
                    raise ValueError(
                        f"Duplicate PV inverter id '{inverter}' returned by multiple providers"
                    )
                pv_arrays[inverter] = self._validate_series(f"pv_{inverter}_wh", arr, n)

        logger.info(
            "prediction_fetch_complete",
            steps=n,
            columns=[
                "electricprice_eur_wh",
                "feedintariff_eur_wh",
                "load_wh",
                *[f"pv_{inverter}_wh" for inverter in pv_arrays],
                *(
                    []
                    if weather_dict is None
                    else [f"weather_{channel}" for channel in weather_dict]
                ),
            ],
        )

        return PredictionData(
            timestamps=timestamps,
            dt_hours=dt_hours,
            load_wh=load_wh,
            electricprice_eur_wh=eprice,
            feedintariff_eur_wh=ftariff,
            pv_by_inverter=pv_arrays,
            weather_by_channel=weather_dict,
        )
