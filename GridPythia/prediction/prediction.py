"""Unified prediction orchestration and consumer-facing prediction contract."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from structlog import get_logger

from GridPythia.prediction.base import floor_to_slot, make_timestamps
from GridPythia.prediction.electricprice.provider import ElecPriceProvider
from GridPythia.prediction.feedintariff.provider import FeedInTariffProvider
from GridPythia.prediction.load.provider import LoadProvider
from GridPythia.prediction.pvforecast.provider import PVForecastProvider
from GridPythia.prediction.weather.provider import WeatherProvider

logger = get_logger(__name__)


async def _return(value):
    """Trivial awaitable that immediately returns *value*."""
    return value


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
        "_requested_start",
        "_timestamps",
        "dt_hours",
        "_base_load_wh",
        "_load_wh",
        "_appliance_load_by_id",
        "_electricprice_eur_wh",
        "_feedintariff_eur_wh",
        "_pv_by_inverter",
        "_weather_by_channel",
        "_solver_view_cache",
    )

    def __init__(
        self,
        *,
        requested_start: datetime | None = None,
        timestamps: Sequence[datetime] | None = None,
        dt_hours: float,
        load_wh: Sequence[float] | np.ndarray | None = None,
        electricprice_eur_wh: Sequence[float] | np.ndarray | None = None,
        feedintariff_eur_wh: Sequence[float] | np.ndarray | None = None,
        pv_by_inverter: Mapping[str, Sequence[float] | np.ndarray] | None = None,
        weather_by_channel: Mapping[str, Sequence[float] | np.ndarray] | None = None,
        appliance_load_by_id: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    ) -> None:
        if dt_hours <= 0:
            raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
        if timestamps is None or load_wh is None:
            raise TypeError("PredictionData requires timestamps and load_wh")
        timestamps_tuple = tuple(timestamps)

        if requested_start is not None and requested_start.tzinfo is None:
            raise ValueError("requested_start must be timezone-aware")

        steps = len(timestamps_tuple)
        if steps == 0:
            raise ValueError("PredictionData requires at least one timestamp")

        self._requested_start = requested_start
        self._timestamps = timestamps_tuple
        self.dt_hours = float(dt_hours)
        self._base_load_wh = self._coerce_series("load_wh", load_wh, steps)
        self._appliance_load_by_id = self._coerce_named_series_map(
            prefix="appliance",
            data=appliance_load_by_id,
            steps=steps,
        )
        # Combined load = base profile + all appliance forecasts (used by solver)
        if self._appliance_load_by_id:
            combined = self._base_load_wh.copy()
            for arr in self._appliance_load_by_id.values():
                combined = combined + arr
            self._load_wh = np.ascontiguousarray(combined, dtype=np.float32)
        else:
            self._load_wh = self._base_load_wh
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
    def requested_start(self) -> datetime | None:
        return self._requested_start

    @property
    def timestamps(self) -> list[datetime]:
        return list(self._timestamps)

    @property
    def steps(self) -> int:
        return len(self._timestamps)

    @property
    def load_wh(self) -> np.ndarray:
        """Combined load (base profile + all active appliance forecasts)."""
        return self._load_wh

    @property
    def base_load_wh(self) -> np.ndarray:
        """Base household load profile without appliance forecasts."""
        return self._base_load_wh

    @property
    def appliance_load_by_id(self) -> dict[str, np.ndarray]:
        """Per-appliance load arrays snapped to the prediction grid."""
        return self._appliance_load_by_id

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

    def slice_from(self, start: datetime) -> "PredictionData":
        """Return a new PredictionData whose first timestamp is *start*.

        Finds the first timestamp in this prediction whose UTC epoch is >=
        *start*'s UTC epoch and returns all series from that index onward.
        All values stay mapped to their original timestamps (no re-indexing
        of prices, load, PV, etc.) – this is the timestamp-indexed slicing
        API that prevents off-by-one errors when the optimisation dispatch
        slot advances by one step.

        Args:
            start: Target start timestamp (timezone-aware recommended).
                   Compared to internal timestamps in UTC.

        Returns:
            A new :class:`PredictionData` starting at or after *start*.

        Raises:
            ValueError: If *start* lies beyond the last prediction timestamp.
        """
        if start.tzinfo is not None:
            start_epoch = start.astimezone(timezone.utc).timestamp()
        else:
            start_epoch = start.replace(tzinfo=timezone.utc).timestamp()

        idx = None
        for i, ts in enumerate(self._timestamps):
            if ts.tzinfo is not None:
                ts_epoch = ts.astimezone(timezone.utc).timestamp()
            else:
                ts_epoch = ts.replace(tzinfo=timezone.utc).timestamp()
            # 0.5 s grace period to absorb floating-point drift in epoch arithmetic
            if ts_epoch >= start_epoch - 0.5:
                idx = i
                break

        if idx is None:
            raise ValueError(
                f"slice_from: start {start.isoformat()} is beyond the last "
                f"prediction timestamp {self._timestamps[-1].isoformat()}"
            )

        if idx == 0:
            return self

        new_timestamps = list(self._timestamps[idx:])
        return PredictionData(
            requested_start=start,
            timestamps=new_timestamps,
            dt_hours=self.dt_hours,
            load_wh=self._base_load_wh[idx:],
            electricprice_eur_wh=(
                self._electricprice_eur_wh[idx:] if self._electricprice_eur_wh is not None else None
            ),
            feedintariff_eur_wh=(
                self._feedintariff_eur_wh[idx:] if self._feedintariff_eur_wh is not None else None
            ),
            pv_by_inverter={k: v[idx:] for k, v in self._pv_by_inverter.items()},
            weather_by_channel={k: v[idx:] for k, v in self._weather_by_channel.items()},
            appliance_load_by_id={k: v[idx:] for k, v in self._appliance_load_by_id.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "requested_start": (
                self._requested_start.isoformat() if self._requested_start is not None else None
            ),
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
    def _normalize_start(start: datetime | None) -> datetime:
        """Return a timezone-aware start datetime.

        Raises:
            ValueError: If *start* is a naive (timezone-unaware) datetime.
                        Pass e.g. ``datetime.now(tz=ZoneInfo('Europe/Berlin'))`` or
                        ``datetime.now(timezone.utc)``.
        """
        if start is None:
            return datetime.now().astimezone()
        if start.tzinfo is None:
            raise ValueError(
                "Prediction.fetch() requires a timezone-aware start datetime. "
                "Pass e.g. datetime.now(tz=ZoneInfo('Europe/Berlin')) or "
                "datetime.now(timezone.utc)."
            )
        return start

    @staticmethod
    def _build_aligned_timestamps(
        start: datetime,
        hours: int | float,
        dt_hours: float,
    ) -> tuple[list[datetime], int]:
        """Build grid-aligned timestamps spanning ``[floor(start), last_slot_start]``.

        The last slot is the largest slot that starts strictly before
        ``start + hours``.  This guarantees that the returned array covers the
        entire requested window without adding spurious extra slots when the
        window already falls on an exact grid boundary.

        Returns:
            timestamps: Aligned list from ``floor(start)`` onward.
            start_idx:  Index of the first complete slot (>= requested *start*).
                        0 when *start* is already aligned, 1 when it is not.
        """
        if dt_hours <= 0:
            raise ValueError(f"dt_hours must be > 0, got {dt_hours}")

        step_s = dt_hours * 3600.0
        start_epoch = start.timestamp()
        end_epoch = start_epoch + float(hours) * 3600.0

        aligned_start = floor_to_slot(start, dt_hours)
        floor_epoch = aligned_start.timestamp()

        # Number of full slots from aligned_start to (but not including) end.
        # Using int() (floor) instead of round() to avoid over-counting when
        # the end falls on an exact boundary.
        raw_end_steps = (end_epoch - floor_epoch) / step_s
        n_slots = int(raw_end_steps)
        if raw_end_steps - n_slots <= 1e-9 and n_slots > 0:
            # end is exactly on a boundary – last slot is the one before it
            n_slots -= 1
        n = n_slots + 1  # inclusive count of slot-start timestamps

        elapsed_in_slot = start_epoch - floor_epoch
        start_idx = 1 if elapsed_in_slot > 1e-9 else 0

        return make_timestamps(aligned_start, n * dt_hours, dt_hours), start_idx

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
    def _validate_series(
        name: str,
        values: np.ndarray | list[float] | dict[str, object] | object,
        n: int,
    ) -> np.ndarray:
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
        start: datetime | None = None,
        hours: int | float = 24,
        dt_hours: float = 1.0,
    ) -> PredictionData:
        """Fetch aligned prediction channels at grid-aligned timestamps.

        Alignment behavior:
        - Timestamps are aligned to the dt-grid (``:00, :15, :30, :45`` for 15 min).
        - The fetch window spans ``[floor(start), last_slot_before(start + hours)]``.
        - All energy values are for complete grid slots.

        Timezone contract:
        - *start* must be timezone-aware (raises ``ValueError`` otherwise).
        - *start* defaults to ``datetime.now().astimezone()`` when ``None``.
        - Internet-backed providers receive UTC timestamps.
        - Load providers receive the local-TZ-aligned timestamps
          (load profiles are date-indexed, not UTC-offset-dependent).
        """
        requested_start = self._normalize_start(start)
        timestamps, start_idx = self._build_aligned_timestamps(
            requested_start,
            hours,
            dt_hours,
        )
        n = len(timestamps)
        internet_provider_timestamps = self._to_utc_timestamps(timestamps)
        load_provider_timestamps = timestamps

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
            start=requested_start.isoformat(),
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
            requested_start=requested_start,
            timestamps=timestamps,
            dt_hours=dt_hours,
            load_wh=load_wh,
            electricprice_eur_wh=eprice,
            feedintariff_eur_wh=ftariff,
            pv_by_inverter=pv_arrays,
            weather_by_channel=weather_dict,
        )

    async def fetch_partial(
        self,
        start: datetime | None = None,
        hours: int | float = 24,
        dt_hours: float = 1.0,
    ) -> "tuple[PredictionData, dict[str, str]]":
        """Like :meth:`fetch` but tolerates individual provider failures.

        Each provider is fetched independently.  Failures are collected in
        *errors* (provider_id → error message) and the remaining channels
        are returned as a :class:`PredictionData` with zeros for the failed
        ones.

        Returns:
            ``(PredictionData, errors)`` where *errors* is empty on full success.
        """
        requested_start = self._normalize_start(start)
        timestamps, _ = self._build_aligned_timestamps(requested_start, hours, dt_hours)
        n = len(timestamps)
        internet_ts = self._to_utc_timestamps(timestamps)
        errors: dict[str, str] = {}

        async def _safe(coro, provider_id: str, fallback):
            try:
                return await coro
            except Exception as exc:  # noqa: BLE001
                errors[provider_id] = str(exc)
                logger.warning("provider_fetch_failed", provider=provider_id, error=str(exc))
                return fallback

        zeros = np.zeros(n, dtype=np.float32)

        eprice = await _safe(
            self.setup.electricprice.fetch(internet_ts)
            if self.setup.electricprice
            else _return(zeros),
            getattr(self.setup.electricprice, "provider_id", "electricprice"),
            zeros.copy(),
        )
        ftariff = await _safe(
            self.setup.feedintariff.fetch(internet_ts)
            if self.setup.feedintariff
            else _return(zeros),
            getattr(self.setup.feedintariff, "provider_id", "feedintariff"),
            zeros.copy(),
        )
        load_wh = await _safe(
            self.setup.load.fetch(timestamps) if self.setup.load else _return(zeros),
            getattr(self.setup.load, "provider_id", "load"),
            zeros.copy(),
        )

        pv_names = list(self.setup.pv)
        pv_arrays: dict[str, np.ndarray] = {}
        for name in pv_names:
            result = await _safe(
                self.setup.pv[name].fetch_by_inverter(internet_ts),
                self.setup.pv[name].provider_id,
                None,
            )
            if result is not None and isinstance(result, Mapping):
                for inv_id, arr in result.items():
                    pv_arrays[str(inv_id)] = np.asarray(arr, dtype=np.float32)

        weather_dict: dict[str, np.ndarray] | None = None
        if self.setup.weather:
            wraw = await _safe(
                self.setup.weather.fetch(internet_ts),
                self.setup.weather.provider_id,
                None,
            )
            if wraw is not None and isinstance(wraw, Mapping):
                weather_dict = {str(k): np.asarray(v, dtype=np.float32) for k, v in wraw.items()}

        # Load is required; if it failed, substitute zeros and keep error
        eprice = self._validate_series("electricprice_eur_wh", eprice, n)
        ftariff = self._validate_series("feedintariff_eur_wh", ftariff, n)
        try:
            load_wh = self._validate_series("load_wh", load_wh, n)
        except Exception:
            load_wh = zeros.copy()

        logger.info(
            "prediction_fetch_partial_complete",
            steps=n,
            failed=list(errors.keys()),
            ok_channels=[
                c
                for c in [
                    "electricprice",
                    "feedintariff",
                    "load",
                    *pv_arrays,
                    *(weather_dict or {}),
                ]
                if c not in errors
            ],
        )

        return PredictionData(
            requested_start=requested_start,
            timestamps=timestamps,
            dt_hours=dt_hours,
            load_wh=load_wh,
            electricprice_eur_wh=eprice,
            feedintariff_eur_wh=ftariff,
            pv_by_inverter=pv_arrays,
            weather_by_channel=weather_dict,
        ), errors
