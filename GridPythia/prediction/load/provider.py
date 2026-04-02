"""Load forecast provider interface."""

from __future__ import annotations

from abc import abstractmethod
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter1d

from GridPythia.prediction.base import PredictionProvider

if TYPE_CHECKING:
    from GridPythia.prediction.load.config import LoadProfileConfig


class DayType(Enum):
    """Day-type categories used to select load profiles."""

    WEEKDAY = "weekday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"
    WEEKEND = "weekend"
    VACATIONS = "vacations"


def day_type_for_date(
    d: date,
    country: str | None = None,
    subdivision: str | None = None,
) -> DayType:
    """Return the :class:`DayType` for *d*, optionally considering public holidays.

    When *country* is provided the ``holidays`` library is consulted.  If *d*
    is a public holiday it is treated as :attr:`DayType.SUNDAY`.  Otherwise
    the calendar weekday determines the type.

    Args:
        d:           The date to classify.
        country:     ISO-3166-1 alpha-2 country code (e.g. ``"DE"``).
        subdivision: Country-specific subdivision (e.g. ``"BW"``).

    Returns:
        The corresponding :class:`DayType`.
    """
    if country is not None:
        import holidays as hol

        kwargs: dict = {"years": d.year}
        if subdivision is not None:
            kwargs["subdiv"] = subdivision
        country_holidays = hol.country_holidays(country, **kwargs)
        if d in country_holidays:
            return DayType.SUNDAY

    wd = d.weekday()  # 0=Mon … 6=Sun
    if wd < 5:
        return DayType.WEEKDAY
    if wd == 5:
        return DayType.SATURDAY
    return DayType.SUNDAY


class LoadProvider(PredictionProvider):
    """Returns electrical load energy in Wh per time step.

    Subclasses only need to implement :meth:`_get_day_profile_w`.
    :meth:`fetch` and :meth:`get_profile_series` are provided by this base class.
    """

    def __init__(
        self,
        country: str | None = None,
        subdivision: str | None = None,
    ) -> None:
        self._country = country
        self._subdivision = subdivision

    @abstractmethod
    def _get_day_profile_w(self, day_type: DayType) -> tuple[list[float], float]:
        """Return ``(power_values_w, source_dt_hours)`` for one day of *day_type*.

        *power_values_w* covers exactly one day (24 h / source_dt_hours slots),
        with values already expressed as average power in **W** (not Wh).
        """
        ...

    async def fetch(self, timestamps: list, *, use_vacation_profile: bool = False) -> np.ndarray:
        """Return Float32 Series of Wh, same length as *timestamps*.

        Profile selection per timestamp:

        * When *use_vacation_profile* is ``True``, every slot uses the vacation
          profile regardless of weekday or holiday status.
        * When *country* is configured, public holidays map to
          :attr:`DayType.SUNDAY`.
        * Mon–Fri → :attr:`DayType.WEEKDAY`
        * Sat → :attr:`DayType.SATURDAY`
        * Sun → :attr:`DayType.SUNDAY`

        Day profiles are cached per :class:`DayType` within each call to avoid
        redundant work across timestamps that share the same day type.
        """
        ts_list = list(timestamps)
        if not ts_list:
            return np.zeros(0, dtype=np.float32)

        target_dt_h = (
            (ts_list[1] - ts_list[0]).total_seconds() / 3600.0 if len(ts_list) >= 2 else 1.0
        )

        # Cache per DayType – at most 5 entries, avoids repeated file/parse work
        profile_cache: dict[DayType, tuple[list[float], float]] = {}

        result: list[float] = []
        for ts in ts_list:
            dt = (
                DayType.VACATIONS
                if use_vacation_profile
                else day_type_for_date(ts.date(), self._country, self._subdivision)
            )
            if dt not in profile_cache:
                profile_cache[dt] = self._get_day_profile_w(dt)
            power_values_w, src_dt = profile_cache[dt]

            hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
            t = hour / src_dt
            lo = int(t) % len(power_values_w)
            frac = t - int(t)
            hi = (lo + 1) % len(power_values_w)
            pw = power_values_w[lo] * (1.0 - frac) + power_values_w[hi] * frac
            result.append(float(pw * target_dt_h))

        return np.array(result, dtype=np.float32)

    async def get_profile_series(
        self, day_timestamps: list, day_types: list[DayType]
    ) -> np.ndarray:
        """Return a concatenated load series for the requested day types.

        Args:
            day_timestamps: Timestamps for **one** representative day at the
                target resolution.  The step width is inferred from the first
                two entries.
            day_types: Ordered list of day types; one day is appended per
                entry.  Example: ``[DayType.WEEKDAY, DayType.WEEKEND]`` →
                two days returned.

        Returns:
            ``np.ndarray`` of ``float32`` of length
            ``len(day_timestamps) * len(day_types)`` with Wh energy values.

        Notes:
            Upsampling (e.g. 1 h source → 15 min target) is performed on the
            **entire** concatenated series via linear interpolation followed by
            a Gaussian smooth so that day-boundary transitions are seamless.
        """
        if not day_types:
            return np.empty(0, dtype=np.float32)

        ts_list: list = day_timestamps
        n_per_day = len(ts_list)
        if n_per_day == 0:
            return np.empty(0, dtype=np.float32)

        target_dt_h = (ts_list[1] - ts_list[0]).total_seconds() / 3600.0 if n_per_day >= 2 else 1.0

        # --- build source (power in W) for all requested days ---
        src_values_w: list[float] = []
        src_dt_h: float | None = None
        for dt in day_types:
            day_w, s_dt = self._get_day_profile_w(dt)
            if src_dt_h is None:
                src_dt_h = s_dt
            src_values_w.extend(day_w)

        if src_dt_h is None:
            src_dt_h = 1.0

        n_src = len(src_values_w)
        n_tgt = n_per_day * len(day_types)

        if n_src == 0:
            return np.zeros(n_tgt, dtype=np.float32)

        # --- interpolate the full concatenated series to target resolution ---
        src_arr = np.asarray(src_values_w, dtype=np.float64)

        if abs(src_dt_h - target_dt_h) < 1e-9:
            # Same resolution – no resampling needed
            tgt_arr = (
                src_arr[:n_tgt]
                if len(src_arr) >= n_tgt
                else np.pad(src_arr, (0, n_tgt - len(src_arr)), mode="edge")
            )
        else:
            # Interpolate on fractional indices
            src_idx = np.arange(n_src, dtype=np.float64)
            # Target sample positions in source units
            tgt_idx = np.linspace(0, n_src - 1, n_tgt)
            tgt_arr = np.interp(tgt_idx, src_idx, src_arr)

            # Smooth only when upsampling (fine target, coarse source)
            upsample_ratio = src_dt_h / target_dt_h
            if upsample_ratio > 1.0:
                # Gaussian sigma ≈ half the upsample ratio → gentle smoothing
                sigma = upsample_ratio / 2.0
                tgt_arr = gaussian_filter1d(tgt_arr, sigma=sigma)

        # Convert average power (W) → energy per target step (Wh)
        return (tgt_arr * target_dt_h).astype(np.float32)


def load_provider_from_config(cfg: LoadProfileConfig) -> LoadProvider:
    """Instantiate the correct :class:`LoadProvider` from a :class:`LoadProfileConfig`.

    CSV files map to :class:`~GridPythia.prediction.load.profilecsv.LoadProfileCSV`.
    """
    # Lazy imports to avoid circular dependency (subclasses import from this module).
    from GridPythia.prediction.load.profilecsv import LoadProfileCSV  # noqa: PLC0415

    return LoadProfileCSV(cfg)
