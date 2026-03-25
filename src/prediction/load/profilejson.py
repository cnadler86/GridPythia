"""Load provider based on JSON profile data with hashed temp-file cache."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.prediction.load.provider import LoadProvider

DEFAULT_DATA_PATH: Path = Path(__file__).parent / "data" / "load_profiles.json"
_CACHE_STEP_HOURS = 0.25
_SLOTS_PER_DAY = int(24 / _CACHE_STEP_HOURS)
_SLOTS_PER_WEEK = 7 * _SLOTS_PER_DAY


class LoadProfileJSON(LoadProvider):
    """Build load forecast from JSON profile data and a hashed weekly cache.

    The input JSON is expected to include profile curves under ``profiles``.
    ``mean_wh`` values are interpreted as energy per source interval and converted
    to average power (W) using ``profile_dt_hours`` (or ``source_dt_hours``).
    """

    def __init__(
        self,
        data_path: Path | None = None,
        profile_dt_hours: float | None = None,
        use_vacation_profile: bool = False,
    ) -> None:
        self._data_path = data_path or DEFAULT_DATA_PATH
        self._profile_dt_override = profile_dt_hours
        self._use_vacation_profile = use_vacation_profile
        self._cached_values_w: list[float] | None = None
        self._cached_hash: str | None = None

    @property
    def provider_id(self) -> str:
        return "LoadProfileJSON"

    @property
    def cache_file(self) -> Path:
        return self._cache_path(self._input_hash())

    def _input_hash(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self._data_path.read_bytes())
        hasher.update(str(self._profile_dt_override).encode("ascii"))
        hasher.update(str(self._use_vacation_profile).encode("ascii"))
        return hasher.hexdigest()

    def _cache_path(self, digest: str) -> Path:
        return Path(tempfile.gettempdir()) / f"eos2_load_profile_week_15m_{digest}.json"

    def _load_input(self) -> dict[str, Any]:
        return json.loads(self._data_path.read_text(encoding="utf-8"))

    def _resolve_profile_dt_hours(self, data: dict[str, Any]) -> float:
        if self._profile_dt_override is not None:
            dt_hours = float(self._profile_dt_override)
        else:
            dt_hours = float(data.get("profile_dt_hours", data.get("source_dt_hours", 1.0)))
        if dt_hours <= 0.0:
            raise ValueError("profile_dt_hours must be > 0")
        return dt_hours

    def _select_profile(self, profiles: dict[str, Any], weekday: int) -> dict[str, Any]:
        if self._use_vacation_profile and "vacation" in profiles:
            selected = profiles["vacation"]
        else:
            key = "weekday" if weekday < 5 else "weekend"
            selected = profiles.get(key) or profiles.get("overall") or profiles.get("vacation")
        if not isinstance(selected, dict):
            raise ValueError("No usable profile data found in JSON")
        return selected

    def _as_power_values_w(self, profile: dict[str, Any], source_dt_hours: float) -> list[float]:
        mean_wh = profile.get("mean_wh")
        if not isinstance(mean_wh, list) or not mean_wh:
            raise ValueError("Profile must contain non-empty 'mean_wh' list")
        return [float(v) / source_dt_hours for v in mean_wh]

    @staticmethod
    def _interp(values: list[float], source_dt_hours: float, hour_of_day: float) -> float:
        if not values:
            return 0.0
        t = hour_of_day / source_dt_hours
        lo = int(t)
        frac = t - lo
        if lo < 0:
            return float(values[0])
        if lo >= len(values) - 1:
            return float(values[-1])
        return float(values[lo]) * (1.0 - frac) + float(values[lo + 1]) * frac

    def _build_week_values(self) -> list[float]:
        data = self._load_input()
        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError("JSON must contain a 'profiles' object")

        source_dt_hours = self._resolve_profile_dt_hours(data)
        week_values_w: list[float] = []
        for weekday in range(7):
            profile = self._select_profile(profiles, weekday)
            day_values_w = self._as_power_values_w(profile, source_dt_hours)
            for slot in range(_SLOTS_PER_DAY):
                hour_of_day = slot * _CACHE_STEP_HOURS
                week_values_w.append(self._interp(day_values_w, source_dt_hours, hour_of_day))
        return week_values_w

    def _ensure_cache(self) -> None:
        digest = self._input_hash()
        if self._cached_values_w is not None and self._cached_hash == digest:
            return

        cache_file = self._cache_path(digest)
        should_write_cache = False
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            values_w = payload.get("values_w")
            if not isinstance(values_w, list) or len(values_w) != _SLOTS_PER_WEEK:
                values_w = self._build_week_values()
                should_write_cache = True
        else:
            values_w = self._build_week_values()
            should_write_cache = True

        if should_write_cache:
            cache_file.write_text(
                json.dumps(
                    {
                        "cache_step_hours": _CACHE_STEP_HOURS,
                        "slots_per_week": _SLOTS_PER_WEEK,
                        "values_w": values_w,
                    }
                ),
                encoding="utf-8",
            )

        self._cached_values_w = [float(v) for v in values_w]
        self._cached_hash = digest

    def _from_cache(self, ts: datetime) -> float:
        values = self._cached_values_w
        if values is None:
            return 0.0

        seconds_of_week = (
            ts.weekday() * 86_400
            + ts.hour * 3_600
            + ts.minute * 60
            + ts.second
            + ts.microsecond / 1_000_000.0
        )
        idx = seconds_of_week / (_CACHE_STEP_HOURS * 3_600.0)
        lo = int(idx) % _SLOTS_PER_WEEK
        frac = idx - int(idx)
        hi = (lo + 1) % _SLOTS_PER_WEEK
        return values[lo] * (1.0 - frac) + values[hi] * frac

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        self._ensure_cache()
        ts_list: list[datetime] = timestamps.to_list()
        values = [self._from_cache(ts) for ts in ts_list]
        return pl.Series(values, dtype=pl.Float32)
