"""Load provider based on JSON profile data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from src.prediction.load.config import LoadProfileConfig
from src.prediction.load.provider import DayType, LoadProvider

DEFAULT_DATA_PATH: Path = Path(__file__).parent / "data" / "load_profiles.json"


class LoadProfileJSON(LoadProvider):
    """Build load forecast from a JSON profile file.

    The input JSON must contain a ``profiles`` object with at least a
    ``weekday`` or ``overall`` key.  Each profile value is a dict with a
    ``mean_wh`` list (energy in Wh per source time step).

    The source time-step is read from the JSON key ``profile_dt_hours`` /
    ``source_dt_hours`` (default 1.0).  The file is read once and kept in
    memory; no temporary files are written.

    Args:
        config: Provider configuration.  ``config.path`` points to the JSON
                file.
    """

    def __init__(self, config: LoadProfileConfig) -> None:
        super().__init__(country=config.country, subdivision=config.subdivision)
        self._data_path = config.path
        self._loaded_data: dict[str, Any] | None = None

    @property
    def provider_id(self) -> str:
        return "LoadProfileJSON"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_data_loaded(self) -> None:
        if self._loaded_data is None:
            self._loaded_data = json.loads(self._data_path.read_text(encoding="utf-8"))

    def _resolve_profile_dt_hours(self, data: dict[str, Any]) -> float:
        dt_hours = float(data.get("profile_dt_hours", data.get("source_dt_hours", 1.0)))
        if dt_hours <= 0.0:
            raise ValueError("profile_dt_hours must be > 0")
        return dt_hours

    def _as_power_values_w(self, profile: dict[str, Any], source_dt_hours: float) -> list[float]:
        mean_wh = profile.get("mean_wh")
        if not isinstance(mean_wh, list) or not mean_wh:
            raise ValueError("Profile must contain non-empty 'mean_wh' list")
        return [float(v) / source_dt_hours for v in mean_wh]

    # ------------------------------------------------------------------
    # LoadProvider implementation
    # ------------------------------------------------------------------

    def _get_day_profile_w(self, day_type: DayType) -> tuple[list[float], float]:
        """Return average power (W) for every source slot of *day_type*'s profile."""
        self._ensure_data_loaded()
        data = self._loaded_data
        if data is None:
            raise RuntimeError("Failed to load profile data")

        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError("JSON must contain a 'profiles' object")

        source_dt_hours = self._resolve_profile_dt_hours(data)

        if day_type == DayType.VACATIONS:
            profile = (
                profiles.get("vacation") or profiles.get("overall") or next(iter(profiles.values()))
            )
        elif day_type in (DayType.SATURDAY, DayType.SUNDAY, DayType.WEEKEND):
            profile = (
                profiles.get("weekend") or profiles.get("overall") or next(iter(profiles.values()))
            )
        else:  # WEEKDAY
            profile = (
                profiles.get("weekday") or profiles.get("overall") or next(iter(profiles.values()))
            )

        if not isinstance(profile, dict):
            raise ValueError(f"No usable profile for {day_type}")

        return self._as_power_values_w(
            cast(dict[str, Any], profile), source_dt_hours
        ), source_dt_hours

    # fetch and get_profile_series are inherited from LoadProvider
    # and use _get_day_profile_w above.
