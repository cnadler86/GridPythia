"""Load provider backed by a CSV profile file.

Expected file format
--------------------
* **Column 0** – time-of-day labels.  Parseable as ``HH:MM`` or ``HH:MM:SS``.
  The step width is derived from the diff of consecutive entries and must be
  uniform (e.g. 1 h, 15 min, 10 min).
* **Required columns** (case-insensitive):

  * ``weekday`` – mean energy in Wh for the corresponding time slot on a weekday.
  * Either ``weekend`` **or** both ``saturday`` and ``sunday``.

* The **vacation** profile is computed automatically as the minimum Wh value
  found anywhere in the data, spread uniformly across all time slots.

All source values are treated as energy in Wh per source time step.
:meth:`fetch` and :meth:`get_profile_series` convert to the requested
target resolution.

When *country* (and optionally *subdivision*) are supplied, public holidays
are detected via the ``holidays`` library and automatically mapped to the
vacation profile.
"""

from __future__ import annotations

import csv
import io

import numpy as np
from structlog import get_logger

from GridPythia.prediction.load.config import LoadProfileConfig
from GridPythia.prediction.load.provider import DayType, LoadProvider

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# column-name aliases (lower-cased)
# ---------------------------------------------------------------------------
_COL_WEEKDAY = "weekday"
_COL_WEEKEND = "weekend"
_COL_SATURDAY = "saturday"
_COL_SUNDAY = "sunday"


class LoadProfileCSV(LoadProvider):
    """Build a load forecast from a CSV profile file.

    The first row must be the header; data starts on the second row.  Both
    comma-separated (``','``) and semicolon-separated (``';'``) files are
    supported.  For semicolon-delimited files the comma is treated as the
    decimal separator (German locale convention).

    Args:
        config: Provider configuration.  ``config.path`` must point to a
                ``.csv`` file.
    """

    def __init__(self, config: LoadProfileConfig) -> None:
        super().__init__(country=config.country, subdivision=config.subdivision)
        self._file_path = config.path
        self._profiles: dict[str, list[float]] | None = None
        self._source_dt_hours: float | None = None

    # ------------------------------------------------------------------
    # PredictionProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        return f"LoadProfileCSV({self._file_path.name})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Parse the CSV and populate ``_profiles`` and ``_source_dt_hours``."""
        raw = self._read_csv_stdlib()

        # Normalise column names
        col_data: dict[str, list[str]] = {k.strip().lower(): v for k, v in raw.items()}
        col_names = list(col_data.keys())

        # Identify time column (first column)
        time_col = col_names[0]
        dt_hours = self._infer_dt_hours(col_data[time_col])
        self._source_dt_hours = dt_hours

        n = len(col_data[time_col])
        expected_slots = round(24.0 / dt_hours)
        if n != expected_slots:
            raise ValueError(f"Expected {expected_slots} rows for dt={dt_hours} h, got {n}")

        profiles: dict[str, list[float]] = {}

        def _col(name: str) -> list[float]:
            if name not in col_data:
                raise ValueError(f"Required column '{name}' not found in {self._file_path.name}")
            return [float(v) for v in col_data[name]]

        profiles["weekday"] = _col(_COL_WEEKDAY)

        if _COL_SATURDAY in col_data and _COL_SUNDAY in col_data:
            profiles["saturday"] = _col(_COL_SATURDAY)
            profiles["sunday"] = _col(_COL_SUNDAY)
            # Derive weekend as mean of sat+sun for convenience
            profiles["weekend"] = [
                (s + su) / 2.0
                for s, su in zip(profiles["saturday"], profiles["sunday"], strict=True)
            ]
        elif _COL_WEEKEND in col_data:
            profiles["weekend"] = _col(_COL_WEEKEND)
            profiles["saturday"] = profiles["weekend"]
            profiles["sunday"] = profiles["weekend"]
        else:
            raise ValueError(
                f"{self._file_path.name}: need 'weekend' or 'saturday'+'sunday' column(s)"
            )

        # Vacation = constant minimum of all energy values across all profiles
        all_values = [v for vals in profiles.values() for v in vals]
        min_wh = float(np.min(all_values)) if all_values else 0.0
        profiles["vacations"] = [min_wh] * expected_slots

        self._profiles = profiles

        logger.info(
            "load_profile_loaded",
            file=self._file_path.name,
            source_dt_hours=dt_hours,
            slots=expected_slots,
            profiles=list(profiles.keys()),
        )

    def _read_csv_stdlib(self) -> dict[str, list[str]]:
        """Read CSV and return a column-keyed dict of string values.

        Delimiter is auto-detected from the header line: if a semicolon is
        present the file is treated as semicolon-separated with comma decimal
        separators (German locale); otherwise a comma delimiter with dot
        decimals is assumed.
        """
        try:
            text = self._file_path.read_text(encoding="utf-8-sig")
            first_line = text.split("\n", 1)[0]
            delimiter = ";" if ";" in first_line else ","

            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")
            rows = list(reader)

            if not rows:
                raise ValueError("CSV has no data rows")

            col_data: dict[str, list[str]] = {k: [] for k in rows[0]}
            for row in rows:
                for k, v in row.items():
                    col_data.setdefault(k, []).append(v or "")

            # Normalise decimal separator for German-locale files
            if delimiter == ";":
                col_data = {k: [v.replace(",", ".") for v in vals] for k, vals in col_data.items()}

            return col_data
        except Exception as exc:
            raise ValueError(f"Cannot read CSV '{self._file_path}': {exc}") from exc

    @staticmethod
    def _infer_dt_hours(time_col: list[str]) -> float:
        """Derive the uniform step width from the time-of-day column."""
        values = time_col
        if len(values) < 2:
            raise ValueError("Time column must have at least 2 entries")

        hours: list[float] = []
        for v in values:
            hours.append(LoadProfileCSV._parse_time_to_hours(str(v).strip()))

        diffs: list[float] = []
        for i in range(1, len(hours)):
            d = hours[i] - hours[i - 1]
            if d < 0:
                d += 24.0
            if d > 0:
                diffs.append(d)

        if not diffs:
            raise ValueError("Could not infer time step from time column")

        dt = float(np.median(diffs))
        if dt <= 0 or dt > 24:
            raise ValueError(f"Implausible time step: {dt} h")
        return dt

    @staticmethod
    def _parse_time_to_hours(s: str) -> float:
        """Parse ``HH:MM`` or ``HH:MM:SS`` (or decimal hours) to fractional hours."""
        parts = s.split(":")
        if len(parts) >= 2:
            h = float(parts[0])
            m = float(parts[1])
            sec = float(parts[2]) if len(parts) >= 3 else 0.0
            return h + m / 60.0 + sec / 3600.0
        return float(s)

    def _ensure_loaded(self) -> None:
        if self._profiles is None:
            self._load()

    def _profile_for(self, day_type: DayType) -> list[float]:
        if self._profiles is None:
            raise RuntimeError("Profiles not loaded; call _ensure_loaded() first")
        mapping = {
            DayType.WEEKDAY: "weekday",
            DayType.WEEKEND: "weekend",
            DayType.SATURDAY: "saturday",
            DayType.SUNDAY: "sunday",
            DayType.VACATIONS: "vacations",
        }
        key = mapping[day_type]
        return self._profiles[key]

    # ------------------------------------------------------------------
    # LoadProvider implementation
    # ------------------------------------------------------------------

    def _get_day_profile_w(self, day_type: DayType) -> tuple[list[float], float]:
        """Return ``(power_values_w, source_dt_hours)`` for *day_type*."""
        self._ensure_loaded()
        if self._source_dt_hours is None:
            raise RuntimeError("source_dt_hours not set; _load() must have failed")
        energy_wh = self._profile_for(day_type)
        power_w = [e / self._source_dt_hours for e in energy_wh]
        return power_w, self._source_dt_hours

    # fetch and get_profile_series are inherited from LoadProvider.
