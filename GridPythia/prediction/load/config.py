"""Pydantic v2 configuration model for load profile providers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, model_validator


class LoadProfileConfig(BaseModel):
    """Configuration for a file-based load profile provider.

    The file format is determined from the path suffix.

    Args:
        path:                 Path to the profile file (``.csv``).
        country:              ISO-3166-1 alpha-2 country code used to look up
                              public holidays (e.g. ``"DE"``).  When ``None``,
                              holiday detection is disabled.
        subdivision:          Country-specific subdivision code (e.g. ``"BW"``
                              for Baden-Württemberg).  Only used when *country*
                              is set.
    """

    path: Path
    country: str | None = None
    subdivision: str | None = None

    @model_validator(mode="after")
    def _validate_path_suffix(self) -> "LoadProfileConfig":
        suffix = self.path.suffix.lower()
        if suffix != ".csv":
            raise ValueError(f"Unsupported profile file extension: {suffix!r} (use .csv)")
        return self
