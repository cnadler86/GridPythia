"""Tests for load forecast providers."""

from datetime import datetime, timezone

import polars as pl
import pytest

from src.prediction.base import make_timestamps

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


def _ts(hours: float = 24, dt: float = 1.0) -> pl.Series:
    return make_timestamps(START, hours, dt)
