"""Tests for feed-in tariff providers."""

from datetime import datetime, timezone

import numpy as np
import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


def _ts(hours: float = 24, dt: float = 1.0) -> list:
    return make_timestamps(START, hours, dt)


class TestFeedInTariffFixed:
    async def test_constant_tariff(self):
        provider = FeedInTariffFixed(tariff_kwh=0.082)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[0] == pytest.approx(0.082 / 1000.0)
        assert all(v == pytest.approx(result[0]) for v in result)

    async def test_quarter_hour(self):
        provider = FeedInTariffFixed(tariff_kwh=0.082)
        result = await provider.fetch(_ts(dt=0.25))
        assert len(result) == 96

    async def test_provider_id(self):
        assert FeedInTariffFixed().provider_id == "FeedInTariffFixed"

    async def test_returns_polars_float32(self):
        result = await FeedInTariffFixed().fetch(_ts())
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


