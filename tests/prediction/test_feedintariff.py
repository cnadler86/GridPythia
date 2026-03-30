"""Tests for feed-in tariff providers."""

from datetime import datetime, timezone

import polars as pl
import pytest

from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed
from GridPythia.prediction.feedintariff.import_ import FeedInTariffImport

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)


def _ts(hours: float = 24, dt: float = 1.0) -> pl.Series:
    return make_timestamps(START, hours, dt)


class TestFeedInTariffFixed:
    async def test_constant_tariff(self):
        provider = FeedInTariffFixed(tariff_kwh=0.082)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[0] == pytest.approx(0.082 / 1000.0)
        assert all(v == pytest.approx(result[0]) for v in result.to_list())

    async def test_quarter_hour(self):
        provider = FeedInTariffFixed(tariff_kwh=0.082)
        result = await provider.fetch(_ts(dt=0.25))
        assert len(result) == 96

    async def test_provider_id(self):
        assert FeedInTariffFixed().provider_id == "FeedInTariffFixed"

    async def test_returns_polars_float32(self):
        result = await FeedInTariffFixed().fetch(_ts())
        assert isinstance(result, pl.Series)
        assert result.dtype == pl.Float32


class TestFeedInTariffImport:
    async def test_exact_length(self):
        tariffs = [0.00008] * 24
        provider = FeedInTariffImport(tariffs_wh=tariffs)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert list(result) == pytest.approx(tariffs)

    async def test_padding(self):
        tariffs = [0.00005, 0.00010]
        provider = FeedInTariffImport(tariffs_wh=tariffs)
        result = await provider.fetch(_ts())
        assert len(result) == 24
        assert result[23] == pytest.approx(0.00010)

    async def test_provider_id(self):
        assert FeedInTariffImport(tariffs_wh=[]).provider_id == "FeedInTariffImport"
