"""Tests for feed-in tariff providers."""

from datetime import datetime, timezone

import pytest

from src.prediction.feedintariff.fixed import FeedInTariffFixed
from src.prediction.feedintariff.import_ import FeedInTariffImport

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)
END_24H = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)


class TestFeedInTariffFixed:
    def test_constant_tariff(self):
        provider = FeedInTariffFixed(tariff_kwh=0.082)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result) == 24
        assert result[0] == pytest.approx(0.082 / 1000.0)
        assert all(v == pytest.approx(result[0]) for v in result)

    def test_quarter_hour(self):
        provider = FeedInTariffFixed(tariff_kwh=0.082)
        result = provider.fetch(START, END_24H, dt_hours=0.25)
        assert len(result) == 96

    def test_provider_id(self):
        assert FeedInTariffFixed().provider_id == "FeedInTariffFixed"


class TestFeedInTariffImport:
    def test_exact_length(self):
        tariffs = [0.00008] * 24
        provider = FeedInTariffImport(tariffs_wh=tariffs)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result) == 24
        assert list(result) == pytest.approx(tariffs)

    def test_padding(self):
        tariffs = [0.00005, 0.00010]
        provider = FeedInTariffImport(tariffs_wh=tariffs)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result) == 24
        assert result[23] == pytest.approx(0.00010)

    def test_provider_id(self):
        assert FeedInTariffImport(tariffs_wh=[]).provider_id == "FeedInTariffImport"
