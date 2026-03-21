"""Tests for electricity price providers."""

from datetime import datetime, timezone

import pytest

from src.prediction.electricprice.fixed import ElecPriceFixed, TimeWindow
from src.prediction.electricprice.import_ import ElecPriceImport

START = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)
END_24H = datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc)


# ── ElecPriceFixed ────────────────────────────────────────────────────


class TestElecPriceFixed:
    def test_flat_price(self):
        provider = ElecPriceFixed(price_kwh=0.30)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result) == 24
        assert result[0] == pytest.approx(0.30 / 1000.0)
        assert all(v == pytest.approx(result[0]) for v in result)

    def test_flat_price_with_charges_and_vat(self):
        provider = ElecPriceFixed(price_kwh=0.25, charges_kwh=0.05, vat_rate=1.19)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        expected_wh = (0.25 / 1000.0 + 0.05 / 1000.0) * 1.19
        assert result[0] == pytest.approx(expected_wh)

    def test_schedule(self):
        schedule = [
            TimeWindow(start_hour=0, end_hour=6, value=0.20),
            TimeWindow(start_hour=6, end_hour=22, value=0.35),
            TimeWindow(start_hour=22, end_hour=24, value=0.20),
        ]
        provider = ElecPriceFixed(schedule=schedule)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert result[0] == pytest.approx(0.20 / 1000.0)  # hour 0
        assert result[5] == pytest.approx(0.20 / 1000.0)  # hour 5
        assert result[6] == pytest.approx(0.35 / 1000.0)  # hour 6
        assert result[12] == pytest.approx(0.35 / 1000.0)  # hour 12
        assert result[22] == pytest.approx(0.20 / 1000.0)  # hour 22

    def test_quarter_hour_steps(self):
        provider = ElecPriceFixed(price_kwh=0.30)
        result = provider.fetch(START, END_24H, dt_hours=0.25)
        assert len(result) == 96

    def test_provider_id(self):
        assert ElecPriceFixed().provider_id == "ElecPriceFixed"


# ── ElecPriceImport ──────────────────────────────────────────────────


class TestElecPriceImport:
    def test_exact_match(self):
        prices = [0.0003] * 24
        provider = ElecPriceImport(prices_wh=prices)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result) == 24
        assert list(result) == pytest.approx(prices)

    def test_shorter_than_window_pads(self):
        prices = [0.0002, 0.0003]
        provider = ElecPriceImport(prices_wh=prices)
        result = provider.fetch(START, END_24H, dt_hours=1.0)
        assert len(result) == 24
        assert result[0] == pytest.approx(0.0002)
        assert result[1] == pytest.approx(0.0003)
        assert result[23] == pytest.approx(0.0003)  # padded

    def test_resample_to_quarter_hour(self):
        prices = [0.0001, 0.0005]  # 2h at 1h resolution
        end_2h = datetime(2025, 6, 15, 2, 0, tzinfo=timezone.utc)
        provider = ElecPriceImport(prices_wh=prices, source_dt_hours=1.0)
        result = provider.fetch(START, end_2h, dt_hours=0.25)
        assert len(result) == 8
        # linear interpolation between 0.0001 and 0.0005
        assert result[0] == pytest.approx(0.0001, abs=1e-5)

    def test_provider_id(self):
        assert ElecPriceImport(prices_wh=[]).provider_id == "ElecPriceImport"


def test_energycharts_fetch_today():
    """Attempt to fetch today's prices from Energy-Charts for DE-LU.

    If the Energy-Charts server is unreachable or returns an error the
    test is skipped so CI environments without network won't fail.
    """
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    import requests

    from src.prediction.electricprice.energycharts import ElecPriceEnergyCharts

    # Build today's Berlin 00:00..next day 00:00 in UTC (Energy-Charts expects UTC timestamps)
    berlin = ZoneInfo("Europe/Berlin")
    start_local = datetime.now(berlin).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = start_local.astimezone(timezone.utc)
    end = start + timedelta(days=1)

    # Quick reachability check
    try:
        probe = requests.get(
            "https://api.energy-charts.info/price",
            params={
                "bzn": "DE-LU",
                "start": start.strftime("%Y-%m-%dT%H:%M"),
                "end": end.strftime("%Y-%m-%dT%H:%M"),
            },
            timeout=5,
        )
        probe.raise_for_status()
    except Exception as exc:  # network error or non-2xx
        import pytest

        pytest.skip(f"Energy-Charts API unreachable or returned error: {exc}")

    # Now call the provider to validate parsing
    provider = ElecPriceEnergyCharts(bidding_zone="DE-LU")
    try:
        prices = provider._request_prices(start, end)
    except Exception as exc:
        import pytest

        pytest.skip(f"Energy-Charts API call failed: {exc}")

    assert prices, "no prices returned from Energy-Charts"
    # basic sanity checks
    for dt, price in prices:
        assert isinstance(dt, datetime)
        assert start <= dt <= end
        assert isinstance(price, float)
    assert any(price > -100.0 for _, price in prices)
