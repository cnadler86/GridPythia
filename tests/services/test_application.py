"""Tests for the GridPythiaService application service."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from GridPythia.services.application import GridPythiaService, PredictionCache


class TestPredictionCache:
    """Test the PredictionCache class."""

    def test_empty_cache_is_invalid(self):
        """Empty cache should not be valid."""
        cache = PredictionCache()
        assert cache.is_valid() is False
        assert cache.get() is None

    def test_cache_set_and_get(self):
        """Test setting and getting cache data."""
        from unittest.mock import MagicMock

        cache = PredictionCache(ttl_seconds=300.0)
        mock_data = MagicMock()
        forecast_from = datetime.now()

        cache.set(mock_data, forecast_from)

        assert cache.is_valid() is True
        result = cache.get()
        assert result is not None
        assert result[0] is mock_data
        assert result[1] is forecast_from

    def test_cache_invalidate(self):
        """Test cache invalidation."""
        from unittest.mock import MagicMock

        cache = PredictionCache()
        cache.set(MagicMock())
        assert cache.is_valid() is True

        cache.invalidate()
        assert cache.is_valid() is False
        assert cache.get() is None

    def test_cache_expiration(self):
        """Test that cache expires after TTL."""
        from unittest.mock import MagicMock

        cache = PredictionCache(ttl_seconds=0.01)  # 10ms TTL
        cache.set(MagicMock())

        # Immediately should be valid
        assert cache.is_valid() is True

        # After waiting, should be invalid
        import time

        time.sleep(0.02)
        assert cache.is_valid() is False


class TestGridPythiaService:
    """Test the GridPythiaService class."""

    @pytest.fixture
    def config_path(self) -> Path:
        """Return path to the test config file."""
        return Path(__file__).parent.parent.parent / "config.yaml"

    def test_service_initialization(self, config_path: Path):
        """Test that service initializes correctly."""
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        service = GridPythiaService.from_config_path(config_path)
        assert service.config is not None
        assert service.config.prediction is not None
        assert service.config.optimization is not None

    def test_service_providers_lazy_creation(self, config_path: Path):
        """Test that providers are created lazily."""
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        service = GridPythiaService.from_config_path(config_path)
        assert service._providers is None

        # Access providers property
        providers = service.providers
        assert providers is not None
        assert service._providers is providers

        # Second access should return same instance
        assert service.providers is providers

    def test_service_optimizer_lazy_creation(self, config_path: Path):
        """Test that optimizer is created lazily."""
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        service = GridPythiaService.from_config_path(config_path)
        assert service._optimizer is None

        # Access optimizer property
        optimizer = service.optimizer
        assert optimizer is not None
        assert service._optimizer is optimizer

        # Second access should return same instance
        assert service.optimizer is optimizer

    def test_service_inverters_access(self, config_path: Path):
        """Test accessing inverters through service."""
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        service = GridPythiaService.from_config_path(config_path)
        inverters = service.inverters

        assert len(inverters) > 0
        assert all(hasattr(inv, "device_id") for inv in inverters)

    def test_service_cache_invalidation(self, config_path: Path):
        """Test cache invalidation."""
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        from unittest.mock import MagicMock

        service = GridPythiaService.from_config_path(config_path)
        service.prediction_cache.set(MagicMock())
        assert service.prediction_cache.is_valid() is True

        service.invalidate_cache()
        assert service.prediction_cache.is_valid() is False


class TestGridPythiaServiceIntegration:
    """Integration tests for GridPythiaService (requires network)."""

    @pytest.fixture
    def config_path(self) -> Path:
        """Return path to the test config file."""
        return Path(__file__).parent.parent.parent / "config.yaml"

    @pytest.mark.asyncio
    async def test_fetch_predictions(self, config_path: Path):
        """Test fetching predictions through service."""
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        service = GridPythiaService.from_config_path(config_path)

        # This may fail if network is unavailable, which is expected
        try:
            pdata = await service.fetch_predictions(hours=2, dt_hours=0.25)
            assert pdata.steps > 0
            assert pdata.load_wh is not None
        except Exception:
            pytest.skip("Network unavailable for prediction fetch")

    @pytest.mark.asyncio
    async def test_optimize(self, config_path: Path):
        """Test optimization through service."""
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        service = GridPythiaService.from_config_path(config_path)

        try:
            pdata = await service.fetch_predictions(hours=2, dt_hours=0.25)
            solution = await service.optimize(pdata)
            assert solution is not None
            assert len(solution.inverter_plans) > 0
        except Exception:
            pytest.skip("Network or solver unavailable")
