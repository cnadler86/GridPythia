"""Tests for the provider registry module."""

import pytest

from GridPythia.prediction.registry import (
    ProviderProtocol,
    ProviderRegistry,
    electricprice_registry,
    feedintariff_registry,
    load_registry,
    provider_registry,
    pvforecast_registry,
    weather_registry,
)


class TestProviderRegistry:
    """Test the generic ProviderRegistry class."""

    class _DummyProvider(ProviderProtocol):
        def __init__(self, value: int = 0):
            self.value = value

        @property
        def provider_id(self) -> str:
            return "dummy"

    def test_register_and_create(self):
        """Test registering and creating a provider."""
        registry: ProviderRegistry[TestProviderRegistry._DummyProvider] = ProviderRegistry(
            _category="test"
        )

        registry.register("Dummy", lambda cfg: self._DummyProvider(cfg.get("value", 0)))

        provider = registry.create("Dummy", {"value": 42})
        assert provider.value == 42

    def test_create_returns_singleton_for_same_config(self):
        """Same provider+config should reuse the cached instance."""
        registry: ProviderRegistry[TestProviderRegistry._DummyProvider] = ProviderRegistry(
            _category="test"
        )
        registry.register("Dummy", lambda cfg: self._DummyProvider(cfg.get("value", 0)))

        a = registry.create("Dummy", {"value": 7})
        b = registry.create("Dummy", {"value": 7})

        assert a is b

    def test_create_fresh_bypasses_singleton_cache(self):
        """fresh=True should force a new provider instance."""
        registry: ProviderRegistry[TestProviderRegistry._DummyProvider] = ProviderRegistry(
            _category="test"
        )
        registry.register("Dummy", lambda cfg: self._DummyProvider(cfg.get("value", 0)))

        a = registry.create("Dummy", {"value": 7})
        b = registry.create("Dummy", {"value": 7}, fresh=True)

        assert a is not b

    def test_unknown_provider_raises(self):
        """Test that creating an unknown provider raises KeyError."""
        registry: ProviderRegistry[TestProviderRegistry._DummyProvider] = ProviderRegistry(
            _category="test"
        )

        with pytest.raises(KeyError, match="Unknown test provider 'NotExists'"):
            registry.create("NotExists", {})

    def test_available_providers(self):
        """Test listing available providers."""
        registry: ProviderRegistry[TestProviderRegistry._DummyProvider] = ProviderRegistry(
            _category="test"
        )
        registry.register("A", lambda cfg: self._DummyProvider())
        registry.register("B", lambda cfg: self._DummyProvider())

        assert set(registry.available()) == {"A", "B"}

    def test_is_registered(self):
        """Test checking if a provider is registered."""
        registry: ProviderRegistry[TestProviderRegistry._DummyProvider] = ProviderRegistry(
            _category="test"
        )
        registry.register("Exists", lambda cfg: self._DummyProvider())

        assert registry.is_registered("Exists") is True
        assert registry.is_registered("NotExists") is False


class TestBuiltinRegistries:
    """Test that all builtin providers are registered."""

    def test_electricprice_providers_registered(self):
        """EnergyCharts and Fixed should be registered."""
        assert "EnergyCharts" in electricprice_registry.available()
        assert "Fixed" in electricprice_registry.available()

    def test_feedintariff_providers_registered(self):
        """Fixed feed-in tariff should be registered."""
        assert "Fixed" in feedintariff_registry.available()

    def test_load_providers_registered(self):
        """ProfileCSV should be registered."""
        assert "ProfileCSV" in load_registry.available()

    def test_pvforecast_providers_registered(self):
        """OpenMeteo and Akkudoktor should be registered."""
        assert "OpenMeteo" in pvforecast_registry.available()
        assert "Akkudoktor" in pvforecast_registry.available()

    def test_weather_providers_registered(self):
        """OpenMeteo and BrightSky should be registered."""
        assert "OpenMeteo" in weather_registry.available()
        assert "BrightSky" in weather_registry.available()


class TestProviderRegistryCreation:
    """Test creating providers from the registry."""

    def test_create_fixed_electricprice(self):
        """Test creating a fixed electricity price provider."""
        provider = provider_registry.create_electricprice(
            "Fixed",
            {"price_kwh": 0.30, "charges_kwh": 0.15, "vat_rate": 0.19},
        )
        # Provider ID is the class name, not the registry key
        assert "Fixed" in provider.provider_id

    def test_create_fixed_feedintariff(self):
        """Test creating a fixed feed-in tariff provider."""
        provider = provider_registry.create_feedintariff("Fixed", {"tariff_kwh": 0.08})
        # Provider ID is the class name
        assert "Fixed" in provider.provider_id

    def test_create_weather_openmeteo(self):
        """Test creating an OpenMeteo weather provider."""
        provider = provider_registry.create_weather(
            "OpenMeteo",
            {"latitude": 48.0, "longitude": 8.0},
        )
        assert "OpenMeteo" in provider.provider_id

    def test_create_weather_brightsky(self):
        """Test creating a BrightSky weather provider."""
        provider = provider_registry.create_weather(
            "BrightSky",
            {"latitude": 48.0, "longitude": 8.0},
        )
        assert "BrightSky" in provider.provider_id

    def test_create_pv_openmeteo(self):
        """Test creating an OpenMeteo PV forecast provider."""
        provider = provider_registry.create_pvforecast(
            "OpenMeteo",
            {
                "latitude": 48.0,
                "longitude": 8.0,
                "plane": {"peak_kw": 5.0, "tilt": 30.0, "azimuth": 180.0},
            },
        )
        assert "OpenMeteo" in provider.provider_id
