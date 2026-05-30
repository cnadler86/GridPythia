"""Tests for MQTT gateway message handling and validation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from GridPythia.config.server import MqttConfig
from GridPythia.server.mqtt_gateway import MqttGateway


@pytest.fixture
def mqtt_cfg():
    return MqttConfig(
        enabled=True,
        broker="mqtt://localhost:1883",
        client_id="test",
        username="",
        password="",
        topic_prefix="gridpythia",
    )


@pytest.fixture
def gateway(mqtt_cfg):
    """Create gateway without connecting to a broker."""
    with patch("paho.mqtt.client.Client"):
        gw = MqttGateway(mqtt_cfg)
    return gw


def _make_msg(topic: str, payload: bytes | str) -> SimpleNamespace:
    """Create a mock MQTTMessage."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return SimpleNamespace(topic=topic, payload=payload)


class TestInverterStatusParsing:
    """Tests for _on_message with inverter status topics."""

    def test_valid_status_updates_coordinator(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            mock_state.coordinator.update_status.return_value = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": 65.0, "mode": 1}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_called_once_with(
                "inv1", soc=65.0, mode=1
            )

    def test_missing_soc_field_is_rejected(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"mode": 1}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_not_called()

    def test_invalid_json_is_rejected(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            msg = _make_msg("gridpythia/inverters/inv1/status", b"not-json")
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_not_called()

    def test_nan_soc_is_rejected(self, gateway):
        """NaN soc values should be rejected by the range check."""
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": float("nan")}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_not_called()

    def test_inf_soc_is_rejected(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": float("inf")}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_not_called()

    def test_soc_over_100_is_rejected(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": 101.0}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_not_called()

    def test_negative_soc_is_rejected(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": -5.0}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_not_called()

    def test_invalid_mode_is_rejected(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": 50.0, "mode": 99}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_not_called()

    def test_mode_defaults_to_zero(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            mock_state.coordinator.update_status.return_value = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": 42.0}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_called_once_with(
                "inv1", soc=42.0, mode=0
            )

    def test_wrong_topic_prefix_ignored(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            msg = _make_msg(
                "other/inverters/inv1/status",
                json.dumps({"soc": 50.0}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_not_called()

    def test_soc_boundary_zero_accepted(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            mock_state.coordinator.update_status.return_value = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": 0.0}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_called_once()

    def test_soc_boundary_hundred_accepted(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.coordinator = MagicMock()
            mock_state.coordinator.update_status.return_value = MagicMock()
            msg = _make_msg(
                "gridpythia/inverters/inv1/status",
                json.dumps({"soc": 100.0}),
            )
            gateway._on_message(None, None, msg)
            mock_state.coordinator.update_status.assert_called_once()


class TestApplianceForecastParsing:
    """Tests for _on_message with appliance load forecast topics."""

    def test_valid_appliance_forecast_stored(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.appliance_forecasts = {}
            slots = [
                {"time": "2026-05-30T10:00:00+00:00", "load_wh": 100.0},
                {"time": "2026-05-30T10:15:00+00:00", "load_wh": 200.0},
            ]
            msg = _make_msg(
                "gridpythia/appliance_load/forecast/dishwasher",
                json.dumps(slots),
            )
            gateway._on_message(None, None, msg)
            assert "dishwasher" in mock_state.appliance_forecasts
            assert len(mock_state.appliance_forecasts["dishwasher"]) == 2

    def test_empty_payload_clears_forecast(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.appliance_forecasts = {"dishwasher": [{"time": "t", "load_wh": 1}]}
            msg = _make_msg(
                "gridpythia/appliance_load/forecast/dishwasher",
                json.dumps([]),
            )
            gateway._on_message(None, None, msg)
            assert "dishwasher" not in mock_state.appliance_forecasts

    def test_malformed_slots_are_filtered(self, gateway):
        """Slots missing required keys should be filtered out."""
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.appliance_forecasts = {}
            slots = [
                {"time": "2026-05-30T10:00:00+00:00", "load_wh": 100.0},  # valid
                {"time": "2026-05-30T10:15:00+00:00"},  # missing load_wh
                {"load_wh": 200.0},  # missing time
                "not_a_dict",  # not a dict
                {"time": "2026-05-30T10:30:00+00:00", "load_wh": "bad"},  # non-numeric
                {"time": "2026-05-30T10:45:00+00:00", "load_wh": 300.0},  # valid
            ]
            msg = _make_msg(
                "gridpythia/appliance_load/forecast/washer",
                json.dumps(slots),
            )
            gateway._on_message(None, None, msg)
            assert "washer" in mock_state.appliance_forecasts
            stored = mock_state.appliance_forecasts["washer"]
            assert len(stored) == 2
            assert stored[0]["load_wh"] == 100.0
            assert stored[1]["load_wh"] == 300.0

    def test_all_invalid_slots_clears_forecast(self, gateway):
        """If all slots are invalid the forecast should be cleared."""
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.appliance_forecasts = {"oven": [{"time": "t", "load_wh": 1}]}
            slots = [{"invalid": True}, {"also_invalid": "yes"}]
            msg = _make_msg(
                "gridpythia/appliance_load/forecast/oven",
                json.dumps(slots),
            )
            gateway._on_message(None, None, msg)
            assert "oven" not in mock_state.appliance_forecasts

    def test_non_list_payload_rejected(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.appliance_forecasts = {}
            msg = _make_msg(
                "gridpythia/appliance_load/forecast/dev",
                json.dumps({"not": "a list"}),
            )
            gateway._on_message(None, None, msg)
            assert "dev" not in mock_state.appliance_forecasts

    def test_non_json_payload_rejected(self, gateway):
        with patch("GridPythia.server.mqtt_gateway.state") as mock_state:
            mock_state.appliance_forecasts = {}
            msg = _make_msg(
                "gridpythia/appliance_load/forecast/dev",
                b"garbage",
            )
            gateway._on_message(None, None, msg)
            assert "dev" not in mock_state.appliance_forecasts
