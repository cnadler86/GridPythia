"""Tests for ServerConfig Pydantic model."""

from pathlib import Path

import pytest

from GridPythia.config import AppConfig, config_log_summary
from GridPythia.config.server import SchedulerConfig, ServerConfig


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.inverter_status_max_age_s == 300.0
        assert cfg.mqtt.enabled is False
        assert cfg.scheduler.optimization_interval_minutes == 15
        assert cfg.scheduler.dispatch_buffer_seconds == 5.0
        assert cfg.scheduler.dispatch_buffer_max_seconds == 30.0

    def test_from_dict(self):
        cfg = ServerConfig.model_validate({
            "inverter_status_max_age_s": 120.0,
            "scheduler": {
                "optimization_interval_minutes": 30,
                "dispatch_buffer_seconds": 7,
                "dispatch_buffer_max_seconds": 25,
            },
            "mqtt": {"enabled": True, "broker": "mqtt://192.168.1.1:1883"},
        })
        assert cfg.inverter_status_max_age_s == 120.0
        assert cfg.scheduler.optimization_interval_minutes == 30
        assert cfg.scheduler.dispatch_buffer_seconds == 7.0
        assert cfg.scheduler.dispatch_buffer_max_seconds == 25.0
        assert cfg.mqtt.enabled is True
        assert cfg.mqtt.broker == "mqtt://192.168.1.1:1883"

    def test_max_age_must_be_positive(self):
        with pytest.raises(Exception):
            ServerConfig(inverter_status_max_age_s=0.0)

    def test_scheduler_interval_bounds(self):
        with pytest.raises(Exception):
            SchedulerConfig(optimization_interval_minutes=0)
        with pytest.raises(Exception):
            SchedulerConfig(optimization_interval_minutes=61)

    def test_scheduler_dispatch_buffer_bounds(self):
        with pytest.raises(Exception):
            SchedulerConfig(dispatch_buffer_seconds=-1)
        with pytest.raises(Exception):
            SchedulerConfig(dispatch_buffer_seconds=31)
        with pytest.raises(Exception):
            SchedulerConfig(dispatch_buffer_max_seconds=0)

    def test_frozen(self):
        cfg = ServerConfig()
        with pytest.raises(Exception):
            cfg.inverter_status_max_age_s = 999.0


class TestAppConfigIncludesServer:
    """Ensure AppConfig correctly loads and exposes the server section."""

    def test_server_defaults_present(self):
        from GridPythia.config import AppConfig
        cfg = AppConfig()
        assert isinstance(cfg.server, ServerConfig)

    def test_server_from_yaml(self):
        from GridPythia.config import AppConfig
        yaml_text = """
prediction:
  horizon: 24
optimization:
  batteries: []
  inverters: []
server:
  inverter_status_max_age_s: 60
  scheduler:
    optimization_interval_minutes: 30
"""
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.server.inverter_status_max_age_s == 60.0
        assert cfg.server.scheduler.optimization_interval_minutes == 30

    def test_server_field_missing_uses_defaults(self):
        cfg = AppConfig.from_yaml("prediction:\n  horizon: 24\n")
        assert cfg.server.inverter_status_max_age_s == 300.0

    def test_log_summary_includes_active_settings_without_secrets(self):
        yaml_text = """
prediction:
  electricprice:
    provider: Fixed
    charges_kwh: 0.3096
    vat_rate: 0.19
  load:
    path: profiles.csv
  pvforecast:
    openmeteo:
      api_key: do-not-log-this
optimization:
  solver:
    solver_opts:
      token: also-do-not-log
server:
  mqtt:
    enabled: true
    password: secret-password
"""
        config = AppConfig.from_yaml(yaml_text)
        summary = config_log_summary(
            config,
            "/opt/gridpythia/config.yaml",
            {"prediction": {"electricprice": {"provider": "Fixed"}}},
        )

        assert summary["prediction"]["electricprice"]["provider"] == "Fixed"
        assert summary["prediction"]["electricprice"]["charges_kwh"] == 0.3096
        assert summary["prediction"]["load"]["profile_path"] == str(
            Path("/opt/gridpythia") / "profiles.csv"
        )
        assert summary["server"]["mqtt_enabled"] is True
        assert "do-not-log-this" not in repr(summary)
        assert "also-do-not-log" not in repr(summary)
        assert "secret-password" not in repr(summary)

    def test_server_logs_loaded_config(self, tmp_path, monkeypatch):
        from structlog.testing import capture_logs

        import GridPythia.server.state as state
        from GridPythia.server import services

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "prediction:\n"
            "  electricprice:\n"
            "    provider: Fixed\n"
            "    charges_kwh: 0.3096\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(state, "config_path", config_path)
        monkeypatch.setattr(state, "config_cache", None)
        monkeypatch.setattr(state, "config_cache_raw", None)
        monkeypatch.setattr(state, "config_cache_mtime", -1.0)

        with capture_logs() as logs:
            services.load_config()

        loaded = next(log for log in logs if log["event"] == "configuration_loaded")
        assert loaded["prediction"]["electricprice"]["provider"] == "Fixed"
        assert loaded["prediction"]["electricprice"]["charges_kwh"] == 0.3096
