from GridPythia.config import AppConfig
from GridPythia.optimization.terminal_soc import TerminalSoCMode
from GridPythia.server import services, state


def test_server_optimizer_receives_provider_terminal_soc_mode(monkeypatch) -> None:
    monkeypatch.setattr(services, "_config_mtime_for_singletons", lambda: 1.0)
    monkeypatch.setattr(services, "build_inverters", lambda cfg: [])
    monkeypatch.setattr(state, "optimizer", None)
    monkeypatch.setattr(state, "optimizer_config_mtime", 0.0)
    monkeypatch.setattr(
        state.coordinator,
        "max_age_s",
        state.coordinator.max_age_s,
    )

    optimizer = services.get_optimizer(AppConfig(), TerminalSoCMode.TARGET)

    assert optimizer._terminal_soc_mode == TerminalSoCMode.TARGET
