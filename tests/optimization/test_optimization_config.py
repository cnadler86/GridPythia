import pytest
from pydantic import ValidationError

from GridPythia.config.optimization import BatteryParameters, OptimizationConfig


def test_solver_config_defaults_to_highs_provider() -> None:
    cfg = OptimizationConfig.model_validate({})
    assert cfg.solver.provider == "highs"
    assert cfg.solver.objective == "cost"
    assert isinstance(cfg.solver.solver_opts, dict)
    assert cfg.solver.solver_opts == {}


def test_solver_config_accepts_solver_opts_mapping() -> None:
    cfg = OptimizationConfig.model_validate(
        {
            "solver": {
                "provider": "highs",
                "objective": "cost",
                "solver_opts": {
                    "time_limit": 45,
                    "mip_rel_gap": 0.03,
                    "presolve": "on",
                },
            }
        }
    )
    assert cfg.solver.provider == "highs"
    assert cfg.solver.solver_opts["time_limit"] == 45
    assert cfg.solver.solver_opts["mip_rel_gap"] == 0.03


def test_terminal_soc_configuration_defaults_disabled() -> None:
    battery = BatteryParameters()

    assert battery.terminal_soc_target_percentage is None
    assert battery.terminal_soc_penalty_eur_per_percentage_point == 0.0


def test_terminal_soc_penalty_requires_target_in_operating_range() -> None:
    with pytest.raises(ValidationError, match="terminal_soc_target_percentage is required"):
        BatteryParameters(terminal_soc_penalty_eur_per_percentage_point=0.01)
    with pytest.raises(ValidationError, match="within min_soc and max_soc"):
        BatteryParameters(
            min_soc_percentage=20,
            max_soc_percentage=90,
            terminal_soc_target_percentage=10,
        )
