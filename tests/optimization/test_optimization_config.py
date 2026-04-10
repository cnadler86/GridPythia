from GridPythia.config.optimization import OptimizationConfig


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
