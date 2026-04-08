from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from GridPythia.config.optimization import OptimizationConfig
from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


@dataclass(frozen=True, slots=True)
class SolverFixtureScenario:
    prediction: PredictionData
    inverters: list[InverterBase]
    payload: dict[str, Any]


def load_solver_fixture_scenario(
    fixture_path: Path | None = None,
    config_path: Path | None = None,
) -> SolverFixtureScenario:
    fixture = fixture_path or Path("tests/optimization/fixtures/prediction_2026_04_06_48h_15m.json")
    config_file = config_path or Path("config.yaml")

    payload = json.loads(fixture.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    opt_cfg = OptimizationConfig.model_validate(config["optimization"])

    prediction = PredictionData(
        timestamps=[datetime.fromisoformat(ts) for ts in payload["timestamps"]],
        dt_hours=float(payload["dt_hours"]),
        load_wh=payload["load_wh"],
        electricprice_eur_wh=payload["electricprice_eur_wh"],
        feedintariff_eur_wh=payload["feedintariff_eur_wh"],
        pv_by_inverter={str(inv_id): values for inv_id, values in payload["pv_by_inverter"].items()},
    )

    batteries = {bat.device_id: Battery(bat) for bat in opt_cfg.batteries}
    inverters = [
        InverterBase(inv, battery=batteries.get(inv.battery_id) if inv.battery_id else None)
        for inv in opt_cfg.inverters
    ]
    return SolverFixtureScenario(prediction=prediction, inverters=inverters, payload=payload)