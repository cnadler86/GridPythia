"""GridPythia - Home Energy Management System.

A modular prediction and optimization framework for residential energy systems.

Public API
----------
Configuration:
    AppConfig           Root configuration from YAML
    PredictionConfig    Prediction settings (providers, horizon, dt)
    OptimizationConfig  Optimization settings (solver, batteries, inverters)

Prediction:
    Prediction          Orchestrator for fetching aligned prediction data
    PredictionData      Consumer-facing prediction contract
    PredictionSetup     Wire providers before fetching

Optimization:
    LinearOptimizer     MILP solver using CVXPY + HiGHS
    LinearSolution      Optimizer output with inverter plans
    InverterPlan        Per-device schedule

Simulation:
    GridSimulation      Step-by-step energy simulation
    SimulationResult    Simulation output

Devices:
    InverterBase        Inverter device model
    Battery             Battery device model
    InverterMode        Operating modes enum

Example:
-------
>>> from GridPythia import AppConfig, Prediction, LinearOptimizer
>>> cfg = AppConfig.from_yaml_file("config.yaml")
>>> pred = Prediction(setup)
>>> data = await pred.fetch(hours=24)
>>> optimizer = LinearOptimizer(inverters)
>>> solution = optimizer.solve(data)
"""

from GridPythia.config import AppConfig
from GridPythia.config.optimization import (
    BatteryParameters,
    InverterParameters,
    OptimizationConfig,
)
from GridPythia.config.prediction import PredictionConfig
from GridPythia.optimization.solution import (
    InverterPlan,
    LinearSolution,
    OptimizationObjective,
)
from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.prediction import Prediction, PredictionData, PredictionSetup
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase
from GridPythia.simulation.grid_simulation import GridSimulation, SimulationResult

__all__ = [
    # Config
    "AppConfig",
    "PredictionConfig",
    "OptimizationConfig",
    "BatteryParameters",
    "InverterParameters",
    # Prediction
    "Prediction",
    "PredictionData",
    "PredictionSetup",
    # Optimization
    "LinearOptimizer",
    "LinearSolution",
    "InverterPlan",
    "OptimizationObjective",
    # Simulation
    "GridSimulation",
    "SimulationResult",
    # Devices
    "InverterBase",
    "Battery",
    "InverterMode",
]

__version__ = "0.1.0"
