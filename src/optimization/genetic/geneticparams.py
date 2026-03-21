"""GENETIC algorithm parameters."""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class GeneticEnergyManagementParameters:
    """Encapsulates energy-related forecasts and costs used in GENETIC optimization."""

    pv_prognose_wh: dict[str, list[float]]
    strompreis_euro_pro_wh: list[float]
    einspeiseverguetung_euro_pro_wh: Union[list[float], float]
    preis_euro_pro_wh_akku: float
    gesamtlast: list[float]

    def __post_init__(self) -> None:
        # Accept legacy list format and convert to dict
        if isinstance(self.pv_prognose_wh, list):
            self.pv_prognose_wh = {"__global__": self.pv_prognose_wh}


@dataclass
class GeneticOptimizationParameters:
    """Main parameter class for running the genetic energy optimization."""

    ems: GeneticEnergyManagementParameters
    pv_akku: Optional[object] = None  # BatteryParameters
    inverter: Optional[object] = None  # InverterParameters
    eauto: Optional[object] = None  # ElectricVehicleParameters
    dishwasher: Optional[object] = None  # HomeApplianceParameters
    temperature_forecast: Optional[list[Optional[float]]] = None
    start_solution: Optional[list[float]] = None
