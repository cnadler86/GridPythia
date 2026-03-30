"""Abstract and base classes for inverter devices."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class InverterMode(IntEnum):
    """Inverter operation modes."""

    IDLE = 0
    DISCHARGE = 1
    DISCHARGE_ZERO_FEED_IN = 2
    AC_CHARGE = 3
    AC_CHARGE_ZERO_FEED_IN = 4


class SystemTopology(StrEnum):
    """Inverter system topology derived from configuration."""

    PV_ONLY = "PV_ONLY"
    PV_BATTERY = "PV_BATTERY"
    PV_HYBRID = "PV_HYBRID"
    AC_BATTERY = "AC_BATTERY"
    EV_CHARGE_ONLY = "EV_CHARGE_ONLY"
    EV_V2G = "EV_V2G"


@dataclass(slots=True)
class EnergyFlowResult:
    """Result of a single inverter's energy processing for one time step.

    All values are in watt-hours (Wh) and non-negative.
    """

    ac_output_wh: float
    """Energy available from inverter to AC bus (PV + battery discharge) [Wh]."""

    ac_input_wh: float
    """Energy drawn from AC bus to charge battery [Wh]."""

    losses_wh: float
    """Total inverter + battery losses [Wh]."""

    pv_ac_wh: float = 0.0
    """PV-sourced portion of ac_output_wh [Wh]."""
