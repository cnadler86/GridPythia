"""Pydantic configuration models for optimization-related settings."""

from __future__ import annotations

from itertools import count
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Auto-incrementing device IDs
_INVERTER_COUNTER = count(1)
_BATTERY_COUNTER = count(1)


class BatteryParameters(BaseModel):
    """Battery configuration parameters (Pydantic v2, frozen)."""

    model_config = {"frozen": True}

    device_id: str = Field(
        default_factory=lambda: f"battery{next(_BATTERY_COUNTER)}",
        description="Unique battery identifier",
    )
    capacity_wh: int = Field(default=8000, gt=0, description="Battery capacity in Wh")
    charging_efficiency: float = Field(
        default=0.98, gt=0.0, le=1.0, description="Charging efficiency (0, 1]"
    )
    discharging_efficiency: float = Field(
        default=0.98, gt=0.0, le=1.0, description="Discharging efficiency (0, 1]"
    )
    max_charge_power_w: float = Field(default=5000, ge=0.0, description="Max charge power in W")
    max_discharge_power_w: float = Field(
        default=5000, ge=0.0, description="Max discharge power in W"
    )
    initial_soc_percentage: int = Field(default=50, ge=0, le=100, description="Initial SoC in %")
    min_soc_percentage: int = Field(default=0, ge=0, le=100, description="Minimum SoC in %")
    max_soc_percentage: int = Field(default=100, ge=0, le=100, description="Maximum SoC in %")

    @field_validator("min_soc_percentage", mode="after")
    @classmethod
    def validate_min_soc(cls, v: int, info) -> int:
        """Ensure min_soc <= max_soc."""
        if info.data.get("max_soc_percentage") is not None:
            if v > info.data["max_soc_percentage"]:
                raise ValueError("min_soc_percentage cannot exceed max_soc_percentage")
        return v

    @field_validator("initial_soc_percentage", mode="after")
    @classmethod
    def validate_initial_soc(cls, v: int, info) -> int:
        """Clamp initial_soc to [min_soc, max_soc]."""
        min_soc = info.data.get("min_soc_percentage", 0)
        max_soc = info.data.get("max_soc_percentage", 100)
        return min(max(v, min_soc), max_soc)


class InverterParameters(BaseModel):
    """Inverter device configuration (Pydantic v2, frozen)."""

    model_config = {"frozen": True}

    device_id: str = Field(
        default_factory=lambda: f"inverter{next(_INVERTER_COUNTER)}",
        description="Unique inverter identifier",
    )
    battery_id: Optional[str] = Field(
        default=None, description="Associated battery device_id (if any)"
    )
    has_pv: bool = Field(
        default=True, description="Whether a PV plane is attached to this inverter"
    )
    max_ac_output_power_w: float = Field(
        default=5000, ge=0.0, description="Max AC output (to grid) in W"
    )
    max_ac_charge_power_w: float = Field(
        default=0.0, ge=0.0, description="Max AC charge (from grid) in W"
    )
    dc_to_ac_efficiency: float = Field(
        default=0.96, ge=0.0, le=1.0, description="DC->AC efficiency [0, 1]"
    )
    ac_to_dc_efficiency: float = Field(
        default=0.96, ge=0.0, le=1.0, description="AC->DC efficiency [0, 1]"
    )
    zero_feed_in: bool = Field(default=True, description="Enable zero-feed-in mode")
    mode_switch_cost: float = Field(
        default=0.005, ge=0.0, description="Cost (EUR) per inverter mode change (wear cost)"
    )
    active_inverter_consumption_w: float = Field(
        default=10.0, ge=0.0, description="Inverter consumption when active (W)"
    )


class OptimizationSolverConfig(BaseModel):
    """Optimizer objective and solver-wide settings."""

    objective: Literal["cost", "self_consumption"] = "cost"


class OptimizationConfig(BaseModel):
    """Top-level optimization section from config.yaml."""

    solver: OptimizationSolverConfig = Field(default_factory=OptimizationSolverConfig)
    batteries: list[BatteryParameters] = Field(default_factory=list)
    inverters: list[InverterParameters] = Field(default_factory=list)
