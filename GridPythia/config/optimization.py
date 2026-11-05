"""Pydantic configuration models for optimization-related settings."""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class BatteryParameters(BaseModel):
    """Battery configuration parameters (Pydantic v2, frozen)."""

    model_config = {"frozen": True}

    device_id: str = Field(
        default_factory=lambda: f"battery_{uuid.uuid4().hex[:8]}",
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
    terminal_soc_target_percentage: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Optional end-of-horizon battery SoC target in percent",
    )
    terminal_soc_penalty_eur_per_percentage_point: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost objective penalty in EUR per terminal SoC percentage point",
    )

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

    @field_validator("terminal_soc_target_percentage", mode="after")
    @classmethod
    def validate_terminal_soc_target(cls, v: float | None, info) -> float | None:
        """Ensure the terminal target is inside the configured operating SoC range."""
        if v is None:
            return None
        min_soc = info.data.get("min_soc_percentage", 0)
        max_soc = info.data.get("max_soc_percentage", 100)
        if not min_soc <= v <= max_soc:
            raise ValueError("terminal_soc_target_percentage must be within min_soc and max_soc")
        return v

    @field_validator("terminal_soc_penalty_eur_per_percentage_point", mode="after")
    @classmethod
    def validate_terminal_soc_penalty(cls, v: float, info) -> float:
        """Require a target whenever a terminal SoC penalty is configured."""
        if v > 0.0 and info.data.get("terminal_soc_target_percentage") is None:
            raise ValueError(
                "terminal_soc_target_percentage is required when the terminal SoC penalty is set"
            )
        return v


class InverterParameters(BaseModel):
    """Inverter device configuration (Pydantic v2, frozen)."""

    model_config = {"frozen": True}

    device_id: str = Field(
        default_factory=lambda: f"inverter_{uuid.uuid4().hex[:8]}",
        description="Unique inverter identifier",
    )
    battery_id: Optional[str] = Field(
        default=None, description="Associated battery device_id (if any)"
    )
    has_pv: bool = Field(
        default=True, description="Whether a PV plane is attached to this inverter"
    )
    min_ac_output_power_w: float = Field(
        default=0.0, ge=0.0, description="Min AC output (to grid) in W"
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

    provider: Literal["highs", "blackbox"] = "highs"
    objective: Literal["cost", "self_consumption"] = "cost"
    solver_opts: dict[str, Any] = Field(default_factory=dict)

    @field_validator("solver_opts", mode="before")
    @classmethod
    def _normalize_solver_opts(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        raise TypeError("solver_opts must be a mapping")


class OptimizationConfig(BaseModel):
    """Top-level optimization section from config.yaml."""

    solver: OptimizationSolverConfig = Field(default_factory=OptimizationSolverConfig)
    batteries: list[BatteryParameters] = Field(default_factory=list)
    inverters: list[InverterParameters] = Field(default_factory=list)
