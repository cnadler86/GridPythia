"""Centralized configuration models for inverters and batteries using Pydantic v2."""

from __future__ import annotations

from itertools import count
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Auto-incrementing device IDs
_INVERTER_COUNTER = count(1)
_BATTERY_COUNTER = count(1)

# Default charge/discharge rates as percent values (10% steps).
DEFAULT_AC_RATES: tuple[int, ...] = tuple(range(10, 101, 10))


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
    initial_soc_percentage: int = Field(default=0, ge=0, le=100, description="Initial SoC in %")
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
    pv_source: Optional[str] = Field(default=None, description="PV source identifier (if any)")
    max_ac_output_power_w: float = Field(default=5000, ge=0.0, description="Max AC output in W")
    max_ac_charge_power_w: float = Field(default=0.0, ge=0.0, description="Max AC charge in W")
    dc_to_ac_efficiency: float = Field(
        default=0.95, ge=0.0, le=1.0, description="DC→AC efficiency [0, 1]"
    )
    ac_to_dc_efficiency: float = Field(
        default=0.95, ge=0.0, le=1.0, description="AC→DC efficiency [0, 1]"
    )
    zero_feed_in: bool = Field(default=True, description="Enable zero-feed-in mode")
    ac_rates_pct: tuple[int, ...] = Field(
        default=DEFAULT_AC_RATES,
        description="Discrete charge/discharge rates in percent (1..100)",
    )
    mode_switch_cost: float = Field(
        default=0.005, ge=0.0, description="Cost (€) per inverter mode change (wear cost)"
    )

    @field_validator("battery_id", "pv_source", mode="before")
    @classmethod
    def validate_topology(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure at least battery_id or pv_source is provided."""
        if info.context and info.context.get("skip_topology_check"):
            return v
        # Check will happen in second pass after both fields are set
        return v

    def model_post_init(self, __context) -> None:  # noqa: ARG002
        """Validate that at least battery_id or pv_source is configured."""
        if self.battery_id is None and self.pv_source is None:
            raise ValueError(
                f"Inverter '{self.device_id}' must have either battery_id or pv_source (or both)."
            )

    @field_validator("ac_rates_pct", mode="before")
    @classmethod
    def normalize_rates(cls, v) -> tuple[int, ...]:
        """Normalize ac_rates_pct to sorted tuple of unique integer percentages."""
        if isinstance(v, (list, tuple)):
            normalized: set[int] = set()
            for r in v:
                if not isinstance(r, int):
                    raise ValueError("ac_rates_pct entries must be integers in [1, 100]")
                if 1 <= r <= 100:
                    normalized.add(r)
            return tuple(sorted(x for x in normalized if 0 < x <= 100))
        return v
