"""Pydantic request / response schemas for the GridPythia API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Requests ──────────────────────────────────────────────────────────────


class FetchRequest(BaseModel):
    """Request body for ``POST /api/predictions/fetch``."""

    timezone: str = Field("UTC", description="IANA timezone name for the forecast start time")


class InverterStatusRequest(BaseModel):
    """Request body for ``POST /api/inverters/{device_id}/status``."""

    soc: float = Field(..., ge=0.0, le=100.0, description="Battery state-of-charge in %")
    mode: int = Field(0, ge=0, le=4, description="Active InverterMode (0=IDLE … 4=AC_CHARGE_ZFI)")


class InverterStatusResponse(BaseModel):
    """Response for ``GET /api/inverters/status``."""

    device_id: str
    soc: float
    mode: int
    mode_name: str
    reported_at: str  # ISO 8601
    age_s: float
    is_fresh: bool


class OptimizeRequest(BaseModel):
    """Request body for ``POST /api/optimize``."""

    timezone: str = Field("UTC", description="IANA timezone name for the forecast start time")

    battery_soc: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Battery SoC overrides: battery_id → percentage [0–100]. "
            "Values are clamped to [min_soc, max_soc] configured for each battery."
        ),
    )
    initial_modes: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Initial inverter mode at the start of the optimisation horizon. "
            "inverter_id → InverterMode int "
            "(0=IDLE, 1=DISCHARGE, 2=DISCHARGE_ZFI, 3=AC_CHARGE, 4=AC_CHARGE_ZFI). "
            "Defaults to IDLE for any inverter not listed."
        ),
    )
    solver_opts: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional HiGHS solver option overrides for this call only. "
            "Merged on top of the config-level solver_opts. "
            'Example: {"time_limit": 10, "mip_rel_gap": 0.05}.'
        ),
    )


# ── Config response ───────────────────────────────────────────────────────


class BatteryInfo(BaseModel):
    device_id: str
    min_soc_percentage: int
    max_soc_percentage: int
    initial_soc_percentage: int
    capacity_wh: float


class InverterInfo(BaseModel):
    device_id: str
    has_pv: bool
    battery_id: str | None
    max_ac_output_power_w: float
    max_ac_charge_power_w: float
    zero_feed_in: bool


class AppConfigResponse(BaseModel):
    """UI bootstrap data returned by ``GET /api/config``."""

    batteries: list[BatteryInfo]
    inverters: list[InverterInfo]
    has_weather: bool
    horizon_h: float
    dt_min: int
    objective: str
    optimization_interval_min: int = 15
    inverter_status_max_age_s: float = 300.0
    mqtt_enabled: bool = False


# ── Prediction status response ────────────────────────────────────────────


class PredictionsStatusResponse(BaseModel):
    """Cache status returned by ``GET /api/predictions/status``."""

    has_cache: bool
    age_s: float | None = None
    ttl_s: float
    forecast_from: str | None = None


# ── Optimization status response ──────────────────────────────────────────


class OptimizeStatusResponse(BaseModel):
    """Cache status returned by ``GET /api/optimize/status``."""

    has_cache: bool
    age_s: float | None = None
    ttl_s: float


# ── Optimization response ─────────────────────────────────────────────────


class InverterPlanStep(BaseModel):
    """One time-slot in an inverter schedule."""

    timestamp: str  # ISO 8601 wall-clock time for this slot
    mode: int  # InverterMode integer value
    mode_name: str  # Human-readable InverterMode name
    charge_ac_wh: float  # AC energy drawn from grid to charge battery [Wh/dt]
    discharge_ac_wh: float  # AC energy delivered from battery to home [Wh/dt]
    pv_to_ac_wh: float  # PV energy routed directly to AC bus [Wh/dt]
    pv_to_battery_wh: float  # PV energy routed into battery [Wh/dt]
    battery_soc_wh: float | None  # SoC at the *end* of this slot [Wh]; None if no battery


class InverterPlanResponse(BaseModel):
    """Full per-inverter schedule for the optimisation horizon."""

    device_id: str
    steps: list[InverterPlanStep]


# ── Appliance load forecast ───────────────────────────────────────────────


class ApplianceForecastSlot(BaseModel):
    """One energy-demand slot from a home appliance."""

    time: str = Field(..., description="ISO 8601 datetime (timezone-aware) for this slot")
    load_wh: float = Field(..., ge=0.0, description="Expected energy demand in Wh for this slot")


class ApplianceForecastRequest(BaseModel):
    """List of forecast slots submitted by a home appliance."""

    slots: list[ApplianceForecastSlot] = Field(
        ...,
        description="Ordered list of (time, load_wh) pairs; may span multiple time steps",
    )


class ApplianceForecastInfo(BaseModel):
    """Summary of one appliance's active forecast."""

    appliance_id: str
    slot_count: int
    first_slot: str | None = None
    last_slot: str | None = None


class OptimizeSummary(BaseModel):
    """Cost / savings numbers and solver metadata."""

    solver_status: str
    solve_time_s: float
    objective: str
    total_cost_eur: float
    total_revenue_eur: float
    net_cost_eur: float
    naive_net_cost_eur: float
    savings_eur: float
    parity_ok: bool | None = None
