"""Battery device simulation."""

from __future__ import annotations

from math import isfinite
from typing import Any

from structlog import get_logger

from GridPythia.config.optimization import BatteryParameters

logger = get_logger(__name__)


class Battery:
    """Represents a battery device with methods to simulate energy charging and discharging."""

    __slots__ = (
        "parameters",
        "prediction_hours",
        "capacity_wh",
        "_initial_soc_percentage",
        "min_soc_percentage",
        "max_soc_percentage",
        "charging_efficiency",
        "discharging_efficiency",
        "max_charge_power_w",
        "max_discharge_power_w",
        "_soc_wh",
        "_soc_percentage",
        "min_soc_wh",
        "max_soc_wh",
        "_initial_soc_wh",
        "_soc_pct_factor",
        "_log",
    )

    def __init__(self, parameters: BatteryParameters) -> None:
        self.parameters = parameters
        self._log = logger.bind(device_id=parameters.device_id, component="battery")
        self._setup()

    def _setup(self) -> None:
        """Sets up the battery parameters based on provided parameters."""
        self.capacity_wh = self.parameters.capacity_wh
        self.min_soc_percentage = self.parameters.min_soc_percentage
        self.max_soc_percentage = self.parameters.max_soc_percentage
        self.charging_efficiency = self.parameters.charging_efficiency
        self.discharging_efficiency = self.parameters.discharging_efficiency

        self.max_charge_power_w = self.parameters.max_charge_power_w
        self.max_discharge_power_w = self.parameters.max_discharge_power_w

        if not (self.capacity_wh > 0):
            raise ValueError("capacity_wh must be > 0")

        for val, name in (
            (self.charging_efficiency, "charging_efficiency"),
            (self.discharging_efficiency, "discharging_efficiency"),
        ):
            if not (0.0 < val <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {val}")

        for perc, name in (
            (self.min_soc_percentage, "min_soc_percentage"),
            (self.max_soc_percentage, "max_soc_percentage"),
            (self.parameters.initial_soc_percentage, "initial_soc_percentage"),
        ):
            if not (0.0 <= perc <= 100.0):
                raise ValueError(f"{name} must be within [0, 100], got {perc}")

        if self.min_soc_percentage > self.max_soc_percentage:
            raise ValueError("Min_soc_percentage cannot be greater than max_soc_percentage")

        # Clamp initial SoC percentage into [min, max]
        self._initial_soc_percentage = min(
            max(float(self.parameters.initial_soc_percentage), self.min_soc_percentage),
            self.max_soc_percentage,
        )

        self.min_soc_wh = (self.min_soc_percentage / 100) * self.capacity_wh
        self.max_soc_wh = (self.max_soc_percentage / 100) * self.capacity_wh

        self._initial_soc_wh = (self._initial_soc_percentage / 100.0) * self.capacity_wh
        self._soc_pct_factor: float = 100.0 / self.capacity_wh
        self._set_soc_wh_unchecked(self._initial_soc_wh)

        self._log.info(
            "battery_setup_complete",
            capacity_wh=self.capacity_wh,
            initial_soc_pct=self.initial_soc_percentage,
            min_soc_pct=self.min_soc_percentage,
            max_soc_pct=self.max_soc_percentage,
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the object to a dictionary representation."""
        return {
            "device_id": self.parameters.device_id,
            "capacity_wh": self.capacity_wh,
            "initial_soc_percentage": self.initial_soc_percentage,
            "soc_wh": self.soc_wh,
            "charging_efficiency": self.charging_efficiency,
            "discharging_efficiency": self.discharging_efficiency,
            "max_charge_power_w": self.max_charge_power_w,
            "max_discharge_power_w": self.max_discharge_power_w,
        }

    @property
    def initial_soc_percentage(self) -> float:
        """Configured reset SoC in percent."""
        return self._initial_soc_percentage

    @property
    def soc_wh(self) -> float:
        """Current state of charge in Wh."""
        return self._soc_wh

    @soc_wh.setter
    def soc_wh(self, value: float) -> None:
        self._set_soc_wh(value)

    @property
    def soc_percentage(self) -> float:
        """Current state of charge in percent."""
        return self._soc_percentage

    @soc_percentage.setter
    def soc_percentage(self, value: float) -> None:
        self._set_soc_percentage(value)

    def reset(self) -> None:
        """Resets the battery state to its initial values."""
        self._set_soc_wh(self._initial_soc_wh)

    def current_soc_percentage(self) -> float:
        """Calculates the current state of charge in percentage."""
        return self._soc_percentage

    def _validate_soc_wh(self, soc_wh: float) -> float:
        if not isfinite(soc_wh):
            raise ValueError(f"soc_wh must be finite, got {soc_wh}")
        if not (self.min_soc_wh <= soc_wh <= self.max_soc_wh):
            raise ValueError(
                f"soc_wh must be within [{self.min_soc_wh}, {self.max_soc_wh}], got {soc_wh}"
            )
        return float(soc_wh)

    def _validate_soc_percentage(self, soc_percentage: float) -> float:
        if not isfinite(soc_percentage):
            raise ValueError(f"soc_percentage must be finite, got {soc_percentage}")
        if not (self.min_soc_percentage <= soc_percentage <= self.max_soc_percentage):
            raise ValueError(
                "soc_percentage must be within "
                f"[{self.min_soc_percentage}, {self.max_soc_percentage}], got {soc_percentage}"
            )
        return float(soc_percentage)

    def _set_soc_wh_unchecked(self, soc_wh: float) -> None:
        self._soc_wh = soc_wh
        self._soc_percentage = soc_wh * self._soc_pct_factor

    def _set_soc_wh(self, soc_wh: float) -> None:
        self._set_soc_wh_unchecked(self._validate_soc_wh(soc_wh))

    def _set_soc_percentage(self, soc_percentage: float) -> None:
        validated_percentage = self._validate_soc_percentage(soc_percentage)
        self._set_soc_wh_unchecked((validated_percentage / 100.0) * self.capacity_wh)

    def discharge_energy(self, wh: float, dt: float = 1.0) -> tuple[float, float]:
        """Discharge energy from the battery.

        Returns:
            tuple[float, float]: (delivered_wh, losses_wh)
        """
        s = self._soc_wh
        min_s = self.min_soc_wh
        eff = self.discharging_efficiency
        max_power = self.max_discharge_power_w * dt

        usable_raw = s - min_s
        if usable_raw <= 0.0:
            return 0.0, 0.0

        deliverable_by_energy = usable_raw * eff
        max_deliverable = deliverable_by_energy if deliverable_by_energy < max_power else max_power
        requested = wh if wh < max_deliverable else max_deliverable
        raw_req = requested / eff
        raw_used = raw_req if raw_req < usable_raw else usable_raw
        delivered = raw_used * eff
        losses = raw_used - delivered

        s -= raw_used
        if s < min_s:
            s = min_s
        self._set_soc_wh_unchecked(s)

        return delivered, losses

    def charge_energy(self, wh: float, dt: float = 1.0) -> tuple[float, float]:
        """Charge energy into the battery.

        Returns:
            tuple[float, float]: (stored_wh, losses_wh)
        """
        s = self._soc_wh
        max_s = self.max_soc_wh
        eff = self.charging_efficiency
        max_power = self.max_charge_power_w * dt

        headroom = max_s - s
        if headroom <= 0.0:
            return 0.0, 0.0

        max_raw_headroom = headroom / eff
        max_raw = max_raw_headroom if max_raw_headroom < max_power else max_power
        raw = wh if wh < max_raw else max_raw
        stored = raw * eff

        s += stored
        if s > max_s:
            s = max_s
        self._set_soc_wh_unchecked(s)

        losses = raw - stored
        return stored, losses

    def current_raw_deliverable_energy_content(self) -> float:
        """Returns the current raw deliverable energy in the battery (pre-efficiency)."""
        return max(self._soc_wh - self.min_soc_wh, 0.0)

    def current_deliverable_energy_content(self) -> float:
        """Returns the current deliverable energy in the battery."""
        usable_energy = (self._soc_wh - self.min_soc_wh) * self.discharging_efficiency
        return max(usable_energy, 0.0)
