"""Home appliance device simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HomeApplianceParameters:
    """Home appliance configuration."""

    device_id: str = "dishwasher"
    consumption_wh: int = 2000
    duration_h: int = 3


@dataclass
class HomeAppliance:
    """Home appliance with schedulable load curve."""

    parameters: HomeApplianceParameters
    optimization_hours: int
    prediction_hours: int
    load_curve: list[float] = field(init=False)
    duration_h: int = field(init=False)
    consumption_wh: int = field(init=False)
    start_earliest: int = field(init=False)
    start_latest: int = field(init=False)

    def __post_init__(self) -> None:
        self.load_curve = [0.0] * self.prediction_hours
        self.duration_h = self.parameters.duration_h
        self.consumption_wh = self.parameters.consumption_wh
        # Default: allow start at any hour within prediction horizon
        self.start_earliest = 0
        self.start_latest = max(0, self.prediction_hours - self.duration_h)

    def set_starting_time(self, start_hour: int, global_start_hour: int = 0) -> int:
        """Sets the start time and generates the load curve."""
        if start_hour < 0 or start_hour >= self.prediction_hours:
            start_hour = max(0, min(start_hour, self.prediction_hours - 1))

        self.reset_load_curve()
        power_per_hour = self.consumption_wh / self.duration_h

        if start_hour < len(self.load_curve):
            end_hour = min(start_hour + self.duration_h, self.prediction_hours)
            for h in range(start_hour, end_hour):
                self.load_curve[h] = power_per_hour

        return start_hour

    def reset_load_curve(self) -> None:
        """Resets the load curve."""
        self.load_curve = [0.0] * self.prediction_hours

    def get_load_curve(self) -> list[float]:
        """Returns the current load curve."""
        return self.load_curve

    def get_load_for_hour(self, hour: int) -> float:
        """Returns the load for a specific hour."""
        if hour < 0 or hour >= self.prediction_hours:
            raise ValueError(
                f"The specified hour {hour} is outside the available time frame {self.prediction_hours}."
            )
        return self.load_curve[hour]
