"""Self-consumption probability interpolator for PV systems."""

import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class SelfConsumptionProbabilityInterpolator:
    def __init__(self, filepath: str | Path):
        self.filepath = filepath
        with open(self.filepath, "rb") as file:
            self.interpolator: RegularGridInterpolator = pickle.load(file)  # noqa: S301

    def calculate_self_consumption(self, load_1h_power: float, pv_power: float) -> float:
        """Calculate the PV self-consumption rate using RegularGridInterpolator.

        Args:
            load_1h_power: 1h power levels (W).
            pv_power: Current PV power output (W).

        Returns:
            Self-consumption rate as a float.
        """
        partial_loads = np.arange(0, pv_power + 50, 50)
        points = np.array([np.full_like(partial_loads, load_1h_power), partial_loads]).T
        probabilities = self.interpolator(points)
        return probabilities.sum()


# Module-level singleton
_interpolator: SelfConsumptionProbabilityInterpolator | None = None


def get_load_interpolator() -> SelfConsumptionProbabilityInterpolator:
    """Get or create the load interpolator singleton."""
    global _interpolator
    if _interpolator is None:
        filepath = Path(__file__).parent.parent.resolve() / "data" / "regular_grid_interpolator.pkl"
        _interpolator = SelfConsumptionProbabilityInterpolator(filepath)
    return _interpolator
