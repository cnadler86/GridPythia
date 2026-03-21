"""Core utilities for the prediction framework."""

from abc import ABC, abstractmethod
from array import array


def make_array(values: list[float] | None = None, size: int = 0) -> array:
    """Create a float array, either from *values* or zero-filled to *size*."""
    if values is not None:
        return array("f", values)
    return array("f", [0.0] * size)


def n_steps(hours: float, dt_hours: float) -> int:
    """Number of discrete time steps that fit in *hours*."""
    return max(1, round(hours / dt_hours))


def resample(values: array, source_dt: float, target_dt: float) -> array:
    """Resample a time series via linear interpolation.

    Args:
        values: Source series (``array('f', ...)``).
        source_dt: Source step width in hours.
        target_dt: Target step width in hours.

    Returns:
        New array at *target_dt* resolution.
    """
    if abs(source_dt - target_dt) < 1e-9:
        return array("f", values)

    n_src = len(values)
    if n_src == 0:
        return array("f")

    total_hours = n_src * source_dt
    n_tgt = max(1, round(total_hours / target_dt))
    result = array("f", [0.0] * n_tgt)

    for i in range(n_tgt):
        t = i * target_dt / source_dt
        idx = int(t)
        frac = t - idx
        if idx >= n_src - 1:
            result[i] = values[n_src - 1]
        else:
            result[i] = values[idx] * (1.0 - frac) + values[idx + 1] * frac

    return result


class PredictionProvider(ABC):
    """Interface every prediction source must implement."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Short unique name, e.g. ``"EnergyCharts"``."""
        ...
