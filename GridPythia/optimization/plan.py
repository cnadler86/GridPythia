"""Inverter plan: per-timestep schedule and time-window consumer API.

``InverterPlan`` is the raw solver output for a single inverter device.
``PlanWindow`` groups consecutive same-mode slots into a compact, readable
block that is easier to act on than raw per-slot arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from GridPythia.simulation.devices import InverterMode


@dataclass(frozen=True, slots=True)
class PlanWindow:
    """Contiguous block of timesteps sharing the same inverter mode.

    Energy arrays inside a window are *views* into the parent :class:`InverterPlan`
    arrays — no copying occurs when iterating windows.
    """

    start_idx: int  # inclusive index into the full plan
    end_idx: int  # exclusive index into the full plan
    mode: InverterMode
    charge_ac_wh: np.ndarray  # AC energy drawn from grid for charging [Wh/slot]
    discharge_ac_wh: np.ndarray  # AC energy delivered to home from battery [Wh/slot]
    pv_to_ac_wh: np.ndarray  # PV energy routed directly to AC bus [Wh/slot]
    pv_to_battery_wh: np.ndarray  # PV energy routed into battery [Wh/slot]
    battery_soc_wh: np.ndarray | None  # SoC at end of each slot [Wh], or None

    @property
    def steps(self) -> int:
        """Number of timestep slots in this window."""
        return self.end_idx - self.start_idx

    @property
    def total_charge_ac_wh(self) -> float:
        """Total AC charge energy across the window [Wh]."""
        return float(self.charge_ac_wh.sum())

    @property
    def total_discharge_ac_wh(self) -> float:
        """Total AC discharge energy across the window [Wh]."""
        return float(self.discharge_ac_wh.sum())

    @property
    def total_pv_to_ac_wh(self) -> float:
        """Total PV-to-AC energy across the window [Wh]."""
        return float(self.pv_to_ac_wh.sum())

    @property
    def total_pv_to_battery_wh(self) -> float:
        """Total PV-to-battery energy across the window [Wh]."""
        return float(self.pv_to_battery_wh.sum())


@dataclass(frozen=True, slots=True)
class InverterPlan:
    """Raw per-timestep schedule produced by the optimizer for one inverter.

    Per-slot arrays are indexed ``[0 … steps-1]`` and correspond 1-to-1 with
    the :class:`~GridPythia.prediction.prediction.PredictionData` timestamps of
    the solution they belong to.

    Use :meth:`windows` to obtain a time-window view that is easier to
    display or act on (consecutive same-mode slots are merged into a single
    :class:`PlanWindow`).
    """

    device_id: str
    modes: np.ndarray  # InverterMode int per timestep (np.int8)
    charge_ac_wh: np.ndarray  # AC energy drawn from grid for charging [Wh/dt]
    discharge_ac_wh: np.ndarray  # AC energy delivered to home from battery [Wh/dt]
    pv_to_ac_wh: np.ndarray  # PV energy routed directly to AC [Wh/dt]
    pv_to_battery_wh: np.ndarray  # PV energy routed into battery [Wh/dt]
    battery_soc_wh: np.ndarray | None = None  # SoC at end of each slot [Wh]

    @property
    def steps(self) -> int:
        """Number of timestep slots in the plan."""
        return int(self.modes.shape[0])

    def windows(self) -> list[PlanWindow]:
        """Group consecutive same-mode slots into contiguous :class:`PlanWindow` objects.

        Returns an empty list for a zero-length plan.  The returned windows
        cover the full plan without gaps and in chronological order.
        Array slices inside each window are views — no data is copied.
        """
        T = self.steps
        if T == 0:
            return []

        result: list[PlanWindow] = []
        start = 0
        current_mode_int = int(self.modes[0])

        for i in range(1, T + 1):
            if i == T or int(self.modes[i]) != current_mode_int:
                sl = slice(start, i)
                result.append(
                    PlanWindow(
                        start_idx=start,
                        end_idx=i,
                        mode=InverterMode(current_mode_int),
                        charge_ac_wh=self.charge_ac_wh[sl],
                        discharge_ac_wh=self.discharge_ac_wh[sl],
                        pv_to_ac_wh=self.pv_to_ac_wh[sl],
                        pv_to_battery_wh=self.pv_to_battery_wh[sl],
                        battery_soc_wh=(
                            self.battery_soc_wh[sl] if self.battery_soc_wh is not None else None
                        ),
                    )
                )
                if i < T:
                    start = i
                    current_mode_int = int(self.modes[i])

        return result

    # ------------------------------------------------------------------
    # Dict-style access kept for backward compatibility with consumers
    # that index plans as plan["charge_ac_wh"] etc.
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return getattr(self, key, default)
