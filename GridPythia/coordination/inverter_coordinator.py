"""Inverter coordination layer – real-time state management.

``InverterCoordinator`` maintains an in-memory snapshot of every inverter's
last-reported runtime state (SoC, active mode, timestamp) and provides the
readiness checks needed before each optimization run.

``next_optimization_slot`` is a pure utility that returns when the scheduler
should next fire, given a fixed-interval grid aligned to the hour boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor
from typing import TYPE_CHECKING

from structlog import get_logger

from GridPythia.simulation.devices import InverterMode

if TYPE_CHECKING:
    from GridPythia.simulation.devices.inverterbase import InverterBase

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class InverterState:
    """Immutable snapshot of an inverter's last-reported runtime state."""

    device_id: str
    soc: float  # 0–100 %
    mode: InverterMode
    reported_at: datetime  # timezone-aware

    def age_s(self, now: datetime | None = None) -> float:
        """Seconds since this state was reported."""
        ts = now or datetime.now(tz=timezone.utc)
        return (ts - self.reported_at).total_seconds()

    def is_fresh(self, max_age_s: float, now: datetime | None = None) -> bool:
        """True when the state is not older than *max_age_s* seconds."""
        return self.age_s(now) <= max_age_s


class InverterCoordinator:
    """Tracks real-time inverter states; gates optimization on freshness.

    States are updated via :meth:`update_status` (called by the MQTT gateway
    or a manual API endpoint).  Before each optimization run the scheduler
    calls :meth:`all_optimizable_ready` to confirm every active inverter has
    a recent status.
    """

    def __init__(self, max_age_s: float = 300.0) -> None:
        self._max_age_s = max_age_s
        self._states: dict[str, InverterState] = {}

    # ── State ingestion ────────────────────────────────────────────────

    def update_status(
        self,
        device_id: str,
        soc: float,
        mode: InverterMode | int = InverterMode.IDLE,
        *,
        reported_at: datetime | None = None,
    ) -> InverterState:
        """Record a new status report for *device_id*.

        Args:
            device_id:   Inverter device identifier.
            soc:         State-of-charge in percent [0–100].
            mode:        Active :class:`InverterMode` (or its int value).
            reported_at: Timestamp; defaults to ``datetime.now(UTC)``.

        Returns:
            The stored :class:`InverterState`.
        """
        if not 0.0 <= soc <= 100.0:
            raise ValueError(f"soc must be in [0, 100], got {soc!r}")
        ts = reported_at or datetime.now(tz=timezone.utc)
        if ts.tzinfo is None:
            raise ValueError("reported_at must be timezone-aware")
        m = mode if isinstance(mode, InverterMode) else InverterMode(int(mode))
        state = InverterState(device_id=device_id, soc=soc, mode=m, reported_at=ts)
        self._states[device_id] = state
        logger.debug("inverter_status_updated", device_id=device_id, soc=soc, mode=m.name)
        return state

    # ── State queries ──────────────────────────────────────────────────

    def get_state(self, device_id: str) -> InverterState | None:
        """Return the latest state for *device_id*, or ``None`` if unknown."""
        return self._states.get(device_id)

    def is_fresh(self, device_id: str, now: datetime | None = None) -> bool:
        """True when *device_id* has a state and it is within the max-age window."""
        state = self._states.get(device_id)
        return state is not None and state.is_fresh(self._max_age_s, now)

    def all_optimizable_ready(
        self,
        inverters: list[InverterBase],
        now: datetime | None = None,
    ) -> tuple[bool, list[str]]:
        """Check whether every optimizable inverter has a fresh status.

        Returns ``(ready, missing)`` where *missing* lists the device IDs
        that are absent or stale.  Inverters with ``is_optimizable=False``
        (pure PV-only) are skipped.
        """
        missing: list[str] = []
        for inv in inverters:
            if not inv.is_optimizable:
                continue
            if not self.is_fresh(inv.device_id, now):
                missing.append(inv.device_id)
        return len(missing) == 0, missing

    # ── Optimizer input helpers ────────────────────────────────────────

    def get_soc_overrides_wh(self, inverters: list[InverterBase]) -> dict[str, float]:
        """Map inverter IDs to current SoC in Wh for the optimizer.

        Only includes inverters that have a battery **and** a fresh status.
        Inverters without a known state are omitted (optimizer uses battery
        default from config in that case).
        """
        result: dict[str, float] = {}
        for inv in inverters:
            if inv.battery is None:
                continue
            state = self._states.get(inv.device_id)
            if state is None:
                continue
            soc_wh = (state.soc / 100.0) * float(inv.battery.capacity_wh)
            result[inv.device_id] = soc_wh
        return result

    def get_initial_modes(self, inverters: list[InverterBase]) -> dict[str, InverterMode]:
        """Return currently reported modes keyed by inverter device ID."""
        return {
            inv.device_id: self._states[inv.device_id].mode
            for inv in inverters
            if inv.device_id in self._states
        }

    def snapshot(self) -> dict[str, InverterState]:
        """Return a shallow copy of all current states."""
        return dict(self._states)


# ── Grid-alignment utility ─────────────────────────────────────────────────


def next_optimization_slot(
    now: datetime,
    interval_minutes: int = 15,
) -> datetime:
    """Return the next optimization slot aligned to the hour boundary.

    Grid slots are defined as :math:`0, N, 2N, \\ldots` minutes past the hour
    where *N* = ``interval_minutes``.  Fires *at* the boundary (not before).

    Args:
        now:               Reference time (timezone-aware recommended).
        interval_minutes:  Step width in minutes; must be a divisor of 60.

    Returns:
        The first grid boundary strictly after *now* (or equal, with a 1-second
        grace period to avoid floating-point drift).

    Raises:
        ValueError: If *interval_minutes* is not a positive divisor of 60.

    Example:
        >>> now = datetime(2026, 4, 23, 14, 7, 0, tzinfo=timezone.utc)
        >>> next_optimization_slot(now, 15)
        datetime(2026, 4, 23, 14, 15, 0, tzinfo=timezone.utc)
    """
    if interval_minutes <= 0 or 60 % interval_minutes != 0:
        raise ValueError(
            f"interval_minutes must be a positive divisor of 60, got {interval_minutes}"
        )

    interval_s = interval_minutes * 60
    epoch = now.timestamp()
    slot_epoch = (floor(epoch / interval_s) + 1) * interval_s

    # Grace: if we're within 1 s of the current slot, stay on it.
    current_slot = floor(epoch / interval_s) * interval_s
    if epoch - current_slot <= 1.0:
        slot_epoch = current_slot

    tz = now.tzinfo or timezone.utc
    return datetime.fromtimestamp(slot_epoch, tz=tz)
