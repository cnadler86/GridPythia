"""Inverter device simulation with topology-aware energy processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

from structlog import get_logger

from GridPythia.config.optimization import DEFAULT_AC_RATES, InverterParameters
from GridPythia.simulation.devices import (
    EnergyFlowResult,
    InverterMode,
    SystemTopology,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from GridPythia.simulation.devices.battery import Battery

logger = get_logger(__name__)


class InverterBase:
    """Topology-aware inverter at the DC/AC boundary."""

    _ZERO_FEED_IN_MODES: ClassVar[frozenset[InverterMode]] = frozenset(
        {InverterMode.DISCHARGE_ZERO_FEED_IN, InverterMode.AC_CHARGE_ZERO_FEED_IN}
    )
    _RATE_REQUIRED_MODES: ClassVar[frozenset[InverterMode]] = frozenset(
        {InverterMode.DISCHARGE, InverterMode.AC_CHARGE}
    )

    __slots__ = (
        "device_id",
        "parameters",
        "battery",
        "current_state",
        "_max_ac_output_power_w",
        "_max_ac_charge_power_w",
        "_dc_to_ac_efficiency",
        "_ac_to_dc_efficiency",
        "_zero_feed_in",
        "_has_pv",
        "charge_rates",
        "discharge_rates",
        "topology",
        "available_modes",
        "_mode_dispatch",
        "is_optimizable",
        "_log",
    )

    def __init__(
        self,
        parameters: InverterParameters,
        battery: Optional[Battery] = None,
    ):
        self.parameters: InverterParameters = parameters
        self.battery: Optional[Battery] = battery
        self.current_state: InverterMode = InverterMode.IDLE
        self.device_id = self.parameters.device_id
        self._log = logger.bind(device_id=self.device_id, component="inverter")

        if self.battery and self.parameters.battery_id != self.battery.parameters.device_id:
            raise ValueError(
                f"Battery ID mismatch - {self.parameters.battery_id} is configured; "
                f"got {self.battery.parameters.device_id}."
            )
        if self.battery is None and self.parameters.battery_id is not None:
            raise ValueError(
                f"Inverter '{self.device_id}' has battery_id '{self.parameters.battery_id}' but no battery provided."
            )

        self._max_ac_output_power_w = self.parameters.max_ac_output_power_w
        self._max_ac_charge_power_w = self.parameters.max_ac_charge_power_w
        self._dc_to_ac_efficiency = self.parameters.dc_to_ac_efficiency
        self._ac_to_dc_efficiency = self.parameters.ac_to_dc_efficiency
        self._zero_feed_in = self.parameters.zero_feed_in
        self._has_pv = self.parameters.pv_source is not None

        self.topology = self._resolve_topology()
        # store available modes as an immutable, ordered tuple (no duplicates)
        self.available_modes = self._resolve_available_modes()

        self.charge_rates = tuple()
        self.discharge_rates = tuple()

        # if any of the rate-required modes are available, populate rates
        if InverterMode.AC_CHARGE in self.available_modes:
            self.charge_rates = DEFAULT_AC_RATES
            if self.parameters.ac_rates_pct:
                self.charge_rates = tuple(sorted({int(r) for r in self.parameters.ac_rates_pct}))
        if InverterMode.DISCHARGE in self.available_modes:
            self.discharge_rates = DEFAULT_AC_RATES
            if self.parameters.ac_rates_pct:
                self.discharge_rates = tuple(
                    sorted({int(r) for r in self.parameters.ac_rates_pct if 0 < int(r) <= 100})
                )

        self.is_optimizable: bool = (
            self.topology != SystemTopology.PV_ONLY and len(self.available_modes) > 1
        )

        self._mode_dispatch: dict[InverterMode, Callable[..., EnergyFlowResult]] = {
            InverterMode.IDLE: self._process_idle,
            InverterMode.DISCHARGE: self._process_discharge,
            InverterMode.DISCHARGE_ZERO_FEED_IN: self._process_discharge,
            InverterMode.AC_CHARGE: self._process_ac_charge,
            InverterMode.AC_CHARGE_ZERO_FEED_IN: self._process_ac_charge,
        }
        self._log.info(
            "inverter_setup_complete",
            topology=self.topology,
            pv_source=self.parameters.pv_source,
            available_modes=[m.name for m in self.available_modes],
            is_optimizable=self.is_optimizable,
        )

    def _resolve_topology(self) -> SystemTopology:
        has_pv = self._has_pv
        has_bat = self.battery is not None
        can_ac_charge = (
            self._ac_to_dc_efficiency > 0 and self._max_ac_charge_power_w > 0 and has_bat
        )
        can_discharge = self._dc_to_ac_efficiency > 0 and self._max_ac_output_power_w > 0

        if has_pv and not has_bat:
            return SystemTopology.PV_ONLY
        elif has_pv and has_bat and not can_ac_charge:
            return SystemTopology.PV_BATTERY
        elif has_pv and has_bat and can_ac_charge:
            return SystemTopology.PV_HYBRID
        elif not has_pv and has_bat and can_ac_charge and can_discharge:
            if self._zero_feed_in:
                return SystemTopology.AC_BATTERY
            else:
                return SystemTopology.EV_V2G
        elif not has_pv and has_bat and can_ac_charge and not can_discharge:
            return SystemTopology.EV_CHARGE_ONLY
        elif not has_pv and has_bat and not can_ac_charge and can_discharge:
            raise ValueError("Battery without input source.")
        else:
            raise ValueError("Invalid inverter configuration: cannot determine topology.")

    def _resolve_available_modes(self) -> tuple[InverterMode, ...]:
        modes: list[InverterMode] = [InverterMode.IDLE]

        can_discharge = (
            self._dc_to_ac_efficiency > 0 and self._max_ac_output_power_w > 0 and self.battery
        )
        can_ac_charge = (
            self._ac_to_dc_efficiency > 0 and self._max_ac_charge_power_w > 0 and self.battery
        )

        if can_discharge:
            # if zero feed-in is allowed, include the zero feed-in variant of DISCHARGE; otherwise just DISCHARGE
            if self._zero_feed_in:
                modes.append(InverterMode.DISCHARGE_ZERO_FEED_IN)
            else:
                modes.append(InverterMode.DISCHARGE)

        if can_ac_charge:
            modes.append(InverterMode.AC_CHARGE)
            if self._zero_feed_in and not self._has_pv:
                modes.append(InverterMode.AC_CHARGE_ZERO_FEED_IN)

        # preserve order as determined by the logic above, remove duplicates
        return tuple(dict.fromkeys(modes))

    def process_energy(
        self,
        generation: float,
        mode: InverterMode,
        dt: float = 1.0,
        *,
        ac_rate_pct: Optional[int] = None,
        energy_wh: Optional[float] = None,
    ) -> EnergyFlowResult:
        """Process energy at DC/AC boundary for one time step."""
        if generation > 0 and not self._has_pv:
            raise ValueError(
                f"Inverter '{self.parameters.device_id}': generation={generation} > 0 but no PV source"
            )
        if mode not in self.available_modes:
            raise ValueError(
                f"Inverter '{self.parameters.device_id}': mode {mode} not in available_modes {self.available_modes}"
            )
        if mode in self._ZERO_FEED_IN_MODES and energy_wh is None:
            raise ValueError(
                f"Inverter '{self.parameters.device_id}': energy_wh must be provided for zero feed-in modes"
            )
        if mode in self._RATE_REQUIRED_MODES and ac_rate_pct is None:
            raise ValueError(
                f"Inverter '{self.parameters.device_id}': ac_rate_pct must be provided for DISCHARGE and AC_CHARGE modes"
            )

        self.current_state = mode
        handler = self._mode_dispatch[mode]

        if mode in self._ZERO_FEED_IN_MODES:
            return handler(generation, dt, energy_wh=energy_wh)
        if mode in self._RATE_REQUIRED_MODES:
            if not isinstance(ac_rate_pct, int):
                raise ValueError(
                    f"Inverter '{self.parameters.device_id}': ac_rate_pct must be an integer percent in [1, 100]"
                )
            if ac_rate_pct < 1 or ac_rate_pct > 100:
                raise ValueError(
                    f"Inverter '{self.parameters.device_id}': ac_rate_pct must be within [1, 100], got {ac_rate_pct}"
                )
            return handler(generation, dt, ac_rate_pct=ac_rate_pct)
        return handler(generation, dt)

    def _process_idle(self, generation_wh: float, dt: float) -> EnergyFlowResult:
        """IDLE: PV → AC; excess charges battery if available."""
        ac_output_wh = 0.0
        ac_input_wh = 0.0
        losses_wh = 0.0

        if generation_wh > 0:
            dc2ac = self._dc_to_ac_efficiency
            max_ac_out_dt = self._max_ac_output_power_w * dt
            if self.battery:
                charged_dc_wh, charge_losses = self.battery.charge_energy(wh=generation_wh, dt=dt)
                losses_wh += charge_losses
                available_after_battery = generation_wh - (charged_dc_wh + charge_losses)
                ac_available_wh = available_after_battery * dc2ac
                losses_wh += available_after_battery - ac_available_wh
            else:
                ac_available_wh = generation_wh * dc2ac
                losses_wh += generation_wh - ac_available_wh

            ac_output_wh = ac_available_wh if ac_available_wh < max_ac_out_dt else max_ac_out_dt
            if ac_output_wh < ac_available_wh:
                losses_wh += ac_available_wh - ac_output_wh

        return EnergyFlowResult(
            ac_output_wh=ac_output_wh,
            ac_input_wh=ac_input_wh,
            losses_wh=losses_wh,
            pv_ac_wh=ac_output_wh,
        )

    def _process_discharge(
        self,
        generation_wh: float,
        dt: float,
        *,
        ac_rate_pct: Optional[int] = None,
        energy_wh: Optional[float] = None,
    ) -> EnergyFlowResult:
        """DISCHARGE: (PV) + (Battery) → AC output."""
        ac_output_wh = 0.0
        ac_input_wh = 0.0
        losses_wh = 0.0

        dc2ac = self._dc_to_ac_efficiency
        max_discharge_ac_wh = self._max_ac_output_power_w * dt
        if ac_rate_pct is not None:
            max_discharge_ac_wh *= ac_rate_pct / 100.0

        if max_discharge_ac_wh <= 0:
            return self._process_idle(generation_wh, dt)

        if self.battery is None:
            if generation_wh > 0:
                ac_available_wh = generation_wh * dc2ac
                losses_wh = generation_wh - ac_available_wh
                ac_output_wh = (
                    ac_available_wh
                    if ac_available_wh < max_discharge_ac_wh
                    else max_discharge_ac_wh
                )
                if ac_output_wh < ac_available_wh:
                    losses_wh += ac_available_wh - ac_output_wh
            return EnergyFlowResult(ac_output_wh, ac_input_wh, losses_wh, pv_ac_wh=ac_output_wh)

        battery = self.battery

        if energy_wh is not None:
            # Mode 1: DISCHARGE_ZERO_FEED_IN
            requested_ac = energy_wh
            ac_budget = requested_ac if requested_ac < max_discharge_ac_wh else max_discharge_ac_wh

            pv_used_dc = 0.0
            ac_from_pv = 0.0
            if generation_wh > 0 and ac_budget > 0:
                dc_needed = ac_budget / dc2ac
                pv_used_dc = generation_wh if generation_wh < dc_needed else dc_needed
                ac_from_pv = pv_used_dc * dc2ac
                losses_wh += pv_used_dc - ac_from_pv

            bat_ac = 0.0
            req = ac_budget - ac_from_pv
            if req > 0:
                needed_dc = req / dc2ac
                delivered_dc, bat_losses_dc = battery.discharge_energy(needed_dc, dt=dt)
                bat_ac = delivered_dc * dc2ac
                losses_wh += bat_losses_dc + (delivered_dc - bat_ac)

            total_ac = ac_from_pv + bat_ac
            ac_output_wh = total_ac if total_ac < max_discharge_ac_wh else max_discharge_ac_wh
            if ac_output_wh < total_ac:
                losses_wh += total_ac - ac_output_wh

            pv_leftover_dc = generation_wh - pv_used_dc
            if pv_leftover_dc > 0:
                stored, charge_losses = battery.charge_energy(wh=pv_leftover_dc, dt=dt)
                losses_wh += charge_losses
                curtailed_dc = pv_leftover_dc - (stored + charge_losses)
                if curtailed_dc > 0:
                    max_forced_ac = curtailed_dc * dc2ac
                    remaining_ac_cap = max_discharge_ac_wh - ac_output_wh
                    forced_ac = (
                        max_forced_ac
                        if max_forced_ac < remaining_ac_cap
                        else max(remaining_ac_cap, 0.0)
                    )
                    losses_wh += curtailed_dc - forced_ac
                    ac_output_wh += forced_ac
                    ac_from_pv += forced_ac

            return EnergyFlowResult(ac_output_wh, ac_input_wh, losses_wh, pv_ac_wh=ac_from_pv)

        # Mode 2: DISCHARGE with ac_rate_pct
        pv_used_dc = 0.0
        pv_ac = 0.0
        if generation_wh > 0:
            dc_needed = max_discharge_ac_wh / dc2ac
            pv_used_dc = generation_wh if generation_wh < dc_needed else dc_needed
            pv_ac = pv_used_dc * dc2ac
            losses_wh += pv_used_dc - pv_ac

        bat_ac = 0.0
        if pv_ac < max_discharge_ac_wh:
            remaining_ac_budget = max_discharge_ac_wh - pv_ac
            needed_dc = remaining_ac_budget / dc2ac
            delivered_dc, bat_losses_dc = battery.discharge_energy(needed_dc, dt=dt)
            bat_ac = delivered_dc * dc2ac
            losses_wh += bat_losses_dc + (delivered_dc - bat_ac)

        total_ac = pv_ac + bat_ac
        ac_output_wh = total_ac if total_ac < max_discharge_ac_wh else max_discharge_ac_wh
        if ac_output_wh < total_ac:
            losses_wh += total_ac - ac_output_wh

        pv_leftover_dc = generation_wh - pv_used_dc
        if pv_leftover_dc > 0:
            stored, charge_losses = battery.charge_energy(wh=pv_leftover_dc, dt=dt)
            losses_wh += charge_losses
            curtailed = pv_leftover_dc - (stored + charge_losses)
            if curtailed > 0:
                losses_wh += curtailed

        return EnergyFlowResult(ac_output_wh, ac_input_wh, losses_wh, pv_ac_wh=pv_ac)

    def _process_ac_charge(
        self,
        generation: float,
        dt: float,
        *,
        ac_rate_pct: Optional[int] = None,
        energy_wh: Optional[float] = None,
    ) -> EnergyFlowResult:
        """AC_CHARGE: AC bus → Battery. PV prioritized over AC."""
        ac_output = 0.0
        ac_input = 0.0
        losses = 0.0

        if self.battery is None:
            if generation > 0:
                ac_available = generation * self._dc_to_ac_efficiency
                losses = generation - ac_available
                ac_output = min(ac_available, self._max_ac_output_power_w)
                if ac_output < ac_available:
                    losses += ac_available - ac_output
            return EnergyFlowResult(ac_output, ac_input, losses, pv_ac_wh=ac_output)

        battery = self.battery
        bat_charge_eff = battery.charging_efficiency
        pv_ac_wh = 0.0

        if generation > 0:
            charged_pv, charge_losses_pv = battery.charge_energy(wh=generation, dt=dt)
            losses += charge_losses_pv
            raw_input_pv = charged_pv / bat_charge_eff
            curtailed_pv_dc = generation - raw_input_pv
            if curtailed_pv_dc > 0:
                curtailed_pv_ac = curtailed_pv_dc * self._dc_to_ac_efficiency
                losses += curtailed_pv_dc - curtailed_pv_ac
                ac_output += curtailed_pv_ac
                pv_ac_wh = curtailed_pv_ac

        if energy_wh is not None:
            max_ac_charge = energy_wh
        elif ac_rate_pct is not None:
            max_ac_charge = self._max_ac_charge_power_w * dt * (ac_rate_pct / 100.0)
        else:
            raise ValueError(
                f"Inverter '{self.parameters.device_id}': either energy_wh or ac_rate_pct must be provided"
            )

        if max_ac_charge > 0:
            headroom_wh = battery.max_soc_wh - battery.soc_wh
            if headroom_wh > 0.0:
                dc_headroom = headroom_wh / bat_charge_eff
                dc_budget = min(max_ac_charge * self._ac_to_dc_efficiency, dc_headroom)

                if dc_budget > 0:
                    charged_ac, charge_losses_ac = battery.charge_energy(wh=dc_budget, dt=dt)
                    losses += charge_losses_ac
                    ac_energy = (charged_ac + charge_losses_ac) / self._ac_to_dc_efficiency
                    ac_input = ac_energy

        return EnergyFlowResult(ac_output, ac_input, losses, pv_ac_wh=pv_ac_wh)
