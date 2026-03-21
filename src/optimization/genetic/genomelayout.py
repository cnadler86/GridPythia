"""Genome encoding/decoding for the genetic optimization algorithm."""

from dataclasses import dataclass, field
from typing import Optional

from src.simulation.devices import InverterMode, SystemTopology
from src.simulation.devices.inverterbase import InverterBase


@dataclass
class InverterSegmentSpec:
    """Per-inverter segment structure in the flat genome."""

    inverter_index: int
    mode_slice: slice
    rate_slice: slice
    mode_count: int
    rate_count: int
    discharge_rate_slice: Optional[slice] = field(default=None)
    discharge_rate_count: int = 0

    @property
    def genome_slice(self) -> slice:
        return self.mode_slice


@dataclass(slots=True)
class DecodedGenome:
    """Decoded representation of a flat genome."""

    inverter_modes: list[list[InverterMode]]
    inverter_ac_rates: list[list[float]]
    home_appliance_starts: list[Optional[int]]


class GenomeLayout:
    """Defines and decodes the flat genome structure."""

    def __init__(
        self,
        inverters: list[InverterBase],
        prediction_hours: int,
        home_appliance_count: int,
    ):
        self.prediction_hours = prediction_hours
        self.inverter_specs: list[InverterSegmentSpec] = []
        offset = 0

        for i, inv in enumerate(inverters):
            if not inv.is_optimizable:
                continue

            n_modes = len(inv.available_modes)

            n_ac_rates = (
                len([r for r in inv.charge_rates if r > 0.0])
                if any(
                    m in (InverterMode.AC_CHARGE, InverterMode.AC_CHARGE_ZERO_FEED_IN)
                    for m in inv.available_modes
                )
                else 0
            )

            is_ev_v2g = inv.topology == SystemTopology.EV_V2G
            n_discharge_rates = (
                len([r for r in inv.charge_rates if r > 0.0]) if is_ev_v2g else 0
            )

            mode_sl = slice(offset, offset + prediction_hours)
            rate_sl = slice(offset + prediction_hours, offset + 2 * prediction_hours)
            offset += 2 * prediction_hours

            discharge_rate_sl: Optional[slice] = None
            if n_discharge_rates > 0:
                discharge_rate_sl = slice(offset, offset + prediction_hours)
                offset += prediction_hours

            self.inverter_specs.append(
                InverterSegmentSpec(
                    inverter_index=i,
                    mode_slice=mode_sl,
                    rate_slice=rate_sl,
                    mode_count=n_modes,
                    rate_count=n_ac_rates,
                    discharge_rate_slice=discharge_rate_sl,
                    discharge_rate_count=n_discharge_rates,
                )
            )

        self.home_appliance_slice: Optional[slice] = (
            slice(offset, offset + home_appliance_count)
            if home_appliance_count > 0
            else None
        )
        self.total_length: int = offset + home_appliance_count

    def decode(
        self,
        genome: list[int],
        inverters: list[InverterBase],
    ) -> DecodedGenome:
        """Decode a flat genome into per-inverter modes and rates."""
        all_modes: list[list[InverterMode]] = []
        all_rates: list[list[float]] = []

        for spec in self.inverter_specs:
            inv = inverters[spec.inverter_index]
            mode_genes = genome[spec.mode_slice]
            ac_rate_genes = genome[spec.rate_slice]
            discharge_rate_genes = (
                genome[spec.discharge_rate_slice] if spec.discharge_rate_slice else []
            )
            modes_h, rates_h = self._decode_segment_split(
                mode_genes, ac_rate_genes, discharge_rate_genes, inv
            )
            all_modes.append(modes_h)
            all_rates.append(rates_h)

        appliance_starts: list[Optional[int]] = (
            list(genome[self.home_appliance_slice]) if self.home_appliance_slice else []
        )

        return DecodedGenome(
            inverter_modes=all_modes,
            inverter_ac_rates=all_rates,
            home_appliance_starts=appliance_starts,
        )

    def _decode_segment_split(
        self,
        mode_genes: list[int],
        ac_rate_genes: list[int],
        discharge_rate_genes: list[int],
        inv: InverterBase,
    ) -> tuple[list[InverterMode], list[float]]:
        modes_list = inv.available_modes
        ac_rates = [r for r in inv.charge_rates if r > 0.0] or [1.0]
        discharge_rates = (
            [r for r in inv.charge_rates if r > 0.0] if discharge_rate_genes else []
        ) or [1.0]

        n_modes = len(modes_list)
        n_ac = len(ac_rates)
        n_dc = len(discharge_rates)

        decoded_modes: list[InverterMode] = []
        decoded_rates: list[float] = []

        for i, mg in enumerate(mode_genes):
            idx = max(0, min(mg, n_modes - 1))
            mode = modes_list[idx]

            if mode in (InverterMode.AC_CHARGE, InverterMode.AC_CHARGE_ZERO_FEED_IN):
                rg = ac_rate_genes[i] if i < len(ac_rate_genes) else 0
                rate = ac_rates[max(0, min(rg, n_ac - 1))]
            elif mode == InverterMode.DISCHARGE and discharge_rate_genes:
                rg = discharge_rate_genes[i] if i < len(discharge_rate_genes) else 0
                rate = discharge_rates[max(0, min(rg, n_dc - 1))]
            else:
                rate = 1.0

            decoded_modes.append(mode)
            decoded_rates.append(rate)

        return decoded_modes, decoded_rates
