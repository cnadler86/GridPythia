"""Tests for GenomeLayout encoding and decoding."""

import pytest

from src.optimization.genetic.genomelayout import (
    DecodedGenome,
    GenomeLayout,
)
from src.simulation.devices import InverterMode
from src.simulation.devices.battery import Battery, BatteryParameters
from src.simulation.devices.inverterbase import InverterBase, InverterParameters


@pytest.fixture
def battery() -> Battery:
    return Battery(
        BatteryParameters(
            device_id="bat1",
            capacity_wh=10000,
            initial_soc_percentage=50,
            min_soc_percentage=10,
        ),
        prediction_hours=6,
    )


@pytest.fixture
def pv_hybrid_inv(battery) -> InverterBase:
    params = InverterParameters(
        device_id="hybrid1",
        max_ac_output_power_w=5000,
        battery_id="bat1",
        pv_source="pv_main",
        dc_to_ac_efficiency=0.95,
        ac_to_dc_efficiency=0.95,
        max_ac_charge_power_w=3000,
        zero_feed_in=True,
        ac_rates=[0.0, 0.5, 1.0],
    )
    return InverterBase(parameters=params, battery=battery)


@pytest.fixture
def pv_only_inv() -> InverterBase:
    params = InverterParameters(
        device_id="pv_only",
        max_ac_output_power_w=800,
        pv_source="pv_balkon",
        dc_to_ac_efficiency=0.97,
    )
    return InverterBase(parameters=params)


@pytest.fixture
def pv_battery_inv(battery) -> InverterBase:
    params = InverterParameters(
        device_id="pv_bat1",
        max_ac_output_power_w=5000,
        battery_id="bat1",
        pv_source="pv_roof",
        dc_to_ac_efficiency=0.95,
        ac_to_dc_efficiency=0.0,
        max_ac_charge_power_w=0,
        zero_feed_in=True,
    )
    return InverterBase(parameters=params, battery=battery)


@pytest.fixture
def ev_v2g_inv(battery) -> InverterBase:
    params = InverterParameters(
        device_id="ev_v2g",
        max_ac_output_power_w=7400,
        battery_id="bat1",
        pv_source=None,
        dc_to_ac_efficiency=0.95,
        ac_to_dc_efficiency=0.95,
        max_ac_charge_power_w=7400,
        zero_feed_in=False,
        ac_rates=[0.0, 0.5, 1.0],
    )
    return InverterBase(parameters=params, battery=battery)


class TestLayoutConstruction:
    def test_single_optimizable_inverter(self, pv_hybrid_inv):
        layout = GenomeLayout(
            [pv_hybrid_inv], prediction_hours=6, home_appliance_count=0
        )
        assert len(layout.inverter_specs) == 1
        spec = layout.inverter_specs[0]
        assert spec.inverter_index == 0
        assert spec.mode_slice == slice(0, 6)
        assert spec.rate_slice == slice(6, 12)
        assert spec.genome_slice == slice(0, 6)
        assert spec.mode_count > 1
        assert layout.total_length == 12
        assert layout.home_appliance_slice is None

    def test_non_optimizable_excluded(self, pv_only_inv):
        layout = GenomeLayout([pv_only_inv], prediction_hours=6, home_appliance_count=0)
        assert len(layout.inverter_specs) == 0
        assert layout.total_length == 0

    def test_mixed_inverters(self, pv_only_inv, pv_hybrid_inv):
        layout = GenomeLayout(
            [pv_only_inv, pv_hybrid_inv], prediction_hours=6, home_appliance_count=0
        )
        assert len(layout.inverter_specs) == 1
        assert layout.inverter_specs[0].inverter_index == 1
        assert layout.total_length == 12

    def test_multiple_optimizable(self, pv_hybrid_inv, pv_battery_inv):
        layout = GenomeLayout(
            [pv_hybrid_inv, pv_battery_inv], prediction_hours=6, home_appliance_count=0
        )
        assert len(layout.inverter_specs) == 2
        spec0, spec1 = layout.inverter_specs
        assert spec0.mode_slice == slice(0, 6)
        assert spec0.rate_slice == slice(6, 12)
        assert spec1.mode_slice == slice(12, 18)
        assert spec1.rate_slice == slice(18, 24)
        assert layout.total_length == 24

    def test_home_appliance_appended(self, pv_hybrid_inv):
        layout = GenomeLayout(
            [pv_hybrid_inv], prediction_hours=6, home_appliance_count=1
        )
        assert layout.home_appliance_slice == slice(12, 13)
        assert layout.total_length == 13

    def test_home_appliance_multiple(self, pv_hybrid_inv):
        layout = GenomeLayout(
            [pv_hybrid_inv], prediction_hours=6, home_appliance_count=2
        )
        assert layout.home_appliance_slice == slice(12, 14)
        assert layout.total_length == 14


class TestModeCount:
    def test_pv_hybrid_mode_count(self, pv_hybrid_inv):
        layout = GenomeLayout(
            [pv_hybrid_inv], prediction_hours=6, home_appliance_count=0
        )
        spec = layout.inverter_specs[0]
        assert spec.mode_count == 3
        assert spec.rate_count == 2
        assert spec.discharge_rate_count == 0
        assert spec.discharge_rate_slice is None

    def test_pv_battery_mode_count(self, pv_battery_inv):
        layout = GenomeLayout(
            [pv_battery_inv], prediction_hours=6, home_appliance_count=0
        )
        spec = layout.inverter_specs[0]
        assert spec.mode_count == 2
        assert spec.rate_count == 0
        assert spec.discharge_rate_count == 0

    def test_ev_v2g_mode_count(self, ev_v2g_inv):
        layout = GenomeLayout([ev_v2g_inv], prediction_hours=6, home_appliance_count=0)
        assert len(layout.inverter_specs) == 1
        spec = layout.inverter_specs[0]
        assert spec.mode_count == 3
        assert spec.rate_count == 2
        assert spec.discharge_rate_count == 2
        assert spec.mode_slice == slice(0, 6)
        assert spec.rate_slice == slice(6, 12)
        assert spec.discharge_rate_slice == slice(12, 18)
        assert layout.total_length == 18


class TestDecode:
    def test_decode_basic(self, pv_hybrid_inv):
        inverters = [pv_hybrid_inv]
        layout = GenomeLayout(inverters, prediction_hours=6, home_appliance_count=0)

        genome = [0] * layout.total_length
        decoded = layout.decode(genome, inverters)

        assert isinstance(decoded, DecodedGenome)
        assert len(decoded.inverter_modes) == 1
        assert len(decoded.inverter_modes[0]) == 6
        assert len(decoded.inverter_ac_rates) == 1
        assert len(decoded.inverter_ac_rates[0]) == 6
        assert decoded.home_appliance_starts == []

        assert all(m == InverterMode.IDLE for m in decoded.inverter_modes[0])
        assert all(r == 1.0 for r in decoded.inverter_ac_rates[0])

    def test_decode_all_modes_covered(self, pv_hybrid_inv):
        inverters = [pv_hybrid_inv]
        layout = GenomeLayout(inverters, prediction_hours=6, home_appliance_count=0)
        spec = layout.inverter_specs[0]

        mode_genes = list(range(min(6, spec.mode_count)))
        while len(mode_genes) < 6:
            mode_genes.append(0)
        rate_genes = [0] * 6
        genome = mode_genes + rate_genes

        decoded = layout.decode(genome, inverters)
        assert len(decoded.inverter_modes[0]) == 6
        # All decoded modes should be valid InverterMode values
        for m in decoded.inverter_modes[0]:
            assert isinstance(m, InverterMode)
