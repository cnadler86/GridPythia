"""Tests for topology-aware InverterBase."""

import pytest

from GridPythia.simulation.devices import InverterMode, SystemTopology
from GridPythia.config.optimization import BatteryParameters, DEFAULT_AC_RATES, InverterParameters
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


@pytest.fixture
def battery() -> Battery:
    return Battery(
        BatteryParameters(
            device_id="battery1",
            capacity_wh=5000,
            initial_soc_percentage=50,
            min_soc_percentage=10,
            max_soc_percentage=100,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            max_charge_power_w=5000,
        ),
        prediction_hours=48,
    )


@pytest.fixture
def pv_hybrid_params() -> InverterParameters:
    return InverterParameters(
        device_id="inverter_hybrid",
        max_ac_output_power_w=2000,
        battery_id="battery1",
        pv_source="pv_main",
        dc_to_ac_efficiency=0.95,
        ac_to_dc_efficiency=0.90,
        max_ac_charge_power_w=2000,
        zero_feed_in=True,
        ac_rates_pct=(25, 50, 75, 100),
    )


@pytest.fixture
def pv_only_params() -> InverterParameters:
    return InverterParameters(
        device_id="inverter_pv_only",
        max_ac_output_power_w=800,
        pv_source="pv_balkon",
        dc_to_ac_efficiency=0.97,
        ac_to_dc_efficiency=0.0,
        max_ac_charge_power_w=0,
        zero_feed_in=False,
    )


@pytest.fixture
def pv_battery_no_ac_params() -> InverterParameters:
    return InverterParameters(
        device_id="inverter_pv_bat",
        max_ac_output_power_w=5000,
        battery_id="battery1",
        pv_source="pv_roof",
        dc_to_ac_efficiency=0.95,
        ac_to_dc_efficiency=0.0,
        max_ac_charge_power_w=0,
        zero_feed_in=True,
    )


@pytest.fixture
def ac_battery_params() -> InverterParameters:
    return InverterParameters(
        device_id="inverter_ac_bat",
        max_ac_output_power_w=3000,
        battery_id="battery1",
        dc_to_ac_efficiency=0.95,
        ac_to_dc_efficiency=0.95,
        max_ac_charge_power_w=3000,
        zero_feed_in=True,
    )


class TestTopologyResolution:
    def test_pv_only_topology(self, pv_only_params):
        inv = InverterBase(pv_only_params, battery=None)
        assert inv.topology == SystemTopology.PV_ONLY

    def test_pv_battery_topology(self, pv_battery_no_ac_params, battery):
        inv = InverterBase(pv_battery_no_ac_params, battery=battery)
        assert inv.topology == SystemTopology.PV_BATTERY

    def test_pv_hybrid_topology(self, pv_hybrid_params, battery):
        inv = InverterBase(pv_hybrid_params, battery=battery)
        assert inv.topology == SystemTopology.PV_HYBRID

    def test_ac_battery_topology(self, ac_battery_params, battery):
        inv = InverterBase(ac_battery_params, battery=battery)
        assert inv.topology == SystemTopology.AC_BATTERY


class TestAvailableModes:
    def test_pv_only_has_idle_only(self, pv_only_params):
        inv = InverterBase(pv_only_params, battery=None)
        assert inv.available_modes == (InverterMode.IDLE,)

    def test_pv_battery_has_discharge(self, pv_battery_no_ac_params, battery):
        inv = InverterBase(pv_battery_no_ac_params, battery=battery)
        modes = inv.available_modes
        assert InverterMode.IDLE in modes
        assert InverterMode.DISCHARGE_ZERO_FEED_IN in modes
        assert InverterMode.DISCHARGE not in modes
        assert InverterMode.AC_CHARGE not in modes

    def test_pv_hybrid_has_all_modes(self, pv_hybrid_params, battery):
        inv = InverterBase(pv_hybrid_params, battery=battery)
        modes = inv.available_modes
        assert InverterMode.IDLE in modes
        assert InverterMode.DISCHARGE_ZERO_FEED_IN in modes
        assert InverterMode.AC_CHARGE in modes
        assert InverterMode.DISCHARGE not in modes
        assert InverterMode.AC_CHARGE_ZERO_FEED_IN not in modes

    def test_ac_battery_has_charge_and_discharge(self, ac_battery_params, battery):
        inv = InverterBase(ac_battery_params, battery=battery)
        modes = inv.available_modes
        assert InverterMode.IDLE in modes
        assert InverterMode.DISCHARGE_ZERO_FEED_IN in modes
        assert InverterMode.DISCHARGE not in modes
        assert InverterMode.AC_CHARGE in modes
        assert InverterMode.AC_CHARGE_ZERO_FEED_IN in modes


class TestOptimizable:
    def test_pv_only_not_optimizable(self, pv_only_params):
        inv = InverterBase(pv_only_params, battery=None)
        assert inv.is_optimizable is False

    def test_pv_hybrid_is_optimizable(self, pv_hybrid_params, battery):
        inv = InverterBase(pv_hybrid_params, battery=battery)
        assert inv.is_optimizable is True

    def test_pv_battery_is_optimizable(self, pv_battery_no_ac_params, battery):
        inv = InverterBase(pv_battery_no_ac_params, battery=battery)
        assert inv.is_optimizable is True


class TestEnergyFlow:
    def test_idle_pv_no_battery(self, pv_only_params):
        inv = InverterBase(pv_only_params, battery=None)
        res = inv.process_energy(generation=500, mode=InverterMode.IDLE, dt=1.0)
        assert res.ac_output_wh > 0
        assert res.ac_input_wh == 0
        assert res.losses_wh >= 0

    def test_idle_pv_with_battery(self, pv_hybrid_params, battery):
        inv = InverterBase(pv_hybrid_params, battery=battery)
        res = inv.process_energy(generation=1000, mode=InverterMode.IDLE, dt=1.0)
        assert res.losses_wh >= 0

    def test_ac_charge(self, pv_hybrid_params, battery):
        inv = InverterBase(pv_hybrid_params, battery=battery)
        initial_soc = battery.soc_wh
        res = inv.process_energy(
            generation=0, mode=InverterMode.AC_CHARGE, dt=1.0, ac_rate_pct=100
        )
        assert res.ac_input_wh > 0
        assert battery.soc_wh > initial_soc

    def test_discharge_zfi(self, pv_battery_no_ac_params, battery):
        inv = InverterBase(pv_battery_no_ac_params, battery=battery)
        initial_soc = battery.soc_wh
        res = inv.process_energy(
            generation=0,
            mode=InverterMode.DISCHARGE_ZERO_FEED_IN,
            dt=1.0,
            energy_wh=500,
        )
        assert res.ac_output_wh == pytest.approx(500, abs=1e-3)
        assert battery.soc_wh < initial_soc


class TestRates:
    def test_hybrid_has_only_charge_rates(self, pv_hybrid_params, battery):
        """Hybrid inverter: should expose charge_rates from config, no discharge_rates."""
        inv = InverterBase(pv_hybrid_params, battery=battery)
        expected_charge = tuple(sorted({int(r) for r in pv_hybrid_params.ac_rates_pct}))
        assert inv.charge_rates == expected_charge
        assert inv.discharge_rates == tuple()

    def test_ac_battery_default_charge_rates(self, ac_battery_params, battery):
        """AC-only battery inverter: should have default charge_rates and no discharge_rates when zero-feed-in is used."""
        inv = InverterBase(ac_battery_params, battery=battery)
        assert inv.charge_rates == DEFAULT_AC_RATES
        assert inv.discharge_rates == tuple()

    def test_pv_battery_no_ac_rates_empty(self, pv_battery_no_ac_params, battery):
        """PV+Battery without AC charge: no charge_rates; discharge rates only relevant for DISCHARGE mode."""
        inv = InverterBase(pv_battery_no_ac_params, battery=battery)
        assert inv.charge_rates == tuple()
        assert inv.discharge_rates == tuple()
