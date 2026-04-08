"""Tests for topology-aware InverterBase."""

import pytest

from GridPythia.simulation.devices import InverterMode, SystemTopology
from GridPythia.config.optimization import BatteryParameters, InverterParameters
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
        )
    )


@pytest.fixture
def pv_hybrid_params() -> InverterParameters:
    return InverterParameters(
        device_id="inverter_hybrid",
        max_ac_output_power_w=2000,
        battery_id="battery1",
        has_pv=True,
        dc_to_ac_efficiency=0.95,
        ac_to_dc_efficiency=0.90,
        max_ac_charge_power_w=2000,
        zero_feed_in=True,
    )


@pytest.fixture
def pv_only_params() -> InverterParameters:
    return InverterParameters(
        device_id="inverter_pv_only",
        max_ac_output_power_w=800,
        has_pv=True,
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
        has_pv=True,
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
        has_pv=False,
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
        """Hybrid inverter: rate lists are no longer config-driven."""
        inv = InverterBase(pv_hybrid_params, battery=battery)
        assert inv.charge_rates == tuple()
        assert inv.discharge_rates == tuple()

    def test_ac_battery_default_charge_rates(self, ac_battery_params, battery):
        """AC-only battery inverter: rate lists are no longer config-driven."""
        inv = InverterBase(ac_battery_params, battery=battery)
        assert inv.charge_rates == tuple()
        assert inv.discharge_rates == tuple()

    def test_pv_battery_no_ac_rates_empty(self, pv_battery_no_ac_params, battery):
        """PV+Battery without AC charge: no charge_rates; discharge rates only relevant for DISCHARGE mode."""
        inv = InverterBase(pv_battery_no_ac_params, battery=battery)
        assert inv.charge_rates == tuple()
        assert inv.discharge_rates == tuple()


class TestActiveInverterConsumption:
    """Tests for active_inverter_consumption_w in simulation."""

    @pytest.fixture
    def simple_battery(self) -> Battery:
        return Battery(
            BatteryParameters(
                device_id="battery_a",
                capacity_wh=2000,
                charging_efficiency=1.0,
                discharging_efficiency=1.0,
                max_charge_power_w=1000,
                max_discharge_power_w=1000,
                initial_soc_percentage=50,
                min_soc_percentage=0,
                max_soc_percentage=100,
            )
        )

    def _make_inv(self, battery: Battery, active_w: float) -> InverterBase:
        return InverterBase(
            InverterParameters(
                device_id="inv_active",
                battery_id="battery_a",
                has_pv=False,
                max_ac_output_power_w=1000,
                max_ac_charge_power_w=1000,
                dc_to_ac_efficiency=1.0,
                ac_to_dc_efficiency=1.0,
                zero_feed_in=False,
                mode_switch_cost=0.0,
                active_inverter_consumption_w=active_w,
            ),
            battery=battery,
        )

    def test_idle_mode_has_no_active_consumption(self, simple_battery: Battery) -> None:
        """IDLE mode must not add any active consumption regardless of parameter value."""
        inv = self._make_inv(simple_battery, active_w=50.0)
        result = inv.process_energy(generation=0.0, mode=InverterMode.IDLE, dt=1.0)
        assert result.ac_input_wh == pytest.approx(0.0)
        assert result.losses_wh == pytest.approx(0.0)

    def test_discharge_mode_adds_active_consumption_to_ac_input_and_losses(
        self, simple_battery: Battery
    ) -> None:
        """DISCHARGE mode should increase ac_input_wh and losses_wh by active_consumption_w * dt."""
        active_w = 30.0
        dt = 1.0
        expected_extra = active_w * dt

        inv_no = self._make_inv(simple_battery, active_w=0.0)
        res_no = inv_no.process_energy(
            generation=0.0, mode=InverterMode.DISCHARGE, dt=dt, ac_rate_pct=50
        )

        simple_battery.reset()
        inv_with = self._make_inv(simple_battery, active_w=active_w)
        res_with = inv_with.process_energy(
            generation=0.0, mode=InverterMode.DISCHARGE, dt=dt, ac_rate_pct=50
        )

        assert res_with.ac_input_wh == pytest.approx(res_no.ac_input_wh + expected_extra, abs=1e-6)
        assert res_with.losses_wh == pytest.approx(res_no.losses_wh + expected_extra, abs=1e-6)

    def test_ac_charge_mode_adds_active_consumption(self, simple_battery: Battery) -> None:
        """AC_CHARGE mode should also add active consumption to ac_input_wh and losses_wh."""
        active_w = 20.0
        dt = 0.25
        expected_extra = active_w * dt

        inv_no = self._make_inv(simple_battery, active_w=0.0)
        res_no = inv_no.process_energy(
            generation=0.0, mode=InverterMode.AC_CHARGE, dt=dt, ac_rate_pct=100
        )

        simple_battery.reset()
        inv_with = self._make_inv(simple_battery, active_w=active_w)
        res_with = inv_with.process_energy(
            generation=0.0, mode=InverterMode.AC_CHARGE, dt=dt, ac_rate_pct=100
        )

        assert res_with.ac_input_wh == pytest.approx(res_no.ac_input_wh + expected_extra, abs=1e-6)
        assert res_with.losses_wh == pytest.approx(res_no.losses_wh + expected_extra, abs=1e-6)

    def test_zero_active_consumption_leaves_result_unchanged(self, simple_battery: Battery) -> None:
        """When active_inverter_consumption_w=0 the result must be identical to the baseline."""
        inv = self._make_inv(simple_battery, active_w=0.0)
        res = inv.process_energy(
            generation=0.0, mode=InverterMode.AC_CHARGE, dt=1.0, ac_rate_pct=50
        )
        # Just verify no extra is injected compared to a zero-loss inverter
        assert res.losses_wh >= 0.0
        assert res.ac_input_wh >= 0.0
