"""Edge-case tests for Battery device simulation (NaN/inf, rapid cycling, boundary)."""

from math import inf, nan

import pytest

from GridPythia.config.optimization import BatteryParameters
from GridPythia.simulation.devices.battery import Battery


@pytest.fixture
def battery():
    params = BatteryParameters(
        device_id="bat_edge",
        capacity_wh=5000,
        initial_soc_percentage=50,
        charging_efficiency=0.90,
        discharging_efficiency=0.90,
        min_soc_percentage=10,
        max_soc_percentage=90,
        max_charge_power_w=5000,
        max_discharge_power_w=5000,
    )
    b = Battery(params)
    b.reset()
    return b


class TestDischargeEdgeCases:
    def test_nan_request_returns_zero(self, battery):
        delivered, losses = battery.discharge_energy(nan)
        assert delivered == 0.0
        assert losses == 0.0
        assert battery.soc_percentage == pytest.approx(50.0)

    def test_inf_request_returns_zero(self, battery):
        delivered, losses = battery.discharge_energy(inf)
        assert delivered == 0.0
        assert losses == 0.0
        assert battery.soc_percentage == pytest.approx(50.0)

    def test_negative_inf_request_returns_zero(self, battery):
        delivered, losses = battery.discharge_energy(-inf)
        assert delivered == 0.0
        assert losses == 0.0

    def test_negative_request_returns_zero(self, battery):
        delivered, losses = battery.discharge_energy(-100.0)
        assert delivered == 0.0
        assert losses == 0.0
        assert battery.soc_percentage == pytest.approx(50.0)

    def test_zero_request_returns_zero(self, battery):
        delivered, losses = battery.discharge_energy(0.0)
        assert delivered == 0.0
        assert losses == 0.0

    def test_exact_usable_energy(self, battery):
        """Discharge exactly the usable energy should land at min_soc."""
        usable_raw = battery.soc_wh - battery.min_soc_wh
        deliverable = usable_raw * battery.discharging_efficiency
        delivered, losses = battery.discharge_energy(deliverable)
        assert battery.soc_wh == pytest.approx(battery.min_soc_wh, abs=1e-6)
        assert delivered == pytest.approx(deliverable, rel=1e-5)
        assert losses >= 0

    def test_small_epsilon_discharge(self, battery):
        """Very small discharge should not corrupt state."""
        initial = battery.soc_wh
        delivered, losses = battery.discharge_energy(1e-10)
        assert delivered >= 0.0
        assert battery.soc_wh <= initial


class TestChargeEdgeCases:
    def test_nan_request_returns_zero(self, battery):
        stored, losses = battery.charge_energy(nan)
        assert stored == 0.0
        assert losses == 0.0
        assert battery.soc_percentage == pytest.approx(50.0)

    def test_inf_request_returns_zero(self, battery):
        stored, losses = battery.charge_energy(inf)
        assert stored == 0.0
        assert losses == 0.0
        assert battery.soc_percentage == pytest.approx(50.0)

    def test_negative_request_returns_zero(self, battery):
        stored, losses = battery.charge_energy(-500.0)
        assert stored == 0.0
        assert losses == 0.0

    def test_exact_headroom_charge(self, battery):
        """Charge exactly the headroom should land at max_soc."""
        headroom = battery.max_soc_wh - battery.soc_wh
        raw_needed = headroom / battery.charging_efficiency
        stored, losses = battery.charge_energy(raw_needed)
        assert battery.soc_wh == pytest.approx(battery.max_soc_wh, abs=1e-6)
        assert stored == pytest.approx(headroom, rel=1e-5)
        assert losses >= 0

    def test_small_epsilon_charge(self, battery):
        """Very small charge should not corrupt state."""
        initial = battery.soc_wh
        stored, losses = battery.charge_energy(1e-10)
        assert stored >= 0.0
        assert battery.soc_wh >= initial


class TestRapidCycling:
    def test_charge_discharge_cycle_preserves_valid_state(self, battery):
        """Rapid charge/discharge cycles should never produce invalid SoC."""
        for _ in range(100):
            battery.charge_energy(200, dt=0.25)
            battery.discharge_energy(200, dt=0.25)

        assert battery.min_soc_wh <= battery.soc_wh <= battery.max_soc_wh
        assert 0.0 <= battery.soc_percentage <= 100.0

    def test_full_charge_then_full_discharge(self, battery):
        """Fully charge then fully discharge should stay within bounds."""
        battery.charge_energy(100000, dt=1.0)
        assert battery.soc_wh == pytest.approx(battery.max_soc_wh, abs=1e-6)

        battery.discharge_energy(100000, dt=1.0)
        assert battery.soc_wh == pytest.approx(battery.min_soc_wh, abs=1e-6)

    def test_alternating_small_cycles_energy_conservation(self, battery):
        """Ensure total AC energy out <= total AC energy in (due to round-trip losses)."""
        total_charge_raw = 0.0
        total_discharge_delivered = 0.0
        for _ in range(50):
            # charge_energy returns (stored_wh_internal, losses_wh)
            # The raw AC input is stored + losses
            stored, charge_losses = battery.charge_energy(100, dt=0.25)
            total_charge_raw += stored + charge_losses
            delivered, _ = battery.discharge_energy(100, dt=0.25)
            total_discharge_delivered += delivered
        # Round-trip efficiency: charge_eff * discharge_eff < 1 → delivered < raw input
        assert total_discharge_delivered <= total_charge_raw


class TestConstructionValidation:
    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity_wh"):
            Battery(
                BatteryParameters(
                    device_id="bad", capacity_wh=0, initial_soc_percentage=50
                )
            )

    def test_efficiency_above_one_raises(self):
        with pytest.raises(ValueError, match="charging_efficiency"):
            Battery(
                BatteryParameters(
                    device_id="bad",
                    capacity_wh=1000,
                    initial_soc_percentage=50,
                    charging_efficiency=1.5,
                )
            )

    def test_min_soc_above_max_soc_raises(self):
        with pytest.raises(ValueError, match="[Mm]in_soc"):
            Battery(
                BatteryParameters(
                    device_id="bad",
                    capacity_wh=1000,
                    initial_soc_percentage=50,
                    min_soc_percentage=90,
                    max_soc_percentage=10,
                )
            )

    def test_initial_soc_clamped_to_range(self):
        """Initial SoC outside [min, max] should be clamped, not rejected."""
        bat = Battery(
            BatteryParameters(
                device_id="clamp",
                capacity_wh=1000,
                initial_soc_percentage=5,
                min_soc_percentage=20,
                max_soc_percentage=80,
            )
        )
        assert bat.initial_soc_percentage == 20.0

    def test_soc_wh_finite_validation(self, battery):
        with pytest.raises(ValueError, match="finite"):
            battery.soc_wh = float("nan")

        with pytest.raises(ValueError, match="finite"):
            battery.soc_wh = float("inf")
