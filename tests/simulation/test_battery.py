"""Tests for Battery device simulation."""

import pytest

from GridPythia.config.optimization import BatteryParameters
from GridPythia.simulation.devices.battery import Battery


@pytest.fixture
def setup_pv_battery():
    params = BatteryParameters(
        device_id="battery1",
        capacity_wh=10000,
        initial_soc_percentage=50,
        charging_efficiency=0.88,
        discharging_efficiency=0.88,
        min_soc_percentage=20,
        max_soc_percentage=80,
        max_charge_power_w=8000,
    )
    battery = Battery(params)
    battery.reset()

    assert battery.parameters.device_id == "battery1"
    assert battery.capacity_wh == 10000
    assert battery.initial_soc_percentage == 50
    assert battery.charging_efficiency == 0.88
    assert battery.discharging_efficiency == 0.88
    assert battery.max_soc_percentage == 80
    assert battery.max_charge_power_w == 8000
    assert battery.soc_wh == float((50 / 100) * 10000)
    assert battery.min_soc_wh == float((20 / 100) * 10000)
    assert battery.max_soc_wh == float((80 / 100) * 10000)

    return battery


def test_initial_state_of_charge(setup_pv_battery):
    battery = setup_pv_battery
    assert battery.current_soc_percentage() == 50.0


def test_battery_discharge_below_min_soc(setup_pv_battery):
    battery = setup_pv_battery
    discharged_wh, loss_wh = battery.discharge_energy(5000, 1)

    assert discharged_wh > 0
    assert battery.current_soc_percentage() >= 20
    assert loss_wh >= 0
    assert discharged_wh == 2640.0


def test_battery_charge_above_max_soc(setup_pv_battery):
    battery = setup_pv_battery
    charged_wh, loss_wh = battery.charge_energy(5000, 1)

    assert charged_wh > 0
    assert battery.current_soc_percentage() <= 80
    assert loss_wh >= 0
    assert charged_wh == 3000.0


def test_battery_charge_when_full(setup_pv_battery):
    battery = setup_pv_battery
    battery.soc_wh = battery.max_soc_wh
    charged_wh, loss_wh = battery.charge_energy(5000, 0)

    assert charged_wh == 0
    assert loss_wh == 0
    assert battery.current_soc_percentage() == 80


def test_battery_discharge_when_empty(setup_pv_battery):
    battery = setup_pv_battery
    battery.soc_wh = battery.min_soc_wh
    discharged_wh, loss_wh = battery.discharge_energy(5000, 0)

    assert discharged_wh == 0
    assert loss_wh == 0
    assert battery.current_soc_percentage() == 20


def test_battery_reset_function(setup_pv_battery):
    battery = setup_pv_battery
    battery.soc_wh = 8000
    battery.reset()
    assert battery.current_soc_percentage() == battery.initial_soc_percentage


def test_soc_limits(setup_pv_battery):
    battery = setup_pv_battery

    with pytest.raises(ValueError, match="soc_wh"):
        battery.soc_wh = battery.max_soc_wh + 1000

    with pytest.raises(ValueError, match="soc_wh"):
        battery.soc_wh = battery.min_soc_wh - 1000


def test_soc_percentage_setter_updates_soc_wh(setup_pv_battery):
    battery = setup_pv_battery

    battery.soc_percentage = 75.0

    assert battery.soc_wh == pytest.approx(7500.0)
    assert battery.current_soc_percentage() == pytest.approx(75.0)


def test_soc_wh_setter_updates_soc_percentage(setup_pv_battery):
    battery = setup_pv_battery

    battery.soc_wh = 6000.0

    assert battery.soc_percentage == pytest.approx(60.0)
    assert battery.current_soc_percentage() == pytest.approx(60.0)


def test_soc_percentage_limits(setup_pv_battery):
    battery = setup_pv_battery

    with pytest.raises(ValueError, match="soc_percentage"):
        battery.soc_percentage = 85.0

    with pytest.raises(ValueError, match="soc_percentage"):
        battery.soc_percentage = 10.0


def test_charge_energy_within_limits(setup_pv_battery):
    battery = setup_pv_battery
    initial_soc_wh = battery.soc_wh

    charged_wh, losses_wh = battery.charge_energy(wh=4000, dt=1)

    assert charged_wh > 0
    assert losses_wh >= 0
    assert battery.soc_wh > initial_soc_wh
    assert battery.soc_wh <= battery.max_soc_wh


@pytest.mark.parametrize(
    "wh, dt",
    [
        pytest.param(1000, 1, id="request_limited"),
        pytest.param(5000, 1, id="headroom_limited"),
        pytest.param(10000, 1, id="headroom_limited_large_request"),
        pytest.param(5000, 0.25, id="power_limited_short_dt"),
        pytest.param(1500, 2, id="request_limited_long_dt"),
        pytest.param(20000, 2, id="headroom_limited_long_dt"),
    ],
)
def test_charge_energy_parametrized(setup_pv_battery, wh, dt):
    battery = setup_pv_battery
    eff = battery.charging_efficiency
    headroom = battery.max_soc_wh - battery.soc_wh
    max_raw = min(headroom / eff, battery.max_charge_power_w * dt)
    expected_stored = min(wh, max_raw) * eff

    charged, losses = battery.charge_energy(wh=wh, dt=dt)
    assert charged == pytest.approx(expected_stored, rel=1e-5)
    assert losses >= 0
    assert battery.soc_wh <= battery.max_soc_wh + 1e-6
