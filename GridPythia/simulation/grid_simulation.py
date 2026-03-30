"""Grid simulation engine."""

from array import array
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger

from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.homeappliance import HomeAppliance
from GridPythia.simulation.devices.inverterbase import InverterBase
from GridPythia.simulation.grid_interpolator import FraunhoferSCModel


@dataclass(slots=True)
class InverterSimulationDataStep:
    """Step data for one inverter during simulation."""

    inverter: InverterBase
    mode: InverterMode
    generation: float = 0.0
    ac_rate: Optional[float] = None


@dataclass(slots=True)
class SimulationResult:
    costs_per_dt: array[float]
    revenue_per_dt: array[float]
    grid_import_wh_per_dt: array[float]
    self_consumption_wh_per_dt: array[float]
    feedin_wh_per_dt: array[float]
    losses_wh_per_dt: array[float]
    electricity_price_per_dt: array[float]
    inverter_modes_per_dt: Dict[str, array[int]]
    inverter_ac_rate_per_dt: Dict[str, array[float]] = field(default_factory=dict)
    solar_generation_wh_per_dt: Dict[str, array[float]] = field(default_factory=dict)
    battery_wh_per_dt: Dict[str, array[float]] = field(default_factory=dict)
    battery_soc_percentage_per_dt: Dict[str, array[float]] = field(default_factory=dict)

    home_appliance_load_per_dt: Optional[array[float]] = None

    @property
    def total_losses(self) -> float:
        return sum(self.losses_wh_per_dt)

    @property
    def total_grid_import(self) -> float:
        return sum(self.grid_import_wh_per_dt)

    @property
    def total_feedin(self) -> float:
        return sum(self.feedin_wh_per_dt)

    @property
    def total_self_consumption(self) -> float:
        return sum(self.self_consumption_wh_per_dt)

    @property
    def total_cost(self) -> float:
        return sum(self.costs_per_dt)

    @property
    def total_revenue(self) -> float:
        return sum(self.revenue_per_dt)

    @property
    def net_balance(self) -> float:
        """Net balance of the simulation in Euros (revenue - cost)."""
        return self.total_revenue - self.total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert the simulation result to a dictionary."""

        def _conv(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return {k: _conv(v) for k, v in obj.items()}
            if hasattr(obj, "tolist"):
                try:
                    return obj.tolist()
                except Exception:
                    logger.warning("Failed to convert object with tolist(): {}", obj)
            try:
                return list(obj)
            except Exception:
                return obj

        return {
            "total_revenue": self.total_revenue,
            "total_cost": self.total_cost,
            "total_losses": self.total_losses,
            "costs_per_dt": _conv(self.costs_per_dt),
            "revenue_per_dt": _conv(self.revenue_per_dt),
            "feedin_wh_per_dt": _conv(self.feedin_wh_per_dt),
            "self_consumption_wh_per_dt": _conv(self.self_consumption_wh_per_dt),
            "grid_import_wh_per_dt": _conv(self.grid_import_wh_per_dt),
            "losses_wh_per_dt": _conv(self.losses_wh_per_dt),
            "solar_generation_wh_per_dt": _conv(self.solar_generation_wh_per_dt or {}),
            "battery_wh_per_dt": _conv(self.battery_wh_per_dt or {}),
            "battery_soc_percentage_per_dt": _conv(self.battery_soc_percentage_per_dt or {}),
            # explicit alias: SOC values refer to the state at the end of each simulated step
            "battery_soc_percentage_at_step_end": _conv(self.battery_soc_percentage_per_dt or {}),
            "inverter_modes_per_dt": _conv(self.inverter_modes_per_dt or {}),
            "inverter_ac_rate_per_dt": _conv(self.inverter_ac_rate_per_dt or {}),
            "electricity_price_per_dt": _conv(self.electricity_price_per_dt),
            "home_appliance_load_per_dt": _conv(self.home_appliance_load_per_dt),
        }


class GridSimulation:
    def __init__(
        self,
        prediction: PredictionData,
        inverters: Optional[list[InverterBase]] = None,
        home_appliances: Optional[list[HomeAppliance]] = None,
    ) -> None:
        dt = prediction.dt_hours
        self.simulation_steps = prediction.steps  # number of simulation steps

        # Load is already in Wh (no conversion needed)
        self.load_energy_array = array("f", prediction.load_wh.to_list())
        electricprice = prediction.electricprice
        if electricprice is not None:
            self.electricity_price = array("f", electricprice.to_list())
        else:
            self.electricity_price = array("f", [0.0] * prediction.steps)
            logger.warning(
                "Electricity price column not found in prediction data; defaulting to 0.0 EUR/Wh for all steps."
            )
        feedintariff = prediction.feedintariff
        if feedintariff is not None:
            self.electricity_revenue = array("f", feedintariff.to_list())
        else:
            self.electricity_revenue = array("f", [0.0] * prediction.steps)
            logger.warning(
                "Feed-in tariff column not found in prediction data; defaulting to 0.0 EUR/Wh for all steps."
            )

        self.pv_prediction_map: Optional[Dict[str, array]] = None
        pv_by_inv = prediction.pv_by_inverter
        if pv_by_inv:
            # PV is already in Wh (no conversion needed)
            self.pv_prediction_map = {k: array("f", v.to_list()) for k, v in pv_by_inv.items()}

        # Build mapping of inverter id -> inverter and ensure uniqueness
        self.inverters: Dict[str, InverterBase] = {}
        if inverters:
            for inv in inverters:
                inv_id = inv.device_id
                if inv_id in self.inverters:
                    raise ValueError(
                        f"Duplicate inverter device_id '{inv_id}' provided to GridSimulation"
                    )
                self.inverters[inv_id] = inv

        self._inv_list: list[InverterBase] = list(self.inverters.values())

        self._pv_per_inv: list[array[float]] = [
            self._get_pv_for_inverter(inv) for inv in self._inv_list
        ]

        self._step_buf: list[InverterSimulationDataStep] = [
            InverterSimulationDataStep(
                inverter=inv, mode=InverterMode.IDLE, generation=0.0, ac_rate=1.0
            )
            for inv in self._inv_list
        ]

        self.home_appliances = home_appliances or []
        self.home_appliance_start_hours = [None] * len(self.home_appliances)
        self.home_appliance_start_hour = None

        min_load_wh = min(self.load_energy_array) if self.load_energy_array else 0.0
        self._fraunhofer_sc_model = FraunhoferSCModel(
            baseload_wh=max(float(min_load_wh), 1e-6),
            dt=dt,
        )

    def reset(self) -> None:
        """Reset all battery states to their initial SoC."""
        for inv in self._inv_list:
            if inv.battery:
                inv.battery.reset()
        self.home_appliance_start_hour = None
        if self.home_appliance_start_hours:
            self.home_appliance_start_hours = [None] * len(self.home_appliances)

    def _get_pv_for_inverter(self, inv: InverterBase) -> array[float]:
        if not inv._has_pv:
            return array("f")
        pv_source = inv.parameters.pv_source
        if self.pv_prediction_map and pv_source and pv_source in self.pv_prediction_map:
            return self.pv_prediction_map[pv_source]
        return array("f")

    def _simulate_step(
        self,
        step_buf: list[InverterSimulationDataStep],
        load_wh: float,
        dt: float,
    ) -> tuple[float, float, float, list[float]]:
        """Process all inverters for one time step.

        Returns (end_load, pv_ac_wh, losses_wh, pv_per_inverter).
        """
        _ZFI_D = InverterMode.DISCHARGE_ZERO_FEED_IN
        _ZFI_C = InverterMode.AC_CHARGE_ZERO_FEED_IN

        static_ie = 0.0
        static_pv = 0.0
        zfi_d = None
        zfi_c = None
        losses = 0.0
        pv_per_inv: list[float] = [0.0] * len(step_buf)

        for idx, sb in enumerate(step_buf):
            m = sb.mode
            if m == _ZFI_D:
                zfi_d = (idx, sb)
            elif m == _ZFI_C:
                zfi_c = (idx, sb)
            else:
                res = sb.inverter.process_energy(
                    generation=sb.generation,
                    mode=m,
                    dt=dt,
                    ac_rate=sb.ac_rate,
                )
                static_ie += res.ac_output_wh - res.ac_input_wh
                static_pv += res.pv_ac_wh
                losses += res.losses_wh
                pv_per_inv[idx] = res.pv_ac_wh

        gb = load_wh - static_ie

        if zfi_d is not None and gb >= 0.0:
            idx, sb = zfi_d
            res = sb.inverter.process_energy(
                generation=sb.generation,
                mode=sb.mode,
                dt=dt,
                energy_wh=gb,
            )
            losses += res.losses_wh
            pv_per_inv[idx] = res.pv_ac_wh
            return (
                gb - (res.ac_output_wh - res.ac_input_wh),
                sum(pv_per_inv),
                losses,
                pv_per_inv,
            )
        if zfi_c is not None and gb <= 0.0:
            idx, sb = zfi_c
            res = sb.inverter.process_energy(
                generation=sb.generation,
                mode=sb.mode,
                dt=dt,
                energy_wh=-gb,
            )
            losses += res.losses_wh
            pv_per_inv[idx] = res.pv_ac_wh
            return (
                gb - (res.ac_output_wh - res.ac_input_wh),
                sum(pv_per_inv),
                losses,
                pv_per_inv,
            )
        return gb, sum(pv_per_inv), losses, pv_per_inv

    def simulate(
        self,
        inverter_modes: Dict[str, array[InverterMode]],
        inverter_ac_rates: Dict[str, array[float]],
        appliance_load: Optional[array[float]] = None,
        start_idx: int = 0,
        dt: float = 1.0,
    ) -> Optional[SimulationResult]:
        """Simulate energy flows and costs for a contiguous time window."""
        total_idx = self.simulation_steps - start_idx
        if total_idx <= 0:
            return None

        self.reset()

        load_arr = self.load_energy_array
        price_arr = self.electricity_price
        revenue_arr = self.electricity_revenue
        inv_list = self._inv_list
        pv_arrs = self._pv_per_inv
        step_buf = self._step_buf
        n_inv = len(inv_list)
        calc_sc_ratio = self._fraunhofer_sc_model.sc_ratio

        # Extract ordered arrays for the hot loop (avoids per-step dict lookups)
        modes_arrs = [inverter_modes.get(inv.device_id, array("i", [])) for inv in inv_list]
        rates_arrs = [inverter_ac_rates.get(inv.device_id, array("f", [])) for inv in inv_list]

        _step = self._simulate_step
        pv_lens = [len(a) for a in pv_arrs]
        appl_len = len(appliance_load) if appliance_load is not None else -1

        costs_per_dt = array("f", [0.0] * total_idx)
        revenue_per_dt = array("f", [0.0] * total_idx)
        grid_import_wh_per_dt = array("f", [0.0] * total_idx)
        feedin_wh_per_dt = array("f", [0.0] * total_idx)
        self_consumption_wh_per_dt = array("f", [0.0] * total_idx)
        losses_wh_per_dt = array("f", [0.0] * total_idx)

        battery_wh_per_dt: Dict[str, array[float]] = {}
        battery_soc_percentage_per_dt: Dict[str, array[float]] = {}
        for inv in inv_list:
            if inv.battery is not None:
                battery_wh_per_dt[inv.device_id] = array("f", [0.0] * total_idx)
                battery_soc_percentage_per_dt[inv.device_id] = array("f", [0.0] * total_idx)

        solar_generation_wh_per_dt: Dict[str, array[float]] = {}
        for inv in inv_list:
            if getattr(inv, "_has_pv", False):
                solar_generation_wh_per_dt[inv.device_id] = array("f", [0.0] * total_idx)

        _bat_tracking = [
            (
                battery_wh_per_dt[inv.device_id],
                battery_soc_percentage_per_dt[inv.device_id],
                inv.battery,
                inv.battery._soc_pct_factor,
            )
            for inv in inv_list
            if inv.battery is not None
        ]

        for h in range(start_idx, start_idx + total_idx):
            i = h - start_idx

            load_wh = load_arr[h] + (appliance_load[h] if h < appl_len else 0.0)

            for j in range(n_inv):
                step = step_buf[j]
                step.mode = modes_arrs[j][h]
                step.generation = pv_arrs[j][h] if h < pv_lens[j] else 0.0
                step.ac_rate = rates_arrs[j][h]

            end_load, pv_ac_wh, losses_wh_per_dt[i], pv_per_inv_list = _step(step_buf, load_wh, dt)

            # record per-inverter PV AC generation
            for j in range(n_inv):
                inv_id = inv_list[j].device_id
                if inv_id in solar_generation_wh_per_dt:
                    solar_generation_wh_per_dt[inv_id][i] = pv_per_inv_list[j]

            if pv_ac_wh > 0.0:
                SCR = calc_sc_ratio(pv_ac_wh, load_wh)
                pv_feedin = pv_ac_wh * (1.0 - SCR)
                corrected_end_load = end_load + pv_feedin
                _gi = corrected_end_load if corrected_end_load > 0.0 else 0.0
                grid_import_wh_per_dt[i] = _gi
                feedin_wh_per_dt[i] = pv_feedin + (
                    -corrected_end_load if corrected_end_load < 0.0 else 0.0
                )
            else:
                _gi = end_load if end_load > 0.0 else 0.0
                grid_import_wh_per_dt[i] = _gi
                feedin_wh_per_dt[i] = -end_load if end_load < 0.0 else 0.0

            self_consumption_wh_per_dt[i] = load_wh - _gi
            costs_per_dt[i] = _gi * price_arr[h]
            revenue_per_dt[i] = feedin_wh_per_dt[i] * revenue_arr[h]

            for _wh_arr, _pct_arr, _bat, _pct_f in _bat_tracking:
                _soc = _bat.soc_wh
                _wh_arr[i] = _soc
                _pct_arr[i] = _soc * _pct_f

        # Build inverter mode/rate output dicts from the input arrays (no per-step copy)
        inverter_modes_per_dt: Dict[str, array] = {
            inv.device_id: array(
                "b", inverter_modes[inv.device_id][start_idx : start_idx + total_idx]
            )
            for inv in inv_list
            if inv.device_id in inverter_modes
        }
        inverter_ac_rate_per_dt: Dict[str, array] = {
            inv.device_id: array(
                "f", inverter_ac_rates[inv.device_id][start_idx : start_idx + total_idx]
            )
            for inv in inv_list
            if inv.device_id in inverter_ac_rates
        }

        elec_price_series = array(
            "f",
            [price_arr[h] for h in range(start_idx, start_idx + total_idx)],
        )

        return SimulationResult(
            costs_per_dt=costs_per_dt,
            revenue_per_dt=revenue_per_dt,
            grid_import_wh_per_dt=grid_import_wh_per_dt,
            self_consumption_wh_per_dt=self_consumption_wh_per_dt,
            feedin_wh_per_dt=feedin_wh_per_dt,
            losses_wh_per_dt=losses_wh_per_dt,
            solar_generation_wh_per_dt=solar_generation_wh_per_dt,
            battery_wh_per_dt=battery_wh_per_dt,
            battery_soc_percentage_per_dt=battery_soc_percentage_per_dt,
            inverter_modes_per_dt=inverter_modes_per_dt,
            inverter_ac_rate_per_dt=inverter_ac_rate_per_dt,
            electricity_price_per_dt=elec_price_series,
            home_appliance_load_per_dt=appliance_load or None,
        )
