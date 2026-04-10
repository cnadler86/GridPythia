"""Grid simulation engine."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from structlog import get_logger

from GridPythia.prediction.prediction import PredictionData
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.homeappliance import HomeAppliance
from GridPythia.simulation.devices.inverterbase import InverterBase
from GridPythia.simulation.grid_interpolator import FraunhoferSCModel

logger = get_logger(__name__)


@dataclass(slots=True)
class InverterSimulationDataStep:
    """Step data for one inverter during simulation."""

    inverter: InverterBase
    mode: InverterMode
    generation: float = 0.0
    ac_rate_pct: int | None = None
    ac_energy_wh: float | None = None


@dataclass(slots=True)
class SimulationResult:
    costs_per_dt: np.ndarray
    revenue_per_dt: np.ndarray
    grid_import_wh_per_dt: np.ndarray
    self_consumption_wh_per_dt: np.ndarray
    feedin_wh_per_dt: np.ndarray
    losses_wh_per_dt: np.ndarray
    inverter_modes_per_dt: dict[str, np.ndarray]
    inverter_ac_rate_per_dt: dict[str, np.ndarray] = field(default_factory=dict)
    battery_wh_per_dt: dict[str, np.ndarray] = field(default_factory=dict)
    battery_soc_percentage_per_dt: dict[str, np.ndarray] = field(default_factory=dict)

    home_appliance_load_per_dt: np.ndarray | None = None

    @property
    def total_losses(self) -> float:
        return float(np.sum(self.losses_wh_per_dt))

    @property
    def total_grid_import(self) -> float:
        return float(np.sum(self.grid_import_wh_per_dt))

    @property
    def total_feedin(self) -> float:
        return float(np.sum(self.feedin_wh_per_dt))

    @property
    def total_self_consumption(self) -> float:
        return float(np.sum(self.self_consumption_wh_per_dt))

    @property
    def total_cost(self) -> float:
        return float(np.sum(self.costs_per_dt))

    @property
    def total_revenue(self) -> float:
        return float(np.sum(self.revenue_per_dt))

    @property
    def net_balance(self) -> float:
        """Net balance of the simulation in Euros (revenue - cost)."""
        return self.total_revenue - self.total_cost

    def to_dict(self) -> dict[str, Any]:
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
                    logger.warning("simulation_result_tolist_failed", obj_type=type(obj).__name__)
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
            "battery_wh_per_dt": _conv(self.battery_wh_per_dt or {}),
            "battery_soc_percentage_per_dt": _conv(self.battery_soc_percentage_per_dt or {}),
            "inverter_modes_per_dt": _conv(self.inverter_modes_per_dt or {}),
            "inverter_ac_rate_per_dt": _conv(self.inverter_ac_rate_per_dt or {}),
            "home_appliance_load_per_dt": _conv(self.home_appliance_load_per_dt),
        }


class GridSimulation:
    def __init__(
        self,
        prediction: PredictionData,
        inverters: list[InverterBase] | None = None,
        home_appliances: list[HomeAppliance] | None = None,
    ) -> None:
        dt = prediction.dt_hours
        self.simulation_steps = prediction.steps  # number of simulation steps

        # Load is already in Wh (no conversion needed)
        self.load_energy_array = np.asarray(prediction.load_wh, dtype=np.float32)
        electricprice = prediction.electricprice
        if electricprice is not None:
            self.electricity_price = np.asarray(electricprice, dtype=np.float32)
        else:
            self.electricity_price = np.zeros(prediction.steps, dtype=np.float32)
            logger.warning(
                "simulation_missing_electricity_price",
                default_value=0.0,
                steps=prediction.steps,
            )
        feedintariff = prediction.feedintariff
        if feedintariff is not None:
            self.electricity_revenue = np.asarray(feedintariff, dtype=np.float32)
        else:
            self.electricity_revenue = np.zeros(prediction.steps, dtype=np.float32)
            logger.warning(
                "simulation_missing_feedin_tariff",
                default_value=0.0,
                steps=prediction.steps,
            )

        self.pv_prediction_map: dict[str, np.ndarray] | None = None
        pv_by_inv = prediction.pv_by_inverter
        if pv_by_inv:
            # PV is already in Wh (no conversion needed)
            self.pv_prediction_map = {
                k: np.asarray(v, dtype=np.float32) for k, v in pv_by_inv.items()
            }

        # Build mapping of inverter id -> inverter and ensure uniqueness
        self.inverters: dict[str, InverterBase] = {}
        if inverters:
            for inv in inverters:
                inv_id = inv.device_id
                if inv_id in self.inverters:
                    raise ValueError(
                        f"Duplicate inverter device_id '{inv_id}' provided to GridSimulation"
                    )
                self.inverters[inv_id] = inv

        self._inv_list: list[InverterBase] = list(self.inverters.values())

        # If PV predictions exist, mark inverters that have PV attached.
        if self.pv_prediction_map:
            for inv in self._inv_list:
                inv._has_pv = inv.device_id in self.pv_prediction_map

        self._pv_per_inv: list[np.ndarray] = [
            self._get_pv_for_inverter(inv) for inv in self._inv_list
        ]

        self._step_buf: list[InverterSimulationDataStep] = [
            InverterSimulationDataStep(
                inverter=inv, mode=InverterMode.IDLE, generation=0.0, ac_rate_pct=100
            )
            for inv in self._inv_list
        ]

        # Save initial inverter states for reset functionality and mode switch cost calculation
        self._initial_inverter_states: dict[str, InverterMode] = {
            inv.device_id: inv.current_state for inv in self._inv_list
        }

        self.home_appliances = home_appliances or []
        self.home_appliance_start_hours = [None] * len(self.home_appliances)
        self.home_appliance_start_hour = None

        min_load_wh = float(np.min(self.load_energy_array)) if self.load_energy_array.size else 0.0
        self._fraunhofer_sc_model = FraunhoferSCModel(
            baseload_wh=max(float(min_load_wh), 1e-6),
            dt=dt,
        )

        self._CHARGE_MODES_ARRAY = np.array(
            [int(InverterMode.AC_CHARGE), int(InverterMode.AC_CHARGE_ZERO_FEED_IN)], dtype=np.int8
        )
        self._DISCHARGE_MODES_ARRAY = np.array(
            [int(InverterMode.DISCHARGE), int(InverterMode.DISCHARGE_ZERO_FEED_IN)], dtype=np.int8
        )
        self._IDLE_INT = int(InverterMode.IDLE)

        logger.info(
            "simulation_initialized",
            steps=self.simulation_steps,
            dt_hours=dt,
            inverters=len(self.inverters),
            home_appliances=len(self.home_appliances),
        )

    def reset(self) -> None:
        """Reset all battery states to their initial SoC and inverter states."""
        for inv in self._inv_list:
            if inv.battery:
                inv.battery.reset()
            inv.current_state = self._initial_inverter_states[inv.device_id]
        self.home_appliance_start_hour = None
        if self.home_appliance_start_hours:
            self.home_appliance_start_hours = [None] * len(self.home_appliances)

    def _get_pv_for_inverter(self, inv: InverterBase) -> np.ndarray:
        if not inv._has_pv:
            return np.empty(0, dtype=np.float32)
        inv_id = inv.device_id
        if self.pv_prediction_map and inv_id and inv_id in self.pv_prediction_map:
            return self.pv_prediction_map[inv_id]
        return np.empty(0, dtype=np.float32)

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
                    ac_rate_pct=sb.ac_rate_pct,
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
        inverter_modes: Mapping[str, Sequence[InverterMode | int] | np.ndarray],
        inverter_ac_rates: Mapping[str, Sequence[int] | np.ndarray],
        appliance_load: Sequence[float] | np.ndarray | None = None,
        start_idx: int = 0,
        dt: float = 1.0,
        inverter_ac_energy_wh: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    ) -> SimulationResult | None:
        """Simulate energy flows and costs for a contiguous time window."""
        if start_idx < 0 or start_idx > self.simulation_steps:
            raise ValueError(f"start_idx must be in [0, {self.simulation_steps}], got {start_idx}")

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
        modes_arrs = [
            np.asarray(
                inverter_modes.get(
                    inv.device_id, np.full(self.simulation_steps, int(InverterMode.IDLE))
                ),
                dtype=np.int32,
            )
            for inv in inv_list
        ]

        for inv, arr in zip(inv_list, modes_arrs, strict=False):
            if inv.device_id in inverter_modes and len(arr) < self.simulation_steps:
                raise ValueError(
                    f"inverter_modes[{inv.device_id!r}] must have at least "
                    f"{self.simulation_steps} entries, got {len(arr)}"
                )

        rates_arrs = [
            np.asarray(
                inverter_ac_rates.get(inv.device_id, np.zeros(self.simulation_steps)),
                dtype=np.int32,
            )
            for inv in inv_list
        ]

        energy_map = inverter_ac_energy_wh if inverter_ac_energy_wh is not None else {}
        energy_arrs = [
            np.asarray(
                energy_map.get(inv.device_id, np.full(self.simulation_steps, np.nan)),
                dtype=np.float32,
            )
            for inv in inv_list
        ]

        for inv, arr in zip(inv_list, rates_arrs, strict=False):
            if inv.device_id in inverter_ac_rates and len(arr) < self.simulation_steps:
                raise ValueError(
                    f"inverter_ac_rates[{inv.device_id!r}] must have at least "
                    f"{self.simulation_steps} entries, got {len(arr)}"
                )

        if inverter_ac_energy_wh is not None:
            for inv, arr in zip(inv_list, energy_arrs, strict=False):
                if inv.device_id in inverter_ac_energy_wh and len(arr) < self.simulation_steps:
                    raise ValueError(
                        f"inverter_ac_energy_wh[{inv.device_id!r}] must have at least "
                        f"{self.simulation_steps} entries, got {len(arr)}"
                    )

        _step = self._simulate_step
        pv_lens = [len(a) for a in pv_arrs]
        mode_lens = [len(a) for a in modes_arrs]
        rate_lens = [len(a) for a in rates_arrs]
        energy_lens = [len(a) for a in energy_arrs]
        appl_arr = (
            np.asarray(appliance_load, dtype=np.float32) if appliance_load is not None else None
        )
        appl_len = len(appl_arr) if appl_arr is not None else -1

        costs_per_dt = np.zeros(total_idx, dtype=np.float32)
        revenue_per_dt = np.zeros(total_idx, dtype=np.float32)
        grid_import_wh_per_dt = np.zeros(total_idx, dtype=np.float32)
        feedin_wh_per_dt = np.zeros(total_idx, dtype=np.float32)
        self_consumption_wh_per_dt = np.zeros(total_idx, dtype=np.float32)
        losses_wh_per_dt = np.zeros(total_idx, dtype=np.float32)

        battery_wh_per_dt: dict[str, np.ndarray] = {}
        battery_soc_percentage_per_dt: dict[str, np.ndarray] = {}
        for inv in inv_list:
            if inv.battery is not None:
                battery_wh_per_dt[inv.device_id] = np.zeros(total_idx, dtype=np.float32)
                battery_soc_percentage_per_dt[inv.device_id] = np.zeros(total_idx, dtype=np.float32)

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

            load_wh = load_arr[h] + (appl_arr[h] if appl_arr is not None and h < appl_len else 0.0)

            for j in range(n_inv):
                step = step_buf[j]
                mode_raw = modes_arrs[j][h] if h < mode_lens[j] else int(InverterMode.IDLE)
                step.mode = InverterMode(int(mode_raw))
                step.generation = pv_arrs[j][h] if h < pv_lens[j] else 0.0
                step.ac_rate_pct = int(rates_arrs[j][h]) if h < rate_lens[j] else 0
                energy_value = energy_arrs[j][h] if h < energy_lens[j] else np.nan
                step.ac_energy_wh = None if np.isnan(energy_value) else float(energy_value)

            end_load, pv_ac_wh, losses_wh_per_dt[i], pv_per_inv_list = _step(step_buf, load_wh, dt)

            # record per-inverter PV AC generation is not stored on SimulationResult anymore

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
        inverter_modes_per_dt: dict[str, np.ndarray] = {
            inv.device_id: np.asarray(
                inverter_modes[inv.device_id][start_idx : start_idx + total_idx],
                dtype=np.int8,
            )
            for inv in inv_list
            if inv.device_id in inverter_modes
        }
        inverter_ac_rate_per_dt: dict[str, np.ndarray] = {
            inv.device_id: np.asarray(
                inverter_ac_rates[inv.device_id][start_idx : start_idx + total_idx],
                dtype=np.int32,
            )
            for inv in inv_list
            if inv.device_id in inverter_ac_rates
        }

        appliance_load_series: np.ndarray | None = None
        if appl_arr is not None:
            appliance_load_series = np.zeros(total_idx, dtype=np.float32)
            if start_idx < appl_len:
                copy_len = min(total_idx, appl_len - start_idx)
                appliance_load_series[:copy_len] = appl_arr[start_idx : start_idx + copy_len]

        for inv in inv_list:
            modes = inverter_modes_per_dt.get(inv.device_id)
            if modes is None or len(modes) < 1:
                continue
            switch_cost = inv.parameters.mode_switch_cost

            # Prepend initial state (vor Simulation) für korrekte erste Mode-Wechsel-Berechnung
            arr = np.asarray(modes, dtype=np.int8)
            initial_int = int(self._initial_inverter_states[inv.device_id])
            arr_with_initial = np.concatenate([np.array([initial_int], dtype=np.int8), arr])
            prev = arr_with_initial[:-1]
            curr = arr_with_initial[1:]

            # Kosten pro Wechsel bestimmen mit vorkompilierten Arrays
            # 1. Idle <-> aktiv
            IDLE_INT = self._IDLE_INT
            idle_to_active = ((prev == IDLE_INT) & (curr != IDLE_INT)) | (
                (prev != IDLE_INT) & (curr == IDLE_INT)
            )
            # 2. Charge <-> Discharge (mit numpy array statt isin())
            in_charge_prev = np.isin(prev, self._CHARGE_MODES_ARRAY)
            in_discharge_prev = np.isin(prev, self._DISCHARGE_MODES_ARRAY)
            in_charge_curr = np.isin(curr, self._CHARGE_MODES_ARRAY)
            in_discharge_curr = np.isin(curr, self._DISCHARGE_MODES_ARRAY)
            charge_to_discharge = (in_charge_prev & in_discharge_curr) | (
                in_discharge_prev & in_charge_curr
            )

            # Direkt in costs_per_dt schreiben (Index 0 ist jetzt initial->first, Index 1+ sind die normalen)
            costs_per_dt[: len(prev)][idle_to_active] += switch_cost
            costs_per_dt[: len(prev)][charge_to_discharge] += 2 * switch_cost

        return SimulationResult(
            costs_per_dt=costs_per_dt,
            revenue_per_dt=revenue_per_dt,
            grid_import_wh_per_dt=grid_import_wh_per_dt,
            self_consumption_wh_per_dt=self_consumption_wh_per_dt,
            feedin_wh_per_dt=feedin_wh_per_dt,
            losses_wh_per_dt=losses_wh_per_dt,
            battery_wh_per_dt=battery_wh_per_dt,
            battery_soc_percentage_per_dt=battery_soc_percentage_per_dt,
            inverter_modes_per_dt=inverter_modes_per_dt,
            inverter_ac_rate_per_dt=inverter_ac_rate_per_dt,
            home_appliance_load_per_dt=appliance_load_series,
        )
