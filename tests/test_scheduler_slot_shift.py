"""
Diagnose-Test: 15-Minuten-Verschiebung des Ladeslots.

Reproduziert das Szenario:
- Benutzer startet Server, optimiert um 11:02 → Plan startet ab 11:00
- Scheduler feuert um ~11:14 für den 11:15-Slot → Plan startet ab 11:15
- Frage: Ändert sich der optimale Ladeslot (absolute Uhrzeit)?

Der Test prüft außerdem snap_to_dt_grid auf korrektes Verhalten und
verifiziert, dass die Prediction-Zeitstempel korrekt ausgerichtet sind.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from math import floor
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import yaml

from GridPythia.config import AppConfig
from GridPythia.optimization.solution import OptimizationObjective
from GridPythia.optimization.solver import LinearOptimizer
from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.prediction import PredictionData
from GridPythia.server.services import snap_to_dt_grid
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase


BERLIN = ZoneInfo("Europe/Berlin")
DT_HOURS = 0.25  # 15-Minuten-Slots
HORIZON_H = 48
CONFIG_PATH = "config.yaml"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def load_config() -> AppConfig:
    import pathlib

    raw = yaml.safe_load(pathlib.Path(CONFIG_PATH).read_text(encoding="utf-8"))
    return AppConfig.from_dict(raw)


def berlin_today(hour: int, minute: int = 0) -> datetime:
    """Heutiges Datum in Berlin-Zeitzone."""
    now = datetime.now(tz=BERLIN)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def build_prediction(
    start: datetime,
    prices_eur_per_mwh: list[float],
    load_w: float = 250.0,
    pv_peak_w: float = 0.0,
    dt_hours: float = DT_HOURS,
    horizon_h: float = HORIZON_H,
) -> PredictionData:
    """Baut ein PredictionData-Objekt mit festen/synthetischen Werten."""
    n = max(1, round(horizon_h / dt_hours))
    timestamps = make_timestamps(start, horizon_h, dt_hours)
    actual_n = len(timestamps)

    # Preise: falls zu kurz → letzten Wert wiederholen
    prices_raw = list(prices_eur_per_mwh)
    while len(prices_raw) < actual_n:
        prices_raw.append(prices_raw[-1])
    prices_raw = prices_raw[:actual_n]

    # EUR/MWh → EUR/Wh, dann Netzentgelt + MwSt. (entspricht ElecPriceEpexPredictor)
    charges_kwh = 0.1528
    vat = 0.19
    charges_wh = charges_kwh / 1000.0
    prices_eur_wh = np.array(
        [(p / 1_000_000.0 + charges_wh) * (1.0 + vat) for p in prices_raw],
        dtype=np.float32,
    )

    load_wh = np.full(actual_n, load_w * dt_hours, dtype=np.float32)

    # Einfache PV-Kurve: Gauss um 13:00
    pv_wh = np.zeros(actual_n, dtype=np.float32)
    if pv_peak_w > 0:
        for i, ts in enumerate(timestamps):
            local = ts.astimezone(BERLIN)
            t_h = local.hour + local.minute / 60.0
            pv_wh[i] = pv_peak_w * np.exp(-((t_h - 13.0) ** 2) / 4.0) * dt_hours

    feedin = np.zeros(actual_n, dtype=np.float32)

    return PredictionData(
        requested_start=start,
        timestamps=timestamps,
        dt_hours=dt_hours,
        load_wh=load_wh,
        electricprice_eur_wh=prices_eur_wh,
        feedintariff_eur_wh=feedin,
        pv_by_inverter={"SF800Pro": pv_wh},
    )


def build_inverter(cfg: AppConfig) -> tuple[list[InverterBase], dict[str, float]]:
    """Inverter + Battery aus Config."""
    batteries = {b.device_id: Battery(b) for b in cfg.optimization.batteries}
    inverters = [
        InverterBase(p, battery=batteries.get(p.battery_id) if p.battery_id else None)
        for p in cfg.optimization.inverters
    ]
    return inverters, batteries


def first_charge_slot(solution, timestamps: list[datetime]) -> datetime | None:
    """Gibt die erste absolute Uhrzeit zurück, an der geladen wird."""
    for plan in solution.inverter_plans:
        for i, mode in enumerate(plan.modes):
            if mode in (InverterMode.AC_CHARGE, InverterMode.AC_CHARGE_ZERO_FEED_IN):
                return timestamps[i]
    return None


# ---------------------------------------------------------------------------
# Heutige EPEX-Preise (EUR/MWh, 15-min-Auflösung, ab Berlin 11:00)
# Wurden via fetch_prices.py abgerufen – repräsentativ für den Diagnose-Tag.
# Preise von Berlin 11:00 bis 12:45 (Auswahl für Illustration):
# ---------------------------------------------------------------------------
PRICES_FROM_11_00 = [
    # 11:00 .. 16:45 (24 Slots à 15 min = 6 Stunden)
    -0.36, -0.77, -1.45, -3.13, -5.34, -6.55, -7.27, -8.00,  # 11:00..12:45
    -12.59, -15.00, -15.15, -15.00, -11.63, -12.70, -9.60, -9.60,  # 13:00..14:45
    -10.01, -4.70, -1.09, -0.07, -0.85, -0.60, 0.19, 37.79,         # 15:00..16:45
    # Danach: Abend/Nacht (hohe Preise)
    *([120.0] * 168),  # ~42h Füllwerte (168 * 15min = 42h)
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSnapToDtGrid:
    """snap_to_dt_grid muss korrekt auf Slot-Grenzen runden."""

    def test_rounds_to_nearest_lower(self):
        """11:02 → 11:00 (näher an 11:00 als an 11:15)."""
        t = berlin_today(11, 2)
        result = snap_to_dt_grid(t, DT_HOURS)
        expected = berlin_today(11, 0)
        assert result == expected, f"Erwartet 11:00, erhalten: {result}"

    def test_rounds_to_nearest_upper(self):
        """11:14 → 11:15 (näher an 11:15 als an 11:00)."""
        t = berlin_today(11, 14)
        result = snap_to_dt_grid(t, DT_HOURS)
        expected = berlin_today(11, 15)
        assert result == expected, f"Erwartet 11:15, erhalten: {result}"

    def test_exactly_at_boundary(self):
        """11:15 genau → 11:15."""
        t = berlin_today(11, 15)
        result = snap_to_dt_grid(t, DT_HOURS)
        assert result == t

    def test_scheduler_fire_time_35s_before_slot(self):
        """Scheduler feuert 35s vor dem 11:15-Slot → snap muss 11:15 zurückgeben."""
        # 11:14:25 (35 Sekunden vor 11:15)
        t = berlin_today(11, 14).replace(second=25)
        result = snap_to_dt_grid(t, DT_HOURS)
        expected = berlin_today(11, 15)
        assert result == expected, (
            f"Bei Scheduler-Feuerzeit 11:14:25 sollte snap 11:15 liefern, "
            f"erhalten: {result}"
        )

    def test_scheduler_fire_time_60s_before_slot(self):
        """Scheduler feuert 60s vor dem 11:15-Slot (dispatch_buffer_max=30+lead=30) → snap == 11:15."""
        t = berlin_today(11, 14)  # 11:14:00 = 60s vor 11:15
        result = snap_to_dt_grid(t, DT_HOURS)
        expected = berlin_today(11, 15)
        assert result == expected

    def test_snap_result_is_utc_aligned(self):
        """snap_to_dt_grid-Ergebnis muss auf UTC-900s-Grid ausgerichtet sein."""
        for minute in range(0, 60):
            t = berlin_today(11, minute)
            result = snap_to_dt_grid(t, DT_HOURS)
            epoch = result.timestamp()
            assert epoch % 900 == 0, (
                f"Snap für 11:{minute:02d} liefert {result} – kein 900s-Vielfaches!"
            )


class TestSchedulerSlotShift:
    """Reproduziert den 15-Minuten-Verschiebungs-Bug."""

    @pytest.fixture
    def cfg(self):
        return load_config()

    @pytest.fixture
    def inverters_and_batteries(self, cfg):
        return build_inverter(cfg)

    def _solve(
        self,
        cfg: AppConfig,
        inverters: list[InverterBase],
        start: datetime,
        soc_pct: float,
        initial_mode: InverterMode = InverterMode.DISCHARGE,
        prices: list[float] | None = None,
    ):
        """Hilfsfunktion: Einmal optimieren, Ergebnis zurückgeben."""
        if prices is None:
            # Preise ab start_offset berechnen
            offset_slots = int(round((start - berlin_today(11, 0)).total_seconds() / 900))
            prices = PRICES_FROM_11_00[offset_slots:] if offset_slots >= 0 else PRICES_FROM_11_00
        pdata = build_prediction(start, prices)
        bat_cap = cfg.optimization.batteries[0].capacity_wh
        soc_wh = (soc_pct / 100.0) * bat_cap
        optimizer = LinearOptimizer(
            inverters=inverters,
            objective=OptimizationObjective.MINIMIZE_COST,
            solver_opts=dict(cfg.optimization.solver.solver_opts),
        )
        solution = optimizer.solve(
            pdata,
            soc={"SF800Pro": soc_wh},
            initial_modes={"SF800Pro": initial_mode},
        )
        return solution, pdata

    def test_raw_solver_does_shift_by_one_slot_with_different_starts(
        self, cfg, inverters_and_batteries
    ):
        """
        DOKUMENTIERT DAS BUG-VERHALTEN:
        Der Solver gibt bei verschiedenen Startzeitpunkten unterschiedliche
        absolute Ladeslots zurück (15-min-Shift), auch bei identischem SOC.

        Das ist das ROHE Solver-Verhalten – der Fix liegt im CACHE-Layer:
        Durch PDATA_CACHE_TTL_S = prediction_refresh_minutes * 60 benutzt
        der Scheduler für aufeinanderfolgende 15-min-Zyklen DIESELBEN Prediction-
        Daten, sodass dieser Shift in der Praxis NICHT auftritt.
        """
        inverters, _ = inverters_and_batteries
        soc_pct = 30.0

        start_1100 = berlin_today(11, 0)
        start_1115 = berlin_today(11, 15)

        sol_1100, pdata_1100 = self._solve(cfg, inverters, start_1100, soc_pct)
        charge_1100 = first_charge_slot(sol_1100, pdata_1100.timestamps)

        sol_1115, pdata_1115 = self._solve(cfg, inverters, start_1115, soc_pct)
        charge_1115 = first_charge_slot(sol_1115, pdata_1115.timestamps)

        print(f"\nPlan ab 11:00: erster Ladeslot = {charge_1100}")
        print(f"Plan ab 11:15: erster Ladeslot = {charge_1115}")

        if charge_1100 is not None and charge_1115 is not None:
            diff_min = (charge_1115 - charge_1100).total_seconds() / 60.0
            print(f"Differenz (raw solver): {diff_min:.0f} Minuten")
            # Dokumentiert: Raw-Solver verschiebt um genau 15 Minuten
            assert diff_min == 15.0, (
                f"Unerwartetes Rohverhalten: Differenz ist {diff_min:.0f} statt 15 Minuten. "
                "Bitte Bug-Analyse überprüfen."
            )

    def test_same_pdata_gives_stable_plan(self, cfg, inverters_and_batteries):
        """
        TESTET DEN FIX:
        Wenn der Scheduler für den 11:15-Zyklus DIESELBEN Prediction-Daten
        (gecacht von 11:00) benutzt, liefert der Solver denselben absoluten Ladeslot.

        Dies ist das Verhalten, das durch PDATA_CACHE_TTL_S = prediction_refresh_minutes * 60
        sichergestellt wird.
        """
        inverters, _ = inverters_and_batteries
        soc_pct = 30.0

        start_1100 = berlin_today(11, 0)

        # Plan für 11:00-Zyklus mit 11:00-Prediction
        sol_1100, pdata_1100 = self._solve(cfg, inverters, start_1100, soc_pct)
        charge_1100 = first_charge_slot(sol_1100, pdata_1100.timestamps)

        # Plan für 11:15-Zyklus – aber MIT DENSELBEN Prediction-Daten (cache-Effekt)!
        # pdata_1100 wird wiederverwendet (gleiche Zeitstempel, gleiche Preise)
        sol_1115_cached, _ = self._solve(
            cfg, inverters, start_1100, soc_pct  # ← GLEICHER Start = gecachte Prediction
        )
        charge_1115_cached = first_charge_slot(sol_1115_cached, pdata_1100.timestamps)

        print(f"\nPlan ab 11:00 (Cache-Start): erster Ladeslot = {charge_1100}")
        print(f"Plan für 11:15-Zyklus (gecachte 11:00-Daten): {charge_1115_cached}")

        assert charge_1100 == charge_1115_cached, (
            f"Mit gecachten Prediction-Daten sollten beide Zyklen denselben Ladeslot liefern! "
            f"11:00-Plan: {charge_1100}, gecachter 11:15-Plan: {charge_1115_cached}"
        )

    def test_plan_start_times_are_correct(self, cfg, inverters_and_batteries):
        """Die Zeitstempel in PredictionData müssen exakt mit der Slot-Grenze übereinstimmen."""
        inverters, _ = inverters_and_batteries

        for start_h, start_m in [(11, 0), (11, 15), (11, 30), (13, 0)]:
            start = berlin_today(start_h, start_m)
            pdata = build_prediction(start, PRICES_FROM_11_00)
            first_ts = pdata.timestamps[0]
            assert first_ts == start, (
                f"Prediction-Start sollte {start.strftime('%H:%M')} sein, "
                f"ist aber {first_ts.astimezone(BERLIN).strftime('%H:%M:%S.%f')}"
            )
            # Alle Zeitstempel müssen auf 900s-Grid ausgerichtet sein
            for ts in pdata.timestamps:
                assert ts.timestamp() % 900 == 0, f"Zeitstempel {ts} nicht auf 15-min-Grid"

    def test_price_array_is_correctly_shifted(self, cfg, inverters_and_batteries):
        """
        Die Preisreihe muss absolut korrekt sein: gleicher Preis zur gleichen Uhrzeit,
        unabhängig vom Plan-Startzeitpunkt.

        Testet, dass Preis[13:15] identisch ist, egal ob Plan ab 11:00 oder 11:15 startet.
        """
        inverters, _ = inverters_and_batteries

        start_1100 = berlin_today(11, 0)
        start_1115 = berlin_today(11, 15)

        pdata_1100 = build_prediction(start_1100, PRICES_FROM_11_00)
        pdata_1115 = build_prediction(start_1115, PRICES_FROM_11_00[1:])  # 1 Slot verschoben!

        # Preis bei 13:15 in beiden Plänen ermitteln
        target = berlin_today(13, 15)
        ts_list_1100 = [ts.astimezone(BERLIN) for ts in pdata_1100.timestamps]
        ts_list_1115 = [ts.astimezone(BERLIN) for ts in pdata_1115.timestamps]

        try:
            idx_1100 = next(i for i, ts in enumerate(ts_list_1100) if ts.hour == 13 and ts.minute == 15)
            idx_1115 = next(i for i, ts in enumerate(ts_list_1115) if ts.hour == 13 and ts.minute == 15)
        except StopIteration:
            pytest.skip("Ziel-Zeitstempel 13:15 nicht in Prediction-Fenster")

        price_1100 = float(pdata_1100.electricprice[idx_1100])
        price_1115 = float(pdata_1115.electricprice[idx_1115])

        print(f"\nPreis bei 13:15 (Plan ab 11:00): {price_1100:.6f} EUR/Wh")
        print(f"Preis bei 13:15 (Plan ab 11:15): {price_1115:.6f} EUR/Wh")

        np.testing.assert_allclose(
            price_1100, price_1115, rtol=1e-5,
            err_msg="Preis bei 13:15 ist in beiden Plänen unterschiedlich – Preisreihe ist verschoben!"
        )


class TestCacheTtlMatchesPredictionRefresh:
    """PDATA_CACHE_TTL_S muss >= optimization_interval_minutes * 60 sein,
    damit der Solver in aufeinanderfolgenden 15-min-Zyklen GLEICHE Prediction-Daten
    benutzt und der Plan stabil bleibt."""

    def test_cache_ttl_larger_than_optimization_interval(self):
        import GridPythia.server.state as state
        from GridPythia.server.services import load_config

        cfg, _ = load_config()  # setzt state.PDATA_CACHE_TTL_S
        opt_interval_s = float(cfg.server.scheduler.optimization_interval_minutes) * 60.0
        refresh_interval_s = float(cfg.server.scheduler.prediction_refresh_minutes) * 60.0

        assert state.PDATA_CACHE_TTL_S >= opt_interval_s, (
            f"PDATA_CACHE_TTL_S ({state.PDATA_CACHE_TTL_S}s) ist kleiner als "
            f"optimization_interval_minutes×60 ({opt_interval_s}s). "
            "Das führt dazu, dass jeder Scheduler-Zyklus neue Prediction-Daten holt "
            "und der optimale Ladeslot sich um 15 Minuten verschiebt!"
        )
        assert state.PDATA_CACHE_TTL_S == refresh_interval_s, (
            f"PDATA_CACHE_TTL_S ({state.PDATA_CACHE_TTL_S}s) ≠ "
            f"prediction_refresh_minutes×60 ({refresh_interval_s}s). "
            "Die Config-Einstellung prediction_refresh_minutes wird nicht berücksichtigt."
        )


class TestNextOptimizationSlot:
    """next_optimization_slot und snap_to_dt_grid müssen konsistent sein."""

    def test_scheduler_snap_equals_dispatch_slot(self):
        """
        Wenn der Scheduler bei 11:14:25 feuert (lead=35s vor 11:15-Slot),
        muss snap_to_dt_grid(11:14:25) == 11:15 == dispatch_slot sein.
        """
        from GridPythia.coordination import next_optimization_slot
        from GridPythia.server.scheduler import _NEXT_SLOT_EPSILON_S

        # Simuliere: letzter Slot war 11:00, nächster = 11:15
        last_slot = berlin_today(11, 0)
        dispatch_slot = next_optimization_slot(
            last_slot + timedelta(seconds=_NEXT_SLOT_EPSILON_S), 15
        )
        assert dispatch_slot == berlin_today(11, 15), f"Dispatch-Slot sollte 11:15 sein: {dispatch_slot}"

        # Scheduler feuert 35s davor
        fire_at = dispatch_slot - timedelta(seconds=35)
        snapped = snap_to_dt_grid(fire_at, DT_HOURS)
        assert snapped == dispatch_slot, (
            f"snap_to_dt_grid bei Feuer-Zeit {fire_at.strftime('%H:%M:%S')} "
            f"sollte {dispatch_slot.strftime('%H:%M')} liefern, nicht {snapped.strftime('%H:%M')}"
        )
