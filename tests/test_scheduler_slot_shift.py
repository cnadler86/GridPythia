"""
Diagnose-Test: 15-Minuten-Verschiebung des Ladeslots.

Szenario:
  - Benutzer optimiert um 11:02 → Prediction ab 11:00, Solver sagt z.B. "lade 12:45"
  - Scheduler feuert um 11:14:30 für den 11:15-Slot
  - Frischer Fetch + gleicher SOC → andere Prediction ab 11:15 → Solver sagt "lade 13:00"
    (15-min-Shift, obwohl der günstigste Slot gleich geblieben ist)

Fix: pdata.slice_from(dispatch_slot) statt frischem Fetch.
  - Die Preise bleiben auf ihre absoluten Zeitstempel verankert
  - Mit projiziertem SOC aus dem vorherigen Plan ist die min-SOC-Frist identisch
  - Der Solver wählt denselben optimalen absoluten Ladeslot
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
    """Heutiges Datum in Berlin-Zeitzone mit sekundgenauer Nullstellung."""
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
    """Baut PredictionData mit festen Werten.

    Wichtig: *prices_eur_per_mwh* ist IMMER eine absolute Preisreihe ab *start*.
    Index 0 = Preis bei Timestamp 0, Index N = Preis bei Timestamp N.
    """
    timestamps = make_timestamps(start, horizon_h, dt_hours)
    actual_n = len(timestamps)

    prices_raw = list(prices_eur_per_mwh)
    while len(prices_raw) < actual_n:
        prices_raw.append(prices_raw[-1])
    prices_raw = prices_raw[:actual_n]

    charges_kwh = 0.1528
    vat = 0.19
    charges_wh = charges_kwh / 1000.0
    prices_eur_wh = np.array(
        [(p / 1_000_000.0 + charges_wh) * (1.0 + vat) for p in prices_raw],
        dtype=np.float32,
    )

    load_wh = np.full(actual_n, load_w * dt_hours, dtype=np.float32)

    pv_wh = np.zeros(actual_n, dtype=np.float32)
    if pv_peak_w > 0:
        for i, ts in enumerate(timestamps):
            local = ts.astimezone(BERLIN)
            t_h = local.hour + local.minute / 60.0
            pv_wh[i] = pv_peak_w * np.exp(-((t_h - 13.0) ** 2) / 4.0) * dt_hours

    return PredictionData(
        requested_start=start,
        timestamps=timestamps,
        dt_hours=dt_hours,
        load_wh=load_wh,
        electricprice_eur_wh=prices_eur_wh,
        feedintariff_eur_wh=np.zeros(actual_n, dtype=np.float32),
        pv_by_inverter={"SF800Pro": pv_wh},
    )


def build_inverters(cfg: AppConfig) -> list[InverterBase]:
    batteries = {b.device_id: Battery(b) for b in cfg.optimization.batteries}
    return [
        InverterBase(p, battery=batteries.get(p.battery_id) if p.battery_id else None)
        for p in cfg.optimization.inverters
    ]


def solve_once(
    cfg: AppConfig,
    inverters: list[InverterBase],
    pdata: PredictionData,
    soc_wh: float,
    initial_mode: InverterMode = InverterMode.DISCHARGE_ZERO_FEED_IN,
):
    optimizer = LinearOptimizer(
        inverters=inverters,
        objective=OptimizationObjective.MINIMIZE_COST,
        solver_opts=dict(cfg.optimization.solver.solver_opts),
    )
    return optimizer.solve(
        pdata,
        soc={"SF800Pro": soc_wh},
        initial_modes={"SF800Pro": initial_mode},
    )


def first_charge_slot(solution, timestamps: list[datetime]) -> datetime | None:
    """Gibt den ersten absoluten Zeitstempel zurück, an dem AC-Laden stattfindet."""
    for plan in solution.inverter_plans:
        for i, mode in enumerate(plan.modes):
            if mode in (InverterMode.AC_CHARGE, InverterMode.AC_CHARGE_ZERO_FEED_IN):
                return timestamps[i]
    return None


# ---------------------------------------------------------------------------
# Preistabelle A: Reale EPEX-Preise (EUR/MWh) ab Berlin 11:00
# Korrekt timestamp-indiziert: Index 0 = 11:00, Index 1 = 11:15, ...
# Wird von TestSliceFrom benutzt (nur Slice-Mechanik, kein Solver).
# ---------------------------------------------------------------------------
PRICES_FROM_11_00: list[float] = [
    # 11:00..12:45
    -0.36, -0.77, -1.45, -3.13, -5.34, -6.55, -7.27, -8.00,
    # 13:00..14:45  (globales Minimum: Index 10 = 13:30 mit -15.15)
    -12.59, -15.00, -15.15, -15.00, -11.63, -12.70, -9.60, -9.60,
    # 15:00..16:45
    -10.01, -4.70, -1.09, -0.07, -0.85, -0.60, 0.19, 37.79,
    # Abend/Nacht: hohe Preise (42h × 4 Slots = 168 Slots)
    *([120.0] * 168),
]

# ---------------------------------------------------------------------------
# Preistabelle B: Eindeutiges Minimum bei 13:00-13:45 (4 gleiche Slots).
# Teuer überall, dann günstig von 13:00-13:45, dann wieder teuer.
# Wird von TestSchedulerSlotShift benutzt:
#   - Fresh pdata ab 11:15 mit PRICES_CHEAP_AT_1300[1:] legt das billige
#     Fenster auf 12:45-13:30 statt 13:00-13:45 → 15-min-Shift (Bug-Demo)
#   - slice_from(11:15) legt das Fenster korrekt auf 13:00-13:45 (Fix)
# SOC=60% + IDLE: empirisch verifiziert, kein Discharge-Arbitrage-Effekt.
# ---------------------------------------------------------------------------
PRICES_CHEAP_AT_1300: list[float] = (
    [120.0] * 8   # 11:00..12:45 (8 Slots): teuer
    + [-80.0] * 4  # 13:00..13:45 (4 Slots): günstig
    + [120.0] * 180  # 14:00+: teuer
)

# ---------------------------------------------------------------------------
# Tests: snap_to_dt_grid
# ---------------------------------------------------------------------------


class TestSnapToDtGrid:
    """snap_to_dt_grid muss korrekt auf Slot-Grenzen runden."""

    def test_rounds_to_nearest_lower(self):
        assert snap_to_dt_grid(berlin_today(11, 2), DT_HOURS) == berlin_today(11, 0)

    def test_rounds_to_nearest_upper(self):
        assert snap_to_dt_grid(berlin_today(11, 14), DT_HOURS) == berlin_today(11, 15)

    def test_exactly_at_boundary(self):
        t = berlin_today(11, 15)
        assert snap_to_dt_grid(t, DT_HOURS) == t

    def test_scheduler_fire_35s_before_slot(self):
        t = berlin_today(11, 14).replace(second=25)
        assert snap_to_dt_grid(t, DT_HOURS) == berlin_today(11, 15)

    def test_scheduler_fire_60s_before_slot(self):
        assert snap_to_dt_grid(berlin_today(11, 14), DT_HOURS) == berlin_today(11, 15)

    def test_result_is_utc_aligned(self):
        for minute in range(60):
            result = snap_to_dt_grid(berlin_today(11, minute), DT_HOURS)
            assert result.timestamp() % 900 == 0


# ---------------------------------------------------------------------------
# Tests: PredictionData.slice_from
# ---------------------------------------------------------------------------


class TestSliceFrom:
    """PredictionData.slice_from() muss korrekt timestamp-indiziert slicen."""

    def _make_pdata(self) -> PredictionData:
        return build_prediction(berlin_today(11, 0), PRICES_FROM_11_00)

    def test_slice_returns_correct_start(self):
        pdata = self._make_pdata()
        sliced = pdata.slice_from(berlin_today(11, 15))
        assert sliced.timestamps[0].astimezone(BERLIN).hour == 11
        assert sliced.timestamps[0].astimezone(BERLIN).minute == 15

    def test_slice_preserves_price_at_absolute_time(self):
        """Preis bei 13:30 muss in Full-Prediction und Slice identisch sein."""
        pdata = self._make_pdata()
        sliced = pdata.slice_from(berlin_today(11, 15))

        def price_at(pd: PredictionData, h: int, m: int) -> float:
            for i, ts in enumerate(pd.timestamps):
                local = ts.astimezone(BERLIN)
                if local.hour == h and local.minute == m:
                    return float(pd.electricprice[i])
            raise ValueError(f"{h}:{m:02d} not in prediction")

        p_full = price_at(pdata, 13, 30)
        p_sliced = price_at(sliced, 13, 30)
        assert abs(p_full - p_sliced) < 1e-7, (
            f"Preis bei 13:30: Full={p_full:.8f}, Sliced={p_sliced:.8f}"
        )

    def test_slice_correct_step_count(self):
        pdata = self._make_pdata()
        sliced = pdata.slice_from(berlin_today(11, 15))
        assert sliced.steps == pdata.steps - 1  # ein Slot abgeschnitten

    def test_slice_at_first_timestamp_returns_self(self):
        pdata = self._make_pdata()
        sliced = pdata.slice_from(berlin_today(11, 0))
        assert sliced is pdata

    def test_slice_preserves_load_alignment(self):
        pdata = self._make_pdata()
        sliced = pdata.slice_from(berlin_today(11, 30))
        np.testing.assert_array_equal(sliced.load_wh, pdata.load_wh[2:])

    def test_slice_beyond_range_raises(self):
        pdata = self._make_pdata()
        far_future = berlin_today(11, 0) + timedelta(days=10)
        with pytest.raises(ValueError, match="slice_from"):
            pdata.slice_from(far_future)


# ---------------------------------------------------------------------------
# Tests: Slot-Shift-Bug (Kern-Diagnose)
# ---------------------------------------------------------------------------


class TestSchedulerSlotShift:
    """Diagnose und Fix für den 15-Minuten-Plan-Shift."""

    @pytest.fixture
    def cfg(self):
        return load_config()

    @pytest.fixture
    def inverters(self, cfg):
        return build_inverters(cfg)

    def test_fresh_pdata_same_soc_causes_shift(self, cfg, inverters):
        """
        DOKUMENTIERT DAS URSACHEN-VERHALTEN (kein Bug im Solver!):

        Mit SOC=30% und DISCHARGE_ZFI laeuft die Batterie in ~4 Slots auf
        min_soc. Der Solver muss VOR dem globalen Minimum laden.

        Von 11:00: Deadline ~12:00 (absolut), cheapest-before-deadline = 12:00.
        Von 11:15 mit GLEICHEM SOC: Deadline ~12:15 (15 min spaeter).
        Cheapest-before-deadline = 12:15. Diff = +15 min. Das ist der Bug.

        Der FIX liegt in pdata.slice_from() + projiziertem SOC (naechster Test).
        """
        bat_cap = cfg.optimization.batteries[0].capacity_wh
        # SOC=30%: ~4 Discharge-Slots bis min_soc -> Deadline im teuren Bereich
        soc_wh = 0.30 * bat_cap

        pdata_1100 = build_prediction(berlin_today(11, 0), PRICES_FROM_11_00)
        # Fresh Fetch: korrekte absolute Preise ab 11:15, aber mit gleichem SOC
        pdata_1115_fresh = build_prediction(berlin_today(11, 15), PRICES_FROM_11_00[1:])

        sol_1100 = solve_once(cfg, inverters, pdata_1100, soc_wh, InverterMode.DISCHARGE_ZERO_FEED_IN)
        sol_1115 = solve_once(cfg, inverters, pdata_1115_fresh, soc_wh, InverterMode.DISCHARGE_ZERO_FEED_IN)

        charge_1100 = first_charge_slot(sol_1100, pdata_1100.timestamps)
        charge_1115 = first_charge_slot(sol_1115, pdata_1115_fresh.timestamps)

        print(f"\n[fresh+same-soc] 11:00 plan -> {charge_1100 and charge_1100.astimezone(BERLIN).strftime('%H:%M')}")
        print(f"[fresh+same-soc] 11:15 fresh -> {charge_1115 and charge_1115.astimezone(BERLIN).strftime('%H:%M')}")

        assert charge_1100 is not None, "11:00-Plan sollte einen Ladeslot enthalten"
        assert charge_1115 is not None, "11:15-Plan sollte einen Ladeslot enthalten"
        diff_min = (charge_1115 - charge_1100).total_seconds() / 60.0
        assert diff_min == 15.0, (
            f"Rohverhalten hat sich geaendert (erwartet +15 min, erhalten {diff_min:.0f} min). "
            "Bitte Diagnose ueberpruefen."
        )

    def test_slice_from_with_projected_soc_gives_stable_plan(self, cfg, inverters):
        """
        TESTET DEN FIX:

        Mit pdata.slice_from(dispatch_slot) + projiziertem SOC aus dem
        Vorplan wählt der Solver denselben absoluten Ladeslot, unabhängig davon
        ob die Optimierung um 11:00 oder 11:15 gestartet wurde.

        Warum das funktioniert:
        - Gleiche absolute Preise (slice teilt dieselbe Preis-Reihe)
        - Projektierer SOC = live-SOC nach 1 Slot → gleiche Ausgangslage
        Der Solver sieht exakt dasselbe Optimierungsproblem.
        """
        bat_cap = cfg.optimization.batteries[0].capacity_wh
        soc_wh = 0.60 * bat_cap  # SOC 60% + IDLE-Modus → kein Discharge-Arbitrage

        # Voller 48h-Plan ab 11:00
        full_pdata = build_prediction(berlin_today(11, 0), PRICES_CHEAP_AT_1300)

        # 11:00-Optimierung
        sol_1100 = solve_once(cfg, inverters, full_pdata, soc_wh, InverterMode.IDLE)
        charge_1100 = first_charge_slot(sol_1100, full_pdata.timestamps)

        # Projizierter SOC bei 11:15 aus dem 11:00-Plan
        # battery_soc_wh[i] = SOC am ENDE von Slot i
        # Slot 0 = 11:00-11:15 → battery_soc_wh[0] = SOC bei 11:15 (Anfang Slot 1)
        plan_1100 = sol_1100.inverter_plans[0]
        assert plan_1100.battery_soc_wh is not None, "Plan muss SOC-Kurve enthalten"
        projected_soc_wh = float(plan_1100.battery_soc_wh[0])

        # FIX: slice_from(11:15) statt frischem Fetch
        pdata_1115 = full_pdata.slice_from(berlin_today(11, 15))

        # 11:15-Optimierung mit projiziertem SOC
        sol_1115 = solve_once(cfg, inverters, pdata_1115, projected_soc_wh, InverterMode.IDLE)
        charge_1115 = first_charge_slot(sol_1115, pdata_1115.timestamps)

        print(f"\n[slice+projected-soc] 11:00 -> {charge_1100 and charge_1100.astimezone(BERLIN).strftime('%H:%M')}")
        print(f"[slice+projected-soc] 11:15 -> {charge_1115 and charge_1115.astimezone(BERLIN).strftime('%H:%M')}")
        print(f"Projizierter SOC: {projected_soc_wh:.1f} Wh ({projected_soc_wh/bat_cap*100:.1f}%)")

        assert charge_1100 is not None, "11:00-Plan sollte einen Ladeslot enthalten"
        assert charge_1115 is not None, "11:15-Plan sollte einen Ladeslot enthalten"
        assert charge_1100 == charge_1115, (
            f"SLOT-SHIFT-BUG nicht behoben: "
            f"11:00-Plan laedt um {charge_1100.astimezone(BERLIN).strftime('%H:%M')}, "
            f"11:15-Plan laedt um {charge_1115.astimezone(BERLIN).strftime('%H:%M')} "
            f"(Diff: {(charge_1115 - charge_1100).total_seconds()/60:.0f} min)."
        )

    def test_plan_timestamps_are_grid_aligned(self, cfg):
        """Prediction-Zeitstempel müssen exakt auf 900s-Grid (UTC) ausgerichtet sein."""
        for start_h, start_m in [(11, 0), (11, 15), (11, 30), (13, 0)]:
            start = berlin_today(start_h, start_m)
            pdata = build_prediction(start, PRICES_FROM_11_00)
            assert pdata.timestamps[0] == start
            for ts in pdata.timestamps:
                assert ts.timestamp() % 900 == 0, f"Zeitstempel {ts} nicht auf 15-min-Grid"


# ---------------------------------------------------------------------------
# Tests: Scheduler-Slot-Konsistenz
# ---------------------------------------------------------------------------


class TestNextOptimizationSlot:
    """next_optimization_slot und snap_to_dt_grid müssen konsistent sein."""

    def test_scheduler_snap_equals_dispatch_slot(self):
        from GridPythia.coordination import next_optimization_slot
        from GridPythia.server.scheduler import _NEXT_SLOT_EPSILON_S

        last_slot = berlin_today(11, 0)
        dispatch_slot = next_optimization_slot(
            last_slot + timedelta(seconds=_NEXT_SLOT_EPSILON_S), 15
        )
        assert dispatch_slot == berlin_today(11, 15)

        fire_at = dispatch_slot - timedelta(seconds=35)
        snapped = snap_to_dt_grid(fire_at, DT_HOURS)
        assert snapped == dispatch_slot

    def test_optimize_request_carries_prediction_start(self):
        """OptimizeRequest.prediction_start muss vorhanden und dokumentiert sein."""
        from GridPythia.server.models import OptimizeRequest

        req = OptimizeRequest(
            timezone="Europe/Berlin",
            prediction_start="2026-05-03T09:15:00+00:00",
        )
        assert req.prediction_start == "2026-05-03T09:15:00+00:00"

        # Default ist None (Browser/API-Pfad nutzt TTL-Cache)
        req_default = OptimizeRequest(timezone="Europe/Berlin")
        assert req_default.prediction_start is None

