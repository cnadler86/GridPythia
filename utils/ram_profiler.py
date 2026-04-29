"""RAM profiling utility for GridPythia.

Führt eine schrittweise RAM-Analyse durch:
  1. Baseline nach Python-Start
  2. Import-Kosten jedes Top-Level-Moduls einzeln messen
  3. Full-App-Start (alle Imports wie im echten Betrieb)
  4. tracemalloc Top-Allokationen
  5. pympler-Analyse der größten Live-Objekte

Usage:
    uv run python -m utils.ram_profiler
    uv run python -m utils.ram_profiler --phase imports   # nur Import-Analyse
    uv run python -m utils.ram_profiler --phase full      # voller App-Start
    uv run python -m utils.ram_profiler --phase objects   # Live-Objekte
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import psutil

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """RSS des aktuellen Prozesses in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def _vms_mb() -> float:
    """VMS (virtuelles Memory) in MB."""
    return psutil.Process().memory_info().vms / 1024 / 1024


def _hr(title: str) -> None:
    w = 70
    print(f"\n{'─' * w}")
    print(f"  {title}")
    print(f"{'─' * w}")


def _row(label: str, value: Any, unit: str = "MB") -> None:
    print(f"  {label:<45} {value:>8.1f} {unit}")


# ---------------------------------------------------------------------------
# Phase 1 – Import-Kosten
# ---------------------------------------------------------------------------

_MODULES_TO_PROBE = [
    # Std-lib / leightweight
    "pathlib",
    "asyncio",
    "json",
    "yaml",
    # Pydantic / structlog
    "pydantic",
    "structlog",
    # Numerical stack
    "numpy",
    "scipy",
    "statsmodels",
    # Optimization
    "cvxpy",
    "highspy",
    # Data / web
    "plotly",
    "fastapi",
    "uvicorn",
    "aiohttp",
    "paho.mqtt.client",
    # Project-intern
    "GridPythia.config",
    "GridPythia.prediction.base",
    "GridPythia.prediction.prediction",
    "GridPythia.prediction.electricprice.provider",
    "GridPythia.prediction.electricprice.epexpredictor",
    "GridPythia.prediction.electricprice.energycharts",
    "GridPythia.prediction.pvforecast.openmeteo",
    "GridPythia.prediction.pvforecast.akkudoktor",
    "GridPythia.prediction.weather.openmeteo",
    "GridPythia.prediction.weather.brightsky",
    "GridPythia.prediction.load.profilecsv",
    "GridPythia.optimization.solver",
    "GridPythia.simulation.grid_simulation",
    "GridPythia.server.services",
    "GridPythia.server.app",
]


def phase_imports() -> None:
    _hr("PHASE 1 – Import-Kosten (inkrementell)")
    print(f"  {'Modul':<45} {'Delta':>8}  {'RSS nach':>8}")

    baseline = _rss_mb()
    cumulative = baseline

    for mod_name in _MODULES_TO_PROBE:
        before = _rss_mb()
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            print(f"  {mod_name:<45} {'FEHLT':>8}  ({e})")
            continue
        gc.collect()
        after = _rss_mb()
        delta = after - before
        print(f"  {mod_name:<45} {delta:>+7.1f}MB  {after:>7.1f}MB")

    _hr("PHASE 1 – Zusammenfassung")
    _row("Baseline (vor Imports)", baseline)
    _row("Nach allen Imports", _rss_mb())
    _row("Gesamt Import-Overhead", _rss_mb() - baseline)


# ---------------------------------------------------------------------------
# Phase 2 – Voller App-Start mit tracemalloc
# ---------------------------------------------------------------------------


def phase_full() -> None:
    _hr("PHASE 2 – Voller App-Start (tracemalloc)")

    tracemalloc.start(25)
    snap_before = tracemalloc.take_snapshot()
    rss_before = _rss_mb()

    # --- echte App-Imports wie im Betrieb ---
    from GridPythia.server.app import create_app  # noqa: F401

    gc.collect()
    rss_after = _rss_mb()
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    _hr("PHASE 2 – RSS-Delta")
    _row("RSS vor App-Import", rss_before)
    _row("RSS nach App-Import", rss_after)
    _row("Delta", rss_after - rss_before)

    # --- Top tracemalloc-Allokationen ---
    _hr("PHASE 2 – Top-30 tracemalloc-Allokationen (nach lineno)")
    stats = snap_after.compare_to(snap_before, "lineno")
    for s in stats[:30]:
        size_kb = s.size_diff / 1024
        count = s.count_diff
        trace = str(s.traceback[0]) if s.traceback else "?"
        # Kürze langen Pfad
        trace = trace.replace(str(Path.cwd()), ".")
        print(f"  {size_kb:>8.1f} KB  {count:>6}× alloc  {trace}")

    _hr("PHASE 2 – Top-20 nach Modul (kumuliert)")
    stats_mod = snap_after.compare_to(snap_before, "filename")
    for s in stats_mod[:20]:
        size_kb = s.size_diff / 1024
        fname = str(s.traceback[0].filename).replace(str(Path.cwd()), ".")
        print(f"  {size_kb:>8.1f} KB   {fname}")


# ---------------------------------------------------------------------------
# Phase 3 – Live-Objekte mit pympler
# ---------------------------------------------------------------------------


def phase_objects() -> None:
    _hr("PHASE 3 – Live-Objekte (pympler asizeof)")
    try:
        from pympler import muppy, summary
    except ImportError:
        print("  pympler nicht installiert – übersprungen")
        return

    # Importiere alles, was real geladen wird
    from GridPythia.server.app import create_app  # noqa: F401

    gc.collect()

    all_objects = muppy.get_objects()
    obj_summary = summary.summarize(all_objects)
    summary.print_(obj_summary)


# ---------------------------------------------------------------------------
# Phase 4 – sys.modules Größenanalyse
# ---------------------------------------------------------------------------


def phase_modules_size() -> None:
    """Zeigt welche bereits importierten Module am meisten sys.modules-Einträge belegen."""
    _hr("PHASE 4 – sys.modules nach Namespace-Gruppe")

    # Alle importierten Module einmal laden
    try:
        import GridPythia.server.app  # noqa: F401
    except Exception:
        pass
    gc.collect()

    from collections import defaultdict

    groups: dict[str, int] = defaultdict(int)
    for mod_name in sys.modules:
        top = mod_name.split(".")[0]
        groups[top] += 1

    sorted_groups = sorted(groups.items(), key=lambda x: -x[1])
    print(f"\n  {'Namespace':<30} {'Anzahl Module':>15}")
    for ns, count in sorted_groups[:30]:
        print(f"  {ns:<30} {count:>15}")

    print(f"\n  Gesamt importierte Module: {len(sys.modules)}")
    print(f"  Aktueller RSS:             {_rss_mb():.1f} MB")
    print(f"  Aktueller VMS:             {_vms_mb():.1f} MB")


# ---------------------------------------------------------------------------
# Phase 5 – Numpy/Scipy Array-Inventory
# ---------------------------------------------------------------------------


def phase_arrays() -> None:
    """Findet alle numpy-Arrays im aktuellen Namespace."""
    _hr("PHASE 5 – numpy-Array Inventory")
    try:
        import numpy as np
        from pympler.asizeof import asizeof
    except ImportError:
        print("  numpy/pympler fehlt – übersprungen")
        return

    try:
        import GridPythia.server.app  # noqa: F401
    except Exception:
        pass
    gc.collect()

    arrays: list[tuple[int, str, tuple]] = []
    for obj in gc.get_objects():
        if isinstance(obj, np.ndarray):
            size = obj.nbytes
            arrays.append((size, obj.dtype.name, obj.shape))

    arrays.sort(key=lambda x: -x[0])
    total = sum(a[0] for a in arrays)
    print(f"\n  Gesamt numpy-Arrays: {len(arrays)}  ({total / 1024 / 1024:.1f} MB)")
    print(f"\n  {'Größe':>10}  {'dtype':<10}  {'Shape'}")
    for size, dtype, shape in arrays[:30]:
        print(f"  {size / 1024:>9.1f}KB  {dtype:<10}  {shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GridPythia RAM Profiler")
    parser.add_argument(
        "--phase",
        choices=["imports", "full", "objects", "modules", "arrays", "all"],
        default="all",
        help="Welche Analyse-Phase ausführen (default: all)",
    )
    args = parser.parse_args()

    _hr("GridPythia RAM Profiler")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  Plattform:    {sys.platform}")
    print(f"  Baseline RSS: {_rss_mb():.1f} MB")
    print(f"  Baseline VMS: {_vms_mb():.1f} MB")

    phase = args.phase

    if phase in ("imports", "all"):
        phase_imports()

    if phase in ("modules", "all"):
        phase_modules_size()

    if phase in ("full", "all"):
        phase_full()

    if phase in ("arrays", "all"):
        phase_arrays()

    if phase in ("objects", "all"):
        phase_objects()

    _hr("FERTIG")
    print(f"  Finaler RSS: {_rss_mb():.1f} MB")
    print(f"  Finaler VMS: {_vms_mb():.1f} MB")


if __name__ == "__main__":
    main()
