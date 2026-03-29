"""Interactive dev-tool GUI for exploring and configuring forecast providers.

Usage::

    uv run python -m utils.gui
    # or
    python utils/gui.py
"""

from __future__ import annotations

import asyncio
import re
import threading
import tkinter as tk
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import matplotlib
from matplotlib.axes import Axes

matplotlib.use("TkAgg")
import json
from array import array

import matplotlib.dates as mdates
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from src.config.models import BatteryParameters, InverterParameters
from src.optimization.solver import LinearOptimizer, LinearSolution, OptimizationObjective
from src.prediction.base import make_timestamps
from src.prediction.electricprice.energycharts import ElecPriceEnergyCharts, EnergyChartsConfig
from src.prediction.electricprice.fixed import ElecPriceFixed
from src.prediction.electricprice.import_ import ElecPriceImport
from src.prediction.feedintariff.fixed import FeedInTariffFixed
from src.prediction.feedintariff.import_ import FeedInTariffImport
from src.prediction.load.profilejson import LoadProfileJSON
from src.prediction.prediction import Prediction, PredictionSetup
from src.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor
from src.prediction.pvforecast.import_ import PVForecastImport
from src.prediction.pvforecast.openmeteo import PVForecastOpenMeteo
from src.prediction.pvforecast.provider import PVPlaneConfig
from src.prediction.weather.brightsky import WeatherBrightSky
from src.prediction.weather.openmeteo import WeatherOpenMeteo
from src.simulation.devices import InverterMode
from src.simulation.devices.battery import Battery
from src.simulation.devices.inverterbase import InverterBase

# ── constants ─────────────────────────────────────────────────────────────

_TZ_CHOICES = [
    "UTC",
    "Europe/Berlin",
    "Europe/London",
    "Europe/Paris",
    "America/New_York",
    "America/Chicago",
    "America/Los_Angeles",
    "Asia/Tokyo",
    "Asia/Shanghai",
]
_BZ_CHOICES = [
    "DE-LU",
    "AT",
    "CH",
    "FR",
    "NL",
    "BE",
    "CZ",
    "DK1",
    "DK2",
    "NO1",
    "SE1",
    "PL",
]
_DT_CHOICES = ["0.25", "0.5", "1.0", "2.0", "4.0"]
_PAD: Mapping[str, Any] = {"padx": 4, "pady": 3}

# ── utilities ─────────────────────────────────────────────────────────────


def _run_async(coro, on_done, on_error) -> None:
    """Run *coro* in a daemon thread; deliver results via callbacks."""

    def _worker():
        try:
            on_done(asyncio.run(coro))
        except Exception as exc:
            on_error(exc, traceback.format_exc())

    threading.Thread(target=_worker, daemon=True).start()


def _csv(text: str) -> list[float]:
    return [float(t) for t in re.split(r"[\s,;]+", text.strip()) if t]


def _field(parent: tk.Misc, row: int, label: str, default: str = "") -> tk.StringVar:
    """Grid label + entry at *row*; return the StringVar."""
    lbl = ttk.Label(parent, text=label)
    lbl.grid(row=row, column=0, sticky="w", **_PAD)
    var = tk.StringVar(value=default)
    ent = ttk.Entry(parent, textvariable=var, width=16)
    ent.grid(row=row, column=1, sticky="ew", **_PAD)
    parent.columnconfigure(1, weight=1)
    return var


def _combofield(
    parent: tk.Misc,
    row: int,
    label: str,
    choices: list[str],
    default: str = "",
) -> tk.StringVar:
    lbl = ttk.Label(parent, text=label)
    lbl.grid(row=row, column=0, sticky="w", **_PAD)
    var = tk.StringVar(value=default or (choices[0] if choices else ""))
    cb = ttk.Combobox(parent, textvariable=var, values=choices, state="readonly", width=16)
    cb.grid(row=row, column=1, sticky="ew", **_PAD)
    return var


def _textarea(parent: tk.Misc, row: int, height: int = 5) -> scrolledtext.ScrolledText:
    w = scrolledtext.ScrolledText(parent, height=height, width=24, wrap="word", font=("Courier", 9))
    w.grid(row=row, column=0, columnspan=2, sticky="ew", **_PAD)
    return w


def _place_hover_annotation(ax, annot, x: float, y: float) -> None:
    """Keep hover annotations inside the visible figure area near the hovered point."""
    bbox = ax.figure.bbox
    px, py = ax.transData.transform((x, y))
    xoff = 10
    yoff = 15
    if px > bbox.x0 + bbox.width * 0.72:
        xoff = -10
    if py > bbox.y0 + bbox.height * 0.72:
        yoff = -15
    annot.set_position((xoff, yoff))
    annot.set_ha("left" if xoff > 0 else "right")
    annot.set_va("bottom" if yoff > 0 else "top")


# ── Hover tooltip ────────────────────────────────────────────────────────


def _wire_hover(
    canvas: FigureCanvasTkAgg,
    ax,
    xs_dt: list,
    ys: list,
    fmt_y: str = "{:.4f}",
    unit: str = "",
) -> int:
    """Attach a value tooltip + crosshair to *ax*. Returns the mpl connection id."""
    if not xs_dt:
        return -1
    xs_num = np.array(mdates.date2num(xs_dt), dtype=float)
    ys_arr = np.array(ys, dtype=float)

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fffff0", ec="#888", lw=0.8, alpha=0.95),
        fontsize=8,
        visible=False,
        zorder=10,
    )
    vline = ax.axvline(x=xs_num[0], color="#999", lw=0.7, ls="--", visible=False, zorder=5)
    annot.set_in_layout(False)
    annot.set_annotation_clip(False)
    annot.set_clip_on(False)
    vline.set_in_layout(False)

    def _on_move(event):
        if event.inaxes is not ax:
            if annot.get_visible():
                annot.set_visible(False)
                vline.set_visible(False)
                canvas.draw_idle()
            return
        if event.xdata is None:
            return
        idx = int(np.argmin(np.abs(xs_num - event.xdata)))
        xi = xs_dt[idx]
        yi = float(ys_arr[idx])
        annot.xy = (xs_num[idx], yi)
        _place_hover_annotation(ax, annot, xs_num[idx], yi)
        suffix = f" {unit}" if unit else ""
        annot.set_text(f"{xi:%Y-%m-%d %H:%M}\n{fmt_y.format(yi)}{suffix}")
        annot.set_visible(True)
        vline.set_xdata([xs_num[idx]])
        vline.set_visible(True)
        canvas.draw_idle()

    return canvas.mpl_connect("motion_notify_event", _on_move)


def _setup_plot_axes(fig, num_rows: int = 111) -> Axes:
    """Common plot setup: clear fig, add subplot. Returns the Axes object."""
    fig.clear()
    ax = fig.add_subplot(num_rows)
    return ax


def _finalize_plot(ax, ylabel: str = "", title: str = "", gridOn: bool = True) -> None:
    """Common plot finalization: grid, labels, date formatting."""
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if gridOn:
        ax.grid(True, alpha=0.3)
    # Date formatting happens in _done() for all tabs


def _wire_pv_hover(
    canvas: FigureCanvasTkAgg,
    ax,
    xs_dt: list,
    ys: list,
    dt_hours: float,
) -> int:
    """PV-specific hover: step energy + daily total + remaining for that day."""
    if not xs_dt:
        return -1

    # Values are already energy per step (Wh), so daily aggregation is a plain sum.
    day_total_wh: dict = {}
    for ti, wi in zip(xs_dt, ys):
        d = ti.date()
        day_total_wh[d] = day_total_wh.get(d, 0.0) + float(wi)

    # Precompute remaining energy from each slot to end-of-day (Wh).
    remaining_wh: list[float] = [0.0] * len(xs_dt)
    running: dict = {}
    for i in range(len(xs_dt) - 1, -1, -1):
        d = xs_dt[i].date()
        running[d] = running.get(d, 0.0) + float(ys[i])
        remaining_wh[i] = running[d]

    xs_num = np.array(mdates.date2num(xs_dt), dtype=float)
    ys_arr = np.array(ys, dtype=float)

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fffff0", ec="#888", lw=0.8, alpha=0.95),
        fontsize=8,
        visible=False,
        zorder=10,
    )
    vline = ax.axvline(x=xs_num[0], color="#999", lw=0.7, ls="--", visible=False, zorder=5)
    annot.set_in_layout(False)
    annot.set_annotation_clip(False)
    annot.set_clip_on(False)
    vline.set_in_layout(False)

    def _on_move(event):
        if event.inaxes is not ax:
            if annot.get_visible():
                annot.set_visible(False)
                vline.set_visible(False)
                canvas.draw_idle()
            return
        if event.xdata is None:
            return
        idx = int(np.argmin(np.abs(xs_num - event.xdata)))
        xi = xs_dt[idx]
        yi = float(ys_arr[idx])
        d = xi.date()
        total_kwh = day_total_wh.get(d, 0.0) / 1000.0
        rem_kwh = remaining_wh[idx] / 1000.0
        annot.xy = (xs_num[idx], yi)
        _place_hover_annotation(ax, annot, xs_num[idx], yi)
        annot.set_text(
            f"{xi:%Y-%m-%d %H:%M}\n"
            f"{yi:.0f} Wh\n"
            f"Day total:  {total_kwh:.2f} kWh\n"
            f"Remaining:  {rem_kwh:.2f} kWh"
        )
        annot.set_visible(True)
        vline.set_xdata([xs_num[idx]])
        vline.set_visible(True)
        canvas.draw_idle()

    return canvas.mpl_connect("motion_notify_event", _on_move)


# ── Base tab ──────────────────────────────────────────────────────────────


class _Tab:
    TITLE = ""
    PROVIDERS: list[str] = []

    def __init__(self, nb: ttk.Notebook, app: "App") -> None:
        self.app = app
        self.frame = ttk.Frame(nb)
        nb.add(self.frame, text=f"  {self.TITLE}  ")
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self._hover_cids: list[int] = []
        self._cached_prov: Any = None
        self._last_prov_sig: str | None = None
        self._build_layout()
        self._rebuild()

    # ── layout ────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # Left panel (fixed width)
        left = ttk.Frame(self.frame, width=260)
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)  # config section expands

        # Row 0: provider combobox
        if self.PROVIDERS:
            pf = ttk.LabelFrame(left, text="Provider", padding=4)
            pf.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
            pf.columnconfigure(0, weight=1)
            self._prov_var = tk.StringVar(value=self.PROVIDERS[0])
            cb = ttk.Combobox(
                pf,
                textvariable=self._prov_var,
                values=self.PROVIDERS,
                state="readonly",
            )
            cb.grid(sticky="ew")
            cb.bind("<<ComboboxSelected>>", lambda _: self._rebuild())

        # Row 1: dynamic config (expandable)
        self._cfg = ttk.LabelFrame(left, text="Configuration", padding=4)
        self._cfg.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        self._cfg.columnconfigure(1, weight=1)

        # Row 2: fetch button + status
        af = ttk.Frame(left)
        af.grid(row=2, column=0, sticky="ew", padx=4, pady=(2, 6))
        af.columnconfigure(0, weight=1)
        ttk.Button(af, text="▶  Fetch", command=self.fetch).grid(row=0, column=0, sticky="ew")
        self._status = tk.StringVar(value="–")
        ttk.Label(
            af,
            textvariable=self._status,
            foreground="#666",
            wraplength=235,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        # Right: matplotlib figure
        right = ttk.Frame(self.frame)
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(8, 5), dpi=100, tight_layout=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb = ttk.Frame(right)
        tb.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self._canvas, tb)

    # ── dynamic config ────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._cached_prov = None
        self._last_prov_sig = None
        for w in self._cfg.winfo_children():
            w.destroy()
        self._build_fields()

    def _build_fields(self) -> None:
        """Override — add widgets to self._cfg using grid rows."""

    # ── provider construction ─────────────────────────────────────────

    def make_provider(self) -> Any:
        raise NotImplementedError

    def _provider_sig(self) -> str | None:
        """Return a stable key for the current config, or None to always recreate."""
        return None

    def _get_provider(self) -> Any:
        """Return a cached provider, creating a new one only when config changed."""
        sig = self._provider_sig()
        if sig is None or sig != self._last_prov_sig:
            self._cached_prov = self.make_provider()
            self._last_prov_sig = sig
        return self._cached_prov

    # ── fetch & plot ──────────────────────────────────────────────────

    def fetch(self) -> None:
        try:
            prov = self._get_provider()
        except Exception as exc:
            self._status.set(f"Config error: {exc}")
            return
        start, hours, dt = self.app.get_time_params()
        ts = make_timestamps(start, hours, dt)
        self._status.set("Fetching…")
        _run_async(
            prov.fetch(ts),
            on_done=lambda r: self.app.root.after(0, lambda: self._done(r, ts)),
            on_error=lambda e, tb: self.app.root.after(0, lambda: self._fail(e, tb)),
        )

    def _done(self, result: pl.Series | pl.DataFrame, ts: pl.Series) -> None:
        # Disconnect stale hover callbacks before clearing
        for cid in self._hover_cids:
            self._canvas.mpl_disconnect(cid)
        self._hover_cids.clear()
        self._fig.clear()
        try:
            self._do_plot(result, ts)
            self._canvas.draw()
            self._status.set(f"OK · {len(ts)} steps · {datetime.now():%H:%M:%S}")
        except Exception as exc:
            self._status.set(f"Plot error: {exc}")

    def _fail(self, exc: Exception, tb: str) -> None:
        self._status.set(f"Error: {exc}")
        messagebox.showerror("Fetch error", f"{exc}\n\n{tb[:1500]}", parent=self.frame)

    def _do_plot(self, result: pl.Series | pl.DataFrame, ts: pl.Series) -> None:
        ax = _setup_plot_axes(self._fig)
        # Default plotting: if we got a Series, plot it; if DataFrame, plot first column
        if isinstance(result, pl.Series):
            ax.plot(ts.to_list(), result.to_list(), linewidth=1.4)
        else:
            cols = result.columns
            if cols:
                ax.plot(ts.to_list(), result[cols[0]].to_list(), linewidth=1.4)
        _finalize_plot(ax, title=self.TITLE)


# ── Electric Price ────────────────────────────────────────────────────────


class ElecPriceTab(_Tab):
    TITLE = "Electric Price"
    PROVIDERS = ["EnergyCharts", "Fixed", "Import"]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0
        p = self._prov_var.get()

        if p == "Fixed":
            self._price = _field(f, row, "Price EUR/kWh", "0.30")
            row += 1
        elif p == "EnergyCharts":
            self._zone = _combofield(f, row, "Bidding zone", _BZ_CHOICES, "DE-LU")
            row += 1
        else:  # Import
            ttk.Label(f, text="Values (EUR/Wh, CSV):").grid(
                row=row, column=0, columnspan=2, sticky="w", **_PAD
            )
            row += 1
            self._ta = _textarea(f, row)
            row += 1
            self._srcdt = _field(f, row, "Source dt [h]", "1.0")
            row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        self._charges = _field(f, row, "Charges EUR/kWh", "0.1528")
        row += 1
        self._vat = _field(f, row, "VAT rate", "0.19")

    def _provider_sig(self) -> str | None:
        p = self._prov_var.get()
        if p == "EnergyCharts":
            return f"EnergyCharts:{self._zone.get()}:{self._charges.get()}:{self._vat.get()}"
        return None  # stateless providers: always recreate is fine

    def make_provider(self):
        ch, vat = float(self._charges.get()), float(self._vat.get())
        p = self._prov_var.get()
        if p == "Fixed":
            return ElecPriceFixed(price_kwh=float(self._price.get()), charges_kwh=ch, vat_rate=vat)
        if p == "EnergyCharts":
            cfg = EnergyChartsConfig(bidding_zone=self._zone.get(), charges_kwh=ch, vat_rate=vat)
            return ElecPriceEnergyCharts(cfg)
        return ElecPriceImport(
            prices_wh=_csv(self._ta.get("1.0", "end")),
            source_dt_hours=float(self._srcdt.get()),
        )

    def _do_plot(self, result: pl.Series | pl.DataFrame, ts: pl.Series) -> None:
        ax = _setup_plot_axes(self._fig)
        t = ts.to_list()
        s = result if isinstance(result, pl.Series) else result[result.columns[0]]
        v = [x * 1000 for x in s.to_list()]  # EUR/Wh → EUR/kWh
        ax.step(t, v, color="#1565C0", linewidth=1.5, where="post", label="Electric Price")
        ax.fill_between(t, v, alpha=0.12, color="#1565C0", step="post")
        _finalize_plot(ax, ylabel="EUR / kWh", title="Electricity Price")
        self._hover_cids.append(_wire_hover(self._canvas, ax, t, v, fmt_y="{:.5f}", unit="EUR/kWh"))


# ── Feed-in Tariff ────────────────────────────────────────────────────────


class FeedInTariffTab(_Tab):
    TITLE = "Feed-in Tariff"
    PROVIDERS = ["Fixed", "Import"]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0
        if self._prov_var.get() == "Fixed":
            self._tariff = _field(f, row, "Tariff EUR/kWh", "0.0")
        else:
            ttk.Label(f, text="Values (EUR/Wh, CSV):").grid(
                row=row, column=0, columnspan=2, sticky="w", **_PAD
            )
            row += 1
            self._ta = _textarea(f, row)
            row += 1
            self._srcdt = _field(f, row, "Source dt [h]", "1.0")

    def make_provider(self):
        if self._prov_var.get() == "Fixed":
            return FeedInTariffFixed(tariff_kwh=float(self._tariff.get()))
        return FeedInTariffImport(
            tariffs_wh=_csv(self._ta.get("1.0", "end")),
            source_dt_hours=float(self._srcdt.get()),
        )

    def _do_plot(self, result: pl.Series | pl.DataFrame, ts: pl.Series) -> None:
        ax = _setup_plot_axes(self._fig)
        t = ts.to_list()
        s = result if isinstance(result, pl.Series) else result[result.columns[0]]
        v = [x * 1000 for x in s.to_list()]  # Convert to kWh
        ax.step(t, v, color="#2E7D32", linewidth=1.5, where="post", label="Feed-in Tariff")
        ax.fill_between(t, v, alpha=0.12, color="#2E7D32", step="post")
        _finalize_plot(ax, ylabel="EUR / kWh", title="Feed-in Tariff")
        self._hover_cids.append(_wire_hover(self._canvas, ax, t, v, fmt_y="{:.5f}", unit="EUR/kWh"))


# ── Load ──────────────────────────────────────────────────────────────────


class LoadTab(_Tab):
    TITLE = "Load"
    PROVIDERS = ["ProfileJSON"]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0
        # Only ProfileJSON provider is supported here (other providers were removed)
        self._profile_json_path = _field(
            f,
            row,
            "Profile JSON",
            str(Path("src/prediction/load/data/load_profiles.json")),
        )
        row += 1
        self._vacation_profile = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f,
            text="Use vacation profile",
            variable=self._vacation_profile,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **_PAD)

    def make_provider(self):
        # Only ProfileJSON provider supported in cleaned GUI
        return LoadProfileJSON(
            data_path=Path(self._profile_json_path.get()),
            use_vacation_profile=self._vacation_profile.get(),
        )

    def _do_plot(self, result: pl.Series | pl.DataFrame, ts: pl.Series) -> None:
        ax = _setup_plot_axes(self._fig)
        t = ts.to_list()
        s = result if isinstance(result, pl.Series) else result[result.columns[0]]
        v = s.to_list()
        ax.step(t, v, color="#E65100", linewidth=1.5, where="post")
        ax.fill_between(t, v, alpha=0.12, color="#E65100", step="post")
        _finalize_plot(ax, ylabel="Energy [Wh / step]", title="Load")
        self._hover_cids.append(_wire_hover(self._canvas, ax, t, v, fmt_y="{:.1f}", unit="Wh"))


# ── PV Forecast ───────────────────────────────────────────────────────────


class PVForecastTab(_Tab):
    TITLE = "PV Forecast"
    PROVIDERS = ["OpenMeteo", "Akkudoktor", "ForecastSolar", "Import"]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0

        # ── Plane config (common to all providers) ─────────────────────
        ttk.Label(f, text="Plane", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0)
        )
        row += 1
        self._peak = _field(f, row, "Peak [kW]", "0.41")
        row += 1
        self._tilt = _field(f, row, "Tilt [°]", "75.0")
        row += 1
        self._az = _field(f, row, "Azimuth [°]", "218.0")
        row += 1
        self._loss = _field(f, row, "Loss [%]", "4.0")
        row += 1
        self._horizon = _field(f, row, "Horizon [°] CSV", "")
        row += 1
        self._damp_morn = _field(f, row, "Damp. morning", "2.0")
        row += 1
        self._damp_eve = _field(f, row, "Damp. evening", "0.2")
        row += 1
        self._partial_shading = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Partial shading", variable=self._partial_shading).grid(
            row=row, column=0, columnspan=2, sticky="w", **_PAD
        )
        row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1

        # ── Provider-specific fields ────────────────────────────────────
        ttk.Label(f, text="Provider", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 2)
        )
        row += 1

        prov = self._prov_var.get()
        if prov in ("OpenMeteo", "Akkudoktor", "ForecastSolar"):
            self._lat = _field(f, row, "Latitude", "47.99545")
            row += 1
            self._lon = _field(f, row, "Longitude", "7.83355")
            row += 1
            if prov == "ForecastSolar":
                self._apikey = _field(f, row, "API key (opt.)", "")
            elif prov == "OpenMeteo":
                self._om_apikey = _field(f, row, "API key (opt.)", "")
                row += 1
                self._om_weather_model = _field(f, row, "Weather model", "")
        else:  # Import
            ttk.Label(f, text="Values (W, CSV):").grid(
                row=row, column=0, columnspan=2, sticky="w", **_PAD
            )
            row += 1
            self._ta = _textarea(f, row, height=4)
            row += 1
            self._srcdt = _field(f, row, "Source dt [h]", "1.0")

    def _plane(self) -> PVPlaneConfig:
        hz_text = self._horizon.get().strip()
        userhorizon = _csv(hz_text) if hz_text else None
        return PVPlaneConfig(
            peak_kw=float(self._peak.get()),
            tilt=float(self._tilt.get()),
            azimuth=float(self._az.get()),
            loss_pct=float(self._loss.get()),
            userhorizon=tuple(userhorizon) if userhorizon else None,
            damping_morning=float(self._damp_morn.get()),
            damping_evening=float(self._damp_eve.get()),
            partial_shading=self._partial_shading.get(),
        )

    def make_provider(self):
        p = self._prov_var.get()
        plane = self._plane()
        tz = self.app._tz.get()
        if p == "OpenMeteo":
            return PVForecastOpenMeteo(
                planes=[plane],
                latitude=float(self._lat.get()),
                longitude=float(self._lon.get()),
                timezone_str=tz,
                api_key=self._om_apikey.get().strip() or None,
                weather_model=self._om_weather_model.get().strip() or None,
            )
        if p == "Akkudoktor":
            return PVForecastAkkudoktor(
                planes=[plane],
                latitude=float(self._lat.get()),
                longitude=float(self._lon.get()),
                timezone_str=tz,
            )
        return PVForecastImport(
            power_w=_csv(self._ta.get("1.0", "end")),
            source_dt_hours=float(self._srcdt.get()),
        )

    def _do_plot(self, result: pl.Series | pl.DataFrame, ts: pl.Series) -> None:
        ax = _setup_plot_axes(self._fig)
        t = ts.to_list()
        s = result if isinstance(result, pl.Series) else result[result.columns[0]]
        v = s.to_list()
        dt_hours = ((t[1] - t[0]).total_seconds() / 3600) if len(t) > 1 else 1.0
        ax.plot(t, v, color="#F57F17", linewidth=1.5)
        ax.fill_between(t, v, alpha=0.20, color="#FDD835")
        _finalize_plot(ax, ylabel="Energy [Wh / step]", title="PV Forecast")
        self._hover_cids.append(_wire_pv_hover(self._canvas, ax, t, v, dt_hours))


# ── Weather ───────────────────────────────────────────────────────────────

_WCOLS = [
    "#1565C0",
    "#2E7D32",
    "#00838F",
    "#6A1B9A",
    "#E65100",
    "#AD1457",
    "#37474F",
    "#558B2F",
    "#4E342E",
]
_WYLABEL: dict[str, str] = {
    "temperature_c": "°C",
    "humidity_pct": "%",
    "cloud_cover_pct": "%",
    "wind_speed_kmh": "km/h",
    "precipitation_mm": "mm",
    "pressure_hpa": "hPa",
    "ghi_wm2": "W/m²",
    "dni_wm2": "W/m²",
    "dhi_wm2": "W/m²",
}
_WTITLE: dict[str, str] = {
    "temperature_c": "Temperature",
    "humidity_pct": "Humidity",
    "cloud_cover_pct": "Cloud cover",
    "wind_speed_kmh": "Wind speed",
    "precipitation_mm": "Precipitation",
    "pressure_hpa": "Pressure",
    "ghi_wm2": "GHI",
    "dni_wm2": "DNI",
    "dhi_wm2": "DHI",
}


class WeatherTab(_Tab):
    TITLE = "Weather"
    PROVIDERS = ["OpenMeteo", "BrightSky"]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0
        self._lat = _field(f, row, "Latitude", "52.52")
        row += 1
        self._lon = _field(f, row, "Longitude", "13.405")
        row += 1
        self._tz = _combofield(f, row, "Timezone", _TZ_CHOICES, "UTC")

    def make_provider(self):
        lat = float(self._lat.get())
        lon = float(self._lon.get())
        tz = self._tz.get()
        if self._prov_var.get() == "OpenMeteo":
            return WeatherOpenMeteo(latitude=lat, longitude=lon, timezone_str=tz)
        return WeatherBrightSky(latitude=lat, longitude=lon, timezone_str=tz)

    def _do_plot(self, result: pl.Series | pl.DataFrame, ts: pl.Series) -> None:
        # Accept either a Series (single-column) or a DataFrame
        if isinstance(result, pl.Series):
            # Convert to single-column DataFrame for uniform handling
            col_name = result.name if result.name is not None else "value"
            df = pl.DataFrame({col_name: result})
        else:
            df = result
        t = ts.to_list()
        cols = df.columns
        n = len(cols)
        if n == 0:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        ncols = 2
        nrows = (n + ncols - 1) // ncols
        for i, col in enumerate(cols):
            ax = self._fig.add_subplot(nrows, ncols, i + 1)
            v = df[col].to_list()
            color = _WCOLS[i % len(_WCOLS)]
            ax.plot(t, v, color=color, linewidth=1.2)
            ax.fill_between(t, v, alpha=0.08, color=color)
            title = _WTITLE.get(col, col.replace("_", " "))
            unit = _WYLABEL.get(col, "")
            ax.set_title(f"{title} [{unit}]" if unit else title, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            self._hover_cids.append(_wire_hover(self._canvas, ax, t, v, fmt_y="{:.2f}", unit=unit))
        self._fig.autofmt_xdate(rotation=20)


# ── Main App ──────────────────────────────────────────────────────────────


class OptimizationTab(_Tab):
    TITLE = "Optimization"
    PROVIDERS = []
    _last_ts: list | None = None
    _last_dt: float | None = None

    def _build_fields(self) -> None:
        f, row = self._cfg, 0

        # ── Optimizer selector ────────────────────────────────────────
        ttk.Label(f, text="Optimizer", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0)
        )
        row += 1
        self._optimizer_type = _combofield(
            f, row, "Optimizer", ["Linear (CVXPY)"], "Linear (CVXPY)"
        )
        self._optimizer_type.trace_add("write", lambda *_: self._rebuild())
        row += 1
        # Linear-specific settings
        self._linear_objective = _combofield(
            f,
            row,
            "Objective",
            ["Minimize Cost", "Maximize Self-consumption"],
            "Minimize Cost",
        )
        row += 1
        self._bat_end_value = _field(f, row, "Battery end value EUR/Wh", "0.0")
        row += 1

        # ── Battery configuration ─────────────────────────────────────
        ttk.Label(f, text="Battery", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 0)
        )
        row += 1
        self._bat_id = _field(f, row, "Battery ID", "battery1")
        row += 1
        self._bat_capacity = _field(f, row, "Battery capacity Wh", "1920")
        row += 1
        self._bat_ch_eff = _field(f, row, "Charging efficiency", "0.98")
        row += 1
        self._bat_dc_eff = _field(f, row, "Discharging efficiency", "0.98")
        row += 1
        self._initial_soc = _field(f, row, "Initial battery SoC (%)", "50.0")
        row += 1
        self._min_soc = _field(f, row, "Min SoC (%)", "20")
        row += 1
        self._max_soc = _field(f, row, "Max SoC (%)", "100")
        row += 1

        # ── Inverter configuration ────────────────────────────────────
        ttk.Label(f, text="Inverter", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 0)
        )
        row += 1
        self._inv_id = _field(f, row, "Device ID", "inverter1")
        row += 1
        self._pv_source = _field(f, row, "PV source key", "inverter1")
        row += 1
        self._inv_max_out = _field(f, row, "Max AC output W", "800")
        row += 1
        self._inv_max_charge = _field(f, row, "Max AC charge W", "1000")
        row += 1
        self._inv_dc2ac = _field(f, row, "DC→AC eff", "0.95")
        row += 1
        self._inv_ac2dc = _field(f, row, "AC→DC eff", "0.95")
        row += 1
        self._inv_mode_switch_cost = _field(f, row, "Mode switch cost EUR/switch", "0.005")
        row += 1
        self._zero_feed_in = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            f,
            text="Zero feed-in (prevent exporting)",
            variable=self._zero_feed_in,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **_PAD)
        row += 1
        # Feed-in default (EUR/Wh) when feedin provider is not configured
        self._feedin_default = _field(f, row, "Feed-in default EUR/Wh", "0.0")
        row += 1
        # Run button
        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Button(f, text="▶ Run Optimization", command=self.run_optimization).grid(
            row=row, column=0, columnspan=2, sticky="ew"
        )

    def make_provider(self):
        # Not used: this tab aggregates other providers
        return None

    def fetch(self) -> None:
        """Override fetch: run optimization flow instead of provider fetch."""
        self.run_optimization()

    def run_optimization(self) -> None:
        try:
            start, hours, dt = self.app.get_time_params()
            logger.info("Optimization start: start=%s hours=%s dt=%s", start, hours, dt)
        except Exception as exc:
            self.app.root.bell()
            messagebox.showerror("Time parsing", str(exc), parent=self.frame)
            return

        # Gather providers from other tabs
        elec_tab = next((t for t in self.app.tabs if isinstance(t, ElecPriceTab)), None)
        feed_tab = next((t for t in self.app.tabs if isinstance(t, FeedInTariffTab)), None)
        load_tab = next((t for t in self.app.tabs if isinstance(t, LoadTab)), None)
        pv_tab = next((t for t in self.app.tabs if isinstance(t, PVForecastTab)), None)
        weather_tab = next((t for t in self.app.tabs if isinstance(t, WeatherTab)), None)

        setup = PredictionSetup(
            electricprice=elec_tab._get_provider() if elec_tab else None,
            feedintariff=feed_tab._get_provider() if feed_tab else None,
            load=load_tab._get_provider() if load_tab else None,
            pv={self._inv_id.get(): pv_tab._get_provider()} if pv_tab else {},
            weather=weather_tab._get_provider() if weather_tab else None,
        )

        pred = Prediction(setup)

        self.app.root.after(0, lambda: self._status.set("Fetching prediction…"))

        # Fetch prediction asynchronously
        def on_pred_done(pdata):
            try:
                # cache timestamps and dt for plotting
                try:
                    self._last_ts = pdata.timestamps.to_list()
                except Exception:
                    self._last_ts = None
                self._last_dt = getattr(pdata, "dt_hours", None)

                # Update other tabs' plots with the freshly fetched prediction
                def _update_tabs():
                    ts = pdata.timestamps
                    try:
                        if elec_tab is not None:
                            try:
                                elec_tab._done(pdata.electricprice, ts)
                            except Exception:
                                logger.exception("Failed to update electric price tab")
                        if feed_tab is not None:
                            try:
                                feed_tab._done(pdata.feedintariff, ts)
                            except Exception:
                                logger.exception("Failed to update feed-in tariff tab")
                        if load_tab is not None:
                            try:
                                load_tab._done(pdata["load_wh"], ts)
                            except Exception:
                                logger.exception("Failed to update load tab")
                        if pv_tab is not None:
                            try:
                                # sum all pv_* columns into one series for plotting
                                pv_series_list = list(pdata.pv_by_inverter.values())
                                if pv_series_list:
                                    import operator
                                    from functools import reduce

                                    pv_sum = reduce(operator.add, pv_series_list)
                                    pv_tab._done(pv_sum, ts)
                            except Exception:
                                logger.exception("Failed to sum PV series and update PV tab")
                        if weather_tab is not None:
                            try:
                                # rebuild weather dataframe without the 'weather_' prefix
                                weather_cols = [
                                    c for c in pdata.df.columns if c.startswith("weather_")
                                ]
                                if weather_cols:
                                    data = {c[len("weather_") :]: pdata.df[c] for c in weather_cols}
                                    wf = pl.DataFrame(data)
                                    weather_tab._done(wf, ts)
                            except Exception:
                                logger.exception(
                                    "Failed to rebuild weather dataframe and update weather tab"
                                )
                    except Exception:
                        # don't let tab-updates break the optimization flow
                        return

                # Schedule GUI updates on the main thread
                try:
                    self.app.root.after(0, _update_tabs)
                except Exception:
                    logger.exception("Failed to schedule GUI update (_update_tabs)")

                feedin_default = float(self._feedin_default.get())

                # ── Build battery + inverter from GUI config ──────────────────
                bat_params = BatteryParameters(
                    device_id=self._bat_id.get(),
                    capacity_wh=int(float(self._bat_capacity.get())),
                    max_charge_power_w=int(float(self._inv_max_charge.get())),
                    max_discharge_power_w=int(float(self._inv_max_out.get())),
                    charging_efficiency=float(self._bat_ch_eff.get()),
                    discharging_efficiency=float(self._bat_dc_eff.get()),
                    initial_soc_percentage=int(float(self._initial_soc.get())),
                    min_soc_percentage=int(float(self._min_soc.get())),
                    max_soc_percentage=int(float(self._max_soc.get())),
                )

                def _make_devices(params):
                    bat = Battery(params, prediction_hours=int(hours))
                    inv_params = InverterParameters(
                        device_id=self._inv_id.get(),
                        battery_id=bat.parameters.device_id,
                        pv_source=self._pv_source.get(),
                        max_ac_output_power_w=float(self._inv_max_out.get()),
                        max_ac_charge_power_w=float(self._inv_max_charge.get()),
                        dc_to_ac_efficiency=float(self._inv_dc2ac.get()),
                        ac_to_dc_efficiency=float(self._inv_ac2dc.get()),
                        zero_feed_in=bool(self._zero_feed_in.get()),
                        mode_switch_cost=float(self._inv_mode_switch_cost.get()),
                    )
                    inv = InverterBase(inv_params, battery=bat)
                    return bat, inv

                optimizer_choice = getattr(self, "_optimizer_type", None)
                use_linear = (
                    optimizer_choice is not None and optimizer_choice.get() == "Linear (CVXPY)"
                )

                async def _run_sim():
                    # Linear (CVXPY + HiGHS) path
                    _, inv_obj = _make_devices(bat_params)
                    obj_str = getattr(self, "_linear_objective", None)
                    if obj_str is not None and "Self" in obj_str.get():
                        objective = OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
                    else:
                        objective = OptimizationObjective.MINIMIZE_COST
                    bat_end_val = 0.0
                    bat_end_field = getattr(self, "_bat_end_value", None)
                    if bat_end_field is not None:
                        try:
                            bat_end_val = float(bat_end_field.get())
                        except ValueError:
                            bat_end_val = 0.0
                    optimizer = LinearOptimizer(
                        inverters=[inv_obj],
                        prediction=pdata,
                        battery_end_value_eur_wh=bat_end_val,
                    )
                    logger.info("Starting LinearOptimizer with objective={}", objective.value)
                    sol: LinearSolution = await asyncio.to_thread(
                        lambda: optimizer.solve(objective=objective)
                    )
                    return sol

                def _sim_done(res_tuple):
                    # res_tuple may be a LinearSolution or legacy tuple
                    inv_modes_arrs = None
                    inv_ac_rates_arrs = None
                    solve_meta: str | None = None

                    if isinstance(res_tuple, LinearSolution):
                        sol = res_tuple
                        res = sol.result
                        solve_meta = (
                            f"Linear ({sol.objective.value}) · "
                            f"status={sol.solver_status} · "
                            f"{sol.solve_time_s:.2f}s"
                        )
                        if sol.inverter_plans:
                            plan = sol.inverter_plans[0]
                            try:
                                inv_modes_arrs = [
                                    array("i", [int(x) for x in plan.get("modes", [])])
                                ]
                                inv_ac_rates_arrs = [
                                    array("f", [float(x) for x in plan.get("rates", [])])
                                ]
                            except Exception:
                                inv_modes_arrs = None
                                inv_ac_rates_arrs = None
                    else:
                        # legacy tuple (res, modes, rates)
                        if isinstance(res_tuple, tuple):
                            res, inv_modes_arrs, inv_ac_rates_arrs = res_tuple
                        else:
                            res = res_tuple

                    if res is None:
                        messagebox.showinfo(
                            "Optimization", "Simulation returned no result.", parent=self.frame
                        )
                        self._status.set("Simulation finished: no result")
                        return
                    logger.info("Simulation finished, preparing plots and JSON output")
                    out = res.to_dict()
                    # store timestamps for plotting
                    ts = getattr(self, "_last_ts", None)
                    dt_hours = getattr(self, "_last_dt", None) or float(dt)

                    # Draw results into the right-side figure of this tab
                    try:
                        self._fig.clear()
                        n = len(res.costs_per_dt)

                        # Convert timestamps to matplotlib-compatible format if available
                        ts = getattr(self, "_last_ts", None)
                        if ts and len(ts) >= n:
                            x = ts[:n]
                        else:
                            x = list(range(n))

                        # Top: energy flows
                        ax = self._fig.add_subplot(311)
                        ax.plot(x, list(res.grid_import_wh_per_dt), label="Grid import (Wh)")
                        ax.plot(
                            x, list(res.self_consumption_wh_per_dt), label="Self-consumption (Wh)"
                        )
                        ax.plot(x, list(res.feedin_wh_per_dt), label="Feed-in (Wh)")
                        ax.plot(x, list(res.losses_wh_per_dt), label="Losses (Wh)")
                        ax.legend(loc="upper right", fontsize=8)
                        ax.set_ylabel("Wh")
                        ax.grid(alpha=0.3)
                        if isinstance(x[0], (int, float)):
                            # numeric x: don't format dates
                            pass
                        else:
                            # Add hover to energy flows
                            try:
                                grid_import_data = list(res.grid_import_wh_per_dt)
                                _wire_hover(
                                    self._canvas,
                                    ax,
                                    x,
                                    grid_import_data,
                                    fmt_y="{:.1f}",
                                    unit=" Wh",
                                )
                            except Exception:
                                logger.exception("Failed to add hover for energy flows")

                        # Middle: PV generation and Load (left axis), electric price (right axis)
                        ax2 = self._fig.add_subplot(312)
                        # compute total load (grid import + self-consumption)
                        load_wh = [
                            g + s
                            for g, s in zip(
                                list(res.grid_import_wh_per_dt),
                                list(res.self_consumption_wh_per_dt),
                            )
                        ]
                        (h_load,) = ax2.plot(
                            x, load_wh, color="#1565C0", linewidth=1.4, label="Load (Wh)"
                        )

                        # Plot PV sources if available
                        pv_handles = []
                        if res.solar_generation_wh_per_dt:
                            for i, (k, arr) in enumerate(
                                (res.solar_generation_wh_per_dt or {}).items()
                            ):
                                # choose color cycle
                                col = f"C{i + 2}"
                                (h,) = ax2.plot(
                                    x, list(arr), color=col, linewidth=1.2, label=f"PV {k} (Wh)"
                                )
                                pv_handles.append(h)

                        ax2.set_ylabel("Wh")
                        ax2.grid(alpha=0.2)

                        # Right axis: electricity price (€/Wh)
                        ax3 = ax2.twinx()
                        price_vals = [float(v) for v in res.electricity_price_per_dt]
                        (h_price,) = ax3.plot(
                            x,
                            price_vals,
                            color="orange",
                            linewidth=1.2,
                            linestyle="-",
                            label="Price (€/Wh)",
                        )
                        ax3.set_ylabel("€/Wh")

                        # Combined legend
                        lines, labels = ax2.get_legend_handles_labels()
                        lines2, labels2 = ax3.get_legend_handles_labels()
                        ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

                        # Add hover annotation for ax2 and ax3
                        if isinstance(x[0], (int, float)):
                            # numeric x: datetime hover not applicable
                            pass
                        else:
                            # Add datetime hover to energy plot
                            try:
                                _wire_hover(
                                    self._canvas, ax2, x, load_wh, fmt_y="{:.1f}", unit=" Wh"
                                )
                            except Exception:
                                logger.exception("Failed to add datetime hover to energy plot")

                        # Bottom: battery SoC (%) (left axis) and inverter modes (right axis)
                        ax4 = self._fig.add_subplot(313)
                        ax4r = ax4.twinx()
                        plotted = False
                        handles = []
                        labels = []
                        # Plot SoC on left axis
                        if res.battery_soc_percentage_per_dt:
                            for k, arr in (res.battery_soc_percentage_per_dt or {}).items():
                                (h,) = ax4.plot(x, list(arr), label=f"SoC {k} (%)")
                                handles.append(h)
                                labels.append(f"SoC {k} (%)")
                                plotted = True

                        # Plot inverter modes on right axis (signed rate: discharge=+, charge=-, idle=0)
                        if inv_modes_arrs and len(inv_modes_arrs) > 0:
                            modes_arr = inv_modes_arrs[0]
                            rates_arr = (
                                inv_ac_rates_arrs[0]
                                if inv_ac_rates_arrs and len(inv_ac_rates_arrs) > 0
                                else None
                            )
                            mode_vals = []
                            for i, m in enumerate(modes_arr):
                                try:
                                    mode_int = int(m)
                                except Exception:
                                    mode_int = int(InverterMode.IDLE)
                                rate = (
                                    float(rates_arr[i])
                                    if rates_arr is not None and i < len(rates_arr)
                                    else 1.0
                                )
                                if mode_int == int(InverterMode.DISCHARGE_ZERO_FEED_IN):
                                    # Zero-feed discharge is energy-target based; visualize as full discharge state.
                                    mode_vals.append(+1.0)
                                elif mode_int == int(InverterMode.DISCHARGE):
                                    mode_vals.append(+rate)
                                elif mode_int in (
                                    int(InverterMode.AC_CHARGE),
                                    int(InverterMode.AC_CHARGE_ZERO_FEED_IN),
                                ):
                                    mode_vals.append(-rate)
                                else:
                                    mode_vals.append(0.0)
                            h2 = ax4r.step(
                                x,
                                mode_vals,
                                where="post",
                                color="#555555",
                                label="Inv mode (signed rate)",
                            )
                            # ax4r.step returns a list of Line2D objects; pick first for legend handle
                            if isinstance(h2, (list, tuple)) and h2:
                                handles.append(h2[0])
                            else:
                                handles.append(h2)
                            labels.append("Inv mode (signed rate)")
                            plotted = True

                        if plotted:
                            ax4.set_ylabel("SoC %")
                            ax4r.set_ylabel("Inv mode (signed rate)")
                            ax4.legend(
                                handles=handles, labels=labels, loc="upper right", fontsize=8
                            )
                        ax4.grid(alpha=0.2)

                        # Add hover to battery SoC plot if timestamps available
                        if isinstance(x[0], (int, float)):
                            # numeric x: don't add hover
                            pass
                        else:
                            # Add hover for SoC
                            soc_data = []
                            if res.battery_soc_percentage_per_dt:
                                for k, arr in (res.battery_soc_percentage_per_dt or {}).items():
                                    soc_data = list(arr)
                                    break
                            if soc_data:
                                try:
                                    _wire_hover(
                                        self._canvas, ax4, x, soc_data, fmt_y="{:.1f}", unit="%"
                                    )
                                except Exception:
                                    logger.exception("Failed to add hover for SoC plot")

                        # Date formatting for all subplots with timestamps
                        if not isinstance(x[0], (int, float)):
                            self._fig.autofmt_xdate(rotation=25)

                        self._canvas.draw()
                    except Exception:
                        # Fall back to JSON popup if plotting fails
                        logger.exception("Plotting failed, falling back to JSON popup")

                    # Show JSON in a popup scrolled window as well
                    w = tk.Toplevel(self.frame)
                    w.title("Simulation Result")
                    txt = scrolledtext.ScrolledText(w, width=120, height=30)
                    txt.pack(fill="both", expand=True)
                    # include config metadata
                    meta: dict = {
                        "optimizer": getattr(self, "_optimizer_type", None)
                        and self._optimizer_type.get(),
                        "initial_soc_pct": float(self._initial_soc.get()),
                    }
                    if solve_meta:
                        meta["solve_info"] = solve_meta
                    txt.insert("1.0", json.dumps({"meta": meta, "result": out}, indent=2))
                    txt.configure(state="disabled")
                    suffix = f" · {solve_meta}" if solve_meta else ""
                    self._status.set(f"Done · Net: {res.net_balance:.2f} €{suffix}")

                _run_async(
                    _run_sim(),
                    on_done=lambda r: _sim_done(r),
                    on_error=lambda e, tb: self._on_error(e, tb),
                )
            except Exception as exc:
                self._on_error(exc, None)

        _run_async(
            pred.fetch(start=start, hours=hours, dt_hours=dt),
            on_done=lambda r: on_pred_done(r),
            on_error=lambda e, tb: self._on_error(e, tb),
        )

    def _on_error(self, exc: Exception, tb: str | None) -> None:
        self._status.set(f"Error: {exc}")
        messagebox.showerror("Error", f"{exc}\n\n{(tb or '')[:1500]}", parent=self.frame)


class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Forecast Preview  —  dev tool")
        self.root.minsize(960, 580)
        self._build_topbar()
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.tabs: list[_Tab] = [
            ElecPriceTab(nb, self),
            FeedInTariffTab(nb, self),
            LoadTab(nb, self),
            PVForecastTab(nb, self),
            WeatherTab(nb, self),
            OptimizationTab(nb, self),
        ]

    def _build_topbar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=6)

        ttk.Label(bar, text="Start:").grid(row=0, column=0)
        self._start = tk.StringVar(value=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        ttk.Entry(bar, textvariable=self._start, width=17).grid(row=0, column=1, padx=2)

        ttk.Label(bar, text="Hours:").grid(row=0, column=2, padx=(10, 0))
        self._hours = tk.StringVar(value="48")
        ttk.Entry(bar, textvariable=self._hours, width=6).grid(row=0, column=3, padx=2)

        ttk.Label(bar, text="Δt [h]:").grid(row=0, column=4, padx=(10, 0))
        self._dt = tk.StringVar(value="0.25")
        ttk.Combobox(
            bar, textvariable=self._dt, values=_DT_CHOICES, state="readonly", width=6
        ).grid(row=0, column=5, padx=2)

        ttk.Label(bar, text="Timezone:").grid(row=0, column=6, padx=(10, 0))
        self._tz = tk.StringVar(value="UTC")
        ttk.Combobox(
            bar, textvariable=self._tz, values=_TZ_CHOICES, state="readonly", width=18
        ).grid(row=0, column=7, padx=2)

        ttk.Separator(bar, orient="vertical").grid(row=0, column=8, sticky="ns", padx=12)
        ttk.Button(bar, text="▶▶ Fetch All", command=self._fetch_all).grid(row=0, column=9)

    def get_time_params(self) -> tuple[datetime, float, float]:
        raw = self._start.get().strip()
        dt = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            raise ValueError(f"Cannot parse start datetime: {raw!r}")
        tz_name = self._tz.get()
        tz = timezone.utc if tz_name == "UTC" else ZoneInfo(tz_name)
        return dt.replace(tzinfo=tz), float(self._hours.get()), float(self._dt.get())

    def _fetch_all(self) -> None:
        for tab in self.tabs:
            tab.fetch()


# ── entry point ───────────────────────────────────────────────────────────


def run() -> None:
    """Launch the interactive forecast preview GUI."""
    # Keep third-party loggers quiet while retaining debug output from this package.
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("src").setLevel(logging.DEBUG)

    app = App()
    app.root.mainloop()


if __name__ == "__main__":
    run()
