"""Interactive dev-tool GUI for exploring and configuring forecast providers.

Usage::

    uv run python -m utils.gui
    # or
    python utils/gui.py
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import tkinter as tk
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable, Coroutine, List, Mapping, Optional, Tuple, cast
from zoneinfo import ZoneInfo

import matplotlib
import structlog
import yaml
from matplotlib.axes import Axes

matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from structlog import get_logger

from GridPythia.config import AppConfig
from GridPythia.config.optimization import BatteryParameters, InverterParameters
from GridPythia.optimization.solver import LinearOptimizer, LinearSolution, OptimizationObjective
from GridPythia.prediction.base import make_timestamps
from GridPythia.prediction.electricprice.energycharts import (
    ElecPriceEnergyCharts,
    EnergyChartsConfig,
)
from GridPythia.prediction.electricprice.fixed import ElecPriceFixed
from GridPythia.prediction.electricprice.provider import ElecPriceProvider
from GridPythia.prediction.feedintariff.fixed import FeedInTariffFixed
from GridPythia.prediction.feedintariff.provider import FeedInTariffProvider
from GridPythia.prediction.load.config import LoadProfileConfig
from GridPythia.prediction.load.provider import LoadProvider, load_provider_from_config
from GridPythia.prediction.prediction import Prediction, PredictionData, PredictionSetup
from GridPythia.prediction.pvforecast.akkudoktor import PVForecastAkkudoktor
from GridPythia.prediction.pvforecast.openmeteo import PVForecastOpenMeteo
from GridPythia.prediction.pvforecast.provider import PVForecastProvider, PVPlaneConfig
from GridPythia.prediction.weather.brightsky import WeatherBrightSky
from GridPythia.prediction.weather.openmeteo import WeatherOpenMeteo
from GridPythia.prediction.weather.provider import WeatherProvider
from GridPythia.simulation.devices import InverterMode
from GridPythia.simulation.devices.battery import Battery
from GridPythia.simulation.devices.inverterbase import InverterBase

logger = get_logger(__name__)

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

# ── GUI Defaults ──────────────────────────────────────────────────────────
_GUI_HOURS_DEFAULT = "48"
_GUI_DT_DEFAULT = "0.25"
_GUI_TIMEZONE_DEFAULT = "UTC"

# ── utilities ─────────────────────────────────────────────────────────────


def _run_async(
    coro: Coroutine[Any, Any, Any],
    on_done: Callable[[Any], None],
    on_error: Callable[[Exception, str], None],
) -> None:
    """Run *coro* in a daemon thread; deliver results via callbacks."""

    def _worker() -> None:
        try:
            on_done(asyncio.run(coro))
        except Exception as exc:
            on_error(exc, traceback.format_exc())

    threading.Thread(target=_worker, daemon=True).start()


async def _with_context(
    coro: Coroutine[Any, Any, Any],
    **bindings: Any,
) -> Any:
    """Wrap *coro* so that structlog contextvars are bound for the whole call chain.

    Every log call inside *coro* — including nested providers and the solver —
    will automatically inherit the bound keys (e.g. ``run_id``, ``tab``,
    ``operation``) without any changes to those modules.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**bindings)
    try:
        return await coro
    finally:
        structlog.contextvars.clear_contextvars()


def _csv(text: str) -> list[float]:
    return [float(t) for t in re.split(r"[\s,;]+", text.strip()) if t]


def _csv_text(values: Any, default: str = "") -> str:
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        return ", ".join(str(v) for v in values)
    return str(values)


def _path_field(
    parent: tk.Misc,
    row: int,
    label: str,
    default: str = "",
    filetypes: list[tuple[str, str]] | None = None,
) -> tk.StringVar:
    """Grid label + entry + browse button at *row*; return the StringVar."""
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", **_PAD)
    var = tk.StringVar(value=default)
    ent = ttk.Entry(parent, textvariable=var, width=16)
    ent.grid(row=row, column=1, sticky="ew", **_PAD)
    parent.columnconfigure(1, weight=1)
    ft = filetypes or [("All files", "*.*")]

    def _browse() -> None:
        path = filedialog.askopenfilename(filetypes=ft)
        if path:
            var.set(path)

    ttk.Button(parent, text="\u2026", width=2, command=_browse).grid(
        row=row, column=2, sticky="w", padx=(0, 4), pady=3
    )
    return var


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


def _place_hover_annotation(ax: Axes, annot: Any, x: float, y: float) -> None:
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
    ax: Axes,
    xs_dt: List[datetime],
    ys: List[float],
    fmt_y: str = "{:.4f}",
    unit: str = "",
) -> int:
    """Attach a value tooltip + crosshair to *ax*. Returns the mpl connection id."""
    if not xs_dt:
        return -1
    xs_num = np.array(mdates.date2num(xs_dt), dtype=float)
    ys_arr = np.array(ys, dtype=float)

    annot: Any = ax.annotate(
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

    def _on_move(event: Any) -> Any:
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


def _setup_plot_axes(fig: Figure, num_rows: int = 111) -> Axes:
    """Common plot setup: clear fig, add subplot. Returns the Axes object."""
    fig.clear()
    ax = fig.add_subplot(num_rows)
    return ax


def _finalize_plot(ax: Axes, ylabel: str = "", title: str = "", gridOn: bool = True) -> None:
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
    ax: Axes,
    xs_dt: List[datetime],
    ys: List[float],
    dt_hours: float,
) -> int:
    """PV-specific hover: step energy + daily total + remaining for that day."""
    if not xs_dt:
        return -1

    # Values are already energy per step (Wh), so daily aggregation is a plain sum.
    day_total_wh: dict = {}
    for ti, wi in zip(xs_dt, ys, strict=False):
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

    annot: Any = ax.annotate(
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

    def _on_move(event: Any) -> Any:
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
    _LEFT_WIDTH: int = 260

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
        left = ttk.Frame(self.frame, width=self._LEFT_WIDTH)
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)  # config section expands

        # Row 0: provider combobox
        if self.PROVIDERS:
            pf = ttk.LabelFrame(left, text="Provider", padding=4)
            pf.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
            pf.columnconfigure(0, weight=1)
            self._prov_var = tk.StringVar(value=self._initial_provider())
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

    def _initial_provider(self) -> str:
        return self.PROVIDERS[0] if self.PROVIDERS else ""

    def load_defaults_from_config(self) -> None:
        if self.PROVIDERS and hasattr(self, "_prov_var"):
            prov = self._initial_provider()
            if prov in self.PROVIDERS:
                self._prov_var.set(prov)
        self._rebuild()

    # ── provider construction ─────────────────────────────────────────

    def make_provider(self) -> object:
        raise NotImplementedError

    def _provider_sig(self) -> str | None:
        """Return a stable key for the current config, or None to always recreate."""
        return None

    def _get_provider(self) -> object:
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
        prov = cast(Any, prov)
        start, hours, dt = self.app.get_time_params()
        ts = make_timestamps(start, hours, dt)
        self._status.set("Fetching…")
        _run_async(
            prov.fetch(ts),
            on_done=lambda r: self.app.root.after(0, lambda: self._done(r, ts)),
            on_error=lambda e, tb: self.app.root.after(0, lambda: self._fail(e, tb)),
        )

    def _done(self, result: np.ndarray | dict, ts: list) -> None:
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
        logger.error(
            "fetch_failed",
            tab=self.TITLE,
            exc_type=type(exc).__name__,
            exc=str(exc),
            traceback=tb,
        )
        messagebox.showerror(
            "Fetch error",
            "An error occurred. See console for details.",
            parent=self.frame,
        )

    def _do_plot(self, result: np.ndarray | dict, ts: list) -> None:
        ax = _setup_plot_axes(self._fig)
        # Default plotting: if we got an array, plot it; if dict, plot first channel
        if isinstance(result, np.ndarray):
            ax.plot(ts, result, linewidth=1.4)
        else:
            keys = list(result.keys())
            if keys:
                ax.plot(ts, result[keys[0]], linewidth=1.4)
        _finalize_plot(ax, title=self.TITLE)


# ── Electric Price ────────────────────────────────────────────────────────


class ElecPriceTab(_Tab):
    """Tab showing electric price providers and price plots."""

    TITLE = "Electric Price"
    PROVIDERS = ["EnergyCharts", "Fixed"]

    def _initial_provider(self) -> str:
        v = self.app.cfg_text("prediction", "electricprice", "provider", default="EnergyCharts")
        return v if v in self.PROVIDERS else self.PROVIDERS[0]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0
        p = self._prov_var.get()

        if p == "Fixed":
            self._price = _field(
                f,
                row,
                "Price EUR/kWh",
                self.app.cfg_text(
                    "prediction", "electricprice", "fixed", "price_kwh", default="0.30"
                ),
            )
            row += 1
        elif p == "EnergyCharts":
            self._zone = _combofield(
                f,
                row,
                "Bidding zone",
                _BZ_CHOICES,
                self.app.cfg_text(
                    "prediction",
                    "electricprice",
                    "energycharts",
                    "bidding_zone",
                    default="DE-LU",
                ),
            )
            row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        self._charges = _field(
            f,
            row,
            "Charges EUR/kWh",
            self.app.cfg_text("prediction", "electricprice", "charges_kwh", default="0.1528"),
        )
        row += 1
        self._vat = _field(
            f,
            row,
            "VAT rate",
            self.app.cfg_text("prediction", "electricprice", "vat_rate", default="0.19"),
        )

    def _provider_sig(self) -> str | None:
        p = self._prov_var.get()
        if p == "EnergyCharts":
            return f"EnergyCharts:{self._zone.get()}:{self._charges.get()}:{self._vat.get()}"
        return None  # stateless providers: always recreate is fine

    def make_provider(self) -> object:
        """Create and return the configured electric price provider object."""
        ch, vat = float(self._charges.get()), float(self._vat.get())
        p = self._prov_var.get()
        if p == "Fixed":
            return ElecPriceFixed(price_kwh=float(self._price.get()), charges_kwh=ch, vat_rate=vat)
        cfg = EnergyChartsConfig(bidding_zone=self._zone.get(), charges_kwh=ch, vat_rate=vat)
        return ElecPriceEnergyCharts(cfg)

    def _do_plot(self, result: np.ndarray | dict, ts: list) -> None:
        ax = _setup_plot_axes(self._fig)
        t = ts
        s = result if isinstance(result, np.ndarray) else next(iter(result.values()))
        v = [x * 1000 for x in s]  # EUR/Wh → EUR/kWh
        ax.step(t, v, color="#1565C0", linewidth=1.5, where="post", label="Electric Price")
        ax.fill_between(t, v, alpha=0.12, color="#1565C0", step="post")
        _finalize_plot(ax, ylabel="EUR / kWh", title="Electricity Price")
        self._hover_cids.append(_wire_hover(self._canvas, ax, t, v, fmt_y="{:.5f}", unit="EUR/kWh"))


# ── Feed-in Tariff ────────────────────────────────────────────────────────


class FeedInTariffTab(_Tab):
    """Tab for feed-in tariff providers and plotting."""

    TITLE = "Feed-in Tariff"
    PROVIDERS = ["Fixed"]

    def _initial_provider(self) -> str:
        v = self.app.cfg_text("prediction", "feedintariff", "provider", default="Fixed")
        return v if v in self.PROVIDERS else self.PROVIDERS[0]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0
        self._tariff = _field(
            f,
            row,
            "Tariff EUR/kWh",
            self.app.cfg_text("prediction", "feedintariff", "tariff_kwh", default="0.0"),
        )

    def make_provider(self) -> object:
        """Create and return the configured feed-in tariff provider."""
        return FeedInTariffFixed(tariff_kwh=float(self._tariff.get()))

    def _do_plot(self, result: np.ndarray | dict, ts: list) -> None:
        ax = _setup_plot_axes(self._fig)
        t = ts
        s = result if isinstance(result, np.ndarray) else next(iter(result.values()))
        v = [x * 1000 for x in s]  # Convert to kWh
        ax.step(t, v, color="#2E7D32", linewidth=1.5, where="post", label="Feed-in Tariff")
        ax.fill_between(t, v, alpha=0.12, color="#2E7D32", step="post")
        _finalize_plot(ax, ylabel="EUR / kWh", title="Feed-in Tariff")
        self._hover_cids.append(_wire_hover(self._canvas, ax, t, v, fmt_y="{:.5f}", unit="EUR/kWh"))


# ── Load ──────────────────────────────────────────────────────────────────


class LoadTab(_Tab):
    """Tab for load providers and plotting."""

    TITLE = "Load"
    PROVIDERS = ["ProfileCSV"]

    def _initial_provider(self) -> str:
        v = self.app.cfg_text("prediction", "load", "provider", default="ProfileCSV")
        return v if v in self.PROVIDERS else self.PROVIDERS[0]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0
        # Global path field (common to all providers)
        self._profile_path = _path_field(
            f,
            row,
            "Profile path",
            self.app.cfg_text("prediction", "load", "path", default=""),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        row += 1

        # Holiday settings
        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Label(f, text="Holidays", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4
        )
        row += 1
        self._country = _field(
            f,
            row,
            "Country (ISO)",
            self.app.cfg_text("prediction", "load", "country", default="DE"),
        )
        row += 1
        self._subdivision = _field(
            f,
            row,
            "Subdivision",
            self.app.cfg_text("prediction", "load", "subdivision", default="BW"),
        )

    def make_provider(self) -> object:
        """Create and return the configured load provider."""
        country = self._country.get().strip() or None
        subdivision = self._subdivision.get().strip() or None
        cfg = LoadProfileConfig(
            path=Path(self._profile_path.get()),
            country=country,
            subdivision=subdivision,
        )
        return load_provider_from_config(cfg)

    def fetch(self) -> None:
        """Override to pass ``use_vacation_profile`` at runtime."""
        try:
            prov = self._get_provider()
        except Exception as exc:
            self._status.set(f"Config error: {exc}")
            return
        prov = cast(Any, prov)
        start, hours, dt = self.app.get_time_params()
        ts = make_timestamps(start, hours, dt)
        vac_var = getattr(self, "_vacation_profile", None)
        use_vac = vac_var.get() if vac_var is not None else False
        self._status.set("Fetching…")
        _run_async(
            prov.fetch(ts, use_vacation_profile=use_vac),
            on_done=lambda r: self.app.root.after(0, lambda: self._done(r, ts)),
            on_error=lambda e, tb: self.app.root.after(0, lambda: self._fail(e, tb)),
        )

    def _do_plot(self, result: np.ndarray | dict, ts: list) -> None:
        ax = _setup_plot_axes(self._fig)
        t = ts
        s = result if isinstance(result, np.ndarray) else next(iter(result.values()))
        v = list(s)
        ax.step(t, v, color="#E65100", linewidth=1.5, where="post")
        ax.fill_between(t, v, alpha=0.12, color="#E65100", step="post")
        _finalize_plot(ax, ylabel="Energy [Wh / step]", title="Load")
        self._hover_cids.append(_wire_hover(self._canvas, ax, t, v, fmt_y="{:.1f}", unit="Wh"))


# ── PV Forecast ───────────────────────────────────────────────────────────


class PVForecastTab(_Tab):
    """Tab for PV forecast providers and plane configuration."""

    TITLE = "PV Forecast"
    PROVIDERS = ["OpenMeteo", "Akkudoktor"]

    def _initial_provider(self) -> str:
        v = self.app.cfg_text("prediction", "pvforecast", "provider", default="OpenMeteo")
        return v if v in self.PROVIDERS else self.PROVIDERS[0]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0

        # ── Plane config (common to all providers) ─────────────────────
        ttk.Label(f, text="Plane", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0)
        )
        row += 1
        self._peak = _field(
            f,
            row,
            "Peak [kW]",
            self.app.cfg_text("prediction", "pvforecast", "plane", "peak_kw", default="0.41"),
        )
        row += 1
        self._tilt = _field(
            f,
            row,
            "Tilt [°]",
            self.app.cfg_text("prediction", "pvforecast", "plane", "tilt", default="75.0"),
        )
        row += 1
        self._az = _field(
            f,
            row,
            "Azimuth [°]",
            self.app.cfg_text("prediction", "pvforecast", "plane", "azimuth", default="218.0"),
        )
        row += 1
        self._loss = _field(
            f,
            row,
            "Loss [%]",
            self.app.cfg_text("prediction", "pvforecast", "plane", "loss_pct", default="4.0"),
        )
        row += 1
        self._horizon = _field(
            f,
            row,
            "Horizon [°] CSV",
            self.app.cfg_csv("prediction", "pvforecast", "plane", "userhorizon", default=""),
        )
        row += 1
        self._damp_morn = _field(
            f,
            row,
            "Damp. morning",
            self.app.cfg_text(
                "prediction", "pvforecast", "openmeteo", "damping_morning", default="2.0"
            ),
        )
        row += 1
        self._damp_eve = _field(
            f,
            row,
            "Damp. evening",
            self.app.cfg_text(
                "prediction", "pvforecast", "openmeteo", "damping_evening", default="0.2"
            ),
        )
        row += 1
        self._partial_shading = tk.BooleanVar(
            value=self.app.cfg_bool(
                "prediction", "pvforecast", "openmeteo", "partial_shading", default=False
            )
        )
        ttk.Checkbutton(f, text="Partial shading", variable=self._partial_shading).grid(
            row=row, column=0, columnspan=2, sticky="w", **_PAD
        )
        row += 1

        # Inverter ID for this PV plane (editable in GUI; defaults from YAML or optimizer)
        opt_inv = (
            self.app.app_config.optimization.inverters[0]
            if self.app.app_config.optimization.inverters
            else None
        )
        default_plane_inv = self.app.cfg_text(
            "prediction",
            "pvforecast",
            "plane",
            "inverter_id",
            default=(opt_inv.device_id if opt_inv is not None else "inverter1"),
        )
        self._plane_inverter_id = _field(f, row, "Inverter ID", default_plane_inv)
        row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1

        # ── Provider-specific fields ────────────────────────────────────
        ttk.Label(f, text="Provider", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 2)
        )
        row += 1

        prov = self._prov_var.get()
        if prov in ("OpenMeteo", "Akkudoktor"):
            # Global PV location
            self._lat = _field(
                f,
                row,
                "Latitude",
                self.app.cfg_text("prediction", "latitude", default="47.99545"),
            )
            row += 1
            self._lon = _field(
                f,
                row,
                "Longitude",
                self.app.cfg_text("prediction", "longitude", default="7.83355"),
            )
            row += 1
            if prov == "OpenMeteo":
                self._om_apikey = _field(
                    f,
                    row,
                    "API key (opt.)",
                    self.app.cfg_text(
                        "prediction", "pvforecast", "openmeteo", "api_key", default=""
                    ),
                )
                row += 1
                self._om_weather_model = _field(
                    f,
                    row,
                    "Weather model",
                    self.app.cfg_text(
                        "prediction",
                        "pvforecast",
                        "openmeteo",
                        "weather_model",
                        default="",
                    ),
                )

    def _plane(self) -> PVPlaneConfig:
        hz_text = self._horizon.get().strip()
        userhorizon = _csv(hz_text) if hz_text else None
        # Determine inverter_id with precedence: GUI field -> YAML -> optimization inverter -> 'inverter1'
        gui_val = None
        if hasattr(self, "_plane_inverter_id"):
            gui_val = self._plane_inverter_id.get().strip()

        yaml_val = self.app.cfg_text(
            "prediction", "pvforecast", "plane", "inverter_id", default=""
        ).strip()

        opt_inv = (
            self.app.app_config.optimization.inverters[0]
            if self.app.app_config.optimization.inverters
            else None
        )
        opt_val = opt_inv.device_id if opt_inv is not None else ""

        # Determine a safe fallback for inverter id. Use GUI field, YAML, optimizer
        # default, or the inverter field if present. Be explicit about the type so
        # the type checker knows `.get()` exists on the fallback.
        inv_fallback: str = "inverter1"
        _inv_attr = getattr(self, "_inv_id", None)
        if _inv_attr is not None and isinstance(_inv_attr, tk.StringVar):
            inv_fallback = _inv_attr.get().strip() or inv_fallback

        inv_id = gui_val or yaml_val or opt_val or inv_fallback

        return PVPlaneConfig(
            peak_kw=float(self._peak.get()),
            tilt=float(self._tilt.get()),
            azimuth=float(self._az.get()),
            loss_pct=float(self._loss.get()),
            userhorizon=tuple(userhorizon) if userhorizon else None,
            damping_morning=float(self._damp_morn.get()),
            damping_evening=float(self._damp_eve.get()),
            partial_shading=self._partial_shading.get(),
            inverter_id=inv_id,
        )

    def make_provider(self) -> object:
        """Create and return the configured PV forecast provider."""
        p = self._prov_var.get()
        plane = self._plane()
        if p == "OpenMeteo":
            return PVForecastOpenMeteo(
                planes=[plane],
                latitude=float(self._lat.get()),
                longitude=float(self._lon.get()),
                api_key=self._om_apikey.get().strip() or None,
                weather_model=self._om_weather_model.get().strip() or None,
            )
        return PVForecastAkkudoktor(
            planes=[plane],
            latitude=float(self._lat.get()),
            longitude=float(self._lon.get()),
        )

    def _do_plot(self, result: np.ndarray | dict, ts: list) -> None:
        ax = _setup_plot_axes(self._fig)
        t = ts
        s = result if isinstance(result, np.ndarray) else next(iter(result.values()))
        v = list(s)
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
    """Tab for weather forecast providers and plotting."""

    TITLE = "Weather"
    PROVIDERS = ["OpenMeteo", "BrightSky"]

    def _initial_provider(self) -> str:
        v = self.app.cfg_text("prediction", "weather", "provider", default="OpenMeteo")
        return v if v in self.PROVIDERS else self.PROVIDERS[0]

    def _build_fields(self) -> None:
        f, row = self._cfg, 0
        # Global weather location
        self._lat = _field(
            f,
            row,
            "Latitude",
            self.app.cfg_text("prediction", "latitude", default="47.99545"),
        )
        row += 1
        self._lon = _field(
            f,
            row,
            "Longitude",
            self.app.cfg_text("prediction", "longitude", default="7.83355"),
        )

    def make_provider(self) -> object:
        """Create and return the configured weather provider."""
        lat = float(self._lat.get())
        lon = float(self._lon.get())
        if self._prov_var.get() == "OpenMeteo":
            return WeatherOpenMeteo(latitude=lat, longitude=lon)
        return WeatherBrightSky(latitude=lat, longitude=lon)

    def _do_plot(self, result: np.ndarray | dict, ts: list) -> None:
        # Accept either an array (single-channel) or a dict of channels
        if isinstance(result, np.ndarray):
            weather_data = {"value": result}
        else:
            weather_data = result
        t = ts
        cols = list(weather_data.keys())
        n = len(cols)
        if n == 0:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        ncols = 2
        nrows = (n + ncols - 1) // ncols
        for i, col in enumerate(cols):
            ax = self._fig.add_subplot(nrows, ncols, i + 1)
            v = list(weather_data[col])
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
    """Tab to configure and run the optimization/simulation flow."""

    TITLE = "Optimization"
    PROVIDERS = []
    _LEFT_WIDTH = 295
    _last_ts: list | None = None
    _last_dt: float | None = None
    _optimizer_cache: LinearOptimizer | None = None
    _optimizer_cache_sig: str | None = None
    _optimization_running: bool = False
    _prediction_cache: PredictionData | None = None
    _prediction_cache_sig: str | None = None

    def _build_fields(self) -> None:
        f, row = self._cfg, 0

        objective_raw = self.app.app_config.optimization.solver.objective
        objective_default = (
            "Maximize Self-consumption" if "self" in objective_raw.lower() else "Minimize Cost"
        )

        self._linear_objective = _combofield(
            f,
            row,
            "Objective",
            ["Minimize Cost", "Maximize Self-consumption"],
            objective_default,
        )
        row += 1

        # ── Battery configuration ─────────────────────────────────────
        ttk.Label(f, text="Battery", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 0)
        )
        row += 1
        bat = (
            self.app.app_config.optimization.batteries[0]
            if self.app.app_config.optimization.batteries
            else BatteryParameters()
        )
        self._bat_id = _field(
            f,
            row,
            "Battery ID",
            bat.device_id,
        )
        row += 1
        self._bat_capacity = _field(
            f,
            row,
            "Battery capacity Wh",
            str(bat.capacity_wh),
        )
        row += 1
        self._bat_ch_eff = _field(
            f,
            row,
            "Charging efficiency",
            str(bat.charging_efficiency),
        )
        row += 1
        self._bat_dc_eff = _field(
            f,
            row,
            "Discharging efficiency",
            str(bat.discharging_efficiency),
        )
        row += 1
        self._initial_soc = _field(
            f,
            row,
            "Initial battery SoC (%)",
            str(bat.initial_soc_percentage),
        )
        row += 1
        self._min_soc = _field(
            f,
            row,
            "Min SoC (%)",
            str(bat.min_soc_percentage),
        )
        row += 1
        self._max_soc = _field(
            f,
            row,
            "Max SoC (%)",
            str(bat.max_soc_percentage),
        )
        row += 1

        # ── Inverter configuration ────────────────────────────────────
        ttk.Label(f, text="Inverter", foreground="#555", font=("TkDefaultFont", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 0)
        )
        row += 1
        inv = (
            self.app.app_config.optimization.inverters[0]
            if self.app.app_config.optimization.inverters
            else InverterParameters(battery_id="battery1")
        )
        self._inv_id = _field(
            f,
            row,
            "Device ID",
            inv.device_id,
        )
        row += 1
        # PV attachment flag: PV planes are linked to inverters by inverter_id
        self._has_pv = tk.BooleanVar(value=getattr(inv, "has_pv", False))
        ttk.Checkbutton(
            f,
            text="Has PV plane (attached)",
            variable=self._has_pv,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **_PAD)
        row += 1
        self._inv_max_out = _field(
            f,
            row,
            "Max AC output W",
            str(inv.max_ac_output_power_w),
        )
        row += 1
        self._inv_max_charge = _field(
            f,
            row,
            "Max AC charge W",
            str(inv.max_ac_charge_power_w),
        )
        row += 1
        self._inv_dc2ac = _field(
            f,
            row,
            "DC→AC eff",
            str(inv.dc_to_ac_efficiency),
        )
        row += 1
        self._inv_ac2dc = _field(
            f,
            row,
            "AC→DC eff",
            str(inv.ac_to_dc_efficiency),
        )
        row += 1
        self._inv_mode_switch_cost = _field(
            f,
            row,
            "Mode switch cost EUR",
            str(inv.mode_switch_cost),
        )
        row += 1
        self._zero_feed_in = tk.BooleanVar(value=inv.zero_feed_in)
        ttk.Checkbutton(
            f,
            text="Zero feed-in (prevent exporting)",
            variable=self._zero_feed_in,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **_PAD)
        row += 1
        # Run button
        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Button(f, text="▶ Run Optimization", command=self.run_optimization).grid(
            row=row, column=0, columnspan=2, sticky="ew"
        )

    def make_provider(self) -> object:
        """Not used for OptimizationTab; return None."""
        # Not used: this tab aggregates other providers
        return None

    def _optimizer_signature(self, inv: InverterBase, hours: float) -> str:
        """Return a stable topology signature for optimizer cache invalidation."""
        bat = inv.battery
        p = inv.parameters
        return "|".join(
            [
                inv.device_id,
                str(bool(getattr(p, "has_pv", False))),
                str(float(getattr(p, "max_ac_output_power_w", 0.0))),
                str(float(getattr(p, "max_ac_charge_power_w", 0.0))),
                str(float(getattr(p, "dc_to_ac_efficiency", 0.0))),
                str(float(getattr(p, "ac_to_dc_efficiency", 0.0))),
                str(bool(getattr(p, "zero_feed_in", False))),
                str(float(getattr(p, "mode_switch_cost", 0.0))),
                str(tuple(getattr(p, "ac_rates_pct", tuple()))),
                str(float(hours)),
                "none"
                if bat is None
                else "|".join(
                    [
                        str(float(bat.capacity_wh)),
                        str(float(bat.max_charge_power_w)),
                        str(float(bat.max_discharge_power_w)),
                        str(float(bat.charging_efficiency)),
                        str(float(bat.discharging_efficiency)),
                        str(float(bat.min_soc_percentage)),
                        str(float(bat.max_soc_percentage)),
                    ]
                ),
            ]
        )

    def _get_cached_optimizer(
        self,
        inv: InverterBase,
        hours: float,
        prediction: PredictionData,
    ) -> LinearOptimizer:
        """Reuse LinearOptimizer if topology is unchanged; otherwise rebuild it."""
        sig = self._optimizer_signature(inv, hours)
        if self._optimizer_cache is None or self._optimizer_cache_sig != sig:
            self._optimizer_cache = LinearOptimizer(inverters=[inv], prediction=prediction)
            self._optimizer_cache_sig = sig
            logger.info("optimizer_instance_rebuilt", tab="optimization")
        else:
            # Keep cached inverter object in sync with current UI-driven device state.
            self._optimizer_cache.inverters = [inv]
            self._optimizer_cache.prediction = prediction
            logger.info("optimizer_instance_reused", tab="optimization")
        return self._optimizer_cache

    def _prediction_signature(self, start: datetime, hours: float, dt: float) -> str:
        """Stable key to reuse the last fetched prediction when inputs are unchanged."""
        elec_tab = next((t for t in self.app.tabs if isinstance(t, ElecPriceTab)), None)
        feed_tab = next((t for t in self.app.tabs if isinstance(t, FeedInTariffTab)), None)
        load_tab = next((t for t in self.app.tabs if isinstance(t, LoadTab)), None)
        pv_tab = next((t for t in self.app.tabs if isinstance(t, PVForecastTab)), None)
        weather_tab = next((t for t in self.app.tabs if isinstance(t, WeatherTab)), None)
        parts = [
            start.isoformat(),
            str(float(hours)),
            str(float(dt)),
            elec_tab._provider_sig() if elec_tab is not None else "none",
            feed_tab._provider_sig() if feed_tab is not None else "none",
            load_tab._provider_sig() if load_tab is not None else "none",
            pv_tab._provider_sig() if pv_tab is not None else "none",
            weather_tab._provider_sig() if weather_tab is not None else "none",
            self._inv_id.get(),
            str(bool(self._has_pv.get())),
        ]
        return "|".join(str(p) for p in parts)

    def fetch(self) -> None:
        """Override fetch: run optimization flow instead of provider fetch."""
        self.run_optimization()

    def run_optimization(self) -> None:
        """Gather providers, fetch predictions and run the optimizer asynchronously."""
        if self._optimization_running:
            self._status.set("Optimization already running…")
            return
        self._optimization_running = True
        try:
            start, hours, dt = self.app.get_time_params()
            logger.info(
                "optimization_run_start",
                start=start.isoformat(),
                hours=hours,
                dt_hours=dt,
            )
        except Exception as exc:
            self._optimization_running = False
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
            electricprice=cast(
                Optional[ElecPriceProvider], elec_tab._get_provider() if elec_tab else None
            ),
            feedintariff=cast(
                Optional[FeedInTariffProvider], feed_tab._get_provider() if feed_tab else None
            ),
            load=cast(Optional[LoadProvider], load_tab._get_provider() if load_tab else None),
            pv={self._inv_id.get(): cast(PVForecastProvider, pv_tab._get_provider())}
            if pv_tab
            else {},
            weather=cast(
                Optional[WeatherProvider], weather_tab._get_provider() if weather_tab else None
            ),
        )

        pred = Prediction(setup)

        self.app.root.after(0, lambda: self._status.set("Fetching prediction…"))

        # Fetch prediction asynchronously
        def on_pred_done(pdata: PredictionData) -> None:
            try:
                # cache timestamps and dt for plotting
                try:
                    self._last_ts = pdata.timestamps
                except Exception:
                    self._last_ts = None
                self._last_dt = getattr(pdata, "dt_hours", None)

                # Update other tabs' plots with the freshly fetched prediction
                def _update_tabs() -> None:
                    ts = pdata.timestamps
                    try:
                        if elec_tab is not None:
                            try:
                                if pdata.electricprice is not None:
                                    elec_tab._done(pdata.electricprice, ts)
                            except Exception:
                                logger.exception("Failed to update electric price tab")
                        if feed_tab is not None:
                            try:
                                if pdata.feedintariff is not None:
                                    feed_tab._done(pdata.feedintariff, ts)
                            except Exception:
                                logger.exception("Failed to update feed-in tariff tab")
                        if load_tab is not None:
                            try:
                                load_tab._done(pdata.load_wh, ts)
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
                                weather_data = pdata.weather_by_channel
                                if weather_data:
                                    weather_tab._done(weather_data, ts)
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

                def _make_devices(params: BatteryParameters) -> Tuple[Battery, InverterBase]:
                    bat = Battery(params, prediction_hours=int(hours))
                    inv_cfg = (
                        self.app.app_config.optimization.inverters[0]
                        if self.app.app_config.optimization.inverters
                        else None
                    )
                    raw_rates = inv_cfg.ac_rates_pct if inv_cfg is not None else None
                    ac_rates: tuple[int, ...] | None = None
                    if isinstance(raw_rates, (list, tuple)):
                        norm = sorted({int(r) for r in raw_rates if 0 < int(r) <= 100})
                        ac_rates = tuple(norm) if norm else None

                    logger.info(
                        "optimizer_ac_rates_selected",
                        tab="optimization",
                        source="config" if ac_rates else "model_default",
                        rates=ac_rates if ac_rates else "DEFAULT_AC_RATES",
                    )

                    inv_kwargs: dict[str, Any] = {
                        "device_id": self._inv_id.get(),
                        "has_pv": bool(self._has_pv.get()),
                        "battery_id": bat.parameters.device_id,
                        "max_ac_output_power_w": float(self._inv_max_out.get()),
                        "max_ac_charge_power_w": float(self._inv_max_charge.get()),
                        "dc_to_ac_efficiency": float(self._inv_dc2ac.get()),
                        "ac_to_dc_efficiency": float(self._inv_ac2dc.get()),
                        "zero_feed_in": bool(self._zero_feed_in.get()),
                        "mode_switch_cost": float(self._inv_mode_switch_cost.get()),
                    }
                    if ac_rates is not None:
                        inv_kwargs["ac_rates_pct"] = ac_rates

                    inv_params = InverterParameters(**inv_kwargs)
                    inv = InverterBase(inv_params, battery=bat)
                    return bat, inv

                async def _run_sim() -> LinearSolution:
                    # Linear (CVXPY + HiGHS) path
                    _, inv_obj = _make_devices(bat_params)
                    obj_str = getattr(self, "_linear_objective", None)
                    if obj_str is not None and "Self" in obj_str.get():
                        objective = OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION
                    else:
                        objective = OptimizationObjective.MINIMIZE_COST
                    optimizer = self._get_cached_optimizer(inv_obj, hours, pdata)
                    # Prediction is runtime data and should be set per solve call.
                    optimizer.prediction = pdata
                    logger.info(
                        "optimizer_build_start",
                        objective=objective.value,
                        inverters=1,
                    )
                    sol: LinearSolution = await asyncio.to_thread(
                        lambda: optimizer.solve(
                            objective=objective,
                            validate_with_simulation=False,
                        )
                    )
                    return sol

                def _sim_done(res_tuple: object) -> None:
                    self._optimization_running = False
                    # res_tuple may be a LinearSolution or legacy tuple
                    inv_modes_arrs: Optional[list[np.ndarray]] = None
                    inv_ac_rates_arrs: Optional[list[np.ndarray]] = None
                    solve_meta: str | None = None
                    simulation_error: dict[str, Any] | None = None

                    if isinstance(res_tuple, LinearSolution):
                        sol = res_tuple
                        res = sol.simulation_result or sol.result
                        solve_meta = (
                            f"Linear ({sol.objective.value}) · "
                            f"status={sol.solver_status} · "
                            f"{sol.solve_time_s:.2f}s"
                        )
                        if sol.parity_report is not None and not sol.parity_report.ok:
                            simulation_error = {
                                "max_abs_soc_error_wh": sol.parity_report.max_abs_soc_error_wh,
                                "max_abs_grid_import_error_wh": sol.parity_report.max_abs_grid_import_error_wh,
                                "max_abs_feedin_error_wh": sol.parity_report.max_abs_feedin_error_wh,
                                "max_abs_cost_error_eur": sol.parity_report.max_abs_cost_error_eur,
                            }
                        if sol.inverter_plans:
                            plan = sol.inverter_plans[0]
                            try:
                                inv_modes_arrs = [np.asarray(plan.get("modes", []), dtype=np.int32)]
                                inv_ac_rates_arrs = [
                                    np.asarray(plan.get("rates", []), dtype=np.int32)
                                ]
                            except Exception:
                                inv_modes_arrs = None
                                inv_ac_rates_arrs = None
                    else:
                        # legacy tuple (res, modes, rates)
                        if isinstance(res_tuple, tuple):
                            t = cast(
                                Tuple[Any, Optional[list[np.ndarray]], Optional[list[np.ndarray]]],
                                res_tuple,
                            )
                            res, inv_modes_arrs, inv_ac_rates_arrs = t
                        else:
                            res = res_tuple

                    if res is None:
                        messagebox.showinfo(
                            "Optimization", "Simulation returned no result.", parent=self.frame
                        )
                        self._status.set("Simulation finished: no result")
                        return
                    # Tell the type checker to treat `res` as dynamic here
                    res = cast(Any, res)
                    logger.info(
                        "optimizer_result_ready",
                        solver_status=sol.solver_status
                        if isinstance(res_tuple, LinearSolution)
                        else "legacy",
                    )
                    # Capture serialized prediction (for plotting only).
                    sol_pred_dict: dict | None = None
                    sol_pred_ts: list[datetime] | None = None
                    try:
                        if isinstance(res_tuple, LinearSolution):
                            pdict = getattr(sol, "prediction", None)
                            if isinstance(pdict, dict):
                                sol_pred_dict = pdict
                                try:
                                    sol_pred_ts = [
                                        datetime.fromisoformat(s)
                                        for s in pdict.get("timestamps", [])
                                    ]
                                except Exception:
                                    sol_pred_ts = None
                    except Exception:
                        logger.exception("Failed to extract solution.prediction for plotting")
                    out = res.to_dict()
                    # Remove duplicated prediction channels from the serialized
                    # simulation result to avoid redundant output (price / PV).
                    try:
                        pred = getattr(sol, "prediction", None)
                        if isinstance(pred, dict):

                            def _close(a, b, rtol=1e-6, atol=1e-9):
                                try:
                                    aa = np.asarray(a, dtype=float)
                                    bb = np.asarray(b, dtype=float)
                                except Exception:
                                    return False
                                if aa.shape != bb.shape:
                                    return False
                                return bool(np.allclose(aa, bb, rtol=rtol, atol=atol))

                            # electricity price in prediction uses key 'electricprice_eur_wh'
                            pp = pred.get("electricprice_eur_wh")
                            if pp is not None and "electricity_price_per_dt" in out:
                                if _close(out.get("electricity_price_per_dt", []), pp):
                                    out.pop("electricity_price_per_dt", None)

                            # PV prediction channels are exposed as 'pv_<inv>_wh' in prediction
                            pv_keys = [k for k in pred.keys() if k.startswith("pv_")]
                            if pv_keys and out.get("solar_generation_wh_per_dt"):
                                # If solar_generation matches the prediction per-inverter arrays,
                                # drop it from the result to avoid duplication.
                                sol_gen = out.get("solar_generation_wh_per_dt") or {}
                                match = True
                                for inv_id, arr in sol_gen.items():
                                    pv_key = f"pv_{inv_id}"
                                    if pv_key not in pred:
                                        match = False
                                        break
                                    if not _close(arr, pred.get(pv_key)):
                                        match = False
                                        break
                                if match:
                                    out.pop("solar_generation_wh_per_dt", None)
                    except Exception:
                        logger.exception("result_dedupe_failed")
                    if simulation_error is not None:
                        out["simulation_error"] = simulation_error
                    # store timestamps for plotting
                    ts = getattr(self, "_last_ts", None)

                    # Draw results into the right-side figure of this tab
                    try:
                        self._fig.clear()
                        n = len(res.costs_per_dt)

                        # Convert timestamps to matplotlib-compatible numeric x values.
                        # Keep a copy of original datetimes in `x_dt` for hover tooltips.
                        x_dt: Optional[List[datetime]] = None
                        if sol_pred_ts and len(sol_pred_ts) >= n:
                            x_dt = list(sol_pred_ts[:n])
                        else:
                            ts = getattr(self, "_last_ts", None)
                            if ts and len(ts) >= n:
                                x_dt = list(ts[:n])

                        if x_dt is not None:
                            x_vals = mdates.date2num(x_dt)
                        else:
                            x_vals = list(range(n))

                        # Top: energy flows
                        ax = self._fig.add_subplot(311)
                        ax.plot(x_vals, list(res.grid_import_wh_per_dt), label="Grid import (Wh)")
                        ax.plot(
                            x_vals,
                            list(res.self_consumption_wh_per_dt),
                            label="Self-consumption (Wh)",
                        )
                        ax.plot(x_vals, list(res.feedin_wh_per_dt), label="Feed-in (Wh)")
                        ax.plot(x_vals, list(res.losses_wh_per_dt), label="Losses (Wh)")
                        ax.legend(loc="upper right", fontsize=8)
                        ax.set_ylabel("Wh")
                        ax.grid(alpha=0.3)
                        if x_dt is None:
                            # numeric x: don't format dates
                            pass
                        else:
                            # Add hover to energy flows
                            try:
                                grid_import_data = list(res.grid_import_wh_per_dt)
                                _wire_hover(
                                    self._canvas,
                                    ax,
                                    x_dt,
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
                                strict=False,
                            )
                        ]
                        (h_load,) = ax2.plot(
                            x_vals, load_wh, color="#1565C0", linewidth=1.4, label="Load (Wh)"
                        )

                        # Plot PV sources: prefer prediction PV if available, otherwise simulation result
                        pv_handles = []
                        if sol_pred_dict:
                            # collect pv_*_wh keys
                            pv_map = {
                                k[len("pv_") : -len("_wh")]: np.asarray(v, dtype=float)
                                for k, v in sol_pred_dict.items()
                                if k.startswith("pv_") and k.endswith("_wh")
                            }
                            if pv_map:
                                for i, (k, arr) in enumerate(pv_map.items()):
                                    col = f"C{i + 2}"
                                    vals = list(arr[:n])
                                    (h,) = ax2.plot(
                                        x_vals,
                                        vals,
                                        color=col,
                                        linewidth=1.2,
                                        label=f"PV {k} (Wh, pred)",
                                    )
                                    pv_handles.append(h)
                        else:
                            sol_gen = getattr(res, "solar_generation_wh_per_dt", None)
                            if sol_gen:
                                for i, (k, arr) in enumerate(sol_gen.items()):
                                    col = f"C{i + 2}"
                                    (h,) = ax2.plot(
                                        x_vals,
                                        list(arr),
                                        color=col,
                                        linewidth=1.2,
                                        label=f"PV {k} (Wh)",
                                    )
                                    pv_handles.append(h)

                        ax2.set_ylabel("Wh")
                        ax2.grid(alpha=0.2)

                        # Right axis: electricity price (€/Wh)
                        ax3 = ax2.twinx()
                        # Plot electricity price: prefer prediction price if available
                        if sol_pred_dict and "electricprice_eur_wh" in sol_pred_dict:
                            price_vals = list(
                                np.asarray(sol_pred_dict["electricprice_eur_wh"], dtype=float)[:n]
                            )
                        else:
                            price_arr = getattr(res, "electricity_price_per_dt", None)
                            if price_arr is not None:
                                try:
                                    price_vals = list(np.asarray(price_arr, dtype=float)[:n])
                                except Exception:
                                    price_vals = [float(v) for v in price_arr[:n]]
                            else:
                                price_vals = [0.0] * n
                        (h_price,) = ax3.plot(
                            x_vals,
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
                        if x_dt is None:
                            # numeric x: datetime hover not applicable
                            pass
                        else:
                            # Add datetime hover to energy plot
                            try:
                                _wire_hover(
                                    self._canvas,
                                    ax2,
                                    x_dt,
                                    load_wh,
                                    fmt_y="{:.1f}",
                                    unit=" Wh",
                                )
                            except Exception:
                                logger.exception("Failed to add datetime hover to energy plot")

                        # Bottom: battery SoC (%) (left axis) and inverter modes (right axis)
                        ax4 = self._fig.add_subplot(313)
                        ax4r = ax4.twinx()
                        plotted = False
                        handles = []
                        labels = []
                        # Plot SoC on left axis. Show initial SoC at t0 and shift
                        # subsequent SoC values one step to the right because the
                        # solver reports SoC after processing each slot.
                        if res.battery_soc_percentage_per_dt:
                            for _k, arr in (res.battery_soc_percentage_per_dt or {}).items():
                                arr_list = list(arr)
                                try:
                                    init_pct = float(bat_params.initial_soc_percentage)
                                except Exception:
                                    init_pct = arr_list[0] if arr_list else 0.0

                                if len(arr_list) >= 1:
                                    soc_plot = [init_pct] + arr_list[:-1]
                                else:
                                    soc_plot = [init_pct]

                                (h,) = ax4.plot(x_vals, soc_plot, label=f"SoC {_k} (%)")
                                handles.append(h)
                                labels.append(f"SoC {_k} (%)")
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
                                    float(rates_arr[i]) / 100.0
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
                                x_vals,
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
                        if x_dt is None:
                            # numeric x: don't add hover
                            pass
                        else:
                            # Add hover for SoC
                            soc_data = []
                            if res.battery_soc_percentage_per_dt:
                                for _k, arr in (res.battery_soc_percentage_per_dt or {}).items():
                                    arr_list = list(arr)
                                    try:
                                        init_pct = float(bat_params.initial_soc_percentage)
                                    except Exception:
                                        init_pct = arr_list[0] if arr_list else 0.0
                                    if len(arr_list) >= 1:
                                        soc_data = [init_pct] + arr_list[:-1]
                                    else:
                                        soc_data = [init_pct]
                                    break
                            if soc_data:
                                try:
                                    _wire_hover(
                                        self._canvas,
                                        ax4,
                                        x_dt,
                                        soc_data,
                                        fmt_y="{:.1f}",
                                        unit="%",
                                    )
                                except Exception:
                                    logger.exception("Failed to add hover for SoC plot")

                        # Date formatting for all subplots with timestamps
                        if x_dt is not None:
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
                        "optimizer": getattr(
                            getattr(self, "_optimizer_type", None), "get", lambda: None
                        )(),
                        "initial_soc_pct": float(self._initial_soc.get()),
                    }
                    if solve_meta:
                        meta["solve_info"] = solve_meta
                    payload: dict[str, Any] = {"meta": meta, "result": out}
                    if sol_pred_dict is not None:
                        payload["prediction"] = sol_pred_dict
                    txt.insert("1.0", json.dumps(payload, indent=2))
                    txt.configure(state="disabled")
                    suffix = f" · {solve_meta}" if solve_meta else ""
                    self._status.set(f"Done · Net: {res.net_balance:.2f} €{suffix}")

                run_id = uuid.uuid4().hex[:8]
                _run_async(
                    _with_context(
                        _run_sim(),
                        run_id=run_id,
                        tab="optimization",
                        operation="optimization",
                    ),
                    on_done=lambda r: _sim_done(r),
                    on_error=lambda e, tb: self._on_error(e, tb, operation="optimization"),
                )
            except Exception as exc:
                self._on_error(exc, None, operation="optimization")

        pred_sig = self._prediction_signature(start, hours, dt)
        if self._prediction_cache is not None and self._prediction_cache_sig == pred_sig:
            logger.info("prediction_fetch_reused", tab="optimization")
            cached = self._prediction_cache
            self.app.root.after(0, lambda c=cached: on_pred_done(c))
            return

        run_id = uuid.uuid4().hex[:8]
        _run_async(
            _with_context(
                pred.fetch(start=start, hours=hours, dt_hours=dt),
                run_id=run_id,
                tab="optimization",
                operation="prediction_fetch",
            ),
            on_done=lambda r: (
                setattr(self, "_prediction_cache", r),
                setattr(self, "_prediction_cache_sig", pred_sig),
                on_pred_done(r),
            ),
            on_error=lambda e, tb: (
                setattr(self, "_optimization_running", False),
                self._on_error(e, tb, operation="prediction_fetch"),
            ),
        )

    def _on_error(self, exc: Exception, tb: str | None, *, operation: str = "unknown") -> None:
        self._optimization_running = False
        self._status.set(f"Error: {exc}")
        logger.error(
            "gui_task_error",
            operation=operation,
            exc_type=type(exc).__name__,
            exc=str(exc),
            traceback=tb or "",
        )
        messagebox.showerror(
            "Error",
            "An error occurred. See console for details.",
            parent=self.frame,
        )


class App:
    """Main GUI application container for the forecast preview tool."""

    def __init__(self) -> None:
        """Initialize the main application window and tabs."""
        self.root = tk.Tk()
        default_config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        self._config_path = tk.StringVar(value=str(default_config_path))
        self._app_config = AppConfig()
        self._yaml_config: dict[str, Any] = {}
        # Raw YAML mapping exactly as loaded from file (no defaults applied)
        self._yaml_raw: dict[str, Any] = {}
        # If config file is present and parsed successfully, only create tabs
        # present in the YAML. If no config file exists, create all tabs.
        self._config_present = self._load_yaml_config(show_error=False)
        self.root.title("Forecast Preview  —  dev tool")
        self.root.minsize(960, 580)
        self._build_topbar()
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # Create tabs only when the corresponding config section exists.
        # Mapping: (TabClass, config path tuple) — use None to always include.
        tab_candidates: list[tuple[type[_Tab], tuple[str, ...] | None]] = [
            (ElecPriceTab, ("prediction", "electricprice")),
            (FeedInTariffTab, ("prediction", "feedintariff")),
            (LoadTab, ("prediction", "load")),
            (PVForecastTab, ("prediction", "pvforecast")),
            (WeatherTab, ("prediction", "weather")),
            (OptimizationTab, ("optimization",)),
        ]

        self.tabs = []
        for cls, path in tab_candidates:
            # If no config file provided, include all tabs with defaults.
            if not self._config_present:
                has_cfg = True
            else:
                # Decide presence based on raw YAML (only create tab if user included section)
                if path is None:
                    has_cfg = True
                else:
                    node: Any = self._yaml_raw
                    has_cfg = True
                    for key in path:
                        if not isinstance(node, Mapping) or key not in node:
                            has_cfg = False
                            break
                        node = node[key]
            if not has_cfg:
                logger.debug("skipping_tab_no_config", tab=cls.__name__, path=path)
                continue
            try:
                self.tabs.append(cls(nb, self))
            except Exception as exc:
                logger.error("tab_init_failed", tab=cls.__name__, exc=exc)

    def _load_yaml_config(self, show_error: bool = True) -> bool:
        path = Path(self._config_path.get()).expanduser()
        if not path.exists():
            self._yaml_config = {}
            if show_error:
                messagebox.showwarning(
                    "Config not found",
                    f"Could not find config file:\n{path}",
                    parent=self.root,
                )
            return False
        try:
            # Read raw YAML first so we know which sections the user explicitly provided.
            text = Path(path).read_text(encoding="utf-8")
            raw = yaml.safe_load(text) or {}
            if not isinstance(raw, dict):
                raise ValueError("Root YAML node must be a mapping")
            self._yaml_raw = raw

            # Validate/build the full config model (fills defaults where missing)
            self._app_config = AppConfig.from_dict(raw)
            self._yaml_config = self._app_config.model_dump(mode="python")
            return True
        except Exception as exc:
            self._app_config = AppConfig()
            self._yaml_config = self._app_config.model_dump(mode="python")
            # If we failed to parse, keep raw as empty mapping so tabs won't be created.
            self._yaml_raw = {}
            if show_error:
                messagebox.showerror(
                    "Config error", f"Failed to load YAML:\n{exc}", parent=self.root
                )
            return False

    @property
    def app_config(self) -> AppConfig:
        """Return the validated root config model."""
        return self._app_config

    def cfg(self, *path: str, default: Any = None) -> Any:
        node: Any = self._yaml_config
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return node

    def cfg_text(self, *path: str, default: str = "") -> str:
        value = self.cfg(*path, default=default)
        return _csv_text(value, default=default)

    def cfg_csv(self, *path: str, default: str = "") -> str:
        value = self.cfg(*path, default=None)
        return _csv_text(value, default=default)

    def cfg_bool(self, *path: str, default: bool = False) -> bool:
        value = self.cfg(*path, default=default)
        return bool(value)

    def cfg_list_item(self, *path: str, idx: int = 0, default: Any = None) -> Any:
        """Get item at idx from a list at *path, or return default."""
        lst = self.cfg(*path, default=None)
        if isinstance(lst, list) and idx < len(lst):
            return lst[idx]
        return default or {}

    def cfg_text_from_dict(self, d: dict[str, Any], key: str, default: str = "") -> str:
        """Extract text value from dict, converting lists/tuples to CSV."""
        value = d.get(key, default) if isinstance(d, dict) else default
        return _csv_text(value, default=default)

    def cfg_bool_from_dict(self, d: dict[str, Any], key: str, default: bool = False) -> bool:
        """Extract bool value from dict."""
        value = d.get(key, default) if isinstance(d, dict) else default
        return bool(value)

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if not path:
            return
        self._config_path.set(path)
        self._reload_config_defaults()

    def _reload_config_defaults(self) -> None:
        if not self._load_yaml_config(show_error=True):
            return
        self._apply_topbar_defaults()
        for tab in self.tabs:
            tab.load_defaults_from_config()

    def _build_topbar(self) -> None:
        bar = ttk.Frame(self.root)
        # anchor="nw": bar sizes to its natural content width; both rows stay equally wide
        bar.pack(anchor="nw", padx=6, pady=6)

        # Row 0: Time parameters – natural (fixed) width determines the bar width
        ttk.Label(bar, text="Start:").grid(row=0, column=0)
        self._start = tk.StringVar()
        ttk.Entry(bar, textvariable=self._start, width=16).grid(row=0, column=1, padx=2)

        ttk.Label(bar, text="Hours:").grid(row=0, column=2, padx=(10, 0))
        self._hours = tk.StringVar()
        ttk.Entry(bar, textvariable=self._hours, width=5).grid(row=0, column=3, padx=2)

        ttk.Label(bar, text="Δt [h]:").grid(row=0, column=4, padx=(10, 0))
        self._dt = tk.StringVar()
        ttk.Combobox(
            bar, textvariable=self._dt, values=_DT_CHOICES, state="readonly", width=5
        ).grid(row=0, column=5, padx=2)

        ttk.Label(bar, text="Time Zone:").grid(row=0, column=6, padx=(10, 0))
        self._tz = tk.StringVar()
        ttk.Combobox(
            bar, textvariable=self._tz, values=_TZ_CHOICES, state="readonly", width=14
        ).grid(row=0, column=7, padx=2)

        ttk.Separator(bar, orient="vertical").grid(row=0, column=8, sticky="ns", padx=12)
        # width≈20% wider than the text character-count default
        ttk.Button(bar, text="▶▶ Fetch All", command=self._fetch_all, width=14).grid(
            row=0, column=9, sticky="w"
        )

        # Row 1: Config file selector
        # Columns 1-7 are sized by row 0; the entry with sticky="ew" fills that exact space.
        ttk.Label(bar, text="Config:").grid(row=1, column=0, pady=(6, 0), sticky="w")
        ttk.Entry(bar, textvariable=self._config_path).grid(
            row=1, column=1, columnspan=7, padx=2, pady=(6, 0), sticky="ew"
        )
        ttk.Button(bar, text="…", width=2, command=self._browse_config).grid(
            row=1, column=8, sticky="w", padx=(2, 0), pady=(6, 0)
        )
        ttk.Button(bar, text="Load", command=self._reload_config_defaults).grid(
            row=1, column=9, padx=2, pady=(6, 0)
        )
        self._apply_topbar_defaults()

    def _apply_topbar_defaults(self) -> None:
        start_default = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self._start.set(start_default)
        self._hours.set(_GUI_HOURS_DEFAULT)
        self._dt.set(_GUI_DT_DEFAULT)
        self._tz.set(_GUI_TIMEZONE_DEFAULT)

    def get_time_params(self) -> tuple[datetime, float, float]:
        """Parse and return the start datetime, hours and dt from the top bar fields.

        Returns:
            Tuple of (start: datetime with tzinfo, hours: float, dt_hours: float).
        """
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
    import logging

    import structlog

    # Configure structlog: timestamp + log level + structured console output.
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    # Silence noisy third-party stdlib loggers.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    app = App()
    app.root.mainloop()


if __name__ == "__main__":
    run()
