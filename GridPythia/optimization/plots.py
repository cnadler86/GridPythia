"""Plotly-based plotter for optimization solutions.

The :class:`SolutionPlotter` renders the full grid-level energy-flow picture
from a :class:`~GridPythia.optimization.solution.LinearSolution` (or any
:class:`~GridPythia.optimization.solution.EnergySolution`).

Typical usage::

    solution = optimizer.solve(...)
    fig = SolutionPlotter().plot(solution)
    fig.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from GridPythia.optimization.solution import EnergySolution

# ── colour constants ──────────────────────────────────────────────────────
_C_IMPORT = "#1565C0"  # grid import  – blue
_C_FEEDIN = "#2E7D32"  # feed-in      – green
_C_SELF = "#F9A825"  # self-consumption – amber
_C_LOSS = "#B71C1C"  # losses       – red
_C_COST = "#AD1457"  # cost curve   – pink
_C_SOC = "#00838F"  # battery SoC  – teal
_C_PV = "#F57F17"  # PV total     – deep amber

# ── per-inverter plan colours ────────────────────────────────────────────
_C_CHARGE_BAR = "#4FC3F7"  # pastel cyan-blue  – AC charge power bars
_C_DISCHARGE_BAR = "#F48FB1"  # pastel pink       – AC discharge power bars
_C_PV_BAT_BAR = "#FFE082"  # pastel amber      – PV→Bat power bars
_BG_CHARGE = "rgba(144,202,249,0.30)"  # pastel blue  – charge mode background
_BG_DISCHARGE = "rgba(255,224,130,0.40)"  # pastel amber – discharge mode background

_LAYOUT = {
    "template": "plotly_white",
    "hovermode": "x unified",
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    "margin": {"l": 60, "r": 20, "t": 60, "b": 40},
}


def _add_mode_backgrounds(
    fig: "go.Figure",
    timestamps: list,
    modes: "np.ndarray",
    dt_hours: float,
    row: int,
    n_rows: int = 3,
) -> None:
    """Shade *row* background by inverter mode (charge=pastel-blue, discharge=pastel-amber).

    Args:
        fig: Plotly figure with subplots.
        timestamps: List of datetime objects.
        modes: Array of inverter modes (int).
        dt_hours: Time delta in hours.
        row: Subplot row number (1-indexed).
        n_rows: Total number of subplot rows (needed to compute paper y-domain).
    """
    from datetime import timedelta

    from GridPythia.simulation.devices import InverterMode

    _CHARGE = {int(InverterMode.AC_CHARGE), int(InverterMode.AC_CHARGE_ZERO_FEED_IN)}
    _DISCHARGE = {int(InverterMode.DISCHARGE), int(InverterMode.DISCHARGE_ZERO_FEED_IN)}
    _BG: dict[str, str] = {"charge": _BG_CHARGE, "discharge": _BG_DISCHARGE}
    dt_td = timedelta(hours=dt_hours)

    # Compute xref for this subplot row
    xref = "x" if row == 1 else f"x{row}"

    # For y-axis, use paper coordinates with domain calculation.
    # Each row gets a fraction of the paper height (accounting for vertical_spacing).
    # By default, make backgrounds stretch across full subplot height using paper coords.
    row_height = 1.0 / n_rows
    y0_paper = 1.0 - (row * row_height)  # Bottom edge of this row in paper coords
    y1_paper = y0_paper + row_height  # Top edge of this row in paper coords

    def _cls(m: int) -> str | None:
        if m in _CHARGE:
            return "charge"
        if m in _DISCHARGE:
            return "discharge"
        return None  # IDLE → no shading

    n = min(len(timestamps), len(modes))
    if n == 0:
        return
    seg_start = 0
    seg_cls = _cls(int(modes[0]))
    for i in range(1, n):
        c = _cls(int(modes[i]))
        if c != seg_cls:
            if seg_cls is not None:
                fig.add_shape(
                    type="rect",
                    x0=timestamps[seg_start],
                    y0=y0_paper,
                    x1=timestamps[i - 1] + dt_td,
                    y1=y1_paper,
                    fillcolor=_BG[seg_cls],
                    layer="below",
                    line_width=0,
                    xref=xref,
                    yref="paper",
                )
            seg_start = i
            seg_cls = c
    if seg_cls is not None:
        fig.add_shape(
            type="rect",
            x0=timestamps[seg_start],
            y0=y0_paper,
            x1=timestamps[n - 1] + dt_td,
            y1=y1_paper,
            fillcolor=_BG[seg_cls],
            layer="below",
            line_width=0,
            xref=xref,
            yref="paper",
        )


class SolutionPlotter:
    """Render an optimization solution as a multi-panel Plotly figure.

    The figure contains up to three stacked sub-plots:

    1. **Energy flows** – grid import, feed-in, self-consumption (stacked bars).
    2. **Battery SoC** – one line per inverter that has a battery.
    3. **Cumulative cost / revenue** – running EUR balance.
    """

    def plot(
        self,
        solution: "EnergySolution",
        *,
        title: str = "Optimization Result",
    ) -> go.Figure:
        """Return a :class:`plotly.graph_objects.Figure` for *solution*.

        Args:
            solution: Any :class:`~GridPythia.optimization.solution.EnergySolution`.
            title:    Plot title shown at the top.
        """
        timestamps = solution.prediction.timestamps
        result = solution.result

        # Decide subplot layout.
        has_soc = bool(result.battery_soc_percentage_per_dt)
        n_rows = 3 if has_soc else 2
        subplot_titles = (
            ["Energy Flows (Wh)", "Battery SoC (%)", "Cumulative Balance (EUR)"]
            if has_soc
            else ["Energy Flows (Wh)", "Cumulative Balance (EUR)"]
        )
        row_heights = [0.45, 0.25, 0.30] if has_soc else [0.55, 0.45]

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
            vertical_spacing=0.06,
        )

        # ── row 1: energy flows ───────────────────────────────────────
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=result.grid_import_wh_per_dt.tolist(),
                name="Grid Import",
                marker_color=_C_IMPORT,
                opacity=0.85,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>Grid Import</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=result.self_consumption_wh_per_dt.tolist(),
                name="Self-consumption",
                marker_color=_C_SELF,
                opacity=0.85,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>Self-consumption</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=(-np.asarray(result.feedin_wh_per_dt)).tolist(),
                name="Feed-in",
                marker_color=_C_FEEDIN,
                opacity=0.85,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>Feed-in</extra>",
            ),
            row=1,
            col=1,
        )
        if result.losses_wh_per_dt is not None and np.any(result.losses_wh_per_dt):
            fig.add_trace(
                go.Bar(
                    x=timestamps,
                    y=(-np.asarray(result.losses_wh_per_dt)).tolist(),
                    name="Losses",
                    marker_color=_C_LOSS,
                    opacity=0.7,
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>Losses</extra>",
                ),
                row=1,
                col=1,
            )

        # PV total from prediction (if available)
        pv_total = (
            sum(
                solution.prediction.pv_by_inverter.values(),
                np.zeros(len(timestamps), dtype=np.float32),
            )
            if solution.prediction.pv_by_inverter
            else None
        )
        if pv_total is not None and np.any(pv_total):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=pv_total.tolist(),
                    name="PV",
                    mode="lines",
                    line={"color": _C_PV, "width": 1.5, "dash": "dot"},
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>PV</extra>",
                ),
                row=1,
                col=1,
            )

        fig.update_yaxes(title_text="Wh", row=1, col=1)
        fig.update_layout(barmode="relative")

        # ── row 2: battery SoC (optional) ────────────────────────────
        soc_row = 2 if has_soc else None
        if soc_row is not None:
            _COLORS_SOC = [_C_SOC, "#6A1B9A", "#E65100", "#2E7D32"]
            for idx, (inv_id, soc_arr) in enumerate(result.battery_soc_percentage_per_dt.items()):
                color = _COLORS_SOC[idx % len(_COLORS_SOC)]
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=np.asarray(soc_arr).tolist(),
                        name=f"SoC {inv_id}",
                        mode="lines",
                        line={"color": color, "width": 1.8},
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} %<extra>SoC "
                        + inv_id
                        + "</extra>",
                    ),
                    row=soc_row,
                    col=1,
                )
            fig.update_yaxes(title_text="%", range=[0, 105], row=soc_row, col=1)

        # ── last row: cumulative balance ──────────────────────────────
        balance_row = n_rows
        cum_cost = np.cumsum(result.costs_per_dt)
        cum_rev = np.cumsum(result.revenue_per_dt)
        cum_balance = cum_rev - cum_cost
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cum_balance.tolist(),
                name="Net balance",
                mode="lines",
                line={"color": _C_COST, "width": 1.8},
                fill="tozeroy",
                fillcolor="rgba(173, 20, 87, 0.08)",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.4f} EUR<extra>Net balance</extra>",
            ),
            row=balance_row,
            col=1,
        )
        fig.update_yaxes(title_text="EUR", row=balance_row, col=1)

        # ── inverter-mode backgrounds (charge/discharge shading) ──────
        # Use the first plan with a battery; fall back to the first plan available.
        _bg_plan = next(
            (p for p in solution.inverter_plans if p.battery_soc_wh is not None),
            solution.inverter_plans[0] if solution.inverter_plans else None,
        )
        if _bg_plan is not None and len(_bg_plan.modes) > 0:
            _dt_h = float(solution.prediction.dt_hours)
            for _row in range(1, n_rows + 1):
                _add_mode_backgrounds(
                    fig, timestamps, _bg_plan.modes, _dt_h, row=_row, n_rows=n_rows
                )

        # ── global layout ────────────────────────────────────────────
        fig.update_layout(
            **_LAYOUT,
            title={
                "text": (
                    f"{title}<br>"
                    f"<sup>cost={result.total_cost:.4f} EUR  |  "
                    f"revenue={result.total_revenue:.4f} EUR  |  "
                    f"import={result.total_grid_import / 1000:.2f} kWh  |  "
                    f"feed-in={result.total_feedin / 1000:.2f} kWh</sup>"
                ),
                "font": {"size": 14},
            },
            height=max(500, 220 * n_rows),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
        return fig

    def plot_inverter(
        self,
        solution: "EnergySolution",
        inv_device_id: str,
        *,
        title: str | None = None,
    ) -> go.Figure:
        """Return a per-inverter detail figure for *inv_device_id*.

        Layout (when battery present):

        - **Row 1** – stacked energy-flow bars (grid import, self-consumption,
          feed-in) plus a PV reference line.
        - **Row 2** – battery SoC line (left axis) and per-slot power bars
          (AC charge, discharge, PV→Bat, right axis) with pastel mode
          background derived from :attr:`~GridPythia.optimization.plan.InverterPlan.modes`.
        """
        if title is None:
            title = f"Inverter Plan \u2013 {inv_device_id}"

        timestamps = solution.prediction.timestamps
        result = solution.result
        pdata = solution.prediction
        dt_hours = float(pdata.dt_hours)

        plan = next(
            (p for p in solution.inverter_plans if p.device_id == inv_device_id),
            None,
        )
        has_battery = (
            plan is not None
            and plan.battery_soc_wh is not None
            and inv_device_id in result.battery_soc_percentage_per_dt
        )

        n_rows = 2 if has_battery else 1
        specs = (
            [[{"secondary_y": False}], [{"secondary_y": True}]]
            if has_battery
            else [[{"secondary_y": False}]]
        )
        subplot_titles = (
            ["Energy Flows (Wh)", "Battery SoC (%) \u2022 Power (W)"]
            if has_battery
            else ["Energy Flows (Wh)"]
        )
        row_heights = [0.42, 0.58] if has_battery else [1.0]

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
            vertical_spacing=0.08,
            specs=specs,
        )

        # ── row 1: energy-flow bars ───────────────────────────────────
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=result.grid_import_wh_per_dt.tolist(),
                name="Grid Import",
                marker_color=_C_IMPORT,
                opacity=0.85,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>Grid Import</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=result.self_consumption_wh_per_dt.tolist(),
                name="Self-consumption",
                marker_color=_C_SELF,
                opacity=0.85,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>Self-consumption</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=(-np.asarray(result.feedin_wh_per_dt)).tolist(),
                name="Feed-in",
                marker_color=_C_FEEDIN,
                opacity=0.85,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>Feed-in</extra>",
            ),
            row=1,
            col=1,
        )
        if inv_device_id in pdata.pv_by_inverter:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=np.asarray(pdata.pv_by_inverter[inv_device_id]).tolist(),
                    name="PV",
                    mode="lines",
                    line={"color": _C_PV, "width": 1.5, "dash": "dot"},
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra>PV</extra>",
                ),
                row=1,
                col=1,
            )
        fig.update_yaxes(title_text="Wh", row=1, col=1)
        fig.update_layout(barmode="relative")

        # ── row 2: mode background + SoC line + power bars ───────────
        if has_battery:
            _add_mode_backgrounds(fig, timestamps, plan.modes, dt_hours, row=2, n_rows=2)

            # SoC is a state variable → line (left axis)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=np.asarray(result.battery_soc_percentage_per_dt[inv_device_id]).tolist(),
                    name="SoC",
                    mode="lines",
                    line={"color": _C_SOC, "width": 2.2},
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} %<extra>SoC</extra>",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig.update_yaxes(title_text="SoC (%)", range=[-2, 107], row=2, col=1, secondary_y=False)

            # Per-slot power flows → bars (right axis)
            if np.any(plan.charge_ac_wh):
                fig.add_trace(
                    go.Bar(
                        x=timestamps,
                        y=(np.asarray(plan.charge_ac_wh) / dt_hours).tolist(),
                        name="AC Charge (W)",
                        marker_color=_C_CHARGE_BAR,
                        opacity=0.85,
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.0f} W<extra>AC Charge</extra>",
                    ),
                    row=2,
                    col=1,
                    secondary_y=True,
                )
            if np.any(plan.discharge_ac_wh):
                fig.add_trace(
                    go.Bar(
                        x=timestamps,
                        y=(np.asarray(plan.discharge_ac_wh) / dt_hours).tolist(),
                        name="Discharge (W)",
                        marker_color=_C_DISCHARGE_BAR,
                        opacity=0.85,
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.0f} W<extra>Discharge</extra>",
                    ),
                    row=2,
                    col=1,
                    secondary_y=True,
                )
            if plan.pv_to_battery_wh is not None and np.any(plan.pv_to_battery_wh):
                fig.add_trace(
                    go.Bar(
                        x=timestamps,
                        y=(np.asarray(plan.pv_to_battery_wh) / dt_hours).tolist(),
                        name="PV\u2192Bat (W)",
                        marker_color=_C_PV_BAT_BAR,
                        opacity=0.85,
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.0f} W<extra>PV\u2192Bat</extra>",
                    ),
                    row=2,
                    col=1,
                    secondary_y=True,
                )
            fig.update_yaxes(title_text="Power (W)", row=2, col=1, secondary_y=True)

        # Build layout with per-inverter-specific overrides (wider margins for secondary y-axis)
        layout_params = {
            **_LAYOUT,
            "margin": {"l": 65, "r": 65, "t": 60, "b": 40},
            "title": {
                "text": (
                    f"{title}<br>"
                    f"<sup>cost={result.total_cost:.4f} EUR  |  "
                    f"revenue={result.total_revenue:.4f} EUR  |  "
                    f"import={result.total_grid_import / 1000:.2f} kWh  |  "
                    f"feed-in={result.total_feedin / 1000:.2f} kWh</sup>"
                ),
                "font": {"size": 14},
            },
            "height": 700,
        }
        fig.update_layout(**layout_params)
        fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
        return fig
