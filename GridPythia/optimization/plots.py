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

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

# plotly is imported lazily in SolutionPlotter.plot() so that importing this
# module at server startup does not pull in narwhals (~56 modules) via
# _plotly_utils.basevalidators.
if TYPE_CHECKING:
    import plotly.graph_objects as go

    from GridPythia.optimization.solution import EnergySolution


def _calc_pretty_ticks(
    vmin: float, vmax: float, target_nticks: int = 6
) -> tuple[float, float, float]:
    """Calculate 'pretty' tick parameters (tick0, dtick) for a given value range.

    This uses the Wilkinson algorithm to find nice, round tick intervals.
    Returns (tick0, dtick, max_val) where:
    - tick0: starting value for the first tick
    - dtick: interval between ticks
    - max_val: rounded maximum value for the axis

    Args:
        vmin: minimum value in data
        vmax: maximum value in data
        target_nticks: desired number of ticks (default 6)
    """
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1

    span = vmax - vmin
    magnitude = 10 ** np.floor(np.log10(span))

    # Try candidate tick intervals: 1, 2, 5 times the magnitude
    candidates = [1, 2, 5, 10]
    best_candidate = 1
    best_error = float("inf")

    for cand in candidates:
        dtick = cand * magnitude
        nticks = span / dtick
        error = abs(nticks - target_nticks)
        if error < best_error:
            best_error = error
            best_candidate = cand

    dtick = best_candidate * magnitude
    tick0 = np.floor(vmin / dtick) * dtick
    max_val = np.ceil(vmax / dtick) * dtick

    return float(tick0), float(dtick), float(max_val)


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
    "margin": {"l": 45, "r": 20, "t": 60, "b": 40},
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
        import plotly.graph_objects as go  # noqa: PLC0415
        from plotly.subplots import make_subplots  # noqa: PLC0415

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
                hovertemplate="%{y:.1f} Wh<extra>Grid Import</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=result.self_consumption_wh_per_dt.tolist(),
                name="Self-cons.",
                marker_color=_C_SELF,
                opacity=0.85,
                hovertemplate="%{y:.1f} Wh<extra>Self-consumption</extra>",
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
                customdata=np.asarray(result.feedin_wh_per_dt).tolist(),
                hovertemplate="%{customdata:.1f} Wh<extra>Feed-in</extra>",
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
                    hovertemplate="%{y:.1f} Wh<extra>Losses</extra>",
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
                    hovertemplate="%{y:.1f} Wh<extra>PV</extra>",
                ),
                row=1,
                col=1,
            )

        # Calculate pretty ticks for energy axis (Wh) in row 1
        wh_values = (
            list(result.grid_import_wh_per_dt)
            + list(result.self_consumption_wh_per_dt)
            + (-np.asarray(result.feedin_wh_per_dt)).tolist()
        )
        if result.losses_wh_per_dt is not None:
            wh_values.extend((-np.asarray(result.losses_wh_per_dt)).tolist())

        if wh_values:
            wh_min, wh_max = min(wh_values), max(wh_values)
            if wh_min < wh_max:
                tick0_wh, dtick_wh, max_wh = _calc_pretty_ticks(wh_min, wh_max, target_nticks=6)
                fig.update_yaxes(
                    tickmode="linear",
                    tick0=tick0_wh,
                    dtick=dtick_wh,
                    range=[tick0_wh, max_wh],
                    row=1,
                    col=1,
                )

        fig.update_layout(barmode="relative")

        # ── row 2: battery SoC (optional) ────────────────────────────
        soc_row = 2 if has_soc else None
        if soc_row is not None:
            _COLORS_SOC = [_C_SOC, "#6A1B9A", "#E65100", "#2E7D32"]
            dt_td = timedelta(hours=float(solution.prediction.dt_hours))
            for idx, (inv_id, soc_arr) in enumerate(result.battery_soc_percentage_per_dt.items()):
                color = _COLORS_SOC[idx % len(_COLORS_SOC)]
                soc_vals = np.asarray(soc_arr, dtype=float)
                if len(timestamps) and soc_vals.size:
                    init_soc = float(
                        result.battery_initial_soc_percentage.get(inv_id, float(soc_vals[0]))
                    )
                    soc_plot = np.concatenate(([init_soc], soc_vals))
                    x_plot = [timestamps[0]] + [ts + dt_td for ts in timestamps]
                else:
                    soc_plot = soc_vals
                    x_plot = timestamps
                fig.add_trace(
                    go.Scatter(
                        x=x_plot,
                        y=soc_plot.tolist(),
                        name=f"SoC {inv_id}",
                        mode="lines",
                        line={"color": color, "width": 1.8},
                        hovertemplate="%{y:.1f} %<extra>SoC " + inv_id + "</extra>",
                    ),
                    row=soc_row,
                    col=1,
                )
            fig.update_yaxes(range=[0, 105], row=soc_row, col=1)

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
                hovertemplate="%{y:.4f} EUR<extra>Net balance</extra>",
            ),
            row=balance_row,
            col=1,
        )
        fig.update_yaxes(row=balance_row, col=1)

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
        fig.update_xaxes(
            tickformat="%d.%m.%y",
            showgrid=True,
            gridcolor="#e8e8e8",
            minor={
                "dtick": 3_600_000,
                "showgrid": True,
                "gridcolor": "rgba(200,200,200,0.35)",
                "ticks": "",
            },
        )
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
        import plotly.graph_objects as go  # noqa: PLC0415
        from plotly.subplots import make_subplots  # noqa: PLC0415

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
                legend="legend",
                marker_color=_C_IMPORT,
                opacity=0.85,
                hovertemplate="%{y:.1f} Wh<extra>Grid Import</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=result.self_consumption_wh_per_dt.tolist(),
                name="Self-cons.",
                legend="legend",
                marker_color=_C_SELF,
                opacity=0.85,
                hovertemplate="%{y:.1f} Wh<extra>Self-consumption</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=(-np.asarray(result.feedin_wh_per_dt)).tolist(),
                name="Feed-in",
                legend="legend",
                marker_color=_C_FEEDIN,
                opacity=0.85,
                customdata=np.asarray(result.feedin_wh_per_dt).tolist(),
                hovertemplate="%{customdata:.1f} Wh<extra>Feed-in</extra>",
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
                    legend="legend",
                    mode="lines",
                    line={"color": _C_PV, "width": 1.5, "dash": "dot"},
                    hovertemplate="%{y:.1f} Wh<extra>PV</extra>",
                ),
                row=1,
                col=1,
            )

        # Calculate pretty ticks for energy axis (Wh) in row 1
        wh_values = (
            list(result.grid_import_wh_per_dt)
            + list(result.self_consumption_wh_per_dt)
            + (-np.asarray(result.feedin_wh_per_dt)).tolist()
        )

        if wh_values:
            wh_min, wh_max = min(wh_values), max(wh_values)
            if wh_min < wh_max:
                tick0_wh, dtick_wh, max_wh = _calc_pretty_ticks(wh_min, wh_max, target_nticks=6)
                fig.update_yaxes(
                    tickmode="linear",
                    tick0=tick0_wh,
                    dtick=dtick_wh,
                    range=[tick0_wh, max_wh],
                    row=1,
                    col=1,
                )

        fig.update_layout(barmode="relative")

        # ── row 2: mode background + SoC line + power bars ───────────
        if has_battery:
            if plan is None:
                raise ValueError("plan should not be None when has_battery is True")
            _add_mode_backgrounds(fig, timestamps, plan.modes, dt_hours, row=2, n_rows=2)

            # SoC is an end-of-interval state. Shift the curve one dt to the right,
            # while keeping an initial anchor at the first timestamp.
            soc_array = np.asarray(result.battery_soc_percentage_per_dt[inv_device_id])

            if len(soc_array) > 0:
                initial_soc = float(
                    result.battery_initial_soc_percentage.get(inv_device_id, soc_array[0])
                )
                soc_shifted = np.concatenate([[initial_soc], soc_array])
            else:
                soc_shifted = soc_array

            # Create shifted time axis: first point at t0 (initial anchor),
            # then end-of-slot states at t + dt.
            if timestamps:
                dt_td = timedelta(hours=dt_hours)
                timestamps_shifted = [timestamps[0]] + [ts + dt_td for ts in timestamps]
            else:
                timestamps_shifted = timestamps

            # SoC is a state variable → line (left axis)
            fig.add_trace(
                go.Scatter(
                    x=timestamps_shifted,
                    y=soc_shifted.tolist(),
                    name="SoC",
                    legend="legend2",
                    mode="lines",
                    line={"color": _C_SOC, "width": 2.2},
                    hovertemplate="%{y:.1f} %<extra>SoC</extra>",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig.update_yaxes(range=[-2, 107], row=2, col=1, secondary_y=False)

            # Per-slot power flows → bars (right axis)
            if np.any(plan.charge_ac_wh):
                fig.add_trace(
                    go.Bar(
                        x=timestamps,
                        y=(np.asarray(plan.charge_ac_wh) / dt_hours).tolist(),
                        name="AC Charge",
                        legend="legend2",
                        marker_color=_C_CHARGE_BAR,
                        opacity=0.85,
                        hovertemplate="%{y:.0f} W<extra>AC Charge</extra>",
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
                        name="Discharge",
                        legend="legend2",
                        marker_color=_C_DISCHARGE_BAR,
                        opacity=0.85,
                        hovertemplate="%{y:.0f} W<extra>Discharge</extra>",
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
                        name="PV→Bat",
                        legend="legend2",
                        marker_color=_C_PV_BAT_BAR,
                        opacity=0.85,
                        hovertemplate="%{y:.0f} W<extra>PV→Bat</extra>",
                    ),
                    row=2,
                    col=1,
                    secondary_y=True,
                )

            # Calculate pretty ticks for power axis (right, secondary_y)
            all_power = []
            if np.any(plan.charge_ac_wh):
                all_power.extend((np.asarray(plan.charge_ac_wh) / dt_hours).tolist())
            if np.any(plan.discharge_ac_wh):
                all_power.extend((np.asarray(plan.discharge_ac_wh) / dt_hours).tolist())
            if plan.pv_to_battery_wh is not None and np.any(plan.pv_to_battery_wh):
                all_power.extend((np.asarray(plan.pv_to_battery_wh) / dt_hours).tolist())

            if all_power:
                power_min, power_max = min(all_power), max(all_power)
                if power_min < power_max:
                    tick0_p, dtick_p, max_p = _calc_pretty_ticks(
                        power_min, power_max, target_nticks=5
                    )
                    fig.update_yaxes(
                        tickmode="linear",
                        tick0=tick0_p,
                        dtick=dtick_p,
                        range=[tick0_p, max_p],
                        row=2,
                        col=1,
                        secondary_y=True,
                    )
                else:
                    fig.update_yaxes(row=2, col=1, secondary_y=True)
            else:
                fig.update_yaxes(row=2, col=1, secondary_y=True)

        # Build layout with per-inverter-specific overrides (wider margins for secondary y-axis)
        layout_params = {
            **_LAYOUT,
            "margin": {"l": 45, "r": 45, "t": 60, "b": 40},
            "legend": {
                "orientation": "v",
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "right",
                "x": 0.99,
                "bgcolor": "rgba(255,255,255,0.75)",
                "bordercolor": "#ccc",
                "borderwidth": 1,
                "font": {"size": 10},
            },
            "legend2": {
                "orientation": "v",
                "yanchor": "top",
                "y": 0.56,
                "xanchor": "right",
                "x": 0.99,
                "bgcolor": "rgba(255,255,255,0.75)",
                "bordercolor": "#ccc",
                "borderwidth": 1,
                "font": {"size": 10},
            },
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
        fig.update_xaxes(
            tickformat="%d.%m.%y",
            showgrid=True,
            gridcolor="#e8e8e8",
            minor={
                "dtick": 3_600_000,
                "showgrid": True,
                "gridcolor": "rgba(200,200,200,0.35)",
                "ticks": "",
            },
        )
        fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
        return fig
