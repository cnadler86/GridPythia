"""Plotly-based plotter for PV forecast providers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from GridPythia.prediction.plots._base import PALETTE, PredictionPlotter

_COLORS = [
    PALETTE["orange"],
    PALETTE["blue"],
    PALETTE["green"],
    PALETTE["purple"],
    PALETTE["teal"],
]


class PVForecastPlotter(PredictionPlotter):
    """Render per-inverter PV energy output as a bar chart with daily totals.

    When only one inverter is present the label is simply "PV".  When
    multiple inverters are present each trace is labelled with its key.
    """

    y_label = "Wh"

    def plot(
        self,
        series_by_inverter: dict[str, np.ndarray],
        timestamps: list[datetime],
        *,
        dt_hours: float = 0.25,
        title: str = "PV Forecast",
    ) -> go.Figure:
        """Return a :class:`plotly.graph_objects.Figure` for *series_by_inverter*.

        Args:
            series_by_inverter: Wh arrays keyed by inverter name.
            timestamps:         Time axis.
            dt_hours:           Time-step width in hours (used for kWh labels).
            title:              Plot title.
        """
        fig = go.Figure()
        multi = len(series_by_inverter) > 1

        for idx, (name, arr) in enumerate(series_by_inverter.items()):
            color = _COLORS[idx % len(_COLORS)]
            label = name if multi else "PV"
            wh = np.asarray(arr, dtype=float)
            fig.add_trace(
                go.Bar(
                    x=timestamps,
                    y=wh.tolist(),
                    customdata=[name] * len(wh),
                    name=label,
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate="%{customdata}: %{y:.1f} Wh<extra></extra>",
                )
            )

        # --- daily-total annotation line ---
        if timestamps:
            day_total: dict = {}
            for t, wh_val in zip(timestamps, next(iter(series_by_inverter.values())), strict=False):
                d = t.date()
                day_total[d] = day_total.get(d, 0.0) + float(wh_val)

            first_ts = timestamps[0]
            for d, total_wh in day_total.items():
                noon = datetime(d.year, d.month, d.day, 12, 0, tzinfo=first_ts.tzinfo)
                if noon < first_ts:
                    continue
                fig.add_annotation(
                    x=noon,
                    y=max(next(iter(series_by_inverter.values()))) * 1.05,
                    text=f"{total_wh / 1000:.2f} kWh",
                    showarrow=False,
                    font={"size": 9, "color": "#555"},
                )

        self._apply_layout(fig, timestamps, title=title, yaxis_title="Wh")
        fig.update_layout(barmode="stack")
        return fig
