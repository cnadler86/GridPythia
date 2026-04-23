"""Plotly-based plotter for feed-in tariff providers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from GridPythia.prediction.plots._base import PALETTE, apply_default_layout


class FeedInTariffPlotter:
    """Render a feed-in tariff series as a step chart."""

    def plot(
        self,
        values: np.ndarray,
        timestamps: list[datetime],
        *,
        title: str = "Feed-in Tariff",
    ) -> go.Figure:
        """Return a :class:`plotly.graph_objects.Figure` for *values*.

        Args:
            values:     EUR/Wh array (same length as *timestamps*).
            timestamps: Time axis.
            title:      Plot title.
        """
        eur_kwh = np.asarray(values, dtype=float) * 1000.0  # EUR/Wh → EUR/kWh

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=eur_kwh.tolist(),
                mode="lines",
                line={"color": PALETTE["green"], "width": 1.8, "shape": "hv"},
                fill="tozeroy",
                fillcolor="rgba(46, 125, 50, 0.10)",
                name="Feed-in Tariff",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.5f} EUR/kWh<extra></extra>",
            )
        )

        apply_default_layout(fig, title=title, xaxis_title="Time", yaxis_title="EUR / kWh")
        return fig
