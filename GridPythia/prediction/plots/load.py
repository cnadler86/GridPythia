"""Plotly-based plotter for load providers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from GridPythia.prediction.plots._base import PALETTE, apply_default_layout


class LoadPlotter:
    """Render a household load series."""

    def plot(
        self,
        values: np.ndarray,
        timestamps: list[datetime],
        *,
        title: str = "Load",
    ) -> go.Figure:
        """Return a :class:`plotly.graph_objects.Figure` for *values*.

        Args:
            values:     Wh array (same length as *timestamps*).
            timestamps: Time axis.
            title:      Plot title.
        """
        wh = np.asarray(values, dtype=float)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=wh.tolist(),
                mode="lines",
                line={"color": PALETTE["orange"], "width": 1.8, "shape": "hv"},
                fill="tozeroy",
                fillcolor="rgba(230, 81, 0, 0.10)",
                name="Load",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} Wh<extra></extra>",
            )
        )

        apply_default_layout(fig, title=title, xaxis_title="Time", yaxis_title="Wh")
        return fig
