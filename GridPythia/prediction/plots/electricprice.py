"""Plotly-based plotter for electric price providers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from GridPythia.prediction.plots._base import PALETTE, add_forecast_region, apply_default_layout


class ElecPricePlotter:
    """Render an electricity price series as a step chart.

    The *forecast_from* parameter, when supplied, adds a light pastel
    background over the synthesised (ETS / statistical) region so the
    viewer can distinguish real API data from model-extended values.
    """

    def plot(
        self,
        values: np.ndarray,
        timestamps: list[datetime],
        *,
        forecast_from: datetime | None = None,
        title: str = "Electricity Price",
    ) -> go.Figure:
        """Return a :class:`plotly.graph_objects.Figure` for *values*.

        Args:
            values:        EUR/Wh array (same length as *timestamps*).
            timestamps:    Time axis.
            forecast_from: First timestamp that comes from the statistical
                           model rather than the live API.  A pastel shaded
                           region is drawn from this point to the end.
            title:         Plot title.
        """
        eur_kwh = np.asarray(values, dtype=float) * 1000.0  # EUR/Wh → EUR/kWh

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=eur_kwh.tolist(),
                mode="lines",
                line={"color": PALETTE["blue"], "width": 1.8, "shape": "hv"},
                fill="tozeroy",
                # Keep the normal area blue; only the forecast window is shaded lavender.
                fillcolor="rgba(21, 101, 192, 0.10)",
                name="Price",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.5f} EUR/kWh<extra></extra>",
            )
        )

        if forecast_from is not None:
            add_forecast_region(fig, forecast_from, timestamps)

        apply_default_layout(fig, title=title, xaxis_title="Time", yaxis_title="EUR / kWh")
        return fig
