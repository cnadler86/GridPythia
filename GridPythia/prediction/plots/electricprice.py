"""Plotly-based plotter for electric price providers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from GridPythia.prediction.plots._base import PALETTE, PredictionPlotter, add_forecast_region


class ElecPricePlotter(PredictionPlotter):
    """Render an electricity price series as a bar chart.

    The *forecast_from* parameter, when supplied, adds a light pastel
    background over the ML-predicted / statistical region so the viewer
    can distinguish real API data from predicted values.
    """

    y_label = "EUR / kWh"

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
            forecast_from: First timestamp that comes from the ML model / statistical
                           model rather than the live API.  A pastel shaded region
                           is drawn from this point to the end.
            title:         Plot title.
        """
        eur_kwh = np.asarray(values, dtype=float) * 1000.0  # EUR/Wh → EUR/kWh

        # Split into confirmed and predicted series for visual distinction.
        if forecast_from is not None:
            split_idx = next(
                (i for i, ts in enumerate(timestamps) if ts >= forecast_from),
                len(timestamps),
            )
        else:
            split_idx = len(timestamps)

        known_ts = timestamps[:split_idx]
        known_vals = eur_kwh[:split_idx].tolist()
        pred_ts = timestamps[split_idx:]
        pred_vals = eur_kwh[split_idx:].tolist()

        fig = go.Figure()

        if known_ts:
            fig.add_trace(
                go.Bar(
                    x=known_ts,
                    y=known_vals,
                    marker_color=PALETTE["blue"],
                    name="Price (real)",
                    hovertemplate="%{y:.4f} EUR/kWh<extra></extra>",
                )
            )

        if pred_ts:
            fig.add_trace(
                go.Bar(
                    x=pred_ts,
                    y=pred_vals,
                    marker_color="rgba(179, 157, 219, 0.85)",  # lavender for predicted
                    name="Price (forecast)",
                    hovertemplate="%{y:.4f} EUR/kWh<extra></extra>",
                )
            )

        if forecast_from is not None:
            add_forecast_region(fig, forecast_from, timestamps)

        self._apply_layout(fig, timestamps, title=title, yaxis_title="EUR / kWh")
        fig.update_layout(bargap=0.02, barmode="overlay")
        return fig
