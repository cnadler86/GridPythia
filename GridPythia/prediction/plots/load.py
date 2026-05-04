"""Plotly-based plotter for load providers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from GridPythia.prediction.plots._base import PALETTE, apply_default_layout

# Distinct colours for individual appliance traces
_APPLIANCE_COLORS = [
    "#0277BD",  # blue
    "#00838F",  # teal
    "#6A1B9A",  # purple
    "#558B2F",  # olive
    "#AD1457",  # pink
    "#4E342E",  # brown
    "#37474F",  # slate
]


class LoadPlotter:
    """Render a household load series with optional appliance breakdowns."""

    def plot(
        self,
        values: np.ndarray,
        timestamps: list[datetime],
        *,
        title: str = "Load",
        appliance_load_by_id: dict[str, np.ndarray] | None = None,
    ) -> go.Figure:
        """Return a :class:`plotly.graph_objects.Figure` for *values*.

        Args:
            values:               Base household load in Wh (same length as *timestamps*).
            timestamps:           Time axis.
            title:                Plot title.
            appliance_load_by_id: Optional per-appliance load arrays (each same
                                  length as *timestamps*) rendered in distinct
                                  colours on top of the base load.
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
                name="Base load",
                hovertemplate="%{y:.1f} Wh<extra></extra>",
            )
        )

        if appliance_load_by_id:
            for i, (appliance_id, app_wh) in enumerate(appliance_load_by_id.items()):
                color = _APPLIANCE_COLORS[i % len(_APPLIANCE_COLORS)]
                # rgba fill derived from hex colour
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fill_color = f"rgba({r}, {g}, {b}, 0.20)"
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=np.asarray(app_wh, dtype=float).tolist(),
                        mode="lines",
                        line={"color": color, "width": 1.8, "shape": "hv", "dash": "dot"},
                        fill="tozeroy",
                        fillcolor=fill_color,
                        name=appliance_id,
                        hovertemplate="%{y:.1f} Wh<extra></extra>",
                    )
                )

        apply_default_layout(fig, title=title, xaxis_title="Time", yaxis_title="Wh")
        return fig
