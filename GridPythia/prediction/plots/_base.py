"""Shared types and utilities for prediction plotters."""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Colour palette shared across all plotters (Plotly-compatible hex strings).
PALETTE = {
    "blue": "#1565C0",
    "green": "#2E7D32",
    "teal": "#00838F",
    "purple": "#6A1B9A",
    "orange": "#E65100",
    "pink": "#AD1457",
    "slate": "#37474F",
    "olive": "#558B2F",
    "brown": "#4E342E",
}

# Pastel fill used to mark forecast / fallback regions.
FORECAST_FILL_COLOR = "rgba(255, 200, 100, 0.18)"
FORECAST_LINE_COLOR = "rgba(200, 140, 40, 0.55)"

_LAYOUT_DEFAULTS: dict[str, Any] = {
    "template": "plotly_white",
    "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
    "hovermode": "x unified",
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
}


def apply_default_layout(
    fig: Any, *, title: str = "", xaxis_title: str = "", yaxis_title: str = ""
) -> None:
    """Apply consistent layout settings to *fig* in-place."""
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=title or None,
        xaxis_title=xaxis_title or None,
        yaxis_title=yaxis_title or None,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")


def add_forecast_region(fig: Any, forecast_from: datetime, timestamps: list[datetime]) -> None:
    """Add a light pastel background rectangle covering the forecast period.

    The shaded region starts at *forecast_from* and extends to the last
    timestamp.  Nothing is drawn when *forecast_from* is after the last
    timestamp (no forecasted values in this window).
    """
    if not timestamps:
        return
    last_ts = timestamps[-1]
    if forecast_from >= last_ts:
        return
    fig.add_vrect(
        x0=forecast_from,
        x1=last_ts,
        fillcolor=FORECAST_FILL_COLOR,
        line={"color": FORECAST_LINE_COLOR, "width": 1, "dash": "dot"},
        annotation_text="Prognose",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#9e7520",
        layer="below",
    )
