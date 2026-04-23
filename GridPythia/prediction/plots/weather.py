"""Plotly-based plotter for weather providers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from GridPythia.prediction.plots._base import PALETTE, apply_default_layout

_CHANNEL_COLORS = {
    "temperature_c": PALETTE["orange"],
    "humidity_pct": PALETTE["teal"],
    "cloud_cover_pct": PALETTE["slate"],
    "wind_speed_kmh": PALETTE["blue"],
    "precipitation_mm": PALETTE["purple"],
    "pressure_hpa": PALETTE["brown"],
    "ghi_wm2": PALETTE["pink"],
    "dni_wm2": PALETTE["olive"],
    "dhi_wm2": PALETTE["green"],
}

_CHANNEL_YLABELS: dict[str, str] = {
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

_CHANNEL_TITLES: dict[str, str] = {
    "temperature_c": "Temperature",
    "humidity_pct": "Humidity",
    "cloud_cover_pct": "Cloud Cover",
    "wind_speed_kmh": "Wind Speed",
    "precipitation_mm": "Precipitation",
    "pressure_hpa": "Pressure",
    "ghi_wm2": "GHI",
    "dni_wm2": "DNI",
    "dhi_wm2": "DHI",
}


class WeatherPlotter:
    """Render multi-channel weather data.

    When a single channel is selected (via *channels*) a simple single-axis
    chart is returned.  When multiple channels are requested the figure uses
    stacked sub-plots, one per channel.
    """

    def plot(
        self,
        weather_by_channel: dict[str, np.ndarray],
        timestamps: list[datetime],
        *,
        channels: list[str] | None = None,
        title: str = "Weather",
    ) -> go.Figure:
        """Return a :class:`plotly.graph_objects.Figure`.

        Args:
            weather_by_channel: Channel-name → float32 array mapping.
            timestamps:         Time axis.
            channels:           Subset of channels to display.  ``None`` = all.
            title:              Plot title.
        """
        available = {k: v for k, v in weather_by_channel.items() if len(v) > 0}
        if channels is not None:
            available = {k: v for k, v in available.items() if k in channels}

        if not available:
            fig = go.Figure()
            apply_default_layout(fig, title=title)
            return fig

        keys = list(available.keys())
        n = len(keys)

        if n == 1:
            key = keys[0]
            arr = np.asarray(available[key], dtype=float)
            color = _CHANNEL_COLORS.get(key, PALETTE["blue"])
            ylabel = _CHANNEL_YLABELS.get(key, "")
            chan_title = _CHANNEL_TITLES.get(key, key)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=arr.tolist(),
                    mode="lines",
                    line={"color": color, "width": 1.8},
                    name=chan_title,
                    hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.2f}} {ylabel}<extra>{chan_title}</extra>",
                )
            )
            apply_default_layout(
                fig, title=f"{title} – {chan_title}", xaxis_title="Time", yaxis_title=ylabel
            )
            return fig

        row_titles = [_CHANNEL_TITLES.get(k, k) for k in keys]
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            subplot_titles=row_titles,
            vertical_spacing=0.06,
        )
        for row_idx, key in enumerate(keys, start=1):
            arr = np.asarray(available[key], dtype=float)
            color = _CHANNEL_COLORS.get(key, PALETTE["blue"])
            ylabel = _CHANNEL_YLABELS.get(key, "")
            chan_title = _CHANNEL_TITLES.get(key, key)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=arr.tolist(),
                    mode="lines",
                    line={"color": color, "width": 1.5},
                    name=chan_title,
                    hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.2f}} {ylabel}<extra>{chan_title}</extra>",
                ),
                row=row_idx,
                col=1,
            )
            fig.update_yaxes(title_text=ylabel, row=row_idx, col=1)

        fig.update_layout(
            title=title,
            template="plotly_white",
            margin={"l": 60, "r": 20, "t": 60, "b": 40},
            hovermode="x unified",
            showlegend=False,
            height=max(300, 200 * n),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
        return fig
