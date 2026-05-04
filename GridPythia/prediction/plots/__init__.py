"""Plotly-based plotters for prediction providers.

Each domain (electricprice, feedintariff, load, pvforecast, weather) has exactly
one plotter class.  Concrete provider implementations delegate to the shared
domain plotter, so there is no per-provider duplication.

Usage::

    from GridPythia.prediction.plots import ElecPricePlotter

    fig = ElecPricePlotter().plot(values, timestamps, forecast_from=last_real_ts)
    fig.show()   # or fig.write_html("out.html")
"""

from GridPythia.prediction.plots._base import PredictionPlotter
from GridPythia.prediction.plots.electricprice import ElecPricePlotter
from GridPythia.prediction.plots.feedintariff import FeedInTariffPlotter
from GridPythia.prediction.plots.load import LoadPlotter
from GridPythia.prediction.plots.pvforecast import PVForecastPlotter
from GridPythia.prediction.plots.weather import WeatherPlotter

__all__ = [
    "PredictionPlotter",
    "ElecPricePlotter",
    "FeedInTariffPlotter",
    "LoadPlotter",
    "PVForecastPlotter",
    "WeatherPlotter",
]
