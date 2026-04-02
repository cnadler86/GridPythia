"""Weather provider interface."""

from abc import abstractmethod

import numpy as np

from GridPythia.prediction.base import PredictionProvider


class WeatherProvider(PredictionProvider):
    """Returns multi-channel weather data for a time window.

    The returned ``dict`` maps channel name to a ``float32`` ndarray.  The two
    mandatory channels are ``temperature_c`` and ``cloud_cover_pct``;
    optional ones (``wind_speed_kmh``, ``humidity_pct``, ``precipitation_mm``,
    ``pressure_hpa``, ``ghi_wm2``, ``dni_wm2``, ``dhi_wm2``) are only present
    when the data source provides them.
    """

    @abstractmethod
    async def fetch(self, timestamps: list) -> dict[str, np.ndarray]:
        """Return a dict mapping channel name to float32 ndarray (one value per timestamp)."""
        ...
