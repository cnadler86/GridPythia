"""Electric-price provider interface."""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

import numpy as np
from structlog import get_logger

from GridPythia.prediction.base import PredictionProvider

logger = get_logger(__name__)


class ElecPriceProvider(PredictionProvider):
    """Returns electricity market price in EUR/Wh per time step."""

    @property
    def last_real_ts(self) -> "datetime | None":
        """Timestamp of the last real (non-predicted) data point.

        Returns ``None`` by default.  Subclasses override this to expose the
        boundary between live API data and statistical / ML predictions so that
        the UI can shade the forecast region.
        """
        return None

    @abstractmethod
    async def fetch(self, timestamps: list) -> np.ndarray:
        """Return float32 ndarray of EUR/Wh, same length as *timestamps*."""
        ...


class ElecPriceFallbackChain(ElecPriceProvider):
    """Decorator provider that fails over to a replacement provider.

    The primary provider is attempted first. If it raises, the fallback
    provider is queried for the same timestamps.
    """

    def __init__(self, primary: ElecPriceProvider, fallback: ElecPriceProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def provider_id(self) -> str:
        return f"{self._primary.provider_id}|fallback:{self._fallback.provider_id}"

    @property
    def last_real_ts(self) -> datetime | None:
        """Return the primary provider's last real timestamp when available, otherwise fall back to the fallback provider's value."""
        return self._primary.last_real_ts or self._fallback.last_real_ts

    async def fetch(self, timestamps: list) -> np.ndarray:
        try:
            return await self._primary.fetch(timestamps)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "electricprice_primary_failed_using_fallback",
                primary_provider=self._primary.provider_id,
                fallback_provider=self._fallback.provider_id,
                error=str(exc),
            )
            return await self._fallback.fetch(timestamps)
