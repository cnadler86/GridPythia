"""Electric-price provider interface."""

from abc import abstractmethod

import polars as pl
from structlog import get_logger

from GridPythia.prediction.base import PredictionProvider

logger = get_logger(__name__)


class ElecPriceProvider(PredictionProvider):
    """Returns electricity market price in EUR/Wh per time step."""

    @abstractmethod
    async def fetch(self, timestamps: pl.Series) -> pl.Series:
        """Return Float32 Series of EUR/Wh, same length as *timestamps*."""
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

    async def fetch(self, timestamps: pl.Series) -> pl.Series:
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
