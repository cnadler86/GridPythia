"""Terminal battery SoC valuation strategies."""

from enum import Enum


class TerminalSoCMode(str, Enum):
    """Choose how terminal battery SoC is valued by the optimizer."""

    # Dynamic tariffs retain the legacy end-of-horizon electricity-price reward.
    PRICE_BASED = "price_based"
    # Fixed tariffs use a configured percentage target and penalty instead.
    TARGET = "target"
