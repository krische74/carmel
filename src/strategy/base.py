"""Abstract strategy interface — consumes storage-shaped OHLCV, emits ``Signal`` objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd

    from src.data.regime import MarketRegime
    from src.models import Signal


class Strategy(ABC):
    """Strategy implementations read pre-fetched OHLCV (no live API calls)."""

    @abstractmethod
    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: MarketRegime | None = None,
    ) -> list[Signal]:
        """Return zero or more signals using only ``data`` up to ``as_of`` (inclusive).

        ``market_regime`` is optional; strategies that are regime-aware use it when set.
        """

    @abstractmethod
    def get_universe(self) -> list[str]:
        """Symbols this strategy needs OHLCV for (uppercase tickers)."""
