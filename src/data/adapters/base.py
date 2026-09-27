"""Abstract base for market data adapters."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class MarketDataAdapter(ABC):
    """Contract for fetching OHLCV, fundamentals, and macro series.

    Downstream code depends on this interface, not on concrete providers.
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return OHLCV bars indexed by datetime."""

    @abstractmethod
    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        """Return provider-specific fundamental fields for ``symbol``."""

    @abstractmethod
    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        """Return a macro time series indexed by observation date."""
