"""Macro time series via FRED (fredapi)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fredapi import Fred

from src.data.adapters.base import MarketDataAdapter


class FredAdapter(MarketDataAdapter):
    """FRED-backed macro series (e.g. DGS10, UNRATE, CPIAUCSL)."""

    def __init__(self, api_key: str) -> None:
        self._fred = Fred(api_key=api_key)

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """FRED does not serve equity OHLCV."""
        return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        """No equity fundamentals from FRED."""
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        """Return a two-column frame: observation ``date``, ``value``."""
        series = self._fred.get_series(series_id, **kwargs)
        if series is None or len(series) == 0:
            return pd.DataFrame(columns=["date", "value"])
        frame = series.to_frame(name="value")
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        out = frame.reset_index()
        return out
