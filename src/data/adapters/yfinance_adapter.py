"""Daily OHLCV for equities/ETFs via yfinance with retry/backoff."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import pandas as pd
import yfinance as yf

from src.data.adapters.base import MarketDataAdapter

logger = logging.getLogger(__name__)

_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Detect Yahoo / CDN throttling (HTTP 429, message variants)."""
    msg = str(exc).lower()
    return "too many requests" in msg or "429" in msg or "rate limit" in msg


class YFinanceAdapter(MarketDataAdapter):
    """Yahoo Finance-backed adapter for OHLCV and lightweight fundamentals."""

    def __init__(
        self,
        *,
        max_retries: int = 6,
        retry_backoff_seconds: float = 2.0,
        max_backoff_seconds: float = 120.0,
    ) -> None:
        self._max_retries = max(1, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._max_backoff_seconds = max(1.0, float(max_backoff_seconds))

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Download daily (or other) bars; maps columns to lowercase OHLCV."""
        sym = symbol.strip().upper()
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                ticker = yf.Ticker(sym)
                # Tier 53A / Q014: auto_adjust=True so Close is total-return
                # (dividends reinvested). auto_adjust=False stored nominal Close
                # and omitted distributions — understating TLT/SHV more than SPY.
                hist = ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    **kwargs,
                )
                if hist is None or hist.empty:
                    return pd.DataFrame(columns=list(_COLUMN_MAP.values()))
                frame = hist.rename(columns=_COLUMN_MAP)
                for col in ("open", "high", "low", "close", "volume"):
                    if col not in frame.columns:
                        msg = f"yfinance history missing column {col!r} for {sym}"
                        raise ValueError(msg)
                frame = frame[list(_COLUMN_MAP.values())].astype(float)
                # auto_adjust can leave close 1 ULP outside [low, high]; clamp so
                # validation stays meaningful without rejecting total-return series.
                frame["high"] = frame[["open", "high", "low", "close"]].max(axis=1)
                frame["low"] = frame[["open", "high", "low", "close"]].min(axis=1)
                if not isinstance(frame.index, pd.DatetimeIndex):
                    frame.index = pd.to_datetime(frame.index)
                frame.index.name = None
                return frame.sort_index()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "yfinance fetch failed for %s (attempt %s/%s): %s",
                    sym,
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt >= self._max_retries:
                    break
                if _is_rate_limit_error(exc):
                    base = self._retry_backoff_seconds * (2.0 ** (attempt - 1))
                    jitter = random.uniform(0.0, min(3.0, base * 0.1))
                    delay = min(self._max_backoff_seconds, base + jitter)
                else:
                    delay = self._retry_backoff_seconds * attempt
                time.sleep(delay)
        msg = f"yfinance failed for {sym} after {self._max_retries} attempts"
        raise RuntimeError(msg) from last_error

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        """Return ``Ticker.info`` (sparse; fields vary by instrument)."""
        sym = symbol.strip().upper()
        ticker = yf.Ticker(sym)
        info = getattr(ticker, "info", None)
        if isinstance(info, dict):
            return dict(info)
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        """Macro series are not provided by Yahoo Finance in this adapter."""
        return pd.DataFrame()
