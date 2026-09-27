"""Pluggable market data source adapters."""

from src.data.adapters.base import MarketDataAdapter
from src.data.adapters.fred_adapter import FredAdapter
from src.data.adapters.yfinance_adapter import YFinanceAdapter

__all__ = ["FredAdapter", "MarketDataAdapter", "YFinanceAdapter"]
