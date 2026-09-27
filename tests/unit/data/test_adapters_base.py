"""Unit tests for MarketDataAdapter ABC."""

import pytest

from src.data.adapters.base import MarketDataAdapter


def test_market_data_adapter_cannot_be_instantiated_without_implementations() -> None:
    """Abstract methods must be implemented by concrete adapters."""

    class IncompleteAdapter(MarketDataAdapter):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteAdapter()  # type: ignore[misc]


def test_market_data_adapter_concrete_subclass_is_instantiable() -> None:
    """A fully implemented adapter can be constructed."""

    class FullAdapter(MarketDataAdapter):
        def fetch_ohlcv(self, symbol: str, **kwargs):
            raise NotImplementedError

        def fetch_fundamentals(self, symbol: str, **kwargs):
            raise NotImplementedError

        def fetch_macro(self, series_id: str, **kwargs):
            raise NotImplementedError

    adapter = FullAdapter()
    assert isinstance(adapter, MarketDataAdapter)
