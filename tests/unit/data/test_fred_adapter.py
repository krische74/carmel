"""Unit tests for FredAdapter (fredapi mocked)."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.adapters.fred_adapter import FredAdapter


def test_fred_adapter_fetch_macro_returns_series_dataframe() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="ME")
    series = pd.Series([1.5, 1.6, 1.4], index=idx)
    mock_fred = MagicMock()
    mock_fred.get_series.return_value = series

    with patch("src.data.adapters.fred_adapter.Fred", return_value=mock_fred):
        adapter = FredAdapter(api_key="test-key")
        out = adapter.fetch_macro("DGS10")

    assert "value" in out.columns
    assert len(out) == 3
    mock_fred.get_series.assert_called_once()


def test_fred_adapter_fetch_ohlcv_returns_empty() -> None:
    with patch("src.data.adapters.fred_adapter.Fred"):
        adapter = FredAdapter(api_key="k")
        out = adapter.fetch_ohlcv("SPY")
        assert out.empty


def test_fred_adapter_fetch_fundamentals_returns_empty_dict() -> None:
    with patch("src.data.adapters.fred_adapter.Fred"):
        adapter = FredAdapter(api_key="k")
        assert adapter.fetch_fundamentals("SPY") == {}
