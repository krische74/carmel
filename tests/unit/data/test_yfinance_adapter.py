"""Unit tests for YFinanceAdapter (external calls mocked)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.adapters.yfinance_adapter import YFinanceAdapter


def test_yfinance_adapter_fetch_ohlcv_maps_columns_and_index() -> None:
    raw = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.5],
            "Close": [10.5],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
    )
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = raw

    with patch("src.data.adapters.yfinance_adapter.yf.Ticker", return_value=mock_ticker):
        adapter = YFinanceAdapter()
        out = adapter.fetch_ohlcv("SPY", interval="1d")

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(out.index, pd.DatetimeIndex)
    mock_ticker.history.assert_called_once()
    # Tier 53A: total-return series (dividends reinvested), not nominal Close.
    kwargs = mock_ticker.history.call_args.kwargs
    assert kwargs.get("auto_adjust") is True


def test_yfinance_adapter_requests_auto_adjust_true_for_total_return() -> None:
    """Q014: auto_adjust=False stored nominal Close and omitted dividends."""
    raw = pd.DataFrame(
        {
            "Open": [98.0],
            "High": [99.0],
            "Low": [97.0],
            "Close": [98.5],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2007-01-03")]),
    )
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = raw

    with patch("src.data.adapters.yfinance_adapter.yf.Ticker", return_value=mock_ticker):
        YFinanceAdapter().fetch_ohlcv("SPY", start="2007-01-01", end="2007-01-10")

    assert mock_ticker.history.call_args.kwargs["auto_adjust"] is True


def test_yfinance_adapter_exponential_backoff_on_repeated_rate_limits() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [1],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
    )
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = [
        Exception("Too Many Requests"),
        Exception("Too Many Requests"),
        Exception("429 Too Many Requests"),
        raw,
    ]
    sleeps: list[float] = []

    with (
        patch("src.data.adapters.yfinance_adapter.yf.Ticker", return_value=mock_ticker),
        patch("src.data.adapters.yfinance_adapter.time.sleep", side_effect=lambda s: sleeps.append(float(s))),
        patch("src.data.adapters.yfinance_adapter.random.uniform", return_value=0.0),
    ):
        adapter = YFinanceAdapter(
            max_retries=4,
            retry_backoff_seconds=2.0,
            max_backoff_seconds=60.0,
        )
        out = adapter.fetch_ohlcv("QQQ")

    assert len(out) == 1
    assert mock_ticker.history.call_count == 4
    assert len(sleeps) == 3
    assert sleeps[0] == pytest.approx(2.0)
    assert sleeps[1] == pytest.approx(4.0)
    assert sleeps[2] == pytest.approx(8.0)


def test_yfinance_adapter_retries_on_rate_limit_then_succeeds() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [1],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
    )
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = [
        Exception("Too Many Requests"),
        raw,
    ]

    with (
        patch("src.data.adapters.yfinance_adapter.yf.Ticker", return_value=mock_ticker),
        patch("src.data.adapters.yfinance_adapter.time.sleep", return_value=None),
    ):
        adapter = YFinanceAdapter(max_retries=2, retry_backoff_seconds=0.01)
        out = adapter.fetch_ohlcv("QQQ")

    assert len(out) == 1
    assert mock_ticker.history.call_count == 2


def test_yfinance_adapter_fetch_fundamentals_returns_dict() -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {"symbol": "SPY", "shortName": "S&P 500"}

    with patch("src.data.adapters.yfinance_adapter.yf.Ticker", return_value=mock_ticker):
        adapter = YFinanceAdapter()
        info = adapter.fetch_fundamentals("SPY")

    assert info["symbol"] == "SPY"


def test_yfinance_adapter_fetch_macro_returns_empty_frame() -> None:
    adapter = YFinanceAdapter()
    out = adapter.fetch_macro("DGS10")
    assert isinstance(out, pd.DataFrame)
    assert out.empty
