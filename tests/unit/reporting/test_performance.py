"""Tests for performance summaries and benchmark equity curve."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.reporting.performance import (
    PerformanceSummary,
    compute_equity_curve,
    compute_performance_summary,
)


def test_compute_performance_summary_counts_trades() -> None:
    executions = [
        {
            "cycle_id": "a",
            "timestamp": "2026-01-01T12:00:00+00:00",
            "symbol": "SPY",
            "submitted": 1,
            "order_id": "1",
            "reason": None,
            "side": "buy",
        },
        {
            "cycle_id": "a",
            "timestamp": "2026-01-02T12:00:00+00:00",
            "symbol": "SPY",
            "submitted": 1,
            "order_id": "2",
            "reason": None,
            "side": "sell",
        },
        {
            "cycle_id": "b",
            "timestamp": "2026-01-03T12:00:00+00:00",
            "symbol": "QQQ",
            "submitted": 1,
            "order_id": "3",
            "reason": None,
            "side": "buy",
        },
        {
            "cycle_id": "b",
            "timestamp": "2026-01-04T12:00:00+00:00",
            "symbol": "QQQ",
            "submitted": 0,
            "order_id": None,
            "reason": "blocked",
            "side": "buy",
        },
        {
            "cycle_id": "c",
            "timestamp": "2026-01-05T12:00:00+00:00",
            "symbol": "VOO",
            "submitted": 1,
            "order_id": "4",
            "reason": None,
            "side": "sell",
        },
    ]
    s = compute_performance_summary(executions)
    assert isinstance(s, PerformanceSummary)
    assert s.total_trades == 5
    assert s.buys == 3
    assert s.sells == 2
    assert s.submitted_count == 4
    assert s.rejected_count == 1
    assert s.first_trade == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert s.last_trade == datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def test_compute_performance_summary_empty() -> None:
    s = compute_performance_summary([])
    assert s.total_trades == 0
    assert s.first_trade is None
    assert s.last_trade is None


def test_compute_equity_curve_returns_dataframe(tmp_path: Path) -> None:
    from src.data.storage.parquet_store import ParquetStore

    store = ParquetStore(tmp_path / "pq")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1e6, 1e6],
        },
        index=pd.date_range("2026-01-02", periods=2, freq="B"),
    )
    store.write_ohlcv("SPY", df)

    curve = compute_equity_curve(store, "SPY")
    assert list(curve.columns) == ["date", "close"]
    assert len(curve) == 2
    assert float(curve["close"].iloc[-1]) == 101.5
