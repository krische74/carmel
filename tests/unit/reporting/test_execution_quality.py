"""Execution quality vs Parquet reference close."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.data.storage.parquet_store import ParquetStore
from src.reporting.execution_quality import (
    reference_price_for_execution,
    slippage_bps,
)


def test_slippage_bps_buy_adverse_when_fill_above_ref() -> None:
    assert slippage_bps("buy", 101.0, 100.0) == pytest.approx(100.0)


def test_slippage_bps_buy_favorable_when_fill_below_ref() -> None:
    assert slippage_bps("buy", 99.0, 100.0) == pytest.approx(-100.0)


def test_slippage_bps_sell_adverse_when_fill_below_ref() -> None:
    assert slippage_bps("sell", 99.0, 100.0) == pytest.approx(100.0)


def test_slippage_bps_sell_favorable_when_fill_above_ref() -> None:
    assert slippage_bps("sell", 101.0, 100.0) == pytest.approx(-100.0)


def test_slippage_bps_none_when_bad_prices() -> None:
    assert slippage_bps("buy", 100.0, 0.0) is None
    assert slippage_bps("buy", 0.0, 100.0) is None


def test_reference_price_no_bar_after_as_of(tmp_path) -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [10.0, 20.0, 30.0], "volume": 1.0},
        index=idx,
    )
    pq = ParquetStore(tmp_path)
    pq.write_ohlcv("SPY", df)
    ts = datetime(2024, 1, 1, 16, 0, tzinfo=UTC)
    assert reference_price_for_execution("SPY", ts, pq) is None


def test_reference_price_uses_last_bar_on_or_before(tmp_path) -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [10.0, 20.0, 30.0], "volume": 1.0},
        index=idx,
    )
    pq = ParquetStore(tmp_path)
    pq.write_ohlcv("SPY", df)
    ts = datetime(2024, 1, 4, 16, 0, tzinfo=UTC)
    assert reference_price_for_execution("SPY", ts, pq) == pytest.approx(30.0)
