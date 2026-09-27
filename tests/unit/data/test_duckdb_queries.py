"""Unit tests for DuckDB analytical queries over Parquet OHLCV."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.storage.duckdb_queries import DuckDBQueries
from src.data.storage.parquet_store import ParquetStore


@pytest.fixture()
def multi_symbol_parquet(tmp_path: Path) -> Path:
    store = ParquetStore(base_path=tmp_path)
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    spy = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5, 101.0, 101.5, 102.0, 102.5],
            "volume": [1e6] * 5,
        },
        index=idx,
    )
    qqq = spy * 1.1
    store.write_ohlcv("SPY", spy)
    store.write_ohlcv("QQQ", qqq)
    return tmp_path


def test_duckdb_queries_date_range_and_multi_symbol(multi_symbol_parquet: Path) -> None:
    dq = DuckDBQueries(parquet_dir=multi_symbol_parquet)
    start = pd.Timestamp("2024-01-03")
    end = pd.Timestamp("2024-01-05")
    df = dq.query_ohlcv(symbols=["SPY", "QQQ"], start=start, end=end)
    assert set(df["symbol"].unique()) == {"SPY", "QQQ"}
    assert df["date"].min() >= start.normalize()
    assert df["date"].max() <= end.normalize()


def test_duckdb_queries_single_symbol(multi_symbol_parquet: Path) -> None:
    dq = DuckDBQueries(parquet_dir=multi_symbol_parquet)
    df = dq.query_ohlcv(symbols=["SPY"], start=None, end=None)
    assert df["symbol"].unique().tolist() == ["SPY"]
    assert len(df) == 5
