"""Unit tests for Parquet OHLCV storage."""

from pathlib import Path

import pandas as pd

from src.data.storage.parquet_store import ParquetStore


def test_parquet_store_write_and_read_round_trip(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    store = ParquetStore(base_path=tmp_path)
    store.write_ohlcv("SPY", sample_ohlcv)
    loaded = store.read_ohlcv("SPY")
    pd.testing.assert_frame_equal(
        loaded.sort_index(),
        sample_ohlcv.sort_index(),
        check_freq=False,
        check_dtype=False,
    )


def test_parquet_store_append_merges_by_index(tmp_path: Path) -> None:
    store = ParquetStore(base_path=tmp_path)
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    first = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100, 200, 300],
        },
        index=idx,
    )
    store.write_ohlcv("QQQ", first)
    more_idx = pd.date_range("2024-01-05", periods=2, freq="B")
    second = pd.DataFrame(
        {
            "open": [4.0, 5.0],
            "high": [4.1, 5.1],
            "low": [3.9, 4.9],
            "close": [4.05, 5.05],
            "volume": [400, 500],
        },
        index=more_idx,
    )
    store.append_ohlcv("QQQ", second)
    full = store.read_ohlcv("QQQ")
    assert len(full) == 5
    assert full.index.is_monotonic_increasing


def test_parquet_store_partitions_by_symbol(tmp_path: Path, sample_ohlcv: pd.DataFrame) -> None:
    store = ParquetStore(base_path=tmp_path)
    store.write_ohlcv("SPY", sample_ohlcv)
    store.write_ohlcv("IWM", sample_ohlcv * 1.01)
    spy_path = tmp_path / "SPY.parquet"
    iwm_path = tmp_path / "IWM.parquet"
    assert spy_path.is_file()
    assert iwm_path.is_file()
