"""Integration: ingest date-range OHLCV (mocked) then run backtest engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, BacktestDataMissingError, BacktestEngine
from src.config import DataConfig, DCATarget, Settings
from src.data.adapters.base import MarketDataAdapter
from src.data.pipeline import DataPipeline
from src.data.storage.parquet_store import ParquetStore
from src.strategy.dca import DCAStrategy


def _ohlcv(idx: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 + np.linspace(0.0, 0.4 * len(idx), len(idx), dtype=float)
    high = close + 1.0
    low = close - 1.0
    open_ = np.r_[close[0], close[:-1]]
    vol = np.full(len(idx), 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


class RangeAdapter(MarketDataAdapter):
    """Returns a long daily series (no real network)."""

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        idx = pd.bdate_range("2023-01-03", periods=500, freq="B")
        return _ohlcv(idx)

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()


def test_full_ingest_to_backtest_pipeline(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="SPY", weight=1.0)],
        ),
    )
    pq = ParquetStore(Path(settings.data.parquet_dir))
    pipeline = DataPipeline(
        adapter=RangeAdapter(),
        parquet_store=pq,
        sqlite_store=None,
        initial_backfill_years=0,
    )
    res = pipeline.ingest_ohlcv("SPY", start="2023-01-01", end="2024-12-31")
    assert res.success is True
    loaded = pq.read_ohlcv("SPY")
    assert len(loaded) == 500

    strat = DCAStrategy(settings)
    engine = BacktestEngine()
    result = engine.run(
        strat,
        {"SPY": loaded},
        start=date(2023, 6, 1),
        end=date(2024, 6, 1),
        config=BacktestConfig(use_regime=False, rebalance_frequency="monthly"),
        settings=settings,
    )
    assert len(result.trades) >= 1
    assert len(result.equity_curve) >= 2


def test_backtest_fails_loudly_when_data_missing(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="SPY", weight=1.0)],
        ),
    )
    idx = pd.bdate_range("2023-01-03", periods=100, freq="B")
    df = _ohlcv(idx)
    strat = DCAStrategy(settings)
    engine = BacktestEngine()
    with pytest.raises(BacktestDataMissingError, match="No OHLCV data found"):
        engine.run(
            strat,
            {"SPY": df},
            start=date(2010, 1, 1),
            end=date(2010, 12, 31),
            config=BacktestConfig(use_regime=False),
            settings=settings,
        )
