"""Integration: cold-start path with mocked adapters (no network)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.config import DataConfig, DCATarget, Settings
from src.data.adapters.base import MarketDataAdapter
from src.data.pipeline import DataPipeline
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.execution.order_manager import OrderManager
from src.portfolio.tax_lots import LotLedger
from src.risk.kill_switch import KillSwitch
from src.strategy.dca import DCAStrategy


def _shv_like_ohlcv(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1e-4, n))
    if n > 15:
        close[15] = float(close[14]) * 0.92
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0001
    low = np.minimum(open_, close) * 0.9999
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


class FreshInstallAdapter(MarketDataAdapter):
    """SHV-like OHLCV for any symbol."""

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        return _shv_like_ohlcv(30)

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-08"]),
                "value": [2.1, float("nan"), 2.15],
            },
        )


@pytest.fixture()
def fresh_settings(tmp_path: Path) -> Settings:
    return Settings(
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        _env_file=None,
    )


def test_fresh_install_cycle_completes_without_errors(
    fresh_settings: Settings,
    tmp_path: Path,
) -> None:
    from src.automation.workflows import TradingWorkflow

    fresh_settings.regime.enabled = False

    parquet = ParquetStore(Path(fresh_settings.data.parquet_dir))
    meta_path = tmp_path / "metadata.sqlite"
    lots_path = tmp_path / "lots.sqlite"
    sqlite = SQLiteStore(meta_path)
    lot_ledger = LotLedger(lots_path)
    dcfg = fresh_settings.data
    pipeline = DataPipeline(
        adapter=FreshInstallAdapter(),
        parquet_store=parquet,
        sqlite_store=sqlite,
        fred_adapter=FreshInstallAdapter(),
        ohlcv_outlier_zscore_threshold=dcfg.ohlcv_outlier_zscore_threshold,
        ohlcv_min_prior_for_zscore=dcfg.ohlcv_min_prior_for_zscore,
        ohlcv_max_abs_daily_return=dcfg.ohlcv_max_abs_daily_return,
    )
    assert pipeline.ingest_macro("DGS10").success is True

    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 25_000.0
    broker.get_position_qty.return_value = 0.0
    broker.submit_market_order.return_value = "ord-fresh-1"
    broker.get_order_fill_price.return_value = 100.25
    broker.list_recent_orders.return_value = [
        {
            "order_id": "ord-fresh-1",
            "symbol": "VOO",
            "side": "buy",
            "qty": None,
            "filled_qty": None,
            "filled_avg_price": 100.25,
            "status": "filled",
        },
    ]

    kill_switch = KillSwitch(fresh_settings.risk.daily_loss_limit_pct)
    order_manager = OrderManager(
        broker=broker,
        settings=fresh_settings,
        kill_switch=kill_switch,
        sqlite_store=sqlite,
    )
    wf = TradingWorkflow(
        settings=fresh_settings,
        data_pipeline=pipeline,
        parquet_store=parquet,
        strategies=[DCAStrategy(fresh_settings)],
        order_manager=order_manager,
        broker=broker,
        sqlite_store=sqlite,
        lot_ledger=lot_ledger,
        kill_switch=kill_switch,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    result = wf.run_cycle(as_of=monday)

    assert any(r.success for _, r in result.ingest_results)
    assert len(result.signals) >= 1
    pq_files = list(Path(fresh_settings.data.parquet_dir).glob("*.parquet"))
    assert pq_files, "expected Parquet files after ingest"
    assert sqlite.get_equity_snapshots()


def _write_pre_tier30_tax_lots_schema(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tax_lots_open (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            cost_per_share REAL NOT NULL,
            opened_at TEXT NOT NULL
        );
        CREATE TABLE tax_lots_closed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lot_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            cost_per_share REAL NOT NULL,
            sell_price REAL NOT NULL,
            opened_at TEXT NOT NULL,
            closed_at TEXT NOT NULL,
            realized_pnl REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tax_open_symbol ON tax_lots_open(symbol);
        CREATE INDEX IF NOT EXISTS idx_tax_closed_symbol ON tax_lots_closed(symbol);
        """
    )
    conn.commit()
    conn.close()


def test_pre_tier30_database_migration_succeeds(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite"
    _write_pre_tier30_tax_lots_schema(db)
    LotLedger(db)
    with sqlite3.connect(db) as conn:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(tax_lots_open)").fetchall()}
        assert "account_id" in cols
