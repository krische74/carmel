"""Integration tests: real storage and pipeline wiring with mocked broker and adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
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


def _smooth_ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100.0, 101.5, n), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


class _StubAdapter(MarketDataAdapter):
    """Deterministic OHLCV; optional failing symbols for ingest tests."""

    def __init__(self, fail_symbols: set[str] | None = None) -> None:
        self._fail = {s.strip().upper() for s in (fail_symbols or set())}

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: object,
    ) -> pd.DataFrame:
        sym = symbol.strip().upper()
        if sym in self._fail:
            raise RuntimeError("simulated fetch failure")
        return _smooth_ohlcv()

    def fetch_fundamentals(self, symbol: str, **kwargs: object) -> dict[str, object]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame()


@pytest.fixture()
def integration_settings(tmp_path: Path) -> Settings:
    """Isolated parquet/SQLite under tmp_path."""
    return Settings(
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        _env_file=None,
    )


def test_full_trading_cycle_end_to_end(integration_settings: Settings, tmp_path: Path) -> None:
    """Data → strategy → execution → lots → snapshot → alerts (clean cycle) in one pass."""
    from src.automation.workflows import TradingWorkflow

    integration_settings.regime.enabled = False
    integration_settings.cash_sweep.enabled = False

    parquet = ParquetStore(Path(integration_settings.data.parquet_dir))
    db_path = tmp_path / "cache" / "hub_metadata.sqlite"
    sqlite = SQLiteStore(db_path)
    lot_ledger = LotLedger(db_path)
    adapter = _StubAdapter()
    pipeline = DataPipeline(adapter=adapter, parquet_store=parquet, sqlite_store=sqlite)

    broker = MagicMock()
    broker.refresh_account = MagicMock()
    cash_box = [25_000.0]
    positions: dict[str, float] = {}
    fill_px = 100.25

    def _equity() -> float:
        return cash_box[0] + sum(q * fill_px for q in positions.values())

    def _submit(symbol: str, qty: float, side: str = "buy", **_kwargs: object) -> str:
        sym = symbol.strip().upper()
        q = float(qty)
        if side == "buy":
            positions[sym] = positions.get(sym, 0.0) + q
            cash_box[0] = max(0.0, cash_box[0] - q * fill_px)
        else:
            positions[sym] = positions.get(sym, 0.0) - q
            cash_box[0] += q * fill_px
        return "ord-int-1"

    broker.get_account_equity.side_effect = _equity
    broker.get_last_equity.side_effect = _equity
    broker.get_cash.side_effect = lambda: cash_box[0]
    broker.get_position_qty.side_effect = lambda s: float(
        positions.get(str(s).strip().upper(), 0.0)
    )
    broker.submit_market_order.side_effect = _submit
    broker.get_order_fill_price.return_value = fill_px
    broker.list_recent_orders.return_value = [
        {
            "order_id": "ord-int-1",
            "symbol": "VOO",
            "side": "buy",
            "qty": None,
            "filled_qty": None,
            "filled_avg_price": 100.25,
            "status": "filled",
        },
    ]

    kill_switch = KillSwitch(integration_settings.risk.daily_loss_limit_pct)
    order_manager = OrderManager(
        broker=broker,
        settings=integration_settings,
        kill_switch=kill_switch,
        sqlite_store=sqlite,
    )
    wf = TradingWorkflow(
        settings=integration_settings,
        data_pipeline=pipeline,
        parquet_store=parquet,
        strategies=[DCAStrategy(integration_settings)],
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
    assert sqlite.get_signals(limit=5)
    assert sqlite.get_executions(limit=5)
    assert sqlite.get_equity_snapshots()
    assert sqlite.get_alerts(limit=10) == []


def test_full_cycle_with_ingest_failure_produces_alert(
    integration_settings: Settings,
    tmp_path: Path,
) -> None:
    """One symbol failing ingest yields a persisted WARNING alert."""
    from src.automation.workflows import TradingWorkflow

    integration_settings.data.dca_targets = [
        DCATarget(symbol="VOO", weight=0.5),
        DCATarget(symbol="VXUS", weight=0.5),
    ]
    parquet = ParquetStore(Path(integration_settings.data.parquet_dir))
    db_path = tmp_path / "cache" / "hub_metadata.sqlite"
    sqlite = SQLiteStore(db_path)
    lot_ledger = LotLedger(db_path)
    adapter = _StubAdapter(fail_symbols={"VXUS"})
    pipeline = DataPipeline(adapter=adapter, parquet_store=parquet, sqlite_store=sqlite)

    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 25_000.0
    broker.get_position_qty.return_value = 0.0
    broker.submit_market_order.return_value = "ord-int-2"
    broker.get_order_fill_price.return_value = 100.0
    broker.list_recent_orders.return_value = []

    kill_switch = KillSwitch(integration_settings.risk.daily_loss_limit_pct)
    order_manager = OrderManager(
        broker=broker,
        settings=integration_settings,
        kill_switch=kill_switch,
        sqlite_store=sqlite,
    )
    wf = TradingWorkflow(
        settings=integration_settings,
        data_pipeline=pipeline,
        parquet_store=parquet,
        strategies=[DCAStrategy(integration_settings)],
        order_manager=order_manager,
        broker=broker,
        sqlite_store=sqlite,
        lot_ledger=lot_ledger,
        kill_switch=kill_switch,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    wf.run_cycle(as_of=monday)

    alerts = sqlite.get_alerts(limit=20)
    assert any(
        a.get("category") == "ingest_failure" and "VXUS" in a.get("message", "") for a in alerts
    )
