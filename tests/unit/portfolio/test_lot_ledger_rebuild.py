"""Rebuild lot ledger from trade_executions after an account reset (Tier 46)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.models import OrderExecutionResult
from src.portfolio.lot_ledger_rebuild import (
    RebuildLotsResult,
    rebuild_equity_snapshots_from_ledger,
    rebuild_lot_ledger_from_executions,
)
from src.portfolio.tax_lots import LotLedger

if TYPE_CHECKING:
    from pathlib import Path


def _ohlcv(closes: list[tuple[str, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in closes])
    vals = [c for _, c in closes]
    return pd.DataFrame(
        {
            "open": vals,
            "high": vals,
            "low": vals,
            "close": vals,
            "volume": [1e6] * len(vals),
        },
        index=idx,
    )


def _log_fill(
    store: SQLiteStore,
    *,
    symbol: str,
    side: str,
    qty: float,
    timestamp: datetime,
    filled_avg_price: float | None,
    cycle_id: str = "c1",
) -> None:
    store.log_execution(
        cycle_id,
        OrderExecutionResult(
            symbol=symbol,
            submitted=True,
            order_id=f"oid-{symbol}-{timestamp.isoformat()}-{side}",
            side=side,
            qty=qty,
            fill_price=filled_avg_price,
            timestamp=timestamp,
            strategy_name="Test",
        ),
        account_id="default",
    )


def test_rebuild_clears_stale_pre_reset_lots_and_replays_fifo(tmp_path: Path) -> None:
    """Stale pre-reset lots must not survive; sells match post-reset buys (FIFO)."""
    db = tmp_path / "hub.sqlite"
    store = SQLiteStore(db)
    ledger = LotLedger(db)
    pq = ParquetStore(tmp_path / "parquet")
    pq.write_ohlcv(
        "QQQ",
        _ohlcv(
            [
                ("2026-04-30", 661.57),
                ("2026-05-16", 700.0),
                ("2026-06-19", 740.0),
                ("2026-08-06", 715.0),
            ],
        ),
    )

    # Stale pre-reset lot (retired account) — exactly the production failure mode.
    ledger.record_buy(
        "QQQ",
        35.0,
        661.57,
        datetime(2026, 4, 30, tzinfo=UTC),
    )
    # Also seed a wrong closed lot that matched the stale basis.
    ledger.record_sell(
        "QQQ",
        1.0,
        740.0,
        datetime(2026, 6, 19, 17, 30, tzinfo=UTC),
    )

    t_buy1 = datetime(2026, 5, 16, 17, 30, 7, tzinfo=UTC)
    t_buy2 = datetime(2026, 5, 20, 17, 30, 0, tzinfo=UTC)
    t_sell1 = datetime(2026, 6, 19, 17, 30, 9, tzinfo=UTC)
    t_sell2 = datetime(2026, 8, 6, 17, 30, 15, tzinfo=UTC)
    _log_fill(store, symbol="QQQ", side="buy", qty=1.1764, timestamp=t_buy1, filled_avg_price=700.0)
    _log_fill(store, symbol="QQQ", side="buy", qty=1.2342, timestamp=t_buy2, filled_avg_price=710.0)
    _log_fill(store, symbol="QQQ", side="sell", qty=1.1764, timestamp=t_sell1, filled_avg_price=740.0)
    _log_fill(store, symbol="QQQ", side="sell", qty=1.2342, timestamp=t_sell2, filled_avg_price=715.0)

    result = rebuild_lot_ledger_from_executions(
        store,
        ledger,
        parquet_store=pq,
        since=datetime(2026, 5, 16, tzinfo=UTC),
        account_id="default",
        method="fifo",
    )
    assert isinstance(result, RebuildLotsResult)
    assert result.buys_replayed == 2
    assert result.sells_replayed == 2

    open_lots = ledger.get_open_lots(account_id="default")
    assert open_lots == []
    # Realized P&L must use post-reset costs (700/710), not stale 661.57.
    closed = ledger.get_closed_lots(account_id="default")
    assert len(closed) == 2
    total_rpnl = sum(c.realized_pnl for c in closed)
    expected = 1.1764 * (740.0 - 700.0) + 1.2342 * (715.0 - 710.0)
    assert total_rpnl == pytest.approx(expected)
    assert all(c.cost_per_share != pytest.approx(661.57) for c in closed)


def test_rebuild_uses_parquet_when_fill_price_missing(tmp_path: Path) -> None:
    db = tmp_path / "hub.sqlite"
    store = SQLiteStore(db)
    ledger = LotLedger(db)
    pq = ParquetStore(tmp_path / "parquet")
    pq.write_ohlcv("VOO", _ohlcv([("2026-05-16", 550.0), ("2026-05-18", 555.0)]))

    _log_fill(
        store,
        symbol="VOO",
        side="buy",
        qty=0.5,
        timestamp=datetime(2026, 5, 16, 17, 30, tzinfo=UTC),
        filled_avg_price=None,
    )
    rebuild_lot_ledger_from_executions(
        store,
        ledger,
        parquet_store=pq,
        since=datetime(2026, 5, 16, tzinfo=UTC),
    )
    lots = ledger.get_open_lots("VOO")
    assert len(lots) == 1
    assert lots[0].qty == pytest.approx(0.5)
    assert lots[0].cost_per_share == pytest.approx(550.0)


def test_rebuild_preserves_expected_open_qty_mix(tmp_path: Path) -> None:
    """Multi-symbol post-reset book matches broker-shaped open quantities."""
    db = tmp_path / "hub.sqlite"
    store = SQLiteStore(db)
    ledger = LotLedger(db)
    pq = ParquetStore(tmp_path / "parquet")
    for sym, px in (("VOO", 600.0), ("VXUS", 80.0), ("BND", 70.0), ("BIL", 91.0)):
        pq.write_ohlcv(sym, _ohlcv([("2026-05-16", px), ("2026-08-14", px)]))

    # Stale pre-reset VOO that would inflate open qty if not cleared.
    ledger.record_buy("VOO", 5.0, 500.0, datetime(2026, 4, 1, tzinfo=UTC))

    fills = [
        ("VOO", "buy", 0.8188, 600.0),
        ("VXUS", "buy", 3.0485, 80.0),
        ("BND", "buy", 1.1845, 70.0),
        ("BIL", "buy", 15.5335, 91.0),
    ]
    t0 = datetime(2026, 5, 16, 17, 30, tzinfo=UTC)
    for i, (sym, side, qty, px) in enumerate(fills):
        _log_fill(
            store,
            symbol=sym,
            side=side,
            qty=qty,
            timestamp=t0.replace(second=i),
            filled_avg_price=px,
        )

    rebuild_lot_ledger_from_executions(
        store,
        ledger,
        parquet_store=pq,
        since=datetime(2026, 5, 16, tzinfo=UTC),
    )
    by_sym = {lot.symbol: lot.qty for lot in ledger.get_open_lots()}
    assert by_sym.get("QQQ", 0.0) == pytest.approx(0.0)
    assert by_sym["VOO"] == pytest.approx(0.8188)
    assert by_sym["VXUS"] == pytest.approx(3.0485)
    assert by_sym["BND"] == pytest.approx(1.1845)
    assert by_sym["BIL"] == pytest.approx(15.5335)
    cost = sum(lot.qty * lot.cost_per_share for lot in ledger.get_open_lots())
    assert cost == pytest.approx(0.8188 * 600 + 3.0485 * 80 + 1.1845 * 70 + 15.5335 * 91)


def test_rebuild_equity_snapshots_overwrites_corrupt_curve(tmp_path: Path) -> None:
    db = tmp_path / "hub.sqlite"
    store = SQLiteStore(db)
    ledger = LotLedger(db)
    pq = ParquetStore(tmp_path / "parquet")
    pq.write_ohlcv("BIL", _ohlcv([("2026-05-16", 91.0), ("2026-05-17", 91.5), ("2026-05-18", 92.0)]))

    # Corrupt snapshot reflecting stale lots (including a pre-cutoff row).
    store.write_equity_snapshot(
        "2026-05-10",
        total_market_value=37_400.0,
        total_cost_basis=34_000.0,
        unrealized_pnl=3_400.0,
        realized_pnl=0.0,
        total_pnl=3_400.0,
    )
    store.write_equity_snapshot(
        "2026-05-17",
        total_market_value=37_400.0,
        total_cost_basis=34_000.0,
        unrealized_pnl=3_400.0,
        realized_pnl=0.0,
        total_pnl=3_400.0,
    )
    _log_fill(
        store,
        symbol="BIL",
        side="buy",
        qty=10.0,
        timestamp=datetime(2026, 5, 16, 17, 30, tzinfo=UTC),
        filled_avg_price=91.0,
    )
    rebuild_lot_ledger_from_executions(
        store,
        ledger,
        parquet_store=pq,
        since=datetime(2026, 5, 16, tzinfo=UTC),
    )
    points = rebuild_equity_snapshots_from_ledger(
        store,
        ledger,
        pq,
        start=date(2026, 5, 16),
        end=date(2026, 5, 18),
        account_id="default",
    )
    assert len(points) == 3
    snaps = store.get_equity_snapshots(account_id="default")
    by_date = {r["date"]: r for r in snaps}
    assert "2026-05-10" not in by_date
    assert by_date["2026-05-17"]["total_market_value"] == pytest.approx(10.0 * 91.5)
    assert by_date["2026-05-17"]["total_cost_basis"] == pytest.approx(910.0)
    assert by_date["2026-05-17"]["total_market_value"] < 1_000.0
