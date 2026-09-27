"""Unit tests for FIFO tax lot ledger."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.portfolio.tax_lots import LotLedger, TaxLot


def test_record_buy_creates_open_lot(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    opened = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    lot = led.record_buy("spy", 10.0, 100.0, opened)
    assert isinstance(lot, TaxLot)
    assert lot.symbol == "SPY"
    assert lot.qty == pytest.approx(10.0)
    assert lot.cost_per_share == pytest.approx(100.0)
    assert lot.opened_at == opened
    open_lots = led.get_open_lots()
    assert len(open_lots) == 1
    assert open_lots[0].id == lot.id


def test_record_buy_multiple_lots_ordered_by_opened_at(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 2, 1, tzinfo=UTC)
    t3 = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t1)
    led.record_buy("SPY", 1.0, 110.0, t2)
    led.record_buy("SPY", 1.0, 120.0, t3)
    lots = led.get_open_lots("SPY")
    assert [lot.opened_at for lot in lots] == [t1, t2, t3]


def test_record_sell_fifo_closes_oldest_lot_first(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 2, 1, tzinfo=UTC)
    led.record_buy("SPY", 5.0, 100.0, t1)
    led.record_buy("SPY", 5.0, 110.0, t2)
    closed_at = datetime(2026, 6, 1, tzinfo=UTC)
    closed = led.record_sell("SPY", 5.0, 200.0, closed_at)
    assert len(closed) == 1
    assert closed[0].cost_per_share == pytest.approx(100.0)
    assert closed[0].qty == pytest.approx(5.0)
    remaining = led.get_open_lots("SPY")
    assert len(remaining) == 1
    assert remaining[0].cost_per_share == pytest.approx(110.0)


def test_record_sell_partial_consumption(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    opened = datetime(2026, 1, 1, tzinfo=UTC)
    led.record_buy("SPY", 10.0, 100.0, opened)
    closed_at = datetime(2026, 6, 1, tzinfo=UTC)
    closed = led.record_sell("SPY", 3.0, 120.0, closed_at)
    assert len(closed) == 1
    assert closed[0].qty == pytest.approx(3.0)
    open_lots = led.get_open_lots("SPY")
    assert len(open_lots) == 1
    assert open_lots[0].qty == pytest.approx(7.0)


def test_record_sell_spans_multiple_lots(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 2, 1, tzinfo=UTC)
    led.record_buy("SPY", 5.0, 100.0, t1)
    led.record_buy("SPY", 5.0, 100.0, t2)
    closed_at = datetime(2026, 6, 1, tzinfo=UTC)
    closed = led.record_sell("SPY", 8.0, 120.0, closed_at)
    assert len(closed) == 2
    assert closed[0].qty == pytest.approx(5.0)
    assert closed[1].qty == pytest.approx(3.0)
    remaining = led.get_open_lots("SPY")
    assert len(remaining) == 1
    assert remaining[0].qty == pytest.approx(2.0)


def test_record_sell_exceeds_qty_raises_value_error(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="exceeds"):
        led.record_sell("SPY", 20.0, 100.0, datetime(2026, 6, 1, tzinfo=UTC))


def test_record_sell_no_open_lots_raises_value_error(tmp_path: Path) -> None:
    """Selling when the symbol has no open lots is an oversell (total_open == 0)."""
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    with pytest.raises(ValueError, match=r"No open lots.*0 available"):
        led.record_sell("SPY", 1.0, 100.0, datetime(2026, 6, 1, tzinfo=UTC))


def test_record_sell_zero_qty_returns_empty(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    led.record_buy("SPY", 5.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    out = led.record_sell("SPY", 0.0, 100.0, datetime(2026, 6, 1, tzinfo=UTC))
    assert out == []
    assert len(led.get_open_lots("SPY")) == 1


def test_unrealized_pnl_computation(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    assert led.unrealized_pnl("SPY", 110.0) == pytest.approx(100.0)


def test_realized_pnl_from_closed_lots(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    led.record_buy("SPY", 5.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_sell("SPY", 5.0, 120.0, datetime(2026, 6, 1, tzinfo=UTC))
    assert led.realized_pnl("SPY") == pytest.approx(100.0)
    assert led.realized_pnl() == pytest.approx(100.0)


def test_total_cost_basis(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_buy("SPY", 5.0, 120.0, datetime(2026, 2, 1, tzinfo=UTC))
    assert led.total_cost_basis("SPY") == pytest.approx(10 * 100 + 5 * 120)


def test_lot_persistence_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    opened = datetime(2026, 1, 1, tzinfo=UTC)
    LotLedger(db).record_buy("SPY", 3.0, 50.0, opened)
    led2 = LotLedger(db)
    lots = led2.get_open_lots("SPY")
    assert len(lots) == 1
    assert lots[0].qty == pytest.approx(3.0)


def test_get_closed_lots_respects_limit(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    opened = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(5):
        led.record_buy("SPY", 1.0, 100.0, opened)
        led.record_sell(
            "SPY",
            1.0,
            110.0,
            datetime(2026, 2, 1, tzinfo=UTC) + timedelta(days=i),
        )
    all_closed = led.get_closed_lots()
    assert len(all_closed) == 5
    limited = led.get_closed_lots(limit=2)
    assert len(limited) == 2
    assert limited[0].closed_at == all_closed[0].closed_at
    assert limited[1].closed_at == all_closed[1].closed_at


def test_get_closed_lots_ordered_newest_first(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    led.record_buy("SPY", 5.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_sell("SPY", 5.0, 110.0, datetime(2026, 3, 1, tzinfo=UTC))
    led.record_buy("SPY", 2.0, 90.0, datetime(2026, 4, 1, tzinfo=UTC))
    led.record_sell("SPY", 2.0, 100.0, datetime(2026, 5, 1, tzinfo=UTC))
    closed = led.get_closed_lots("SPY")
    assert len(closed) == 2
    assert closed[0].closed_at >= closed[1].closed_at


def test_closed_lot_aggregates_matches_closed_rows(tmp_path: Path) -> None:
    db = tmp_path / "lots.db"
    led = LotLedger(db)
    t = datetime(2026, 1, 1, tzinfo=UTC)
    led.record_buy("SPY", 10.0, 100.0, t)
    led.record_buy("QQQ", 5.0, 200.0, t)
    led.record_sell("SPY", 10.0, 120.0, datetime(2026, 2, 1, tzinfo=UTC))
    led.record_sell("QQQ", 5.0, 210.0, datetime(2026, 2, 2, tzinfo=UTC))
    by_sym, total_cb = led.closed_lot_aggregates()
    manual = led.get_closed_lots()
    assert sum(by_sym.values()) == pytest.approx(sum(cl.realized_pnl for cl in manual))
    assert total_cb == pytest.approx(sum(cl.qty * cl.cost_per_share for cl in manual))
    assert by_sym["SPY"] == pytest.approx(200.0)
    assert by_sym["QQQ"] == pytest.approx(50.0)


def test_closed_lot_aggregates_empty(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "empty.db")
    assert led.closed_lot_aggregates() == ({}, 0.0)


def test_record_sell_hifo_sells_highest_cost_first(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "hifo.db")
    t = datetime(2026, 1, 1, tzinfo=UTC)
    led.record_buy("SPY", 10.0, 50.0, t)
    led.record_buy("SPY", 10.0, 100.0, t)
    led.record_buy("SPY", 10.0, 75.0, t)
    closed = led.record_sell("SPY", 15.0, 80.0, datetime(2026, 6, 1, tzinfo=UTC), method="hifo")
    assert closed[0].cost_per_share == pytest.approx(100.0)
    assert closed[0].qty == pytest.approx(10.0)
    assert closed[1].cost_per_share == pytest.approx(75.0)
    assert closed[1].qty == pytest.approx(5.0)


def test_record_sell_hifo_partial_lot(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "hifo2.db")
    t = datetime(2026, 1, 1, tzinfo=UTC)
    led.record_buy("SPY", 10.0, 100.0, t)
    led.record_buy("SPY", 10.0, 50.0, t)
    closed = led.record_sell("SPY", 3.0, 90.0, datetime(2026, 6, 1, tzinfo=UTC), method="hifo")
    assert len(closed) == 1
    assert closed[0].qty == pytest.approx(3.0)
    assert closed[0].cost_per_share == pytest.approx(100.0)


def test_record_sell_fifo_unchanged_default(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "fifo.db")
    t = datetime(2026, 1, 1, tzinfo=UTC)
    led.record_buy("SPY", 10.0, 50.0, t)
    led.record_buy("SPY", 10.0, 100.0, t)
    closed = led.record_sell("SPY", 5.0, 80.0, datetime(2026, 6, 1, tzinfo=UTC))
    assert closed[0].cost_per_share == pytest.approx(50.0)


def test_record_sell_lot_closes_entire_lot(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "lot.db")
    lot = led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    cl = led.record_sell_lot(lot.id, 110.0, datetime(2026, 2, 1, tzinfo=UTC))
    assert cl.qty == pytest.approx(10.0)
    assert led.get_open_lots("SPY") == []


def test_record_sell_lot_partial_qty(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "lot2.db")
    lot = led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_sell_lot(lot.id, 110.0, datetime(2026, 2, 1, tzinfo=UTC), qty=5.0)
    open_lots = led.get_open_lots("SPY")
    assert len(open_lots) == 1
    assert open_lots[0].qty == pytest.approx(5.0)


def test_record_sell_lot_raises_on_unknown_id(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "lot3.db")
    with pytest.raises(ValueError, match="No open lot"):
        led.record_sell_lot("nope", 1.0, datetime(2026, 1, 1, tzinfo=UTC))


def test_record_sell_lot_raises_on_excess_qty(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "lot4.db")
    lot = led.record_buy("SPY", 3.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="exceeds"):
        led.record_sell_lot(lot.id, 1.0, datetime(2026, 2, 1, tzinfo=UTC), qty=10.0)


def _write_pre_tier30_tax_lots_schema(path: Path) -> None:
    """SQLite file matching Tier-29-era schema (no account_id on open/closed)."""
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


def test_lot_ledger_opens_pre_tier30_database(tmp_path: Path) -> None:
    db = tmp_path / "legacy_lots.db"
    _write_pre_tier30_tax_lots_schema(db)
    LotLedger(db)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("PRAGMA table_info(tax_lots_open)").fetchall()
        names = {str(r[1]) for r in rows}
        assert "account_id" in names


def test_lot_ledger_fresh_database_still_works(tmp_path: Path) -> None:
    db = tmp_path / "fresh_lots.db"
    LotLedger(db)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("PRAGMA table_info(tax_lots_open)").fetchall()
        names = {str(r[1]) for r in rows}
        assert "account_id" in names
        idx_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='tax_lots_open'",
        ).fetchall()
        idx_names = {str(r[0]) for r in idx_rows}
        assert "idx_tax_open_account" in idx_names
