"""Tests for daily equity snapshots and historical backfill."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.reporting.equity_curve import (
    backfill_equity_curve,
    record_equity_snapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.portfolio.tax_lots import LotLedger


def test_record_equity_snapshot_computes_and_stores(tmp_path: Path) -> None:
    db = tmp_path / "hub.sqlite"
    led = LotLedger(db)
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 5, 16, 0, tzinfo=UTC))
    pq = ParquetStore(tmp_path / "parquet")
    idx = pd.DatetimeIndex([pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")])
    df = pd.DataFrame(
        {
            "open": [100.0, 110.0],
            "high": [101.0, 111.0],
            "low": [99.0, 109.0],
            "close": [100.0, 110.0],
            "volume": [1e6, 1e6],
        },
        index=idx,
    )
    pq.write_ohlcv("SPY", df)
    store = SQLiteStore(db)
    pt = record_equity_snapshot(store, led, pq, as_of_date=date(2026, 1, 6), cash=500.0)
    assert pt.total_market_value == pytest.approx(1100.0)
    assert pt.cash == pytest.approx(500.0)
    rows = store.get_equity_snapshots()
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-01-06"
    assert float(rows[0]["total_market_value"]) == pytest.approx(1100.0)
    assert float(rows[0]["cash"]) == pytest.approx(500.0)


def test_backfill_reconstructs_from_lot_events(tmp_path: Path) -> None:
    db = tmp_path / "hub.sqlite"
    led = LotLedger(db)
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, 16, 0, tzinfo=UTC))
    pq = ParquetStore(tmp_path / "parquet")
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-02"),
            pd.Timestamp("2026-01-03"),
        ],
    )
    df = pd.DataFrame(
        {
            "open": [100.0, 105.0, 110.0],
            "high": [101.0, 106.0, 111.0],
            "low": [99.0, 104.0, 109.0],
            "close": [100.0, 105.0, 110.0],
            "volume": [1e6, 1e6, 1e6],
        },
        index=idx,
    )
    pq.write_ohlcv("SPY", df)
    store = SQLiteStore(db)
    start = date(2026, 1, 1)
    end = date(2026, 1, 3)
    out = backfill_equity_curve(store, led, pq, start, end)
    assert len(out) == 3
    assert out[0].total_market_value == pytest.approx(1000.0)
    assert out[1].total_market_value == pytest.approx(1050.0)
    assert out[2].total_market_value == pytest.approx(1100.0)
    rows = store.get_equity_snapshots()
    assert len(rows) == 3
    assert all(r["cash"] is None for r in rows)


def test_backfill_preserves_existing_cash_and_broker_equity_on_upsert(tmp_path: Path) -> None:
    """48-0: lot rebuild must not NULL broker columns when rewriting MV."""
    from datetime import date

    from src.portfolio.lot_ledger_rebuild import rebuild_equity_snapshots_from_ledger

    db = tmp_path / "hub.sqlite"
    store = SQLiteStore(db)
    store.write_equity_snapshot(
        "2026-08-19",
        total_market_value=2800.0,
        total_cost_basis=2700.0,
        unrealized_pnl=100.0,
        realized_pnl=0.0,
        total_pnl=100.0,
        cash=240.44,
        broker_equity=3436.33,
    )
    led = LotLedger(db)
    led.record_buy("BIL", 10.0, 90.0, datetime(2026, 8, 19, 17, 30, tzinfo=UTC))
    pq = ParquetStore(tmp_path / "parquet")
    pq.write_ohlcv("BIL", _ohlcv([("2026-08-19", 91.0)]))
    rebuild_equity_snapshots_from_ledger(
        store,
        led,
        pq,
        start=date(2026, 8, 19),
        end=date(2026, 8, 19),
    )
    rows = store.get_equity_snapshots()
    assert len(rows) == 1
    assert float(rows[0]["cash"]) == pytest.approx(240.44)
    assert float(rows[0]["broker_equity"]) == pytest.approx(3436.33)
    assert float(rows[0]["total_market_value"]) == pytest.approx(910.0)


def _ohlcv(dates_closes: list[tuple[str, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in dates_closes])
    vals = [c for _, c in dates_closes]
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
