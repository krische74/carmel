"""Tests for cash backfill onto equity_snapshots (Tier 46C)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pytest

from src.data.storage.sqlite_store import SQLiteStore
from src.models import OrderExecutionResult
from src.reporting.backfill_snapshot_cash import (
    backfill_snapshot_cash,
    cash_by_date_from_executions,
)

if TYPE_CHECKING:
    from pathlib import Path


def _log(
    store: SQLiteStore,
    *,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    ts: datetime,
) -> None:
    store.log_execution(
        "c1",
        OrderExecutionResult(
            symbol=symbol,
            submitted=True,
            order_id=f"oid-{symbol}-{ts.isoformat()}-{side}",
            side=side,  # type: ignore[arg-type]
            qty=qty,
            fill_price=price,
            timestamp=ts,
            strategy_name="Test",
        ),
    )


def test_cash_by_date_replay_buy_and_sell() -> None:
    rows = [
        {
            "timestamp": "2026-05-16T17:30:00+00:00",
            "side": "buy",
            "qty": 1.0,
            "filled_avg_price": 100.0,
        },
        {
            "timestamp": "2026-05-17T17:30:00+00:00",
            "side": "sell",
            "qty": 1.0,
            "filled_avg_price": 110.0,
        },
    ]
    by, skipped = cash_by_date_from_executions(rows, since=date(2026, 5, 16), seed=1_000.0)
    assert skipped == 0
    assert by["2026-05-16"] == pytest.approx(900.0)
    assert by["2026-05-17"] == pytest.approx(1_010.0)


def test_backfill_snapshot_cash_writes_series_and_reports_delta(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "hub.sqlite")
    for d in ("2026-05-16", "2026-05-17", "2026-05-18"):
        store.write_equity_snapshot(
            d,
            total_market_value=100.0,
            total_cost_basis=100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            cash=None,
        )
    _log(
        store,
        symbol="QQQ",
        side="buy",
        qty=1.0,
        price=100.0,
        ts=datetime(2026, 5, 16, 17, 30, tzinfo=UTC),
    )
    _log(
        store,
        symbol="QQQ",
        side="buy",
        qty=2.0,
        price=50.0,
        ts=datetime(2026, 5, 17, 17, 30, tzinfo=UTC),
    )
    result = backfill_snapshot_cash(
        store,
        since=date(2026, 5, 16),
        seed=1_000.0,
        broker_cash=800.0,  # true ending after 100+100 buys
    )
    assert result.rows_updated == 3
    assert result.final_cash == pytest.approx(800.0)
    assert result.delta_vs_broker == pytest.approx(0.0)
    snaps = {r["date"]: r for r in store.get_equity_snapshots()}
    assert float(snaps["2026-05-16"]["cash"]) == pytest.approx(900.0)
    assert float(snaps["2026-05-17"]["cash"]) == pytest.approx(800.0)
    assert float(snaps["2026-05-18"]["cash"]) == pytest.approx(800.0)  # carry forward


def test_backfill_snapshot_cash_flags_phantom_delta(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "hub.sqlite")
    store.write_equity_snapshot(
        "2026-05-16",
        total_market_value=0.0,
        total_cost_basis=0.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        total_pnl=0.0,
    )
    result = backfill_snapshot_cash(
        store,
        since=date(2026, 5, 16),
        seed=3_336.0,
        broker_cash=3_000.0,
    )
    assert result.delta_vs_broker == pytest.approx(336.0)
