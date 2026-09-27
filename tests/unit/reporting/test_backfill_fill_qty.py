"""Backfill trade_executions.filled_qty from broker fill quotes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from src.data.storage.sqlite_store import SQLiteStore
from src.models import OrderExecutionResult
from src.reporting.backfill_fill_qty import (
    BrokerOrderMissingError,
    FillQuote,
    backfill_filled_qty,
)

if TYPE_CHECKING:
    from pathlib import Path

BIL_ORDER = "e93fe3f7-c8e2-48e3-b56c-cdc4b8cc4917"


def _log_null_fill(
    store: SQLiteStore,
    *,
    order_id: str,
    symbol: str,
    side: str,
    qty: float,
    ts: datetime,
    submitted: bool = True,
    account_id: str = "default",
) -> None:
    store.log_execution(
        "c-null",
        OrderExecutionResult(
            symbol=symbol,
            submitted=submitted,
            order_id=order_id,
            side=side,  # type: ignore[arg-type]
            qty=qty,
            fill_price=None,
            filled_qty=None,
            timestamp=ts,
            order_status="filled",
            strategy_name="Test",
        ),
        account_id=account_id,
    )


def test_backfill_writes_fill_fields_and_never_touches_qty(tmp_path: Path) -> None:
    """Partial fill is written back; requested qty, side, symbol, submitted stay put."""
    store = SQLiteStore(tmp_path / "hub.sqlite")
    _log_null_fill(
        store,
        order_id=BIL_ORDER,
        symbol="BIL",
        side="buy",
        qty=15.533477,
        ts=datetime(2026, 8, 10, 17, 30, 14, tzinfo=UTC),
    )

    def fetch(order_id: str) -> FillQuote:
        assert order_id == BIL_ORDER
        return FillQuote(filled_qty=6.133477, filled_avg_price=91.55)

    result = backfill_filled_qty(
        store,
        since=datetime(2026, 5, 16, tzinfo=UTC),
        account_id="default",
        fetch_fill=fetch,
        dry_run=False,
    )

    rows = store.get_executions(limit=5)
    assert len(rows) == 1
    row = rows[0]
    assert float(row["filled_qty"]) == pytest.approx(6.133477)
    assert float(row["filled_avg_price"]) == pytest.approx(91.55)
    assert float(row["qty"]) == pytest.approx(15.533477)
    assert row["side"] == "buy"
    assert row["symbol"] == "BIL"
    assert int(row["submitted"]) == 1
    assert row["order_status"] == "filled"
    assert result.rows_updated == 1
    assert result.rows_left_null == 0
    assert len(result.partials) == 1
    assert result.partials[0].order_id == BIL_ORDER
    assert result.partials[0].filled_qty == pytest.approx(6.133477)


def test_backfill_leaves_row_null_when_broker_order_missing(tmp_path: Path) -> None:
    """A gone Alpaca order stays NULL — never default filled_qty to requested qty."""
    store = SQLiteStore(tmp_path / "hub.sqlite")
    _log_null_fill(
        store,
        order_id="gone-1",
        symbol="VOO",
        side="buy",
        qty=0.011047,
        ts=datetime(2026, 5, 16, 17, 30, tzinfo=UTC),
    )

    def fetch(order_id: str) -> FillQuote:
        raise BrokerOrderMissingError(order_id)

    result = backfill_filled_qty(
        store,
        since=datetime(2026, 5, 16, tzinfo=UTC),
        account_id="default",
        fetch_fill=fetch,
        dry_run=False,
    )

    row = store.get_executions(limit=5)[0]
    assert row["filled_qty"] is None
    assert row["filled_avg_price"] is None
    assert float(row["qty"]) == pytest.approx(0.011047)
    assert result.rows_updated == 0
    assert result.rows_left_null == 1
    assert result.missing_order_ids == ["gone-1"]


def test_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "hub.sqlite")
    _log_null_fill(
        store,
        order_id=BIL_ORDER,
        symbol="BIL",
        side="buy",
        qty=15.533477,
        ts=datetime(2026, 8, 10, 17, 30, tzinfo=UTC),
    )

    def fetch(order_id: str) -> FillQuote:
        return FillQuote(filled_qty=6.133477, filled_avg_price=91.55)

    result = backfill_filled_qty(
        store,
        since=datetime(2026, 5, 16, tzinfo=UTC),
        account_id="default",
        fetch_fill=fetch,
        dry_run=True,
    )

    row = store.get_executions(limit=5)[0]
    assert row["filled_qty"] is None
    assert row["filled_avg_price"] is None
    assert float(row["qty"]) == pytest.approx(15.533477)
    assert result.dry_run is True
    assert result.rows_updated == 0
    assert len(result.partials) == 1
