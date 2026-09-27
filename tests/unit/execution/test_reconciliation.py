"""Tests for execution log vs broker reconciliation."""

from __future__ import annotations

import pytest

from src.execution.reconciliation import (
    broker_orders_in_cycle_window,
    reconcile_executions,
    submitted_execution_order_ids,
)


def test_broker_orders_in_cycle_window_includes_orphan_in_window() -> None:
    """Orders in the cycle time window must be visible even with no SQLite row."""
    from datetime import UTC, datetime, timedelta

    as_of = datetime(2026, 6, 9, 17, 30, 13, tzinfo=UTC)
    sqlite_rows = [
        {
            "order_id": "known-bnd",
            "submitted": 1,
            "symbol": "BND",
            "side": "buy",
            "timestamp": as_of.isoformat(),
        },
    ]
    broker_orders = [
        {
            "order_id": "known-bnd",
            "symbol": "BND",
            "side": "buy",
            "timestamp": as_of.isoformat(),
        },
        {
            "order_id": "orphan-voo",
            "symbol": "VOO",
            "side": "buy",
            "timestamp": (as_of - timedelta(seconds=2)).isoformat(),
        },
        {
            "order_id": "ancient",
            "symbol": "SPY",
            "side": "buy",
            "timestamp": (as_of - timedelta(days=30)).isoformat(),
        },
    ]
    scoped = broker_orders_in_cycle_window(
        broker_orders,
        sqlite_rows,
        as_of=as_of,
        lookback=timedelta(hours=1),
    )
    ids = {str(o["order_id"]) for o in scoped}
    assert ids == {"known-bnd", "orphan-voo"}


def test_submitted_execution_order_ids_skips_unsubmitted() -> None:
    rows = [
        {"order_id": "x", "submitted": 1},
        {"order_id": "y", "submitted": 0},
        {"order_id": None, "submitted": 1},
    ]
    assert submitted_execution_order_ids(rows) == {"x"}


def test_reconciliation_all_matched() -> None:
    sqlite_rows = [
        {
            "order_id": "a1",
            "symbol": "SPY",
            "side": "buy",
            "submitted": 1,
            "qty": 10.0,
            "filled_avg_price": 400.0,
        },
    ]
    broker_orders = [
        {
            "order_id": "a1",
            "symbol": "SPY",
            "side": "buy",
            "qty": 10.0,
            "filled_qty": 10.0,
            "filled_avg_price": 400.0,
            "status": "filled",
        },
    ]
    r = reconcile_executions(sqlite_rows, broker_orders)
    assert r.matched == 1
    assert r.discrepancies == 0
    assert all(e.status == "matched" for e in r.entries)


def test_reconciliation_missing_from_broker() -> None:
    sqlite_rows = [
        {"order_id": "gone", "symbol": "QQQ", "side": "buy", "submitted": 1, "qty": 1.0},
    ]
    broker_orders: list[dict[str, object]] = []
    r = reconcile_executions(sqlite_rows, broker_orders)
    assert r.matched == 0
    assert r.discrepancies == 1
    assert r.entries[0].status == "missing_from_broker"


def test_reconciliation_missing_from_log() -> None:
    sqlite_rows: list[dict[str, object]] = []
    broker_orders = [
        {
            "order_id": "orphan",
            "symbol": "SPY",
            "side": "sell",
            "qty": 2.0,
            "filled_qty": 2.0,
            "filled_avg_price": 100.0,
            "status": "filled",
        },
    ]
    r = reconcile_executions(sqlite_rows, broker_orders)
    assert r.matched == 0
    assert r.discrepancies == 1
    assert r.entries[0].status == "missing_from_log"


def test_reconcile_price_mismatch() -> None:
    sqlite_rows = [
        {
            "order_id": "p1",
            "symbol": "SPY",
            "side": "buy",
            "submitted": 1,
            "qty": 10.0,
            "filled_avg_price": 400.0,
        },
    ]
    broker_orders = [
        {
            "order_id": "p1",
            "symbol": "SPY",
            "side": "buy",
            "qty": 10.0,
            "filled_qty": 10.0,
            "filled_avg_price": 400.02,
            "status": "filled",
        },
    ]
    r = reconcile_executions(sqlite_rows, broker_orders)
    assert r.matched == 0
    assert r.discrepancies == 1
    e = r.entries[0]
    assert e.status == "price_mismatch"
    assert e.local_price == pytest.approx(400.0)
    assert e.broker_price == pytest.approx(400.02)


def test_reconcile_field_mismatch_symbol() -> None:
    sqlite_rows = [
        {
            "order_id": "sym1",
            "symbol": "SPY",
            "side": "buy",
            "submitted": 1,
            "qty": 1.0,
            "filled_avg_price": 100.0,
        },
    ]
    broker_orders = [
        {
            "order_id": "sym1",
            "symbol": "QQQ",
            "side": "buy",
            "qty": 1.0,
            "filled_qty": 1.0,
            "filled_avg_price": 100.0,
            "status": "filled",
        },
    ]
    r = reconcile_executions(sqlite_rows, broker_orders)
    assert r.entries[0].status == "field_mismatch"


def test_reconcile_field_mismatch_side() -> None:
    sqlite_rows = [
        {
            "order_id": "side1",
            "symbol": "SPY",
            "side": "buy",
            "submitted": 1,
            "qty": 1.0,
            "filled_avg_price": 100.0,
        },
    ]
    broker_orders = [
        {
            "order_id": "side1",
            "symbol": "SPY",
            "side": "sell",
            "qty": 1.0,
            "filled_qty": 1.0,
            "filled_avg_price": 100.0,
            "status": "filled",
        },
    ]
    r = reconcile_executions(sqlite_rows, broker_orders)
    assert r.entries[0].status == "field_mismatch"


def test_reconciliation_qty_mismatch() -> None:
    sqlite_rows = [
        {"order_id": "x", "symbol": "SPY", "side": "buy", "submitted": 1, "qty": 5.0},
    ]
    broker_orders = [
        {
            "order_id": "x",
            "symbol": "SPY",
            "side": "buy",
            "qty": 5.0,
            "filled_qty": 9.0,
            "filled_avg_price": 400.0,
            "status": "filled",
        },
    ]
    r = reconcile_executions(sqlite_rows, broker_orders)
    assert r.matched == 0
    assert r.discrepancies == 1
    assert r.entries[0].status == "qty_mismatch"
