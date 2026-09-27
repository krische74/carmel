"""Deterministic Alpaca ``client_order_id`` helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.execution.client_order_id import build_exec_client_order_id, sanitize_client_order_id


def test_build_exec_client_order_id_stable_within_cycle_across_minutes() -> None:
    """Tier 49C: same cycle → same cid even when the clock crosses a minute."""
    t1 = datetime(2026, 7, 3, 17, 30, 0, tzinfo=UTC)
    t2 = datetime(2026, 7, 3, 18, 0, 0, tzinfo=UTC)
    a = build_exec_client_order_id(
        account_id="default",
        symbol="SPY",
        qty=1.0,
        side="buy",
        intent="mkt",
        anchor=t1,
        cycle_id="cycle-friday-am",
    )
    b = build_exec_client_order_id(
        account_id="default",
        symbol="SPY",
        qty=1.0,
        side="buy",
        intent="mkt",
        anchor=t2,
        cycle_id="cycle-friday-am",
    )
    assert a == b


def test_build_exec_client_order_id_differs_across_cycles_same_minute() -> None:
    """Friday 17:30 / 18:00 pair: distinct cycles → distinct cids."""
    anchor = datetime(2026, 7, 3, 17, 30, 0, tzinfo=UTC)
    first = build_exec_client_order_id(
        account_id="default",
        symbol="SPY",
        qty=1.0,
        side="buy",
        intent="mkt",
        anchor=anchor,
        cycle_id="cycle-a",
    )
    second = build_exec_client_order_id(
        account_id="default",
        symbol="SPY",
        qty=1.0,
        side="buy",
        intent="mkt",
        anchor=anchor,
        cycle_id="cycle-b",
    )
    assert first != second


def test_build_exec_client_order_id_stable_for_same_inputs() -> None:
    anchor = datetime(2026, 4, 13, 1, 2, tzinfo=UTC)
    a = build_exec_client_order_id(
        account_id="paper-1",
        symbol="QQQ",
        qty=40.91,
        side="buy",
        intent="mkt",
        anchor=anchor,
    )
    b = build_exec_client_order_id(
        account_id="paper-1",
        symbol="QQQ",
        qty=40.91,
        side="buy",
        intent="mkt",
        anchor=anchor,
    )
    assert a == b
    assert a.startswith("cml-")
    assert len(a) <= 48


def test_build_exec_client_order_id_differs_when_minute_changes() -> None:
    t1 = datetime(2026, 4, 13, 1, 2, 0, tzinfo=UTC)
    t2 = datetime(2026, 4, 13, 1, 3, 0, tzinfo=UTC)
    a = build_exec_client_order_id(
        account_id="x",
        symbol="QQQ",
        qty=1.0,
        side="buy",
        intent="mkt",
        anchor=t1,
    )
    b = build_exec_client_order_id(
        account_id="x",
        symbol="QQQ",
        qty=1.0,
        side="buy",
        intent="mkt",
        anchor=t2,
    )
    assert a != b


def test_build_exec_client_order_id_differs_by_intent() -> None:
    anchor = datetime(2026, 4, 13, 12, 0, tzinfo=UTC)
    m = build_exec_client_order_id(
        account_id="a",
        symbol="SPY",
        qty=2.0,
        side="buy",
        intent="mkt",
        anchor=anchor,
    )
    limit_cid = build_exec_client_order_id(
        account_id="a",
        symbol="SPY",
        qty=2.0,
        side="buy",
        intent="lmt",
        anchor=anchor,
        price_key="400.000000",
    )
    assert m != limit_cid


def test_sanitize_client_order_id_strips_invalid_chars() -> None:
    assert sanitize_client_order_id("cml-abc123") == "cml-abc123"


def test_sanitize_client_order_id_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        sanitize_client_order_id("@@@")
