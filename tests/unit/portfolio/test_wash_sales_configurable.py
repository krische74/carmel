"""Tier 31: configurable wash-sale window."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from src.portfolio.tax_lots import ClosedLot, TaxLot
from src.portfolio.wash_sales import detect_cross_account_wash_sales, detect_wash_sales


def test_wash_sale_custom_window_45_days() -> None:
    sell_d = date(2026, 2, 1)
    buy_d = sell_d + timedelta(days=35)
    loss = ClosedLot(
        lot_id="la",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=80.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime.combine(sell_d, datetime.min.time()).replace(tzinfo=UTC),
        realized_pnl=-20.0,
        account_id="a",
    )
    repl = TaxLot(
        id="rb",
        symbol="SPY",
        qty=1.0,
        cost_per_share=85.0,
        opened_at=datetime.combine(buy_d, datetime.min.time()).replace(tzinfo=UTC),
        account_id="a",
    )
    assert detect_wash_sales([loss], [repl], window_days=30) == []
    ws = detect_wash_sales([loss], [repl], window_days=45)
    assert len(ws) == 1


def test_wash_sale_custom_window_15_days() -> None:
    sell_d = date(2026, 3, 1)
    buy_d = sell_d + timedelta(days=20)
    loss = ClosedLot(
        lot_id="lb",
        symbol="QQQ",
        qty=1.0,
        cost_per_share=200.0,
        sell_price=180.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime.combine(sell_d, datetime.min.time()).replace(tzinfo=UTC),
        realized_pnl=-20.0,
        account_id="x",
    )
    repl = TaxLot(
        id="r",
        symbol="QQQ",
        qty=1.0,
        cost_per_share=185.0,
        opened_at=datetime.combine(buy_d, datetime.min.time()).replace(tzinfo=UTC),
        account_id="x",
    )
    assert detect_wash_sales([loss], [repl], window_days=15) == []
    assert len(detect_wash_sales([loss], [repl], window_days=30)) == 1


def test_cross_account_wash_respects_custom_window() -> None:
    sell_d = date(2026, 4, 1)
    buy_d = sell_d + timedelta(days=40)
    loss_a = ClosedLot(
        lot_id="la",
        symbol="IWM",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=85.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime.combine(sell_d, datetime.min.time()).replace(tzinfo=UTC),
        realized_pnl=-15.0,
        account_id="a",
    )
    repl_b = TaxLot(
        id="rb",
        symbol="IWM",
        qty=1.0,
        cost_per_share=86.0,
        opened_at=datetime.combine(buy_d, datetime.min.time()).replace(tzinfo=UTC),
        account_id="b",
    )
    assert detect_cross_account_wash_sales({"a": [loss_a]}, {"b": [repl_b]}, window_days=30) == []
    flags = detect_cross_account_wash_sales({"a": [loss_a]}, {"b": [repl_b]}, window_days=45)
    assert len(flags) == 1


def test_cross_account_partial_replacement_qty_still_flags_full_loss_amount() -> None:
    """Cross-account detector does not prorate by replacement size (L3 / informational)."""
    sell_d = date(2026, 5, 10)
    buy_d = sell_d + timedelta(days=5)
    loss_a = ClosedLot(
        lot_id="big_loss",
        symbol="XLF",
        qty=50.0,
        cost_per_share=40.0,
        sell_price=35.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime.combine(sell_d, datetime.min.time()).replace(tzinfo=UTC),
        realized_pnl=50.0 * (35.0 - 40.0),
        account_id="a",
    )
    repl_b = TaxLot(
        id="small_buy",
        symbol="XLF",
        qty=25.0,
        cost_per_share=36.0,
        opened_at=datetime.combine(buy_d, datetime.min.time()).replace(tzinfo=UTC),
        account_id="b",
    )
    flags = detect_cross_account_wash_sales({"a": [loss_a]}, {"b": [repl_b]}, window_days=30)
    assert len(flags) == 1
    assert flags[0].loss_amount == pytest.approx(250.0)
    assert flags[0].buying_account == "b"
