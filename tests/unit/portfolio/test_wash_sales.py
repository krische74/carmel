"""Tests for IRS wash-sale detection."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from src.portfolio.tax_lots import ClosedLot, LotLedger, TaxLot
from src.portfolio.wash_sales import WashSale, detect_wash_sales


def test_wash_sale_detected_when_buy_within_30_days_after_loss_sale() -> None:
    """Replacement purchase after loss sale within 30 days."""
    loss = ClosedLot(
        lot_id="lot-a",
        symbol="SPY",
        qty=10.0,
        cost_per_share=100.0,
        sell_price=90.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 2, 15, 12, 0, tzinfo=UTC),
        realized_pnl=-100.0,
    )
    replacement = TaxLot(
        id="open-1",
        symbol="SPY",
        qty=10.0,
        cost_per_share=92.0,
        opened_at=datetime(2026, 2, 20, 12, 0, tzinfo=UTC),
    )
    ws = detect_wash_sales([loss], [replacement])
    assert len(ws) == 1
    assert ws[0].symbol == "SPY"
    assert ws[0].disallowed_loss == pytest.approx(100.0)
    assert ws[0].replacement_lot_id == "open-1"


def test_wash_sale_detected_when_buy_within_30_days_before_loss_sale() -> None:
    """Purchase before loss sale still inside ±30 day window."""
    loss = ClosedLot(
        lot_id="lot-b",
        symbol="QQQ",
        qty=5.0,
        cost_per_share=200.0,
        sell_price=180.0,
        opened_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        realized_pnl=-100.0,
    )
    replacement = TaxLot(
        id="open-2",
        symbol="QQQ",
        qty=5.0,
        cost_per_share=190.0,
        opened_at=datetime(2026, 3, 25, 12, 0, tzinfo=UTC),
    )
    ws = detect_wash_sales([loss], [replacement])
    assert len(ws) == 1


def test_no_wash_sale_when_buy_outside_window() -> None:
    loss = ClosedLot(
        lot_id="lot-c",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=80.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        realized_pnl=-20.0,
    )
    replacement = TaxLot(
        id="open-3",
        symbol="SPY",
        qty=1.0,
        cost_per_share=85.0,
        opened_at=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
    )
    assert detect_wash_sales([loss], [replacement]) == []


def test_no_wash_sale_on_profitable_close() -> None:
    gain = ClosedLot(
        lot_id="lot-d",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=110.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        realized_pnl=10.0,
    )
    replacement = TaxLot(
        id="open-4",
        symbol="SPY",
        qty=1.0,
        cost_per_share=105.0,
        opened_at=datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
    )
    assert detect_wash_sales([gain], [replacement]) == []


def test_wash_sale_different_symbol_no_match() -> None:
    loss = ClosedLot(
        lot_id="lot-e",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=80.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        realized_pnl=-20.0,
    )
    other = TaxLot(
        id="open-5",
        symbol="QQQ",
        qty=1.0,
        cost_per_share=300.0,
        opened_at=datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
    )
    assert detect_wash_sales([loss], [other]) == []


def test_wash_sale_buy_exactly_30_days_after_sell() -> None:
    """Replacement on sell_date + 30 calendar days is still inside the inclusive window."""
    sell_d = date(2026, 2, 15)
    buy_d = sell_d + timedelta(days=30)
    loss = ClosedLot(
        lot_id="loss-30",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=80.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime.combine(sell_d, datetime.min.time()).replace(tzinfo=UTC),
        realized_pnl=-20.0,
    )
    replacement = TaxLot(
        id="rep-30",
        symbol="SPY",
        qty=1.0,
        cost_per_share=85.0,
        opened_at=datetime.combine(buy_d, datetime.min.time()).replace(tzinfo=UTC),
    )
    ws = detect_wash_sales([loss], [replacement])
    assert len(ws) == 1
    assert ws[0].buy_date == buy_d


def test_no_wash_sale_buy_31_days_after_sell() -> None:
    """Day 31 after the loss sale is outside the ±30 day window."""
    sell_d = date(2026, 2, 15)
    buy_d = sell_d + timedelta(days=31)
    loss = ClosedLot(
        lot_id="loss-31",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=80.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime.combine(sell_d, datetime.min.time()).replace(tzinfo=UTC),
        realized_pnl=-20.0,
    )
    replacement = TaxLot(
        id="rep-31",
        symbol="SPY",
        qty=1.0,
        cost_per_share=85.0,
        opened_at=datetime.combine(buy_d, datetime.min.time()).replace(tzinfo=UTC),
    )
    assert detect_wash_sales([loss], [replacement]) == []


def test_wash_sale_same_day_buy_and_sell() -> None:
    """Loss sale and replacement purchase on the same calendar day."""
    d = date(2026, 6, 10)
    loss = ClosedLot(
        lot_id="loss-sd",
        symbol="QQQ",
        qty=2.0,
        cost_per_share=200.0,
        sell_price=180.0,
        opened_at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime.combine(d, datetime.min.time()).replace(tzinfo=UTC),
        realized_pnl=-40.0,
    )
    replacement = TaxLot(
        id="rep-sd",
        symbol="QQQ",
        qty=2.0,
        cost_per_share=185.0,
        opened_at=datetime.combine(d, datetime.min.time()).replace(tzinfo=UTC),
    )
    ws = detect_wash_sales([loss], [replacement])
    assert len(ws) == 1
    assert ws[0].buy_date == d


def test_wash_sale_replacement_from_closed_lot() -> None:
    """Replacement purchase tracked as another closed lot (opened_at in window)."""
    loss = ClosedLot(
        lot_id="loss-cl",
        symbol="IWM",
        qty=10.0,
        cost_per_share=100.0,
        sell_price=90.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        realized_pnl=-100.0,
    )
    replacement_closed = ClosedLot(
        lot_id="rep-cl",
        symbol="IWM",
        qty=10.0,
        cost_per_share=92.0,
        sell_price=95.0,
        opened_at=datetime(2026, 3, 10, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
        realized_pnl=30.0,
    )
    ws = detect_wash_sales([loss, replacement_closed], [])
    assert len(ws) == 1
    assert ws[0].replacement_lot_id == "rep-cl"


def test_lot_ledger_detect_wash_sales_convenience(tmp_path) -> None:
    db = tmp_path / "tax.db"
    led = LotLedger(db)
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    lots = led.record_sell("SPY", 10.0, 90.0, datetime(2026, 2, 1, tzinfo=UTC))
    assert lots[0].realized_pnl < 0
    led.record_buy("SPY", 10.0, 92.0, datetime(2026, 2, 10, tzinfo=UTC))
    out = led.detect_wash_sales()
    assert len(out) >= 1
    assert isinstance(out[0], WashSale)
