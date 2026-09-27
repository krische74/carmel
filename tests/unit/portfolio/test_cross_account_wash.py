"""Cross-account wash sale detection (informational)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from src.portfolio.tax_lots import ClosedLot, TaxLot
from src.portfolio.wash_sales import (
    detect_cross_account_wash_sales,
    detect_wash_sales,
    partition_lots_by_account,
)


def test_intra_account_wash_ignores_replacement_in_other_account() -> None:
    """Same-symbol replacement in B does not count as intra-account wash for loss in A."""
    loss = ClosedLot(
        lot_id="la",
        symbol="SPY",
        qty=10.0,
        cost_per_share=100.0,
        sell_price=90.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
        realized_pnl=-100.0,
        account_id="acct_a",
    )
    repl_b = TaxLot(
        id="rb",
        symbol="SPY",
        qty=10.0,
        cost_per_share=92.0,
        opened_at=datetime(2026, 2, 10, 12, 0, tzinfo=UTC),
        account_id="acct_b",
    )
    assert detect_wash_sales([loss], [repl_b]) == []


def test_cross_account_wash_detected() -> None:
    """Loss in A and replacement buy in B within ±30 days → flagged."""
    loss_a = ClosedLot(
        lot_id="la",
        symbol="SPY",
        qty=10.0,
        cost_per_share=100.0,
        sell_price=90.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
        realized_pnl=-100.0,
        account_id="acct_a",
    )
    repl_b = TaxLot(
        id="rb",
        symbol="SPY",
        qty=10.0,
        cost_per_share=88.0,
        opened_at=datetime(2026, 2, 10, 12, 0, tzinfo=UTC),
        account_id="acct_b",
    )
    by_c = {"acct_a": [loss_a], "acct_b": []}
    by_o = {"acct_b": [repl_b]}
    flags = detect_cross_account_wash_sales(by_c, by_o, window_days=30)
    assert len(flags) == 1
    assert flags[0].selling_account == "acct_a"
    assert flags[0].buying_account == "acct_b"
    assert flags[0].symbol == "SPY"
    assert flags[0].loss_amount == pytest.approx(100.0)
    assert flags[0].replacement_date == date(2026, 2, 10)


def test_cross_account_wash_not_detected_outside_window() -> None:
    """Replacement more than 30 days after sale → not flagged."""
    loss_a = ClosedLot(
        lot_id="la",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=80.0,
        opened_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
        realized_pnl=-20.0,
        account_id="acct_a",
    )
    repl_b = TaxLot(
        id="rb",
        symbol="SPY",
        qty=1.0,
        cost_per_share=85.0,
        opened_at=datetime(2026, 3, 20, 12, 0, tzinfo=UTC),
        account_id="acct_b",
    )
    by_c = {"acct_a": [loss_a]}
    by_o = {"acct_b": [repl_b]}
    assert detect_cross_account_wash_sales(by_c, by_o, window_days=30) == []


def test_cross_account_empty_when_single_account_partition() -> None:
    loss = ClosedLot(
        lot_id="x",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=90.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime(2026, 2, 1, tzinfo=UTC),
        realized_pnl=-10.0,
    )
    assert detect_cross_account_wash_sales({"default": [loss]}, {"default": []}) == []


def test_partition_lots_by_account_empty_inputs() -> None:
    by_c, by_o = partition_lots_by_account([], [])
    assert by_c == {}
    assert by_o == {}


def test_partition_lots_by_account_blank_account_id_becomes_default() -> None:
    loss = ClosedLot(
        lot_id="l1",
        symbol="X",
        qty=1.0,
        cost_per_share=1.0,
        sell_price=0.5,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime(2026, 2, 1, tzinfo=UTC),
        realized_pnl=-0.5,
        account_id="   ",
    )
    by_c, _ = partition_lots_by_account([loss], [])
    assert "default" in by_c
    assert len(by_c["default"]) == 1


def test_cross_account_replacement_on_day_30_after_sell_is_flagged() -> None:
    """Inclusive ±30 window: replacement on sell_date + 30 calendar days counts."""
    sell_d = date(2026, 2, 1)
    buy_d = sell_d + timedelta(days=30)
    loss_a = ClosedLot(
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
    repl_b = TaxLot(
        id="rb",
        symbol="SPY",
        qty=1.0,
        cost_per_share=85.0,
        opened_at=datetime.combine(buy_d, datetime.min.time()).replace(tzinfo=UTC),
        account_id="b",
    )
    flags = detect_cross_account_wash_sales({"a": [loss_a]}, {"b": [repl_b]}, window_days=30)
    assert len(flags) == 1
    assert flags[0].replacement_date == buy_d


def test_cross_account_replacement_on_day_31_after_sell_not_flagged() -> None:
    sell_d = date(2026, 2, 1)
    buy_d = sell_d + timedelta(days=31)
    loss_a = ClosedLot(
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
    repl_b = TaxLot(
        id="rb",
        symbol="SPY",
        qty=1.0,
        cost_per_share=85.0,
        opened_at=datetime.combine(buy_d, datetime.min.time()).replace(tzinfo=UTC),
        account_id="b",
    )
    assert detect_cross_account_wash_sales({"a": [loss_a]}, {"b": [repl_b]}, window_days=30) == []


def test_cross_account_picks_earliest_replacement_among_two_buying_accounts() -> None:
    loss_a = ClosedLot(
        lot_id="la",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=90.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        realized_pnl=-10.0,
        account_id="a",
    )
    late_b = TaxLot(
        id="late",
        symbol="SPY",
        qty=1.0,
        cost_per_share=88.0,
        opened_at=datetime(2026, 3, 10, 12, 0, tzinfo=UTC),
        account_id="b",
    )
    early_c = TaxLot(
        id="early",
        symbol="SPY",
        qty=1.0,
        cost_per_share=87.0,
        opened_at=datetime(2026, 3, 5, 12, 0, tzinfo=UTC),
        account_id="c",
    )
    flags = detect_cross_account_wash_sales(
        {"a": [loss_a]},
        {"b": [late_b], "c": [early_c]},
        window_days=30,
    )
    assert len(flags) == 1
    assert flags[0].buying_account == "c"
    assert flags[0].replacement_date == date(2026, 3, 5)


def test_cross_account_replacement_from_closed_lot_in_other_account() -> None:
    """Purchase in B recorded as a closed lot (opened_at in window) still qualifies."""
    loss_a = ClosedLot(
        lot_id="la",
        symbol="IWM",
        qty=10.0,
        cost_per_share=100.0,
        sell_price=90.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        realized_pnl=-100.0,
        account_id="a",
    )
    buy_then_sell_b = ClosedLot(
        lot_id="lb",
        symbol="IWM",
        qty=10.0,
        cost_per_share=92.0,
        sell_price=95.0,
        opened_at=datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
        realized_pnl=30.0,
        account_id="b",
    )
    flags = detect_cross_account_wash_sales(
        {"a": [loss_a], "b": [buy_then_sell_b]}, {}, window_days=30
    )
    assert len(flags) == 1
    assert flags[0].buying_account == "b"


def test_cross_account_no_flag_for_gain_in_selling_account() -> None:
    gain_a = ClosedLot(
        lot_id="g",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        sell_price=120.0,
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        closed_at=datetime(2026, 2, 1, tzinfo=UTC),
        realized_pnl=20.0,
        account_id="a",
    )
    repl_b = TaxLot(
        id="rb",
        symbol="SPY",
        qty=1.0,
        cost_per_share=110.0,
        opened_at=datetime(2026, 2, 5, tzinfo=UTC),
        account_id="b",
    )
    assert detect_cross_account_wash_sales({"a": [gain_a]}, {"b": [repl_b]}) == []
