"""Tier 30: tax lot ledger partitioned by ``account_id``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.portfolio.tax_lots import LotLedger

if TYPE_CHECKING:
    from pathlib import Path


def test_lot_ledger_partitions_by_account(tmp_path: Path) -> None:
    """Buys in two accounts stay isolated when filtering ``get_open_lots``."""
    db = tmp_path / "lots_multi.db"
    led = LotLedger(db)
    t = datetime(2026, 1, 15, 16, 0, tzinfo=UTC)
    led.record_buy("SPY", 10.0, 400.0, t, account_id="acct_a")
    led.record_buy("QQQ", 5.0, 350.0, t, account_id="acct_b")

    open_a = led.get_open_lots(account_id="acct_a")
    open_b = led.get_open_lots(account_id="acct_b")
    assert len(open_a) == 1
    assert len(open_b) == 1
    assert open_a[0].symbol == "SPY"
    assert open_a[0].account_id == "acct_a"
    assert open_b[0].symbol == "QQQ"
    assert open_b[0].account_id == "acct_b"

    all_open = led.get_open_lots(account_id=None)
    assert len(all_open) == 2


def test_detect_wash_sales_none_merges_intra_account_per_hub_account(tmp_path: Path) -> None:
    """``account_id=None`` with multiple hub accounts runs wash detection per account, then merges."""
    db = tmp_path / "wash_merge.db"
    led = LotLedger(db)
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    t_loss_a = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    t_repl_a = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t0, account_id="a")
    led.record_sell("SPY", 1.0, 80.0, t_loss_a, account_id="a")
    led.record_buy("SPY", 1.0, 85.0, t_repl_a, account_id="a")
    led.record_buy("QQQ", 1.0, 50.0, t0, account_id="b")
    led.record_sell("QQQ", 1.0, 60.0, t_loss_a, account_id="b")

    merged = led.detect_wash_sales(account_id=None)
    assert len(merged) == 1
    assert merged[0].symbol == "SPY"


def test_detect_wash_sales_none_sorted_by_sell_date_across_accounts(tmp_path: Path) -> None:
    """Merged washes from two accounts are ordered by ``sell_date``."""
    db = tmp_path / "wash_sort.db"
    led = LotLedger(db)
    o_a = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    sell_a = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    repl_a = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)
    o_b = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    sell_b = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    repl_b = datetime(2026, 6, 4, 12, 0, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, o_a, account_id="a")
    led.record_sell("SPY", 1.0, 80.0, sell_a, account_id="a")
    led.record_buy("SPY", 1.0, 85.0, repl_a, account_id="a")
    led.record_buy("IWM", 1.0, 200.0, o_b, account_id="b")
    led.record_sell("IWM", 1.0, 180.0, sell_b, account_id="b")
    led.record_buy("IWM", 1.0, 185.0, repl_b, account_id="b")

    merged = led.detect_wash_sales(account_id=None)
    assert len(merged) == 2
    assert merged[0].symbol == "SPY"
    assert merged[1].symbol == "IWM"
    assert merged[0].sell_date < merged[1].sell_date


def test_detect_wash_sales_scoped_account_ignores_other_accounts(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "wash_scope.db")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t_loss = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    t_repl = datetime(2026, 4, 5, 12, 0, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t0, account_id="a")
    led.record_sell("SPY", 1.0, 80.0, t_loss, account_id="a")
    led.record_buy("SPY", 1.0, 85.0, t_repl, account_id="a")
    led.record_buy("QQQ", 1.0, 40.0, t0, account_id="b")
    led.record_sell("QQQ", 1.0, 30.0, t_loss, account_id="b")
    led.record_buy("QQQ", 1.0, 32.0, t_repl, account_id="b")

    only_a = led.detect_wash_sales(account_id="a")
    assert len(only_a) == 1
    assert only_a[0].symbol == "SPY"

    only_b = led.detect_wash_sales(account_id="b")
    assert len(only_b) == 1
    assert only_b[0].symbol == "QQQ"
    assert only_a[0].closed_lot_id != only_b[0].closed_lot_id


def test_detect_wash_sales_other_accounts_replacement_does_not_wash_loss(tmp_path: Path) -> None:
    """Cross-account replacement is not intra-account wash; merged ``None`` path stays per-account."""
    led = LotLedger(tmp_path / "wash_xacct.db")
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    t_loss = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    t_repl = datetime(2026, 5, 5, 12, 0, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t0, account_id="a")
    led.record_sell("SPY", 1.0, 70.0, t_loss, account_id="a")
    led.record_buy("SPY", 1.0, 75.0, t_repl, account_id="b")

    assert led.detect_wash_sales(account_id=None) == []
    assert led.detect_wash_sales(account_id="a") == []
