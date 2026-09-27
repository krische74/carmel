"""Tier 31: Form 8949 Part I/II and IRS adjustment columns."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.portfolio.tax_lots import LotLedger
from src.portfolio.wash_sales import WashSale
from src.reporting.tax_report import (
    build_form_8949_rows,
    compute_tax_summary,
    form_8949_csv_string,
)


def test_form_8949_part_i_short_term_only(tmp_path) -> None:
    led = LotLedger(tmp_path / "f8949_st.sqlite")
    t = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t)
    led.record_sell("SPY", 1.0, 110.0, t)
    s = compute_tax_summary(led, 2026)
    csv_text = form_8949_csv_string(s)
    assert "Form 8949 Part I - Short-Term" in csv_text
    assert "Form 8949 Part II - Long-Term" in csv_text
    assert "SPY" in csv_text
    rows = build_form_8949_rows(s, None)
    assert len(rows) == 1
    assert rows[0].term == "short"


def test_form_8949_part_ii_long_term_only(tmp_path) -> None:
    led = LotLedger(tmp_path / "f8949_lt.sqlite")
    o = datetime(2024, 1, 1, tzinfo=UTC)
    c = datetime(2026, 2, 1, tzinfo=UTC)
    led.record_buy("QQQ", 2.0, 200.0, o)
    led.record_sell("QQQ", 2.0, 250.0, c)
    s = compute_tax_summary(led, 2026)
    csv_text = form_8949_csv_string(s)
    part_i = csv_text.index("Form 8949 Part I - Short-Term")
    part_ii = csv_text.index("Form 8949 Part II - Long-Term")
    assert part_i < part_ii
    rows = build_form_8949_rows(s, None)
    assert len(rows) == 1
    assert rows[0].term == "long"
    assert "QQQ" in csv_text


def test_form_8949_wash_sale_code_w(tmp_path) -> None:
    led = LotLedger(tmp_path / "f8949_w.sqlite")
    t = datetime(2026, 4, 1, tzinfo=UTC)
    led.record_buy("X", 1.0, 50.0, t)
    closed = led.record_sell("X", 1.0, 40.0, t)
    lid = closed[0].lot_id
    ca = closed[0].closed_at
    if ca.tzinfo is None:
        ca = ca.replace(tzinfo=UTC)
    ws = [
        WashSale(
            closed_lot_id=f"{lid}|{ca.isoformat()}",
            symbol="X",
            disallowed_loss=5.0,
            replacement_lot_id=None,
            sell_date=ca.date(),
            buy_date=ca.date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    rows = build_form_8949_rows(s, ws)
    assert len(rows) == 1
    assert rows[0].adjustment_code == "W"
    assert rows[0].adjustment_amount == pytest.approx(5.0)


def test_form_8949_no_adjustment_empty_code(tmp_path) -> None:
    led = LotLedger(tmp_path / "f8949_no_w.sqlite")
    t = datetime(2026, 5, 1, tzinfo=UTC)
    led.record_buy("Z", 1.0, 10.0, t)
    led.record_sell("Z", 1.0, 12.0, t)
    s = compute_tax_summary(led, 2026)
    rows = build_form_8949_rows(s, [])
    assert rows[0].adjustment_code == ""
    assert rows[0].adjustment_amount == pytest.approx(0.0)


def test_form_8949_adjusted_basis(tmp_path) -> None:
    led = LotLedger(tmp_path / "f8949_adj.sqlite")
    t = datetime(2026, 6, 10, tzinfo=UTC)
    led.record_buy("A", 1.0, 100.0, t)
    closed = led.record_sell("A", 1.0, 80.0, t)
    lid = closed[0].lot_id
    ca = closed[0].closed_at
    if ca.tzinfo is None:
        ca = ca.replace(tzinfo=UTC)
    ws = [
        WashSale(
            closed_lot_id=lid,
            symbol="A",
            disallowed_loss=10.0,
            replacement_lot_id=None,
            sell_date=ca.date(),
            buy_date=ca.date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    rows = build_form_8949_rows(s, ws)
    assert rows[0].cost_basis == pytest.approx(100.0)
    assert rows[0].adjustment_amount == pytest.approx(10.0)
    assert rows[0].gain_or_loss == pytest.approx(80.0 - (100.0 + 10.0))


def test_form_8949_per_account_with_parts(tmp_path) -> None:
    led = LotLedger(tmp_path / "f8949_acct.sqlite")
    t = datetime(2026, 7, 1, tzinfo=UTC)
    led.record_buy("M", 1.0, 1.0, t, account_id="ira")
    led.record_sell("M", 1.0, 2.0, t, account_id="ira")
    s = compute_tax_summary(led, 2026, account_id="ira")
    csv_text = form_8949_csv_string(s, account_label="ira")
    assert "Account: ira" in csv_text
    assert "M" in csv_text
    assert len(build_form_8949_rows(s, None)) == 1


def test_form_8949_cumulative_wash_adjustments_same_lot(tmp_path) -> None:
    """Two ``WashSale`` rows for the same lot id sum into one adjustment (H1)."""
    led = LotLedger(tmp_path / "f8949_cum.sqlite")
    t = datetime(2026, 8, 15, tzinfo=UTC)
    led.record_buy("CUM", 1.0, 50.0, t)
    closed = led.record_sell("CUM", 1.0, 40.0, t)
    lid = closed[0].lot_id
    ca = closed[0].closed_at
    if ca.tzinfo is None:
        ca = ca.replace(tzinfo=UTC)
    ws = [
        WashSale(
            closed_lot_id=lid,
            symbol="CUM",
            disallowed_loss=3.0,
            replacement_lot_id=None,
            sell_date=ca.date(),
            buy_date=ca.date(),
        ),
        WashSale(
            closed_lot_id=f"{lid}|{ca.isoformat()}",
            symbol="CUM",
            disallowed_loss=7.0,
            replacement_lot_id=None,
            sell_date=ca.date(),
            buy_date=ca.date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    rows = build_form_8949_rows(s, ws)
    assert len(rows) == 1
    assert rows[0].adjustment_amount == pytest.approx(10.0)
    assert rows[0].adjustment_code == "W"


def test_form_8949_per_account_multiple_lots_same_account(tmp_path) -> None:
    """Per-account export includes every closed lot for that hub id (L2)."""
    led = LotLedger(tmp_path / "f8949_multi.sqlite")
    t = datetime(2026, 9, 1, tzinfo=UTC)
    led.record_buy("P", 1.0, 10.0, t, account_id="ira")
    led.record_sell("P", 1.0, 11.0, t, account_id="ira")
    led.record_buy("Q", 2.0, 5.0, t, account_id="ira")
    led.record_sell("Q", 2.0, 6.0, t, account_id="ira")
    led.record_buy("R", 1.0, 1.0, t, account_id="tax")
    led.record_sell("R", 1.0, 2.0, t, account_id="tax")
    s = compute_tax_summary(led, 2026, account_id="ira")
    rows = build_form_8949_rows(s, None)
    assert len(rows) == 2
    syms = {r.description.split()[-1] for r in rows}
    assert syms == {"P", "Q"}
    csv_text = form_8949_csv_string(s, account_label="ira")
    assert "sh P" in csv_text
    assert "sh Q" in csv_text
    assert "sh R" not in csv_text
