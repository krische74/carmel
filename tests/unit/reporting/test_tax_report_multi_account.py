"""Per-account tax summary and Form 8949 scope row."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.portfolio.tax_lots import LotLedger
from src.portfolio.wash_sales import WashSale
from src.reporting.tax_report import (
    compute_tax_summary,
    export_form_8949_csv,
    form_8949_csv_string,
)


def test_tax_summary_filtered_by_account(tmp_path) -> None:
    led = LotLedger(tmp_path / "hub.sqlite")
    t = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="a1")
    led.record_sell("SPY", 1.0, 110.0, t, account_id="a1")
    led.record_buy("QQQ", 2.0, 50.0, t, account_id="a2")
    led.record_sell("QQQ", 2.0, 60.0, t, account_id="a2")
    s1 = compute_tax_summary(led, 2026, account_id="a1")
    assert len(s1.lots) == 1
    assert s1.lots[0].symbol == "SPY"
    assert s1.lots[0].account_id == "a1"
    s2 = compute_tax_summary(led, 2026, account_id="a2")
    assert len(s2.lots) == 1
    assert s2.lots[0].symbol == "QQQ"


def test_tax_summary_all_accounts_consolidated(tmp_path) -> None:
    led = LotLedger(tmp_path / "hub2.sqlite")
    t = datetime(2026, 4, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="x")
    led.record_sell("SPY", 1.0, 120.0, t, account_id="x")
    led.record_buy("GLD", 1.0, 200.0, t, account_id="y")
    led.record_sell("GLD", 1.0, 190.0, t, account_id="y")
    s = compute_tax_summary(led, 2026, account_id=None)
    assert len(s.lots) == 2
    syms = {lot.symbol for lot in s.lots}
    assert syms == {"SPY", "GLD"}


def test_export_form_8949_csv_per_account_single_lot_row(tmp_path) -> None:
    """Tier 30: per-account summary → CSV has one lot row (plus scope + header)."""
    led = LotLedger(tmp_path / "hub_export.sqlite")
    t = datetime(2026, 5, 1, tzinfo=UTC)
    led.record_buy("Z", 1.0, 10.0, t, account_id="ira")
    led.record_sell("Z", 1.0, 12.0, t, account_id="ira")
    led.record_buy("Y", 1.0, 20.0, t, account_id="taxable")
    led.record_sell("Y", 1.0, 22.0, t, account_id="taxable")
    s = compute_tax_summary(led, 2026, account_id="ira")
    out = tmp_path / "f8949_ira.csv"
    export_form_8949_csv(s, out, wash_sales=[], account_label="ira")
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 6
    assert "Report scope" in lines[0]
    assert "Form 8949 Part I - Short-Term" in lines[1]
    assert "(a) Description" in lines[2]
    assert any("Z" in ln and "Part" not in ln for ln in lines)
    assert not any("Y" in ln and "sh Y" in ln for ln in lines)


def test_form_8949_csv_includes_account_scope_row(tmp_path) -> None:
    led = LotLedger(tmp_path / "hub3.sqlite")
    t = datetime(2026, 5, 1, tzinfo=UTC)
    led.record_buy("Z", 1.0, 10.0, t, account_id="ira")
    led.record_sell("Z", 1.0, 12.0, t, account_id="ira")
    s = compute_tax_summary(led, 2026, account_id="ira")
    csv_text = form_8949_csv_string(s, account_label="ira")
    lines = csv_text.strip().splitlines()
    assert "Report scope" in lines[0]
    assert "Account: ira" in lines[0]
    assert "Form 8949 Part I - Short-Term" in lines[1]
    assert "(a) Description" in lines[2]


def test_form_8949_wash_column_keys_by_lot_id(tmp_path) -> None:
    """Wash adjustment column matches ``TaxLotSummary.lot_id`` via ``closed_lot_id`` prefix."""
    led = LotLedger(tmp_path / "hub4.sqlite")
    t = datetime(2026, 6, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t)
    closed = led.record_sell("SPY", 1.0, 90.0, t)
    lot_id = closed[0].lot_id
    closed_at = closed[0].closed_at
    if closed_at.tzinfo is None:
        closed_at = closed_at.replace(tzinfo=UTC)
    ws = [
        WashSale(
            closed_lot_id=f"{lot_id}|{closed_at.isoformat()}",
            symbol="SPY",
            disallowed_loss=10.0,
            replacement_lot_id=None,
            sell_date=closed_at.date(),
            buy_date=closed_at.date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    csv_text = form_8949_csv_string(s, wash_sales=ws)
    assert "10.00" in csv_text
    assert ",W," in csv_text


def test_tax_summary_unknown_account_id_returns_no_lots(tmp_path) -> None:
    led = LotLedger(tmp_path / "hub_empty_acct.sqlite")
    t = datetime(2026, 7, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="real")
    led.record_sell("SPY", 1.0, 110.0, t, account_id="real")
    s = compute_tax_summary(led, 2026, account_id="no_such_hub_account")
    assert s.lots == []
    assert s.total_net == 0.0


def test_form_8949_wash_column_sums_multiple_rows_same_lot(tmp_path) -> None:
    """Several ``WashSale`` rows keyed to the same lot id aggregate in the CSV."""
    led = LotLedger(tmp_path / "hub_wash_sum.sqlite")
    t = datetime(2026, 8, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t)
    closed = led.record_sell("SPY", 1.0, 90.0, t)
    lid = closed[0].lot_id
    ca = closed[0].closed_at
    if ca.tzinfo is None:
        ca = ca.replace(tzinfo=UTC)
    cid = f"{lid}|{ca.isoformat()}"
    ws = [
        WashSale(
            closed_lot_id=cid,
            symbol="SPY",
            disallowed_loss=3.0,
            replacement_lot_id=None,
            sell_date=ca.date(),
            buy_date=ca.date(),
        ),
        WashSale(
            closed_lot_id=cid,
            symbol="SPY",
            disallowed_loss=5.0,
            replacement_lot_id=None,
            sell_date=ca.date(),
            buy_date=ca.date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    csv_text = form_8949_csv_string(s, wash_sales=ws)
    assert "8.00" in csv_text


def test_form_8949_closed_lot_id_without_pipe_still_maps_wash(tmp_path) -> None:
    led = LotLedger(tmp_path / "hub_pipe.sqlite")
    t = datetime(2026, 9, 1, tzinfo=UTC)
    led.record_buy("QQQ", 1.0, 50.0, t)
    closed = led.record_sell("QQQ", 1.0, 40.0, t)
    lid = closed[0].lot_id
    ws = [
        WashSale(
            closed_lot_id=lid,
            symbol="QQQ",
            disallowed_loss=2.5,
            replacement_lot_id=None,
            sell_date=closed[0].closed_at.date(),
            buy_date=closed[0].closed_at.date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    csv_text = form_8949_csv_string(s, wash_sales=ws)
    assert "2.50" in csv_text


def test_compute_tax_summary_wash_year_mismatch_excluded_from_disallowed(tmp_path) -> None:
    """``wash_sale_disallowed`` only counts washes whose ``sell_date`` matches summary year."""
    led = LotLedger(tmp_path / "hub_wash_year.sqlite")
    ws = [
        WashSale(
            closed_lot_id="x",
            symbol="SPY",
            disallowed_loss=99.0,
            replacement_lot_id=None,
            sell_date=datetime(2025, 6, 1, tzinfo=UTC).date(),
            buy_date=datetime(2025, 6, 2, tzinfo=UTC).date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    assert s.wash_sale_disallowed == pytest.approx(0.0)
