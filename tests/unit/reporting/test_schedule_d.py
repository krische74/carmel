"""Tier 31: Schedule D summary from tax lots + wash rows."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.portfolio.tax_lots import LotLedger
from src.portfolio.wash_sales import WashSale
from src.reporting.tax_report import (
    compute_tax_summary,
    export_schedule_d_csv,
    generate_schedule_d_summary,
    schedule_d_csv_string,
)


def test_schedule_d_short_and_long_totals(tmp_path) -> None:
    led = LotLedger(tmp_path / "sd1.sqlite")
    t = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("A", 1.0, 100.0, t)
    led.record_sell("A", 1.0, 130.0, t)
    lo = datetime(2024, 1, 1, tzinfo=UTC)
    lc = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("B", 1.0, 200.0, lo)
    led.record_sell("B", 1.0, 250.0, lc)
    s = compute_tax_summary(led, 2026)
    sd = generate_schedule_d_summary(s, wash_sales=None, account_id=None)
    assert sd.short_term_proceeds == pytest.approx(130.0)
    assert sd.short_term_cost_basis == pytest.approx(100.0)
    assert sd.short_term_net == pytest.approx(30.0)
    assert sd.long_term_proceeds == pytest.approx(250.0)
    assert sd.long_term_cost_basis == pytest.approx(200.0)
    assert sd.long_term_net == pytest.approx(50.0)
    assert sd.total_net_gain_or_loss == pytest.approx(80.0)


def test_schedule_d_wash_adjustments_included(tmp_path) -> None:
    led = LotLedger(tmp_path / "sd_w.sqlite")
    t = datetime(2026, 4, 1, tzinfo=UTC)
    led.record_buy("S", 1.0, 100.0, t)
    closed = led.record_sell("S", 1.0, 85.0, t)
    lid = closed[0].lot_id
    ca = closed[0].closed_at
    if ca.tzinfo is None:
        ca = ca.replace(tzinfo=UTC)
    ws = [
        WashSale(
            closed_lot_id=lid,
            symbol="S",
            disallowed_loss=15.0,
            replacement_lot_id=None,
            sell_date=ca.date(),
            buy_date=ca.date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    sd = generate_schedule_d_summary(s, wash_sales=ws, account_id=None)
    assert sd.short_term_wash_adjustments == pytest.approx(15.0)
    assert sd.short_term_net == pytest.approx(85.0 - (100.0 + 15.0))


def test_schedule_d_per_account(tmp_path) -> None:
    led = LotLedger(tmp_path / "sd_ac.sqlite")
    t = datetime(2026, 5, 1, tzinfo=UTC)
    led.record_buy("X", 1.0, 10.0, t, account_id="a")
    led.record_sell("X", 1.0, 20.0, t, account_id="a")
    led.record_buy("Y", 1.0, 5.0, t, account_id="b")
    led.record_sell("Y", 1.0, 8.0, t, account_id="b")
    s = compute_tax_summary(led, 2026, account_id="a")
    sd = generate_schedule_d_summary(s, wash_sales=None, account_id="a")
    assert sd.account_id == "a"
    assert sd.short_term_proceeds == pytest.approx(20.0)
    assert sd.total_net_gain_or_loss == pytest.approx(10.0)


def test_schedule_d_all_losses_year_negative_total(tmp_path) -> None:
    """Net can be negative when the year is all losses (L1)."""
    led = LotLedger(tmp_path / "sd_loss.sqlite")
    t = datetime(2026, 10, 1, tzinfo=UTC)
    led.record_buy("L1", 1.0, 100.0, t)
    led.record_sell("L1", 1.0, 80.0, t)
    led.record_buy("L2", 1.0, 50.0, t)
    led.record_sell("L2", 1.0, 40.0, t)
    s = compute_tax_summary(led, 2026)
    sd = generate_schedule_d_summary(s, wash_sales=None, account_id=None)
    assert sd.short_term_net < 0
    assert sd.total_net_gain_or_loss == pytest.approx(-30.0)


def test_schedule_d_csv_export(tmp_path) -> None:
    led = LotLedger(tmp_path / "sd_csv.sqlite")
    t = datetime(2026, 6, 1, tzinfo=UTC)
    led.record_buy("Z", 1.0, 1.0, t)
    led.record_sell("Z", 1.0, 3.0, t)
    s = compute_tax_summary(led, 2026)
    sd = generate_schedule_d_summary(s, account_id="main")
    txt = schedule_d_csv_string(sd)
    assert "Part I - Short-term proceeds" in txt
    assert "Tax year" in txt
    assert "main" in txt
    p = tmp_path / "sd_out.csv"
    export_schedule_d_csv(sd, p)
    disk = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert disk == txt.replace("\r\n", "\n")
