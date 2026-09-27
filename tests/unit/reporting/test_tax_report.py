"""Tax summary and Form 8949 CSV (informational)."""

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


def test_tax_summary_classifies_short_term_under_365_days(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    o = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    c = datetime(2025, 7, 20, 12, 0, tzinfo=UTC)
    led.record_buy("SPY", 10.0, 100.0, o)
    led.record_sell("SPY", 10.0, 110.0, c)
    s = compute_tax_summary(led, 2025)
    assert len(s.lots) == 1
    assert s.lots[0].term == "short"
    assert s.short_term_net == pytest.approx(100.0)


def test_tax_summary_classifies_long_term_over_365_days(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    o = datetime(2024, 1, 1, tzinfo=UTC)
    c = datetime(2026, 2, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, o)
    led.record_sell("SPY", 1.0, 200.0, c)
    s = compute_tax_summary(led, 2026)
    assert s.lots[0].term == "long"
    assert s.long_term_net == pytest.approx(100.0)


def test_tax_summary_separates_gains_and_losses(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    t = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("A", 1.0, 100.0, t)
    led.record_sell("A", 1.0, 150.0, t)
    led.record_buy("B", 1.0, 100.0, t)
    led.record_sell("B", 1.0, 80.0, t)
    s = compute_tax_summary(led, 2026)
    assert s.short_term_gains == pytest.approx(50.0)
    assert s.short_term_losses == pytest.approx(-20.0)


def test_tax_summary_filters_to_requested_year(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    led.record_buy("X", 1.0, 10.0, datetime(2025, 1, 1, tzinfo=UTC))
    led.record_sell("X", 1.0, 20.0, datetime(2025, 6, 1, tzinfo=UTC))
    led.record_buy("Y", 1.0, 10.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_sell("Y", 1.0, 12.0, datetime(2026, 2, 1, tzinfo=UTC))
    s = compute_tax_summary(led, 2026)
    assert len(s.lots) == 1
    assert s.lots[0].symbol == "Y"


def test_tax_summary_includes_wash_sale_disallowed(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    ws = [
        WashSale(
            closed_lot_id="x",
            symbol="SPY",
            disallowed_loss=100.0,
            replacement_lot_id=None,
            sell_date=datetime(2026, 6, 1, tzinfo=UTC).date(),
            buy_date=datetime(2026, 6, 15, tzinfo=UTC).date(),
        ),
    ]
    s = compute_tax_summary(led, 2026, wash_sales=ws)
    assert s.wash_sale_disallowed == pytest.approx(100.0)


def test_tax_summary_empty_year(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    s = compute_tax_summary(led, 2020)
    assert s.total_net == 0.0
    assert s.lots == []


def test_export_form_8949_creates_csv_with_headers(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    led.record_buy("SPY", 1.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_sell("SPY", 1.0, 90.0, datetime(2026, 6, 1, tzinfo=UTC))
    s = compute_tax_summary(led, 2026)
    p = tmp_path / "out.csv"
    export_form_8949_csv(s, p)
    text = p.read_text(encoding="utf-8")
    assert "(a) Description" in text
    assert "Form 8949 Part I - Short-Term" in text
    assert "Form 8949 Part II - Long-Term" in text
    assert "(f) Code" in text
    assert "(g) Adjustment" in text


def test_export_form_8949_includes_lot_rows(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    led.record_buy("SPY", 2.0, 50.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_sell("SPY", 2.0, 60.0, datetime(2026, 6, 1, tzinfo=UTC))
    s = compute_tax_summary(led, 2026)
    csv_text = form_8949_csv_string(s)
    lines = [ln for ln in csv_text.strip().splitlines() if ln.strip()]
    assert len(lines) >= 5
    assert any("SPY" in ln and "Part I" not in ln for ln in lines)
    assert "/2026" in csv_text


def test_form_8949_formats_dates_mm_dd_yyyy(tmp_path) -> None:
    led = LotLedger(tmp_path / "t.db")
    led.record_buy("Z", 1.0, 10.0, datetime(2026, 3, 5, tzinfo=UTC))
    led.record_sell("Z", 1.0, 12.0, datetime(2026, 4, 10, tzinfo=UTC))
    s = compute_tax_summary(led, 2026)
    csv_text = form_8949_csv_string(s)
    assert "03/05/2026" in csv_text
    assert "04/10/2026" in csv_text


def test_holding_period_calendar_date_leap_year(tmp_path) -> None:
    """C6: Calendar-date comparison handles leap year correctly.

    Bought on 2024-02-29 (leap), sold on 2025-02-28 = short-term
    (one year later would be 2025-02-28, must be *after* that date).
    """
    led = LotLedger(tmp_path / "t.db")
    led.record_buy("SPY", 1.0, 100.0, datetime(2024, 2, 29, tzinfo=UTC))
    led.record_sell("SPY", 1.0, 110.0, datetime(2025, 2, 28, tzinfo=UTC))
    s = compute_tax_summary(led, 2025)
    assert len(s.lots) == 1
    assert s.lots[0].term == "short"


def test_holding_period_exactly_one_year_is_short_term(tmp_path) -> None:
    """C6: Held exactly one year is still short-term (IRS = 'more than one year')."""
    led = LotLedger(tmp_path / "t.db")
    led.record_buy("SPY", 1.0, 100.0, datetime(2025, 3, 15, tzinfo=UTC))
    led.record_sell("SPY", 1.0, 110.0, datetime(2026, 3, 15, tzinfo=UTC))
    s = compute_tax_summary(led, 2026)
    assert len(s.lots) == 1
    assert s.lots[0].term == "short"


def test_holding_period_one_year_plus_one_day_is_long_term(tmp_path) -> None:
    """C6: One year and one day is long-term."""
    led = LotLedger(tmp_path / "t.db")
    led.record_buy("SPY", 1.0, 100.0, datetime(2025, 3, 15, tzinfo=UTC))
    led.record_sell("SPY", 1.0, 110.0, datetime(2026, 3, 16, tzinfo=UTC))
    s = compute_tax_summary(led, 2026)
    assert len(s.lots) == 1
    assert s.lots[0].term == "long"
