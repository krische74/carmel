"""Tier 32: IRA tax-advantaged treatment in tax summaries (informational)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.portfolio.tax_lots import LotLedger
from src.reporting.tax_report import (
    compute_tax_summary,
    consolidated_taxable_and_advantaged,
    form_8949_csv_string,
    generate_schedule_d_summary,
)


def test_ira_traditional_excluded_from_taxable_summary(tmp_path) -> None:
    led = LotLedger(tmp_path / "ira_tr.sqlite")
    t = datetime(2026, 4, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="ira")
    led.record_sell("SPY", 1.0, 200.0, t, account_id="ira")
    hub = {"ira": "ira_traditional"}
    s = compute_tax_summary(led, 2026, account_id="ira", hub_account_types=hub)
    assert s.short_term_net == pytest.approx(0.0)
    assert s.total_net == pytest.approx(0.0)
    assert len(s.lots) == 1
    assert s.lots[0].tax_treatment == "tax_deferred"


def test_ira_roth_marked_tax_free(tmp_path) -> None:
    led = LotLedger(tmp_path / "ira_r.sqlite")
    t = datetime(2026, 5, 1, tzinfo=UTC)
    led.record_buy("VOO", 2.0, 300.0, t, account_id="roth")
    led.record_sell("VOO", 2.0, 400.0, t, account_id="roth")
    hub = {"roth": "ira_roth"}
    s = compute_tax_summary(led, 2026, account_id="roth", hub_account_types=hub)
    assert s.lots[0].tax_treatment == "tax_free"


def test_brokerage_included_in_taxable_summary(tmp_path) -> None:
    led = LotLedger(tmp_path / "brk.sqlite")
    t = datetime(2026, 6, 1, tzinfo=UTC)
    led.record_buy("QQQ", 1.0, 100.0, t, account_id="main")
    led.record_sell("QQQ", 1.0, 130.0, t, account_id="main")
    hub = {"main": "brokerage"}
    s = compute_tax_summary(led, 2026, account_id="main", hub_account_types=hub)
    assert s.short_term_net == pytest.approx(30.0)
    assert s.lots[0].tax_treatment == "taxable"


def test_consolidated_separates_taxable_and_ira(tmp_path) -> None:
    led = LotLedger(tmp_path / "mix.sqlite")
    t = datetime(2026, 7, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="main")
    led.record_sell("SPY", 1.0, 120.0, t, account_id="main")
    led.record_buy("GLD", 1.0, 200.0, t, account_id="ira")
    led.record_sell("GLD", 1.0, 210.0, t, account_id="ira")
    hub = {"main": "brokerage", "ira": "ira_traditional"}
    s = compute_tax_summary(led, 2026, account_id=None, hub_account_types=hub)
    assert s.short_term_net == pytest.approx(20.0)
    tax_part, adv_part = consolidated_taxable_and_advantaged(s)
    assert len(tax_part.lots) == 1
    assert len(adv_part.lots) == 1
    assert tax_part.lots[0].symbol == "SPY"
    assert adv_part.lots[0].symbol == "GLD"


def test_form_8949_ira_header_note(tmp_path) -> None:
    led = LotLedger(tmp_path / "ira_hdr.sqlite")
    t = datetime(2026, 8, 1, tzinfo=UTC)
    led.record_buy("X", 1.0, 10.0, t, account_id="ira")
    led.record_sell("X", 1.0, 12.0, t, account_id="ira")
    hub = {"ira": "ira_traditional"}
    s = compute_tax_summary(led, 2026, account_id="ira", hub_account_types=hub)
    csv_text = form_8949_csv_string(s)
    assert "IRA — Not Reportable on Form 8949" in csv_text


def test_schedule_d_excludes_ira_when_consolidated(tmp_path) -> None:
    led = LotLedger(tmp_path / "sd_ira.sqlite")
    t = datetime(2026, 9, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="main")
    led.record_sell("SPY", 1.0, 150.0, t, account_id="main")
    led.record_buy("GLD", 1.0, 50.0, t, account_id="ira")
    led.record_sell("GLD", 1.0, 40.0, t, account_id="ira")
    hub = {"main": "brokerage", "ira": "ira_traditional"}
    s = compute_tax_summary(led, 2026, account_id=None, hub_account_types=hub)
    sd = generate_schedule_d_summary(s, wash_sales=None)
    assert sd.short_term_net == pytest.approx(50.0)
    assert "GLD" not in repr(sd)
