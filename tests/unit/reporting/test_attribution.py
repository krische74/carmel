"""Tests for position-level P&L contribution."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003

import pytest

from src.portfolio.tax_lots import LotLedger
from src.reporting.attribution import compute_brinson_attribution, compute_position_contributions


def test_contributions_empty_portfolio(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "a.db")
    assert compute_position_contributions(led, {}) == []


def test_contributions_single_symbol_unrealized_only(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "a.db")
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    rows = compute_position_contributions(led, {"SPY": 120.0})
    assert len(rows) == 1
    r = rows[0]
    assert r.symbol == "SPY"
    assert r.total_qty == pytest.approx(10.0)
    assert r.realized_pnl == pytest.approx(0.0)
    assert r.unrealized_pnl == pytest.approx(200.0)
    assert r.total_pnl == pytest.approx(200.0)
    assert r.weight_pct == pytest.approx(100.0)
    assert r.contribution_pct == pytest.approx((200.0 / 1000.0) * 100.0)


def test_contributions_multi_symbol_with_realized(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "a.db")
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_buy("QQQ", 5.0, 200.0, datetime(2026, 1, 2, tzinfo=UTC))
    led.record_sell("SPY", 5.0, 110.0, datetime(2026, 2, 1, tzinfo=UTC))
    prices = {"SPY": 105.0, "QQQ": 220.0}
    rows = compute_position_contributions(led, prices)
    by = {x.symbol: x for x in rows}
    assert by["SPY"].realized_pnl == pytest.approx(5.0 * (110.0 - 100.0))
    assert by["SPY"].total_qty == pytest.approx(5.0)
    assert by["QQQ"].realized_pnl == pytest.approx(0.0)
    assert by["QQQ"].unrealized_pnl == pytest.approx(5.0 * (220.0 - 200.0))


def test_contributions_weight_pct_sums_to_approximately_100(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "a.db")
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_buy("QQQ", 10.0, 100.0, datetime(2026, 1, 2, tzinfo=UTC))
    prices = {"SPY": 100.0, "QQQ": 100.0}
    rows = compute_position_contributions(led, prices)
    s = sum(r.weight_pct for r in rows)
    assert s == pytest.approx(100.0)


def test_brinson_equal_weights_matching_returns_zero_effects() -> None:
    wp = {"A": 0.5, "B": 0.5}
    wb = {"A": 0.5, "B": 0.5}
    rp = {"A": 0.1, "B": 0.1}
    rb = {"A": 0.1, "B": 0.1}
    s = compute_brinson_attribution(wp, wb, rp, rb)
    assert abs(s.total_active_return) < 1e-9
    assert abs(s.total_allocation) < 1e-9
    assert abs(s.total_selection) < 1e-9
    assert abs(s.total_interaction) < 1e-9


def test_brinson_overweight_better_benchmark_asset_positive_allocation() -> None:
    wp = {"A": 0.7, "B": 0.3}
    wb = {"A": 0.5, "B": 0.5}
    rp = {"A": 0.1, "B": 0.1}
    rb = {"A": 0.2, "B": 0.0}
    s = compute_brinson_attribution(wp, wb, rp, rb)
    assert s.total_allocation > 0.0


def test_brinson_selection_effect_matches_benchmark_weight_times_excess_return() -> None:
    wp = {"A": 0.5, "B": 0.5}
    wb = {"A": 0.5, "B": 0.5}
    rp = {"A": 0.1, "B": 0.05}
    rb = {"A": 0.05, "B": 0.05}
    s = compute_brinson_attribution(wp, wb, rp, rb)
    expected_sel = 0.5 * (0.1 - 0.05) + 0.5 * (0.05 - 0.05)
    assert abs(s.total_selection - expected_sel) < 1e-9


def test_brinson_three_effects_sum_to_active_return() -> None:
    wp = {"A": 0.6, "B": 0.4}
    wb = {"A": 0.5, "B": 0.5}
    rp = {"A": 0.1, "B": 0.05}
    rb = {"A": 0.08, "B": 0.02}
    s = compute_brinson_attribution(wp, wb, rp, rb)
    rb_total = sum(float(wb[k]) * float(rb[k]) for k in wb)
    rp_total = sum(float(wp[k]) * float(rp[k]) for k in wp)
    assert abs(s.total_active_return - (rp_total - rb_total)) < 1e-9
    summed = s.total_allocation + s.total_selection + s.total_interaction
    assert abs(summed - s.total_active_return) < 1e-9
