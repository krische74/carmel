"""Unit tests for portfolio return metrics from equity snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reporting.returns import (
    compute_daily_returns,
    compute_drawdown_series,
    compute_return_metrics,
    compute_rolling_sharpe,
)


def test_daily_returns_empty_snapshots() -> None:
    assert len(compute_daily_returns([])) == 0


def test_daily_returns_single_snapshot() -> None:
    snaps = [
        {
            "date": "2026-01-01",
            "total_pnl": 0.0,
            "total_market_value": 10_000.0,
            "cash": 0.0,
        },
    ]
    assert len(compute_daily_returns(snaps)) == 0


def test_daily_returns_uses_equity_denominator() -> None:
    """r_t = dP&L / (MV + cash) on the prior row — not positions-only MV."""
    snaps = [
        {
            "date": "2026-01-01",
            "total_pnl": 0.0,
            "total_market_value": 5_000.0,
            "cash": 5_000.0,
        },
        {
            "date": "2026-01-02",
            "total_pnl": 500.0,
            "total_market_value": 5_500.0,
            "cash": 5_000.0,
        },
    ]
    r = compute_daily_returns(snaps)
    assert len(r) == 1
    assert r.iloc[0] == pytest.approx(0.05)  # 500 / 10000, not 500 / 5000


def test_daily_returns_skips_pair_when_cash_is_null() -> None:
    snaps = [
        {
            "date": "2026-01-01",
            "total_pnl": 0.0,
            "total_market_value": 10_000.0,
            "cash": None,
        },
        {
            "date": "2026-01-02",
            "total_pnl": 500.0,
            "total_market_value": 10_500.0,
            "cash": 0.0,
        },
        {
            "date": "2026-01-03",
            "total_pnl": 600.0,
            "total_market_value": 10_600.0,
            "cash": 0.0,
        },
    ]
    r = compute_daily_returns(snaps)
    assert len(r) == 1
    assert str(r.index[0]) == "2026-01-03"
    assert r.iloc[0] == pytest.approx(100.0 / 10_500.0)


def test_daily_returns_skips_nonpositive_equity() -> None:
    snaps = [
        {
            "date": "2026-01-01",
            "total_pnl": 0.0,
            "total_market_value": 0.0,
            "cash": 0.0,
        },
        {
            "date": "2026-01-02",
            "total_pnl": 10.0,
            "total_market_value": 10.0,
            "cash": 0.0,
        },
    ]
    assert len(compute_daily_returns(snaps)) == 0


def test_daily_returns_adjusts_for_deposits() -> None:
    """DCA at unchanged prices: P&L flat — daily return is 0%."""
    snaps = [
        {
            "date": "2026-01-01",
            "total_pnl": 0.0,
            "total_market_value": 10_000.0,
            "cash": 0.0,
        },
        {
            "date": "2026-01-02",
            "total_pnl": 0.0,
            "total_market_value": 15_000.0,
            "cash": 0.0,
        },
    ]
    r = compute_daily_returns(snaps)
    assert len(r) == 1
    assert r.iloc[0] == pytest.approx(0.0)


def test_daily_returns_sweep_does_not_inflate_return() -> None:
    """Cash→BIL sweep raises MV but equity is flat → 0% return (Tier 46 cliff fix)."""
    snaps = [
        {
            "date": "2026-08-09",
            "total_pnl": 100.0,
            "total_market_value": 1_745.0,
            "cash": 1_662.0,
        },
        {
            "date": "2026-08-10",
            "total_pnl": 106.0,
            "total_market_value": 3_172.0,
            "cash": 241.0,
        },
    ]
    r = compute_daily_returns(snaps)
    assert len(r) == 1
    # 6 / 3407 ≈ 0.18%, not 6/1745 ≈ 0.34% and nowhere near +82%
    assert r.iloc[0] == pytest.approx(6.0 / 3_407.0)
    assert abs(r.iloc[0]) < 0.01


def test_daily_returns_prefers_broker_equity_denominator() -> None:
    snaps = [
        {
            "date": "2026-08-17",
            "total_pnl": 100.0,
            "total_market_value": 2_000.0,
            "cash": 1_000.0,
            "broker_equity": 3_434.0,
        },
        {
            "date": "2026-08-18",
            "total_pnl": 134.34,
            "total_market_value": 2_100.0,
            "cash": 1_000.0,
            "broker_equity": 3_468.34,
        },
    ]
    r = compute_daily_returns(snaps)
    assert len(r) == 1
    assert r.iloc[0] == pytest.approx(34.34 / 3_434.0)


def test_daily_returns_skips_first_basis_change_pair() -> None:
    """Cutover MV+cash -> broker_equity must not book the ~$592 / -17% phantom."""
    snaps = [
        {
            "date": "2026-08-17",
            "total_pnl": 0.0,
            "total_market_value": 2_842.0,
            "cash": 0.0,
            "broker_equity": None,
        },
        {
            "date": "2026-08-18",
            "total_pnl": 0.0,
            "total_market_value": 2_842.0,
            "cash": 0.0,
            "broker_equity": 3_434.0,
        },
        {
            "date": "2026-08-19",
            "total_pnl": 10.0,
            "total_market_value": 2_852.0,
            "cash": 0.0,
            "broker_equity": 3_444.0,
        },
    ]
    r = compute_daily_returns(snaps)
    assert len(r) == 1
    assert str(r.index[0]) == "2026-08-19"
    assert r.iloc[0] == pytest.approx(10.0 / 3_434.0)
    assert abs(r.iloc[0]) < 0.05


def test_return_metrics_uptrend() -> None:
    daily = pd.Series([0.001, 0.002, 0.0015, 0.001])
    m = compute_return_metrics(daily)
    assert m.trading_days == 4
    assert m.total_return_pct > 0
    assert m.cagr_pct > 0
    assert m.sharpe_ratio is not None
    assert m.sharpe_ratio > 0
    assert m.max_drawdown_pct >= 0
    assert m.max_drawdown_pct < 1.0


def test_return_metrics_with_drawdown() -> None:
    """Peak, trough, recovery — known max drawdown depth and duration."""
    r_list = [0.10, -0.05, -0.10, 0.15]
    daily = pd.Series(r_list)
    m = compute_return_metrics(daily)
    wealth = np.cumprod(1.0 + np.array(r_list))
    rm = np.maximum.accumulate(wealth)
    expected_dd = float(abs(np.min(wealth / rm - 1.0)) * 100.0)
    assert m.max_drawdown_pct == pytest.approx(expected_dd, rel=1e-9)
    assert m.max_drawdown_duration_days >= 1


def test_return_metrics_empty_returns_all_zero_or_none() -> None:
    m = compute_return_metrics(pd.Series(dtype=float))
    assert m.trading_days == 0
    assert m.total_return_pct == 0.0
    assert m.sharpe_ratio is None
    assert m.sortino_ratio is None
    assert m.calmar_ratio is None


def test_compute_drawdown_series_correct_shape_and_values() -> None:
    """5-day path with dip and full recovery to prior peak."""
    r = pd.Series([0.0, 0.2, -0.16666666666666666, 0.2])
    dd = compute_drawdown_series(r)
    assert len(dd) == len(r)
    assert dd.iloc[0] == pytest.approx(0.0)
    w = np.cumprod(1.0 + r.to_numpy(dtype=float))
    rm = np.maximum.accumulate(w)
    expected = w / rm - 1.0
    np.testing.assert_allclose(dd.to_numpy(), expected, rtol=1e-9)
    trough = float(dd.min())
    assert trough < -0.01
    assert dd.iloc[-1] == pytest.approx(0.0)


def test_compute_drawdown_series_empty_input() -> None:
    out = compute_drawdown_series(pd.Series(dtype=float))
    assert len(out) == 0


def test_compute_rolling_sharpe_correct_length() -> None:
    rng = np.random.default_rng(42)
    r = pd.Series(rng.standard_normal(60) * 0.01)
    out = compute_rolling_sharpe(r, window=30)
    assert len(out) == 31


def test_excess_return_sharpe_near_zero_for_risk_free_series() -> None:
    """Constant daily return at the risk-free rate yields Sharpe ~ 0."""
    daily_rf = 0.04 / 252.0
    daily = pd.Series([daily_rf] * 60)
    m = compute_return_metrics(daily, risk_free_rate_annual=4.0)
    assert m.sharpe_ratio == pytest.approx(0.0, abs=1e-9)
    assert m.sortino_ratio == pytest.approx(0.0, abs=1e-9)


def test_excess_sharpe_uses_per_day_risk_free_series() -> None:
    idx = pd.Index([f"2020-01-{i:02d}" for i in range(2, 32)], name="date")
    daily = pd.Series([0.001] * len(idx), index=idx)
    rf = pd.Series([0.001] * len(idx), index=idx)
    m = compute_return_metrics(daily, risk_free_rate_annual=0.0, risk_free_daily=rf)
    assert m.sharpe_ratio == pytest.approx(0.0, abs=1e-9)


def test_compute_rolling_sharpe_insufficient_data() -> None:
    r = pd.Series(np.linspace(0.001, 0.01, 10))
    out = compute_rolling_sharpe(r, window=30)
    assert len(out) == 0


def test_compute_rolling_sharpe_positive_for_uptrend() -> None:
    """Monotonically increasing daily returns -> all rolling Sharpe values positive."""
    n = 60
    inc = np.linspace(0.0001, 0.002, n)
    r = pd.Series(inc)
    out = compute_rolling_sharpe(r, window=30)
    assert len(out) == n - 30 + 1
    assert (out > 0).all()
