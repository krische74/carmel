"""Unit tests for drawdown-episode slicing and the 75/25 static blend."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.reporting.episodes import (
    DRAWDOWN_EPISODES,
    EpisodeSlice,
    buy_and_hold_equity_curve,
    episode_slice_metrics,
    recovery_trading_days,
    static_blend_equity_curve,
    strategy_usable_start_dates,
)


def test_drawdown_episodes_cover_five_named_crises() -> None:
    names = {e.name for e in DRAWDOWN_EPISODES}
    assert names == {
        "gfc_2008",
        "correction_2011",
        "q4_2018",
        "covid_2020",
        "bear_2022",
    }


def test_episode_slice_return_and_max_dd() -> None:
    """Known path: +10%, then -20% from peak, then recover."""
    curve = [
        {"date": "2020-02-03", "equity": 100.0},
        {"date": "2020-02-10", "equity": 110.0},
        {"date": "2020-02-18", "equity": 88.0},
        {"date": "2020-03-23", "equity": 88.0},
        {"date": "2020-04-01", "equity": 110.0},
    ]
    ep = EpisodeSlice(name="covid_2020", start=date(2020, 2, 3), end=date(2020, 3, 23))
    m = episode_slice_metrics(curve, ep)
    assert m.total_return_pct == pytest.approx(-12.0, rel=1e-9)
    assert m.max_drawdown_pct == pytest.approx(20.0, rel=1e-9)


def test_recovery_trading_days_from_trough_to_prior_peak() -> None:
    curve = [
        {"date": "2020-02-03", "equity": 100.0},
        {"date": "2020-02-10", "equity": 110.0},
        {"date": "2020-02-18", "equity": 88.0},
        {"date": "2020-03-23", "equity": 88.0},
        {"date": "2020-03-24", "equity": 100.0},
        {"date": "2020-04-01", "equity": 110.0},
    ]
    days = recovery_trading_days(curve, start=date(2020, 2, 3), end=date(2020, 3, 23))
    # Trough 2020-02-18; first restore of 110 is 2020-04-01 → 3 later curve points.
    assert days == 3


def test_recovery_trading_days_none_if_never_recovers() -> None:
    curve = [
        {"date": "2022-01-03", "equity": 100.0},
        {"date": "2022-06-01", "equity": 80.0},
        {"date": "2022-10-12", "equity": 75.0},
    ]
    days = recovery_trading_days(curve, start=date(2022, 1, 3), end=date(2022, 10, 12))
    assert days is None


def test_static_blend_is_75_spy_25_cash() -> None:
    """Daily-rebalanced 75/25: r = 0.75 * spy + 0.25 * cash_yield_daily."""
    idx = pd.bdate_range("2024-01-02", periods=3, freq="B")
    spy = pd.DataFrame({"close": [100.0, 110.0, 110.0]}, index=idx)
    curve = static_blend_equity_curve(
        spy,
        initial_capital=10_000.0,
        spy_weight=0.75,
        cash_yield_annual_pct=0.0,
    )
    # Day 0: 10000. Day 1: 10000 * (1 + 0.75 * 0.10) = 10750.
    assert curve[0]["equity"] == pytest.approx(10_000.0)
    assert curve[1]["equity"] == pytest.approx(10_750.0)


def test_buy_and_hold_equity_tracks_close_ratio() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3, freq="B")
    spy = pd.DataFrame({"close": [100.0, 50.0, 100.0]}, index=idx)
    curve = buy_and_hold_equity_curve(spy, initial_capital=10_000.0)
    assert curve[0]["equity"] == pytest.approx(10_000.0)
    assert curve[1]["equity"] == pytest.approx(5_000.0)
    assert curve[2]["equity"] == pytest.approx(10_000.0)


def test_strategy_usable_start_dates_are_max_of_required_symbols() -> None:
    idx_early = pd.bdate_range("2006-01-03", periods=5, freq="B")
    idx_shv = pd.bdate_range("2007-01-11", periods=5, freq="B")
    idx_vxus = pd.bdate_range("2011-01-26", periods=5, freq="B")
    dummy = pd.DataFrame({"close": [100.0] * 5}, index=idx_early)
    data = {
        "SPY": dummy,
        "QQQ": dummy,
        "TLT": dummy,
        "GLD": dummy,
        "SHV": pd.DataFrame({"close": [100.0] * 5}, index=idx_shv),
        "VOO": dummy,
        "VXUS": pd.DataFrame({"close": [100.0] * 5}, index=idx_vxus),
        "BND": dummy,
    }
    starts = strategy_usable_start_dates(data)
    assert starts["momentum"] == date(2007, 1, 11)
    assert starts["dca"] == date(2011, 1, 26)
    assert starts["mean_reversion"] == date(2007, 1, 11)
