"""Tests for the idle-cash sweep math (Tier 44B / 45A)."""

from __future__ import annotations

import math

import pytest

from src.execution.cash_sweep import SweepAction, compute_sweep_action


def _action(**kwargs) -> SweepAction:
    base = dict(
        cash=1_000.0,
        equity=10_000.0,
        sweep_position_notional=0.0,
        min_cash_reserve_pct=0.05,
        buffer_pct=0.02,
        min_trade_usd=50.0,
        min_order_notional_usd=5.0,
    )
    base.update(kwargs)
    return compute_sweep_action(**base)


def test_buy_when_cash_above_band() -> None:
    # reserve=500, buffer=200 → target_cash=700. cash=2000 → buy 1300 (>= 50 floor).
    a = _action(cash=2_000.0)
    assert a.action == "buy"
    assert a.notional == pytest.approx(1_300.0)


def test_buy_boundary_exactly_min_trade_submits() -> None:
    # target_cash=700; cash=750 → surplus 50 == min_trade → buy (>=, not >).
    a = _action(cash=750.0, min_trade_usd=50.0)
    assert a.action == "buy"
    assert a.notional == pytest.approx(50.0)


def test_hold_inside_band() -> None:
    # low_water=600, target=700; cash=740 → surplus 40 < min_trade 50 and >= low_water → hold.
    a = _action(cash=740.0)
    assert a.action == "hold"
    assert a.notional == pytest.approx(0.0)


def test_hold_at_low_water_boundary() -> None:
    # cash == low_water (600) → soft-refill does not fire; surplus negative → hold.
    a = _action(cash=600.0, sweep_position_notional=5_000.0)
    assert a.action == "hold"
    assert a.notional == pytest.approx(0.0)


def test_hold_at_reserve_boundary_without_holdings() -> None:
    # cash == reserve (500) is in soft-refill band, but no holdings → hold.
    a = _action(cash=500.0, sweep_position_notional=0.0)
    assert a.action == "hold"


def test_soft_refill_sells_when_below_low_water() -> None:
    # reserve=500, low_water=600, target=700. cash=550 → soft refill sell 150.
    a = _action(cash=550.0, sweep_position_notional=5_000.0)
    assert a.action == "sell"
    assert a.notional == pytest.approx(150.0)


def test_soft_refill_at_reserve_boundary_restores_to_target() -> None:
    # cash == reserve → soft refill (not hard breach); sell to target.
    a = _action(cash=500.0, sweep_position_notional=5_000.0)
    assert a.action == "sell"
    assert a.notional == pytest.approx(200.0)  # 700 - 500


def test_soft_refill_just_below_low_water() -> None:
    # cash=599.99 < low_water 600 → sell ~100.01 to restore target.
    a = _action(cash=599.99, sweep_position_notional=5_000.0)
    assert a.action == "sell"
    assert a.notional == pytest.approx(100.01)


def test_soft_refill_applies_min_order_floor_not_min_trade() -> None:
    # Soft refill of $34 must clear the $5 floor even when min_trade_usd is $50.
    # equity=3400 → reserve=170, buffer=68, low_water=204, target=238.
    # cash=204 - epsilon ≈ 203.99 → refill ≈ 34.01 (>= 5, < 50).
    a = _action(
        cash=203.99,
        equity=3_400.0,
        sweep_position_notional=1_400.0,
        min_cash_reserve_pct=0.05,
        buffer_pct=0.02,
        min_trade_usd=50.0,
        min_order_notional_usd=5.0,
    )
    assert a.action == "sell"
    assert a.notional == pytest.approx(34.01)
    assert a.notional < 50.0
    assert a.notional >= 5.0


def test_soft_refill_holds_when_below_min_order_floor() -> None:
    # Tiny holdings in soft band → notional capped below $5 floor → hold.
    a = _action(cash=550.0, sweep_position_notional=3.0, min_order_notional_usd=5.0)
    assert a.action == "hold"
    assert a.notional == pytest.approx(0.0)


def test_soft_refill_capped_by_holdings() -> None:
    # Soft band; needed 150 but only 80 held → sell 80 (>= $5 floor).
    a = _action(cash=550.0, sweep_position_notional=80.0)
    assert a.action == "sell"
    assert a.notional == pytest.approx(80.0)


def test_sell_when_below_reserve() -> None:
    # cash=480 < reserve 500; holdings 5000 → sell min(700-480=220, 5000)=220.
    a = _action(cash=480.0, sweep_position_notional=5_000.0)
    assert a.action == "sell"
    assert a.notional == pytest.approx(220.0)


def test_sell_below_reserve_ignores_all_minimums() -> None:
    # Hard breach: ignores min_trade_usd and min_order_notional_usd.
    a = _action(
        cash=499.0,
        sweep_position_notional=5_000.0,
        min_trade_usd=10_000.0,
        min_order_notional_usd=10_000.0,
    )
    assert a.action == "sell"
    assert a.notional == pytest.approx(201.0)  # 700 - 499
    assert a.notional < 10_000.0


def test_sell_capped_by_holdings() -> None:
    # cash=100 < reserve 500; needed 700-100=600 but only 250 held → sell 250.
    a = _action(cash=100.0, sweep_position_notional=250.0)
    assert a.action == "sell"
    assert a.notional == pytest.approx(250.0)


def test_below_reserve_but_no_holdings_holds() -> None:
    # cash below reserve but nothing to sell → hold (no zero-qty sell).
    a = _action(cash=100.0, sweep_position_notional=0.0)
    assert a.action == "hold"
    assert a.notional == pytest.approx(0.0)


def test_deadlock_regression_drain_then_soft_refill() -> None:
    """Tier-44 one-sided thermostat deadlocked when cash sat above reserve but

    below the level where DCA could clear the cash-reserve pre-trade check.
    Soft refill must sell once cash drops below low_water.
    """
    equity = 3_437.0
    reserve_pct = 0.05
    buffer_pct = 0.02
    min_trade = 50.0
    holdings = 1_420.0
    # reserve≈171.85, buffer≈68.74, low_water≈206.22, target≈240.59
    cash = 240.0  # post-sweep target-ish
    daily_dca_drain = 7.7

    saw_refill = False
    for _ in range(12):
        a = _action(
            cash=cash,
            equity=equity,
            sweep_position_notional=holdings,
            min_cash_reserve_pct=reserve_pct,
            buffer_pct=buffer_pct,
            min_trade_usd=min_trade,
            min_order_notional_usd=5.0,
        )
        reserve = reserve_pct * equity
        low_water = reserve + 0.5 * buffer_pct * equity
        target = reserve + buffer_pct * equity
        if cash < low_water:
            assert a.action == "sell", f"expected soft refill at cash={cash:.2f}"
            assert a.notional == pytest.approx(min(target - cash, holdings))
            saw_refill = True
            cash = cash + a.notional  # restore toward target
            holdings -= a.notional
            break
        assert a.action == "hold"
        cash -= daily_dca_drain

    assert saw_refill
    # After refill, cash is back at/near target → hold (not permanent deadlock).
    a_after = _action(
        cash=cash,
        equity=equity,
        sweep_position_notional=holdings,
        min_cash_reserve_pct=reserve_pct,
        buffer_pct=buffer_pct,
        min_trade_usd=min_trade,
    )
    assert a_after.action == "hold"


def test_zero_equity_holds() -> None:
    a = _action(equity=0.0, cash=1_000.0)
    assert a.action == "hold"


def test_nonfinite_equity_holds() -> None:
    assert _action(equity=math.nan, cash=1_000.0).action == "hold"
    assert _action(equity=math.inf, cash=1_000.0).action == "hold"


def test_negative_cash_holds() -> None:
    a = _action(cash=-10.0, sweep_position_notional=100.0)
    assert a.action == "hold"


def test_negative_holdings_holds() -> None:
    a = _action(cash=100.0, sweep_position_notional=-5.0)
    assert a.action == "hold"


def test_action_has_reason_string() -> None:
    for a in (
        _action(cash=2_000.0),
        _action(cash=740.0),
        _action(cash=100.0, sweep_position_notional=250.0),
        _action(cash=550.0, sweep_position_notional=5_000.0),
    ):
        assert isinstance(a.reason, str) and a.reason


def test_pending_buy_notional_prevents_819_sweep_buy() -> None:
    """Tier 48A-1: economic cash after SHV buy leaves no sweep surplus."""
    a = _action(
        cash=1_099.56,
        equity=3_436.33,
        sweep_position_notional=0.0,
        min_cash_reserve_pct=0.05,
        buffer_pct=0.02,
        min_trade_usd=50.0,
        pending_buy_notional=859.06,
    )
    assert a.action == "hold"
