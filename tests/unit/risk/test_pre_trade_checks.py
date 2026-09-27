"""Tests for pre-trade risk checks."""

from src.risk.pre_trade_checks import evaluate_pre_trade_checks


def test_blocks_when_kill_switch_halted() -> None:
    r = evaluate_pre_trade_checks(
        notional=100.0,
        price=50.0,
        equity=10_000.0,
        cash=5_000.0,
        current_qty=0.0,
        max_position_pct=0.25,
        min_cash_reserve_pct=0.05,
        kill_switch_halted=True,
    )
    assert r.ok is False
    assert r.reason is not None and "kill" in r.reason.lower()


def test_blocks_insufficient_cash_reserve() -> None:
    r = evaluate_pre_trade_checks(
        notional=9_900.0,
        price=100.0,
        equity=10_000.0,
        cash=9_900.0,
        current_qty=0.0,
        max_position_pct=0.99,
        min_cash_reserve_pct=0.05,
        kill_switch_halted=False,
    )
    assert r.ok is False
    assert "cash" in (r.reason or "").lower()


def test_allows_when_within_limits() -> None:
    r = evaluate_pre_trade_checks(
        notional=500.0,
        price=100.0,
        equity=10_000.0,
        cash=5_000.0,
        current_qty=0.0,
        max_position_pct=0.25,
        min_cash_reserve_pct=0.05,
        kill_switch_halted=False,
    )
    assert r.ok is True


def test_blocks_when_position_limit_exceeded() -> None:
    """Adding 2 shares at $100 would push 24→26 shares above 25% of $10k equity."""
    r = evaluate_pre_trade_checks(
        notional=200.0,
        price=100.0,
        equity=10_000.0,
        cash=5_000.0,
        current_qty=24.0,
        max_position_pct=0.25,
        min_cash_reserve_pct=0.05,
        kill_switch_halted=False,
    )
    assert r.ok is False
    assert r.reason is not None
    assert "position" in r.reason.lower() or "maximum" in r.reason.lower()
