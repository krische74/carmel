"""Tests for portfolio limit helpers."""

from src.risk.portfolio_limits import position_value_pct, would_exceed_max_position


def test_position_value_pct() -> None:
    assert abs(position_value_pct(10.0, 100.0, 10_000.0) - 0.1) < 1e-9


def test_would_exceed_max_position() -> None:
    assert would_exceed_max_position(
        current_qty=0.0,
        add_qty=30.0,
        price=100.0,
        equity=10_000.0,
        max_position_pct=0.25,
    )
    assert not would_exceed_max_position(
        current_qty=0.0,
        add_qty=20.0,
        price=100.0,
        equity=10_000.0,
        max_position_pct=0.25,
    )
