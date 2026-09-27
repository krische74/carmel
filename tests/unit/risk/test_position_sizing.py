"""Tests for position sizing helpers."""

import pytest

from src.risk.position_sizing import compute_atr_notional, compute_order_notional


def test_dca_notional_uses_budget() -> None:
    n = compute_order_notional(
        signal_weight=1.0 / 3.0,
        equity=100_000.0,
        max_position_pct=0.25,
        dca_budget=300.0,
        use_dca_budget=True,
    )
    assert abs(n - 100.0) < 1e-6


def test_regime_multiplier_reduces_notional() -> None:
    base = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=1.0,
        dca_budget=None,
        use_dca_budget=False,
        regime_multiplier=1.0,
    )
    half = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=1.0,
        dca_budget=None,
        use_dca_budget=False,
        regime_multiplier=0.5,
    )
    assert half == pytest.approx(base * 0.5)


def test_regime_multiplier_1_unchanged() -> None:
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
        regime_multiplier=1.0,
    )
    assert abs(n - 2500.0) < 1e-6


def test_regime_multiplier_ignored_for_dca() -> None:
    n = compute_order_notional(
        signal_weight=1.0 / 3.0,
        equity=100_000.0,
        max_position_pct=0.25,
        dca_budget=300.0,
        use_dca_budget=True,
        regime_multiplier=0.5,
    )
    assert abs(n - 100.0) < 1e-6


def test_regime_multiplier_applies_to_atr_sizing() -> None:
    n1 = compute_atr_notional(
        atr_value=2.0,
        price=50.0,
        equity=100_000.0,
        risk_pct=0.01,
        max_position_pct=0.99,
        regime_multiplier=1.0,
    )
    n2 = compute_atr_notional(
        atr_value=2.0,
        price=50.0,
        equity=100_000.0,
        risk_pct=0.01,
        max_position_pct=0.99,
        regime_multiplier=0.75,
    )
    assert n2 == pytest.approx(n1 * 0.75)


def test_non_dca_caps_at_max_position_pct() -> None:
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
    )
    assert abs(n - 2500.0) < 1e-6


def test_compute_order_notional_returns_zero_when_already_at_target() -> None:
    """Position already at target notional → delta is 0 (no top-up needed)."""
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
        current_position_notional=2_500.0,  # target = min(10_000, 2_500) = 2_500
    )
    assert n == pytest.approx(0.0)


def test_compute_order_notional_returns_delta_when_below_target() -> None:
    """Position below target → returns the difference, not the full target."""
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
        current_position_notional=1_000.0,  # target 2_500 - held 1_000 = 1_500
    )
    assert n == pytest.approx(1_500.0)


def test_compute_order_notional_floors_at_zero_when_position_exceeds_target() -> None:
    """Position above target (e.g. price drift) → 0.0, never negative."""
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
        current_position_notional=3_000.0,  # target 2_500 - held 3_000 -> floored at 0
    )
    assert n == pytest.approx(0.0)


def test_compute_order_notional_full_target_when_no_existing_position() -> None:
    """Regression: default current_position_notional=0.0 behaves exactly as before."""
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
    )
    assert n == pytest.approx(2_500.0)


def test_compute_order_notional_dca_path_ignores_current_position_notional() -> None:
    """DCA branch returns signal_weight * dca_budget regardless of current position."""
    n = compute_order_notional(
        signal_weight=1.0 / 3.0,
        equity=100_000.0,
        max_position_pct=0.25,
        dca_budget=300.0,
        use_dca_budget=True,
        current_position_notional=99_999.0,
    )
    assert n == pytest.approx(100.0)


def test_compute_atr_notional_sizes_inversely_to_volatility() -> None:
    """Higher ATR implies smaller dollar exposure (same risk budget)."""
    equity = 100_000.0
    risk_pct = 0.01
    price = 50.0
    low_atr = compute_atr_notional(
        atr_value=1.0,
        price=price,
        equity=equity,
        risk_pct=risk_pct,
        max_position_pct=0.99,
    )
    high_atr = compute_atr_notional(
        atr_value=4.0,
        price=price,
        equity=equity,
        risk_pct=risk_pct,
        max_position_pct=0.99,
    )
    assert low_atr > high_atr > 0.0


def test_compute_atr_notional_caps_at_max_position() -> None:
    n = compute_atr_notional(
        atr_value=0.01,
        price=100.0,
        equity=10_000.0,
        risk_pct=0.5,
        max_position_pct=0.25,
    )
    assert abs(n - 2500.0) < 1e-3


def test_compute_atr_notional_zero_atr_returns_zero() -> None:
    assert (
        compute_atr_notional(
            atr_value=0.0,
            price=100.0,
            equity=10_000.0,
            risk_pct=0.01,
            max_position_pct=0.25,
        )
        == 0.0
    )
    assert (
        compute_atr_notional(
            atr_value=-1.0,
            price=100.0,
            equity=10_000.0,
            risk_pct=0.01,
            max_position_pct=0.25,
        )
        == 0.0
    )


def test_compute_atr_notional_zero_equity_returns_zero() -> None:
    n = compute_atr_notional(
        atr_value=2.0,
        price=50.0,
        equity=0.0,
        risk_pct=0.01,
        max_position_pct=0.25,
    )
    assert n == 0.0


def test_compute_atr_notional_zero_price_returns_zero() -> None:
    n = compute_atr_notional(
        atr_value=2.0,
        price=0.0,
        equity=100_000.0,
        risk_pct=0.01,
        max_position_pct=0.25,
    )
    assert n == 0.0


def test_compute_atr_notional_negative_risk_pct_returns_zero() -> None:
    n = compute_atr_notional(
        atr_value=2.0,
        price=50.0,
        equity=100_000.0,
        risk_pct=-0.01,
        max_position_pct=0.25,
    )
    assert n == 0.0


def test_compute_atr_notional_exact_formula() -> None:
    """Hand check: qty = (0.01 * 100_000) / 2 = 500; notional = 500 * 50 = 25_000 = cap."""
    equity = 100_000.0
    risk_pct = 0.01
    atr_value = 2.0
    price = 50.0
    max_position_pct = 0.25
    qty = (risk_pct * equity) / atr_value
    assert qty == 500.0
    n = compute_atr_notional(
        atr_value=atr_value,
        price=price,
        equity=equity,
        risk_pct=risk_pct,
        max_position_pct=max_position_pct,
    )
    assert abs(n - 25_000.0) < 1e-6
    assert abs(n - max_position_pct * equity) < 1e-6


def test_regime_multiplier_zero_returns_zero() -> None:
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
        regime_multiplier=0.0,
    )
    assert n == 0.0


def test_regime_multiplier_negative_returns_zero() -> None:
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
        regime_multiplier=-0.5,
    )
    assert n == 0.0


def test_regime_multiplier_nan_returns_zero() -> None:
    n = compute_order_notional(
        signal_weight=1.0,
        equity=10_000.0,
        max_position_pct=0.25,
        dca_budget=None,
        use_dca_budget=False,
        regime_multiplier=float("nan"),
    )
    assert n == 0.0


def test_regime_multiplier_above_cap_clamped() -> None:
    """Multiplier > 2.0 should be clamped to 2.0 (matching MarketRegime.sizing_multiplier bound)."""
    n = compute_order_notional(
        signal_weight=0.5,
        equity=10_000.0,
        max_position_pct=1.0,
        dca_budget=None,
        use_dca_budget=False,
        regime_multiplier=3.0,
    )
    assert n == pytest.approx(0.5 * 10_000.0 * 2.0)


def test_atr_regime_multiplier_nan_returns_zero() -> None:
    n = compute_atr_notional(
        atr_value=2.0,
        price=50.0,
        equity=100_000.0,
        risk_pct=0.01,
        max_position_pct=0.25,
        regime_multiplier=float("nan"),
    )
    assert n == 0.0


def test_atr_regime_multiplier_negative_returns_zero() -> None:
    n = compute_atr_notional(
        atr_value=2.0,
        price=50.0,
        equity=100_000.0,
        risk_pct=0.01,
        max_position_pct=0.25,
        regime_multiplier=-1.0,
    )
    assert n == 0.0
