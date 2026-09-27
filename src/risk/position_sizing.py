"""Convert signals and account state into order notional amounts."""

from __future__ import annotations

import math

_MAX_REGIME_MULTIPLIER = 2.0


def _validate_regime_multiplier(regime_multiplier: float) -> float | None:
    """Return clamped multiplier or ``None`` when non-finite / negative."""
    rm = float(regime_multiplier)
    if math.isnan(rm) or math.isinf(rm) or rm < 0.0:
        return None
    return min(rm, _MAX_REGIME_MULTIPLIER)


def compute_atr_notional(
    *,
    atr_value: float,
    price: float,
    equity: float,
    risk_pct: float,
    max_position_pct: float,
    regime_multiplier: float = 1.0,
) -> float:
    """Dollar notional sized so that one ATR move equals ``risk_pct * equity``.

    ``qty = (risk_pct * equity) / atr_value``, ``notional = qty * price``, capped by
    ``max_position_pct * equity``. Signal **weight** is not used in ATR mode — risk
    is fixed by ``risk_pct`` and ``atr_value`` only.

    Returns ``0.0`` when inputs are non-finite, ``atr_value``/``price``/``equity`` are
    non-positive, ``risk_pct <= 0``, or ``regime_multiplier`` is non-finite/negative.
    """
    if risk_pct <= 0.0 or atr_value <= 0.0 or price <= 0.0 or equity <= 0.0:
        return 0.0
    if math.isnan(float(atr_value)) or math.isnan(float(price)) or math.isnan(float(equity)):
        return 0.0
    if math.isnan(float(risk_pct)) or math.isnan(float(max_position_pct)):
        return 0.0
    rm = _validate_regime_multiplier(regime_multiplier)
    if rm is None:
        return 0.0
    risk_dollars = float(risk_pct) * float(equity)
    qty = risk_dollars / float(atr_value)
    raw_notional = qty * float(price) * rm
    cap = float(max_position_pct) * float(equity)
    return float(min(raw_notional, cap))


def compute_order_notional(
    *,
    signal_weight: float,
    equity: float,
    max_position_pct: float,
    dca_budget: float | None,
    use_dca_budget: bool,
    regime_multiplier: float = 1.0,
    current_position_notional: float = 0.0,
) -> float:
    """Return dollar notional for a single order.

    When ``use_dca_budget`` is True and ``dca_budget`` is set, notional is
    ``signal_weight * dca_budget`` (typical for DCA contributions) — a periodic
    new-money contribution, so ``current_position_notional`` is ignored here.
    Otherwise the **target** position notional is
    ``signal_weight * equity * regime_multiplier`` capped by
    ``max_position_pct * equity``, and the return value is the **delta** needed to
    reach that target from ``current_position_notional`` (floored at 0), not the
    full target. A position already at or above target returns ``0.0``. DCA ignores
    ``regime_multiplier``.

    Returns ``0.0`` when ``regime_multiplier`` is non-finite or negative (DCA exempt).
    """
    if use_dca_budget and dca_budget is not None:
        return float(signal_weight) * float(dca_budget)
    rm = _validate_regime_multiplier(regime_multiplier)
    if rm is None:
        return 0.0
    cap = float(max_position_pct) * float(equity)
    raw = float(signal_weight) * float(equity) * rm
    target = min(raw, cap)
    return max(0.0, target - float(current_position_notional))


def compute_target_position_notional(
    *,
    signal_weight: float,
    equity: float,
    max_position_pct: float,
    regime_multiplier: float = 1.0,
) -> float:
    """Absolute target position notional (not delta) for rebalance sizing.

    Matches ``compute_order_notional`` when ``current_position_notional`` is zero.
    """
    rm = _validate_regime_multiplier(regime_multiplier)
    if rm is None:
        return 0.0
    cap = float(max_position_pct) * float(equity)
    raw = float(signal_weight) * float(equity) * rm
    return float(min(raw, cap))
