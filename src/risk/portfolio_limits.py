"""Portfolio concentration limits (post-trade checks)."""

from __future__ import annotations


def position_value_pct(qty: float, price: float, equity: float) -> float:
    """Absolute market value of ``qty`` shares as a fraction of ``equity``."""
    if equity <= 0.0:
        return 0.0
    return abs(float(qty) * float(price)) / float(equity)


def would_exceed_max_position(
    *,
    current_qty: float,
    add_qty: float,
    price: float,
    equity: float,
    max_position_pct: float,
) -> bool:
    """Return True if ``current_qty + add_qty`` would exceed ``max_position_pct``."""
    new_qty = float(current_qty) + float(add_qty)
    return position_value_pct(new_qty, price, equity) > float(max_position_pct) + 1e-12
