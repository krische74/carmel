"""DCA as a periodic contribution (flow), not a target-weight jump (Tier 52B)."""

from __future__ import annotations


def contribution_notional(
    *,
    signal_weight: float,
    dca_budget: float,
    equity: float,
    current_position_notional: float,
    max_position_pct: float,
    min_order_notional_usd: float,
    spendable_cash: float,
) -> float:
    """Dollar amount to buy this cycle for one DCA leg.

    Gross size matches live ``compute_order_notional(..., use_dca_budget=True)``
    (``signal_weight * dca_budget``), then is clipped to remaining room under
    ``max_position_pct``, spendable cash, and the min-order floor.
    """
    raw = float(signal_weight) * float(dca_budget)
    room = max(0.0, float(max_position_pct) * float(equity) - float(current_position_notional))
    n = min(raw, room, max(0.0, float(spendable_cash)))
    if n < float(min_order_notional_usd):
        return 0.0
    return float(n)
