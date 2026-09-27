"""Account leverage signals for the no-leverage invariant (Tier 48B)."""

from __future__ import annotations

from pydantic import BaseModel


class LeverageSnapshot(BaseModel):
    """Point-in-time leverage indicators from the broker account object."""

    cash: float
    buying_power: float | None = None
    maintenance_margin: float | None = None
    broker_reports_leverage: bool = False

    @property
    def cash_negative(self) -> bool:
        return float(self.cash) < -1e-6

    @property
    def leveraged(self) -> bool:
        """True when cash is negative or the broker reports margin borrowing."""
        return self.cash_negative or self.broker_reports_leverage


def format_cycle_order_sequence(
    execution_results: list[object],
) -> str:
    """Compact order sequence for leverage CRITICAL messages."""
    parts: list[str] = []
    for res in execution_results:
        submitted = getattr(res, "submitted", False)
        if not submitted:
            continue
        side = getattr(res, "side", "?")
        symbol = getattr(res, "symbol", "?")
        qty = getattr(res, "qty", None)
        strategy = getattr(res, "strategy_name", None)
        tag = f"{strategy} " if strategy else ""
        qty_s = f" qty={float(qty):.4f}" if qty is not None else ""
        parts.append(f"{tag}{side} {symbol}{qty_s}".strip())
    return "; ".join(parts) if parts else "none"


def leverage_violation_message(
    snap: LeverageSnapshot,
    order_sequence: str,
) -> str | None:
    """Return a CRITICAL message when the account is levered, else None."""
    if not snap.leveraged:
        return None
    detail = [f"cash={snap.cash:.2f}"]
    if snap.maintenance_margin is not None and snap.maintenance_margin > 1e-6:
        detail.append(f"maintenance_margin={snap.maintenance_margin:.2f}")
    return f"Account leveraged ({', '.join(detail)}); cycle orders: {order_sequence}"
