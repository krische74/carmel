"""Account-level snapshot for risk checks and order execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.execution.broker_interface import BrokerInterface


class PortfolioSnapshot(BaseModel):
    """Point-in-time equity, cash, and per-symbol quantities from the broker."""

    equity: float
    cash: float
    positions: dict[str, float] = Field(default_factory=dict)
    daily_pnl_pct: float = 0.0


def snapshot_from_broker(
    broker: BrokerInterface,
    symbols: Iterable[str],
    *,
    daily_pnl_pct: float | None = None,
) -> PortfolioSnapshot:
    """Build a snapshot using ``BrokerInterface`` (no strategies in this call).

    ``symbols`` is the set of tickers to query for position size (typically the
    union of strategy universes). Positions come from one ``get_all_positions()``
    when the adapter implements it; otherwise per-symbol ``get_position_qty``.
    Extra broker positions not in ``symbols`` are included (lot-reconciliation).
    When ``daily_pnl_pct`` is omitted, day P&L is computed from
    ``(equity - last_equity) / last_equity`` using ``broker.get_last_equity()``.
    """
    equity = float(broker.get_account_equity())
    cash = float(broker.get_cash())
    positions: dict[str, float] = {}
    wanted = [raw.strip().upper() for raw in symbols]
    try:
        raw_batch = broker.get_all_positions()
    except (NotImplementedError, AttributeError):
        raw_batch = None
    if isinstance(raw_batch, dict):
        positions = {sym: 0.0 for sym in wanted}
        for key, qty in raw_batch.items():
            positions[str(key).strip().upper()] = float(qty)
    else:
        for raw in wanted:
            positions[raw] = float(broker.get_position_qty(raw))
    if daily_pnl_pct is None:
        last_eq = float(broker.get_last_equity())
        pnl = (equity - last_eq) / last_eq if last_eq > 0.0 else 0.0
    else:
        pnl = float(daily_pnl_pct)
    return PortfolioSnapshot(equity=equity, cash=cash, positions=positions, daily_pnl_pct=pnl)
