"""Pre-trade risk checks before any order reaches the broker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.risk.pdt_tracker import evaluate_pdt_risk, projected_day_trades_after_sell

if TYPE_CHECKING:
    from datetime import datetime
from src.risk.portfolio_limits import would_exceed_max_position


class PreTradeCheckResult(BaseModel):
    """Outcome of evaluating risk gates for a proposed buy."""

    ok: bool
    reason: str | None = Field(default=None)
    warning: str | None = Field(
        default=None,
        description="Non-blocking advisory (e.g. near PDT limit on an allowed sell).",
    )


def evaluate_pre_trade_checks(
    *,
    notional: float,
    price: float,
    equity: float,
    cash: float,
    current_qty: float,
    max_position_pct: float,
    min_cash_reserve_pct: float,
    kill_switch_halted: bool,
) -> PreTradeCheckResult:
    """Validate a proposed **buy** for ``notional`` dollars at ``price``."""
    if kill_switch_halted:
        return PreTradeCheckResult(ok=False, reason="Kill switch active — new orders blocked.")
    if notional <= 0.0 or price <= 0.0:
        return PreTradeCheckResult(ok=False, reason="Notional and price must be positive.")
    qty = notional / price
    if would_exceed_max_position(
        current_qty=current_qty,
        add_qty=qty,
        price=price,
        equity=equity,
        max_position_pct=max_position_pct,
    ):
        return PreTradeCheckResult(
            ok=False,
            reason="Order would exceed maximum position size as a fraction of equity.",
        )
    min_cash = float(min_cash_reserve_pct) * float(equity)
    if float(cash) - float(notional) + 1e-9 < min_cash:
        return PreTradeCheckResult(
            ok=False,
            reason="Order would violate minimum cash reserve after settlement.",
        )
    return PreTradeCheckResult(ok=True, reason=None)


def evaluate_pdt_before_sell(
    *,
    symbol: str,
    equity: float,
    as_of: datetime,
    executions: list[dict[str, Any]],
    pdt_protection: bool,
    pdt_threshold: int = 4,
    equity_floor: float = 25_000.0,
    lookback_days: int = 5,
) -> PreTradeCheckResult:
    """Block or warn on sells that would breach PDT rules for sub-$25k accounts."""
    if not pdt_protection:
        return PreTradeCheckResult(ok=True, reason=None, warning=None)
    projected = projected_day_trades_after_sell(
        executions,
        symbol=symbol,
        as_of=as_of,
        lookback_days=lookback_days,
    )
    pdt = evaluate_pdt_risk(
        projected,
        equity,
        pdt_threshold=pdt_threshold,
        equity_floor=equity_floor,
    )
    if not pdt.allowed:
        return PreTradeCheckResult(
            ok=False,
            reason=(
                f"PDT protection: projected {pdt.day_trade_count} day trades in "
                f"{lookback_days} business days with equity ${equity:,.2f} "
                f"below ${equity_floor:,.0f} (limit {pdt_threshold})."
            ),
            warning=None,
        )
    return PreTradeCheckResult(ok=True, reason=None, warning=pdt.warning)
