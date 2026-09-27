"""Idle-cash sweep math (Tier 44B / 45A).

Pure functions, no I/O. Decides whether to buy into, sell out of, or hold a
short-term T-bill ETF so cash above the reserve+buffer band earns the risk-free
rate. This is cash management, not a strategy: no signals, no regime multipliers.

Thermostat with hysteresis (two-sided):

- ``reserve = min_cash_reserve_pct * equity`` — the hard floor.
- ``target_cash = reserve + buffer_pct * equity`` — the level the sweep aims for.
- ``low_water = reserve + 0.5 * buffer_pct * equity`` — mid-band soft refill trigger.
- **Buy** when surplus cash ``cash - target_cash >= min_trade_usd`` (avoids daily
  dribble as DCA drains the buffer).
- **Hard-breach sell** when ``cash < reserve``, restoring to target. Ignores all
  minimums (refill whatever is needed, bounded by holdings).
- **Soft-refill sell** when ``reserve <= cash < low_water``, restoring to target.
  Applies ``min_order_notional_usd`` ($5), not ``min_trade_usd`` ($50) — half a
  buffer is typically below the $50 gate.
- **Hold** everywhere else: ``low_water <= cash < target_cash + min_trade_usd``.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel


class SweepAction(BaseModel):
    """Decision from :func:`compute_sweep_action`."""

    action: Literal["buy", "sell", "hold"]
    notional: float = 0.0
    reason: str = ""


def _finite(*values: float) -> bool:
    """True only when every value is a finite number."""
    return all(math.isfinite(float(v)) for v in values)


def compute_sweep_action(
    *,
    cash: float,
    equity: float,
    sweep_position_notional: float,
    min_cash_reserve_pct: float,
    buffer_pct: float,
    min_trade_usd: float,
    min_order_notional_usd: float = 5.0,
    pending_buy_notional: float = 0.0,
) -> SweepAction:
    """Decide the sweep action for the current cash/holdings state.

    Returns a ``hold`` on any degenerate input (non-finite anywhere, non-positive
    ``equity``, or negative ``cash``/``sweep_position_notional``) so the caller can
    log a WARNING and take no action rather than trade on garbage.

    ``pending_buy_notional`` (Tier 48A-1) subtracts same-cycle submitted buys whose
    cash debits may not have landed yet — the sweep must size on economic cash,
    not the stale broker read.
    """
    if not _finite(
        cash,
        equity,
        sweep_position_notional,
        min_cash_reserve_pct,
        buffer_pct,
        min_trade_usd,
        min_order_notional_usd,
        pending_buy_notional,
    ):
        return SweepAction(action="hold", notional=0.0, reason="Non-finite input; holding.")
    if equity <= 0.0:
        return SweepAction(action="hold", notional=0.0, reason="Non-positive equity; holding.")
    effective_cash = float(cash) - float(pending_buy_notional)
    if effective_cash < 0.0 or sweep_position_notional < 0.0:
        return SweepAction(
            action="hold",
            notional=0.0,
            reason="Negative cash/holdings; holding.",
        )

    reserve = float(min_cash_reserve_pct) * float(equity)
    buffer = float(buffer_pct) * float(equity)
    target_cash = reserve + buffer
    low_water = reserve + 0.5 * buffer

    # Below the hard reserve: sell to restore target, ignoring all minimums, bounded by holdings.
    if effective_cash < reserve:
        if sweep_position_notional <= 0.0:
            return SweepAction(
                action="hold",
                notional=0.0,
                reason=f"Cash ${effective_cash:.2f} below reserve ${reserve:.2f} but no sweep holdings to sell.",
            )
        notional = min(target_cash - effective_cash, sweep_position_notional)
        return SweepAction(
            action="sell",
            notional=float(notional),
            reason=(
                f"Cash ${effective_cash:.2f} below reserve ${reserve:.2f}; selling ${notional:.2f} "
                f"to restore reserve+buffer (target ${target_cash:.2f})."
            ),
        )

    # Soft refill: cash in [reserve, low_water) — restore to target, gated by $5 floor.
    if effective_cash < low_water:
        if sweep_position_notional <= 0.0:
            return SweepAction(
                action="hold",
                notional=0.0,
                reason=(
                    f"Cash ${effective_cash:.2f} below low-water ${low_water:.2f} but no sweep holdings to sell."
                ),
            )
        notional = min(target_cash - effective_cash, sweep_position_notional)
        if notional < float(min_order_notional_usd):
            return SweepAction(
                action="hold",
                notional=0.0,
                reason=(
                    f"Cash ${effective_cash:.2f} below low-water ${low_water:.2f} but refill "
                    f"${notional:.2f} < min order ${float(min_order_notional_usd):.2f}; holding."
                ),
            )
        return SweepAction(
            action="sell",
            notional=float(notional),
            reason=(
                f"Cash ${effective_cash:.2f} below low-water ${low_water:.2f}; selling ${notional:.2f} "
                f"to restore target ${target_cash:.2f}."
            ),
        )

    # Above target by at least the min trade: sweep the surplus into the T-bill ETF.
    surplus = effective_cash - target_cash
    if surplus >= float(min_trade_usd):
        return SweepAction(
            action="buy",
            notional=float(surplus),
            reason=(
                f"Cash ${effective_cash:.2f} above target ${target_cash:.2f} by ${surplus:.2f} "
                f"(>= min trade ${float(min_trade_usd):.2f}); sweeping surplus."
            ),
        )

    return SweepAction(
        action="hold",
        notional=0.0,
        reason=(
            f"Cash ${effective_cash:.2f} within band [${low_water:.2f}, ${target_cash:.2f}+"
            f"${float(min_trade_usd):.2f}); holding."
        ),
    )
