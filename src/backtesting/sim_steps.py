"""Trade-step helpers for the backtest engine (Tier 52)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.backtesting.dca_contrib import contribution_notional

if TYPE_CHECKING:
    from datetime import date

    from src.backtesting.engine import BacktestTrade
    from src.config import Settings
    from src.models import Signal


def mtm_now(
    cash: float,
    positions: dict[str, float],
    last_marks: dict[str, float],
    normalized: dict[str, Any],
    d: date,
) -> float:
    """Cash plus marked positions using today's close when available."""
    from src.backtesting.engine import _close_price_from_normalized

    mtm = cash
    for sym, qty in positions.items():
        df = normalized.get(sym)
        px = None
        if df is not None:
            px = _close_price_from_normalized(df, d)
        if px is not None:
            last_marks[sym] = px
        else:
            px = last_marks.get(sym)
        if px is not None:
            mtm += qty * px
    return mtm


def compute_risk_exposure_pct(
    *,
    mtm: float,
    positions: dict[str, float],
    last_marks: dict[str, float],
    cash_symbols: frozenset[str],
    sweep_symbol: str = "",
) -> float:
    """Risk-asset weight: sum(risk positions) / equity.

    Excludes the momentum cash leg (e.g. SHV) and the sweep symbol (e.g. BIL).
    """
    if mtm <= 0.0:
        return 0.0
    sweep = sweep_symbol.strip().upper()
    risk_mv = 0.0
    for sym, qty in positions.items():
        sym_u = sym.strip().upper()
        if sweep and sym_u == sweep:
            continue
        if sym_u in cash_symbols:
            continue
        px = last_marks.get(sym_u) or last_marks.get(sym)
        if px is not None and px > 0.0:
            risk_mv += qty * px
    return risk_mv / mtm


def rebalance_to_targets(
    *,
    d: date,
    targets: dict[str, float],
    equity: float,
    cash: float,
    positions: dict[str, float],
    prices: dict[str, float],
    bps: float,
    min_trade_usd: float,
    min_order_floor: float,
    reserve_cash: float,
    use_risk: bool,
    trades: list[BacktestTrade],
    skip_symbols: set[str],
) -> tuple[float, dict[str, float]]:
    """Adjust holdings toward ``targets`` (weight * equity). Mutates ``positions``."""
    from src.backtesting.engine import BacktestTrade

    skip = {s.strip().upper() for s in skip_symbols if s}
    for sym in list(positions):
        if sym in skip:
            continue
        tgt = targets.get(sym, 0.0)
        px = prices.get(sym)
        if px is None:
            continue
        cur_val = positions[sym] * px
        tgt_val = tgt * equity
        diff = tgt_val - cur_val
        if diff < -min_trade_usd:
            sell_px = px * (1.0 - bps)
            sell_notional = min(-diff, cur_val)
            qty_sell = sell_notional / sell_px if sell_px > 0 else 0.0
            qty_sell = min(qty_sell, positions[sym])
            if qty_sell <= 0.0:
                continue
            slip_cost = (px - sell_px) * qty_sell
            cash += qty_sell * sell_px
            positions[sym] -= qty_sell
            if positions[sym] <= 1e-12:
                del positions[sym]
            trades.append(
                BacktestTrade(
                    date=d.isoformat(),
                    symbol=sym,
                    side="sell",
                    qty=qty_sell,
                    price=sell_px,
                    slippage_cost=float(slip_cost),
                ),
            )

    for sym, tgt in targets.items():
        if tgt <= 0.0 or sym in skip:
            continue
        px = prices.get(sym)
        if px is None:
            continue
        cur_qty = positions.get(sym, 0.0)
        cur_val = cur_qty * px
        tgt_val = tgt * equity
        diff = tgt_val - cur_val
        if diff > min_trade_usd:
            buy_px = px * (1.0 + bps)
            spendable = max(0.0, cash - reserve_cash) if use_risk else cash
            diff = min(diff, spendable)
            if diff <= min_order_floor:
                continue
            qty_buy = diff / buy_px if buy_px > 0 else 0.0
            if qty_buy <= 0.0:
                continue
            slip_cost = (buy_px - px) * qty_buy
            cost = qty_buy * buy_px
            if cost > cash + 1e-6:
                continue
            cash -= cost
            positions[sym] = cur_qty + qty_buy
            trades.append(
                BacktestTrade(
                    date=d.isoformat(),
                    symbol=sym,
                    side="buy",
                    qty=qty_buy,
                    price=buy_px,
                    slippage_cost=float(slip_cost),
                ),
            )
    return cash, positions


def drift_band_exceeded(
    *,
    targets: dict[str, float],
    positions: dict[str, float],
    prices: dict[str, float],
    equity: float,
    skip_symbols: set[str],
    band_pct: float,
) -> bool:
    """True when any held leg's weight deviates from target by more than ``band_pct``.

    ``band_pct`` is an absolute percentage-point band (0.05 → +/-5 pp around target).
    """
    if equity <= 0.0 or band_pct >= 1.0:
        return False
    skip = {s.strip().upper() for s in skip_symbols if s}
    symbols = (set(targets) | set(positions)) - skip
    for sym in symbols:
        px = prices.get(sym)
        if px is None or px <= 0.0:
            continue
        tgt_w = float(targets.get(sym, 0.0))
        cur_w = float(positions.get(sym, 0.0)) * px / equity
        if abs(cur_w - tgt_w) > band_pct:
            return True
    return False


def target_weights_changed(
    previous: dict[str, float] | None,
    current: dict[str, float],
    *,
    tol: float = 1e-9,
) -> bool:
    """True when any target weight moved (e.g. regime multiplier changed, same symbol)."""
    if previous is None:
        return True
    keys = set(previous) | set(current)
    return any(abs(float(previous.get(k, 0.0)) - float(current.get(k, 0.0))) > tol for k in keys)


def apply_dca_contributions(
    *,
    d: date,
    signals: list[Signal],
    dca_budget: float,
    equity: float,
    cash: float,
    positions: dict[str, float],
    last_marks: dict[str, float],
    normalized: dict[str, Any],
    bps: float,
    max_position_pct: float,
    min_order_floor: float,
    reserve_cash: float,
    use_risk: bool,
    trades: list[BacktestTrade],
) -> tuple[float, dict[str, float]]:
    """Buy DCA legs as flows, capped per symbol."""
    from src.backtesting.engine import BacktestTrade, _close_price_from_normalized

    for s in signals:
        if s.direction not in ("long", "cash"):
            continue
        sym = s.symbol.strip().upper()
        df = normalized.get(sym)
        if df is None:
            continue
        px = _close_price_from_normalized(df, d)
        if px is None:
            px = last_marks.get(sym)
        if px is None or px <= 0.0:
            continue
        last_marks[sym] = px
        cur_qty = positions.get(sym, 0.0)
        spendable = max(0.0, cash - reserve_cash) if use_risk else cash
        n = contribution_notional(
            signal_weight=float(s.weight),
            dca_budget=dca_budget,
            equity=equity,
            current_position_notional=cur_qty * px,
            max_position_pct=max_position_pct,
            min_order_notional_usd=min_order_floor,
            spendable_cash=spendable,
        )
        if n <= 0.0:
            continue
        buy_px = px * (1.0 + bps)
        qty_buy = n / buy_px if buy_px > 0 else 0.0
        cost = qty_buy * buy_px
        if qty_buy <= 0.0 or cost > cash + 1e-6:
            continue
        slip_cost = (buy_px - px) * qty_buy
        cash -= cost
        positions[sym] = cur_qty + qty_buy
        trades.append(
            BacktestTrade(
                date=d.isoformat(),
                symbol=sym,
                side="buy",
                qty=qty_buy,
                price=buy_px,
                slippage_cost=float(slip_cost),
            ),
        )
    return cash, positions


def apply_cash_sweep(
    *,
    d: date,
    cash: float,
    positions: dict[str, float],
    last_marks: dict[str, float],
    normalized: dict[str, Any],
    sweep_sym: str,
    settings: Settings,
    bps: float,
    trades: list[BacktestTrade],
) -> tuple[float, dict[str, float]]:
    """Park or restore cash via ``compute_sweep_action`` (import allowed; no live orders)."""
    from src.backtesting.engine import BacktestTrade, _close_price_from_normalized
    from src.execution.cash_sweep import compute_sweep_action

    df = normalized.get(sweep_sym)
    px = _close_price_from_normalized(df, d) if df is not None else None
    if px is None:
        px = last_marks.get(sweep_sym)
    if px is None or px <= 0.0:
        return cash, positions
    last_marks[sweep_sym] = px
    equity = mtm_now(cash, positions, last_marks, normalized, d)
    sweep_qty = positions.get(sweep_sym, 0.0)
    action = compute_sweep_action(
        cash=cash,
        equity=equity,
        sweep_position_notional=sweep_qty * px,
        min_cash_reserve_pct=float(settings.risk.min_cash_reserve_pct),
        buffer_pct=float(settings.cash_sweep.buffer_pct),
        min_trade_usd=float(settings.cash_sweep.min_trade_usd),
        min_order_notional_usd=float(settings.risk.min_order_notional_usd),
    )
    if action.action == "hold" or action.notional <= 0.0:
        return cash, positions
    if action.action == "buy":
        buy_px = px * (1.0 + bps)
        qty = action.notional / buy_px if buy_px > 0 else 0.0
        cost = qty * buy_px
        if qty <= 0.0 or cost > cash + 1e-6:
            return cash, positions
        cash -= cost
        positions[sweep_sym] = sweep_qty + qty
        trades.append(
            BacktestTrade(
                date=d.isoformat(),
                symbol=sweep_sym,
                side="buy",
                qty=qty,
                price=buy_px,
                slippage_cost=float((buy_px - px) * qty),
            ),
        )
        return cash, positions
    sell_px = px * (1.0 - bps)
    qty = min(action.notional / sell_px if sell_px > 0 else 0.0, sweep_qty)
    if qty <= 0.0:
        return cash, positions
    cash += qty * sell_px
    positions[sweep_sym] = sweep_qty - qty
    if positions[sweep_sym] <= 1e-12:
        del positions[sweep_sym]
    trades.append(
        BacktestTrade(
            date=d.isoformat(),
            symbol=sweep_sym,
            side="sell",
            qty=qty,
            price=sell_px,
            slippage_cost=float((px - sell_px) * qty),
        ),
    )
    return cash, positions
