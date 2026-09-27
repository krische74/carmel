"""Position-level P&L contribution (prerequisite to full Brinson attribution)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel

from src.reporting.portfolio_analytics import compute_portfolio_valuation, load_current_prices

if TYPE_CHECKING:
    from datetime import date

    from src.data.storage.parquet_store import ParquetStore
    from src.portfolio.tax_lots import LotLedger


class PositionContribution(BaseModel):
    """Per-symbol P&L contribution to total portfolio return."""

    symbol: str
    total_qty: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    weight_pct: float
    contribution_pct: float


def compute_position_contributions(
    lot_ledger: LotLedger,
    current_prices: dict[str, float],
) -> list[PositionContribution]:
    """Break down P&L contribution by symbol from lots and current prices."""
    valuation = compute_portfolio_valuation(lot_ledger, current_prices)
    pos_by_sym = {p.symbol: p for p in valuation.positions}

    realized_by_sym, portfolio_closed_cost = lot_ledger.closed_lot_aggregates()

    portfolio_open_basis = float(valuation.total_cost_basis)
    portfolio_basis = portfolio_open_basis + portfolio_closed_cost

    total_mv = float(valuation.total_market_value)
    symbols = sorted(set(pos_by_sym.keys()) | set(realized_by_sym.keys()))
    if not symbols:
        return []

    out: list[PositionContribution] = []
    for sym in symbols:
        pos = pos_by_sym.get(sym)
        total_qty = float(pos.total_qty) if pos is not None else 0.0
        unrealized = float(pos.unrealized_pnl) if pos is not None else 0.0
        market_value = float(pos.market_value) if pos is not None else 0.0
        realized = float(realized_by_sym.get(sym, 0.0))
        total_pnl = unrealized + realized
        weight_pct = (market_value / total_mv) * 100.0 if total_mv > 1e-12 else 0.0
        contribution_pct = (
            (total_pnl / portfolio_basis) * 100.0 if portfolio_basis > 1e-12 else 0.0
        )

        out.append(
            PositionContribution(
                symbol=sym,
                total_qty=total_qty,
                realized_pnl=realized,
                unrealized_pnl=unrealized,
                total_pnl=total_pnl,
                weight_pct=weight_pct,
                contribution_pct=contribution_pct,
            ),
        )

    out.sort(key=lambda x: x.total_pnl, reverse=True)
    return out


class BrinsonAttribution(BaseModel):
    """Brinson-Fachler attribution for one symbol."""

    symbol: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float


class BrinsonSummary(BaseModel):
    """Aggregated Brinson attribution across symbols."""

    positions: list[BrinsonAttribution]
    total_allocation: float
    total_selection: float
    total_interaction: float
    total_active_return: float


def compute_brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
) -> BrinsonSummary:
    """Brinson-Fachler attribution using per-symbol weights and simple period returns."""
    symbols = sorted(
        set(portfolio_weights)
        | set(benchmark_weights)
        | set(portfolio_returns)
        | set(benchmark_returns),
    )
    rb_total = sum(
        float(benchmark_weights.get(s, 0.0)) * float(benchmark_returns.get(s, 0.0))
        for s in symbols
    )
    rp_total = sum(
        float(portfolio_weights.get(s, 0.0)) * float(portfolio_returns.get(s, 0.0))
        for s in symbols
    )

    positions: list[BrinsonAttribution] = []
    total_a = total_s = total_i = 0.0
    for sym in symbols:
        wp = float(portfolio_weights.get(sym, 0.0))
        wb = float(benchmark_weights.get(sym, 0.0))
        rp = float(portfolio_returns.get(sym, 0.0))
        rb = float(benchmark_returns.get(sym, 0.0))
        alloc = (wp - wb) * (rb - rb_total)
        sel = wb * (rp - rb)
        inter = (wp - wb) * (rp - rb)
        total_a += alloc
        total_s += sel
        total_i += inter
        positions.append(
            BrinsonAttribution(
                symbol=sym,
                portfolio_weight=wp,
                benchmark_weight=wb,
                portfolio_return=rp,
                benchmark_return=rb,
                allocation_effect=alloc,
                selection_effect=sel,
                interaction_effect=inter,
            ),
        )

    active = rp_total - rb_total
    return BrinsonSummary(
        positions=sorted(positions, key=lambda p: p.symbol),
        total_allocation=total_a,
        total_selection=total_s,
        total_interaction=total_i,
        total_active_return=active,
    )


def _symbol_total_return(
    parquet_store: ParquetStore,
    symbol: str,
    start: date,
    end: date,
) -> float | None:
    df = parquet_store.read_ohlcv(symbol)
    if df.empty or "close" not in df.columns:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    frame = df.copy()
    frame.index = idx.normalize()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    window = frame.loc[start_ts:end_ts]
    if window.empty or len(window) < 2:
        return None
    c0 = float(window["close"].astype(float).iloc[0])
    c1 = float(window["close"].astype(float).iloc[-1])
    if c0 <= 1e-18:
        return None
    return c1 / c0 - 1.0


def compute_brinson_from_snapshots(
    lot_ledger: LotLedger,
    parquet_store: ParquetStore,
    benchmark_symbols: list[str],
    *,
    start_date: date,
    end_date: date,
) -> BrinsonSummary:
    """Equal-weight benchmark vs current portfolio weights from open lots.

    Per-symbol total returns are the same for portfolio and benchmark (same price
    history), so **selection effect is always zero** in this helper; active return
    appears in allocation and interaction only. This matches Brinson-Fachler for
    same-universe ETF portfolios where "picking" is weighting, not different assets.
    """
    bench_syms = [s.strip().upper() for s in benchmark_symbols if s.strip()]
    if not bench_syms:
        return BrinsonSummary(
            positions=[],
            total_allocation=0.0,
            total_selection=0.0,
            total_interaction=0.0,
            total_active_return=0.0,
        )
    open_syms = {lot.symbol for lot in lot_ledger.get_open_lots()}
    all_syms = sorted(set(bench_syms) | open_syms)
    prices = load_current_prices(parquet_store, all_syms)
    valuation = compute_portfolio_valuation(lot_ledger, prices)
    total_mv = float(valuation.total_market_value)
    portfolio_weights: dict[str, float] = {}
    for p in valuation.positions:
        if total_mv > 1e-12:
            portfolio_weights[p.symbol] = float(p.market_value) / total_mv
    for sym in bench_syms:
        portfolio_weights.setdefault(sym, 0.0)

    n_b = len(bench_syms)
    wb = 1.0 / float(n_b)
    benchmark_weights = {s: wb for s in bench_syms}

    portfolio_returns: dict[str, float] = {}
    benchmark_returns: dict[str, float] = {}
    for sym in all_syms:
        pr = _symbol_total_return(parquet_store, sym, start_date, end_date)
        br = _symbol_total_return(parquet_store, sym, start_date, end_date)
        if pr is not None:
            portfolio_returns[sym] = pr
        if br is not None:
            benchmark_returns[sym] = br

    return compute_brinson_attribution(
        portfolio_weights,
        benchmark_weights,
        portfolio_returns,
        benchmark_returns,
    )
