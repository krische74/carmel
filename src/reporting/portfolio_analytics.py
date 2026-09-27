"""Portfolio valuation from tax lots and stored market prices (no live broker calls)."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.data.storage.parquet_store import ParquetStore
    from src.portfolio.tax_lots import LotLedger


class PositionSummary(BaseModel):
    """Per-symbol aggregation of open lots with mark-to-market valuation."""

    symbol: str
    open_lots: int
    total_qty: float
    avg_cost_per_share: float
    total_cost_basis: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float = Field(
        description="unrealized_pnl / total_cost_basis, or 0 if no basis",
    )


class PortfolioValuation(BaseModel):
    """Full portfolio valuation at a point in time."""

    positions: list[PositionSummary]
    total_market_value: float
    total_cost_basis: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float


def compute_portfolio_valuation(
    lot_ledger: LotLedger,
    current_prices: dict[str, float],
    *,
    account_id: str | None = None,
) -> PortfolioValuation:
    """Build a portfolio valuation from open lots and current market prices."""
    open_lots = lot_ledger.get_open_lots(account_id=account_id)
    by_symbol: defaultdict[str, list[Any]] = defaultdict(list)
    for lot in open_lots:
        by_symbol[lot.symbol].append(lot)

    positions: list[PositionSummary] = []
    for sym in sorted(by_symbol.keys()):
        lots = by_symbol[sym]
        total_qty = sum(float(lot.qty) for lot in lots)
        total_cost_basis = sum(float(lot.qty) * float(lot.cost_per_share) for lot in lots)
        avg_cost = total_cost_basis / total_qty if total_qty > 1e-12 else 0.0
        current_price = float(current_prices.get(sym, 0.0))
        market_value = total_qty * current_price
        unrealized_pnl = market_value - total_cost_basis
        unrealized_pnl_pct = unrealized_pnl / total_cost_basis if total_cost_basis > 1e-12 else 0.0
        positions.append(
            PositionSummary(
                symbol=sym,
                open_lots=len(lots),
                total_qty=total_qty,
                avg_cost_per_share=avg_cost,
                total_cost_basis=total_cost_basis,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
            ),
        )

    total_market_value = sum(p.market_value for p in positions)
    total_cost_basis_open = sum(p.total_cost_basis for p in positions)
    total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    total_realized_pnl = float(lot_ledger.realized_pnl(account_id=account_id))
    total_pnl = total_unrealized_pnl + total_realized_pnl

    return PortfolioValuation(
        positions=positions,
        total_market_value=total_market_value,
        total_cost_basis=total_cost_basis_open,
        total_unrealized_pnl=total_unrealized_pnl,
        total_realized_pnl=total_realized_pnl,
        total_pnl=total_pnl,
    )


def load_current_prices(
    parquet_store: ParquetStore,
    symbols: list[str],
) -> dict[str, float]:
    """Extract the latest close price from Parquet for each symbol."""
    out: dict[str, float] = {}
    for raw in symbols:
        sym = raw.strip().upper()
        df = parquet_store.read_ohlcv(sym)
        if df is None or df.empty or "close" not in df.columns:
            continue
        out[sym] = float(df["close"].astype(float).iloc[-1])
    return out
