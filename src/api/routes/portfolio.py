"""Portfolio valuation endpoints (read-only, same logic as dashboard)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_lot_ledger, get_parquet_store
from src.data.storage.parquet_store import ParquetStore
from src.portfolio.tax_lots import LotLedger
from src.reporting.portfolio_analytics import (
    PortfolioValuation,
    PositionSummary,
    compute_portfolio_valuation,
    load_current_prices,
)

router = APIRouter(tags=["portfolio"])


def _valuation(ledger: LotLedger, parquet_store: ParquetStore) -> PortfolioValuation:
    open_lots = ledger.get_open_lots()
    closed = ledger.get_closed_lots()
    symbols = sorted({lot.symbol for lot in open_lots} | {c.symbol for c in closed})
    prices = load_current_prices(parquet_store, symbols) if symbols else {}
    return compute_portfolio_valuation(ledger, prices)


@router.get("/portfolio", response_model=PortfolioValuation)
def get_portfolio(
    ledger: LotLedger = Depends(get_lot_ledger),
    parquet_store: ParquetStore = Depends(get_parquet_store),
) -> PortfolioValuation:
    """Full portfolio valuation from tax lots and latest Parquet closes."""
    return _valuation(ledger, parquet_store)


@router.get("/portfolio/positions", response_model=list[PositionSummary])
def get_positions(
    ledger: LotLedger = Depends(get_lot_ledger),
    parquet_store: ParquetStore = Depends(get_parquet_store),
) -> list[PositionSummary]:
    """Open position rows with mark-to-market metrics."""
    return _valuation(ledger, parquet_store).positions
