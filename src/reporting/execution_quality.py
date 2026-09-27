"""Execution quality: slippage vs daily close reference (no lookahead, not live mid)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.data.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


class ExecutionQualityRow(BaseModel):
    """One fill compared to a point-in-time daily reference close."""

    execution_id: int
    cycle_id: str
    execution_ts: str
    symbol: str
    side: str
    qty: float | None = None
    fill_price: float | None = None
    reference_price: float | None = None
    slippage_bps: float | None = None
    note: str = Field(
        default="Reference = last Parquet daily close on or before execution calendar date (UTC).",
    )


def slippage_bps(side: str, fill: float, reference: float) -> float | None:
    """Signed basis points: positive = adverse vs reference.

    Buy: adverse when fill > reference → positive bps.
    Sell: adverse when fill < reference → positive bps.
    """
    if reference <= 0.0 or fill <= 0.0:
        return None
    s = str(side).strip().lower()
    if s == "buy":
        return float((fill - reference) / reference * 10_000.0)
    if s == "sell":
        return float((reference - fill) / reference * 10_000.0)
    return None


def reference_price_for_execution(
    symbol: str,
    execution_ts: datetime,
    parquet_store: ParquetStore,
) -> float | None:
    """Last available daily **close** with bar date ≤ execution calendar date (UTC).

    Same-day or prior close only — not an intraday mid quote.
    """
    sym = symbol.strip().upper()
    df = parquet_store.read_ohlcv(sym)
    if df is None or df.empty or "close" not in df.columns:
        return None
    exec_day = execution_ts.astimezone(UTC).date()
    frame = df.sort_index().copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    mask = frame.index.map(lambda t: t.date() <= exec_day)
    sub = frame.loc[mask]
    if sub.empty:
        return None
    return float(sub["close"].astype(float).iloc[-1])


def build_execution_quality_row(
    *,
    execution_id: int,
    cycle_id: str,
    result_side: str,
    result_qty: float | None,
    fill_price: float | None,
    execution_ts: datetime,
    symbol: str,
    parquet_store: ParquetStore,
) -> ExecutionQualityRow | None:
    """Compute reference and slippage for a submitted execution with a fill."""
    if fill_price is None or float(fill_price) <= 0.0:
        return None
    ref = reference_price_for_execution(symbol, execution_ts, parquet_store)
    if ref is None or ref <= 0.0:
        logger.debug("No reference price for %s at %s", symbol, execution_ts.date())
        return ExecutionQualityRow(
            execution_id=execution_id,
            cycle_id=cycle_id,
            execution_ts=execution_ts.astimezone(UTC).isoformat(),
            symbol=symbol.strip().upper(),
            side=str(result_side).lower(),
            qty=float(result_qty) if result_qty is not None else None,
            fill_price=float(fill_price),
            reference_price=None,
            slippage_bps=None,
        )
    slip = slippage_bps(result_side, float(fill_price), ref)
    return ExecutionQualityRow(
        execution_id=execution_id,
        cycle_id=cycle_id,
        execution_ts=execution_ts.astimezone(UTC).isoformat(),
        symbol=symbol.strip().upper(),
        side=str(result_side).lower(),
        qty=float(result_qty) if result_qty is not None else None,
        fill_price=float(fill_price),
        reference_price=ref,
        slippage_bps=slip,
    )


def execution_quality_row_to_dict(row: ExecutionQualityRow) -> dict[str, Any]:
    """JSON/API-friendly dict."""
    return row.model_dump()
