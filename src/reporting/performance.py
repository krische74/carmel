"""Performance summaries from stored execution history (no live broker calls)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.data.storage.parquet_store import ParquetStore


class PerformanceSummary(BaseModel):
    """Aggregated counts from ``trade_executions`` rows."""

    total_trades: int = 0
    buys: int = 0
    sells: int = 0
    submitted_count: int = 0
    rejected_count: int = 0
    first_trade: datetime | None = None
    last_trade: datetime | None = None


def compute_performance_summary(executions: list[dict[str, Any]]) -> PerformanceSummary:
    """Aggregate ``get_executions()`` rows (newest-first order is fine)."""
    if not executions:
        return PerformanceSummary()

    submitted = 0
    rejected = 0
    buys = 0
    sells = 0
    times: list[datetime] = []

    for row in executions:
        side = str(row.get("side") or "buy").lower()
        if side == "sell":
            sells += 1
        else:
            buys += 1
        is_sub = bool(row.get("submitted"))
        if is_sub:
            submitted += 1
        else:
            rejected += 1
        ts_raw = row.get("timestamp")
        if ts_raw:
            dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            times.append(dt)

    first = min(times) if times else None
    last = max(times) if times else None

    return PerformanceSummary(
        total_trades=len(executions),
        buys=buys,
        sells=sells,
        submitted_count=submitted,
        rejected_count=rejected,
        first_trade=first,
        last_trade=last,
    )


def compute_equity_curve(
    store: ParquetStore,
    symbol: str,
    *,
    start: str | None = None,
) -> pd.DataFrame:
    """Load benchmark close prices from Parquet (placeholder for portfolio equity)."""
    df = store.read_ohlcv(symbol.strip().upper())
    if df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame(
        {
            "date": df.index,
            "close": df["close"].astype(float).values,
        }
    )
    if start:
        start_ts = pd.Timestamp(start)
        out = out[out["date"] >= start_ts]
    return out.reset_index(drop=True)
