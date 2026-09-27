"""Rebuild tax-lot ledger and equity snapshots from trade_executions (Tier 46).

Use after an account reset (or any event that leaves ``tax_lots_*`` out of sync with
broker fills). Clears the ledger for an account, replays submitted fills since a
cutoff in chronological order with the configured lot method, then optionally
rewrites ``equity_snapshots`` from the corrected lots + Parquet closes.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.reporting.equity_curve import DailyEquityPoint, backfill_equity_curve

if TYPE_CHECKING:
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore
    from src.portfolio.tax_lots import LotLedger

logger = logging.getLogger(__name__)


class RebuildLotsResult(BaseModel):
    """Summary of a lot-ledger rebuild."""

    account_id: str
    since: datetime
    buys_replayed: int = 0
    sells_replayed: int = 0
    skipped: int = 0
    open_symbols: dict[str, float] = Field(default_factory=dict)
    total_cost_basis: float = 0.0
    realized_pnl: float = 0.0


def _parse_ts(raw: str | datetime) -> datetime:
    if isinstance(raw, datetime):
        dt = raw
    else:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _parquet_close_on_or_before(parquet_store: ParquetStore, symbol: str, d: date) -> float:
    """Latest close on or before calendar date ``d`` (no lookahead)."""
    df = parquet_store.read_ohlcv(symbol)
    if df.empty or "close" not in df.columns:
        return 0.0
    idx_sel = [i for i, ix in enumerate(df.index) if pd.Timestamp(ix).date() <= d]
    if not idx_sel:
        return 0.0
    last_i = idx_sel[-1]
    return float(df.iloc[last_i]["close"])


def _resolve_fill_price(
    row: dict,
    parquet_store: ParquetStore,
    when: datetime,
) -> float | None:
    """Prefer ``filled_avg_price``; else Parquet close on or before the trade date."""
    px = row.get("filled_avg_price")
    if px is not None and float(px) > 0.0:
        return float(px)
    sym = str(row.get("symbol", "")).strip().upper()
    if not sym:
        return None
    close = _parquet_close_on_or_before(parquet_store, sym, when.astimezone(UTC).date())
    return close if close > 0.0 else None


def _fill_qty(row: dict) -> float | None:
    fq = row.get("filled_qty")
    if fq is not None and float(fq) > 0.0:
        return float(fq)
    q = row.get("qty")
    if q is not None and float(q) > 0.0:
        return float(q)
    return None


def rebuild_lot_ledger_from_executions(
    sqlite_store: SQLiteStore,
    lot_ledger: LotLedger,
    *,
    parquet_store: ParquetStore,
    since: datetime,
    account_id: str = "default",
    method: Literal["fifo", "hifo"] = "fifo",
) -> RebuildLotsResult:
    """Clear lots for ``account_id`` and replay submitted fills since ``since``.

    Skips rows without a resolvable qty or price. Sell qty that exceeds open lots
    raises (same as live ``record_sell``) so the operator sees a hard failure
    rather than a silently truncated book.
    """
    aid = account_id.strip() or "default"
    since_aware = _parse_ts(since)
    lot_ledger.clear_account(aid)

    rows = sqlite_store.get_executions_chronological(
        since_timestamp=since_aware.isoformat(),
        account_id=aid,
        submitted_only=True,
    )

    buys = 0
    sells = 0
    skipped = 0
    for row in rows:
        side = str(row.get("side", "")).strip().lower()
        qty = _fill_qty(row)
        when = _parse_ts(str(row.get("timestamp")))
        if qty is None:
            skipped += 1
            continue
        price = _resolve_fill_price(row, parquet_store, when)
        if price is None:
            logger.warning(
                "rebuild_lots skip %s %s @ %s: no fill price and no Parquet close",
                side,
                row.get("symbol"),
                when.isoformat(),
            )
            skipped += 1
            continue
        sym = str(row.get("symbol", "")).strip().upper()
        if side == "buy":
            lot_ledger.record_buy(sym, qty, price, when, account_id=aid)
            buys += 1
        elif side == "sell":
            lot_ledger.record_sell(sym, qty, price, when, method=method, account_id=aid)
            sells += 1
        else:
            skipped += 1

    lot_ledger.prune_dust_lots(aid)
    open_lots = lot_ledger.get_open_lots(account_id=aid)
    by_sym: dict[str, float] = {}
    cost = 0.0
    for lot in open_lots:
        by_sym[lot.symbol] = by_sym.get(lot.symbol, 0.0) + float(lot.qty)
        cost += float(lot.qty) * float(lot.cost_per_share)

    return RebuildLotsResult(
        account_id=aid,
        since=since_aware,
        buys_replayed=buys,
        sells_replayed=sells,
        skipped=skipped,
        open_symbols=by_sym,
        total_cost_basis=cost,
        realized_pnl=float(lot_ledger.realized_pnl(account_id=aid)),
    )


def rebuild_equity_snapshots_from_ledger(
    sqlite_store: SQLiteStore,
    lot_ledger: LotLedger,
    parquet_store: ParquetStore,
    *,
    start: date,
    end: date,
    account_id: str = "default",
) -> list[DailyEquityPoint]:
    """Delete then rewrite daily equity snapshots for ``[start, end]`` from the ledger.

    Also deletes any snapshots **before** ``start`` for the account — those typically
    belong to a prior account life and would poison the curve.
    """
    aid = account_id.strip() or "default"
    existing_rows = sqlite_store.get_equity_snapshots(
        start=start.isoformat(),
        end=end.isoformat(),
        account_id=aid,
    )
    preserve_cash = {str(r["date"]): r.get("cash") for r in existing_rows}
    preserve_be = {str(r["date"]): r.get("broker_equity") for r in existing_rows}
    sqlite_store.delete_equity_snapshots(
        account_id=aid,
        end=(start - timedelta(days=1)).isoformat(),
    )
    sqlite_store.delete_equity_snapshots(
        account_id=aid,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    return backfill_equity_curve(
        sqlite_store,
        lot_ledger,
        parquet_store,
        start,
        end,
        account_id=aid,
        preserve_cash_by_date=preserve_cash,
        preserve_broker_equity_by_date=preserve_be,
    )
