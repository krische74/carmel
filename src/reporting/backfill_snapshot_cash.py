"""Backfill broker cash onto equity_snapshots by replaying trade_executions (Tier 46C)."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class BackfillSnapshotCashResult(BaseModel):
    """Summary of a cash backfill onto equity_snapshots."""

    since: date
    seed: float
    rows_updated: int = 0
    final_cash: float
    broker_cash: float | None = None
    delta_vs_broker: float | None = Field(
        default=None,
        description="final_cash - broker_cash when broker_cash is known.",
    )
    skipped_fills: int = 0


def _parse_ts(raw: str) -> datetime:
    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _parquet_close_on_or_before(parquet_store: ParquetStore, symbol: str, d: date) -> float:
    df = parquet_store.read_ohlcv(symbol)
    if df.empty or "close" not in df.columns:
        return 0.0
    idx_sel = [i for i, ix in enumerate(df.index) if pd.Timestamp(ix).date() <= d]
    if not idx_sel:
        return 0.0
    return float(df.iloc[idx_sel[-1]]["close"])


def _resolve_fill_price(
    row: dict[str, Any],
    *,
    when: datetime,
    parquet_store: ParquetStore | None,
) -> float | None:
    px = row.get("filled_avg_price")
    if px is not None and float(px) > 0.0:
        return float(px)
    if parquet_store is None:
        return None
    sym = str(row.get("symbol", "")).strip().upper()
    if not sym:
        return None
    close = _parquet_close_on_or_before(parquet_store, sym, when.astimezone(UTC).date())
    return close if close > 0.0 else None


def _fill_qty(row: dict[str, Any]) -> float | None:
    fq = row.get("filled_qty")
    if fq is not None and float(fq) > 0.0:
        return float(fq)
    q = row.get("qty")
    if q is not None and float(q) > 0.0:
        return float(q)
    return None


def cash_by_date_from_executions(
    executions: list[dict[str, Any]],
    *,
    since: date,
    seed: float,
    parquet_store: ParquetStore | None = None,
) -> tuple[dict[str, float], int]:
    """Replay fills chronologically; return (end-of-day cash by date, skipped count).

    Buys subtract ``qty * fill_price``; sells add. Missing fill prices fall back to
    Parquet close on or before the trade date when ``parquet_store`` is provided.
    """
    since_dt = datetime(since.year, since.month, since.day, tzinfo=UTC)
    cash = float(seed)
    by_date: dict[str, float] = {}
    skipped = 0
    for row in executions:
        when = _parse_ts(str(row.get("timestamp")))
        if when < since_dt:
            continue
        qty = _fill_qty(row)
        price = _resolve_fill_price(row, when=when, parquet_store=parquet_store)
        if qty is None or price is None:
            skipped += 1
            continue
        side = str(row.get("side", "")).strip().lower()
        notional = qty * price
        if side == "buy":
            cash -= notional
        elif side == "sell":
            cash += notional
        else:
            skipped += 1
            continue
        by_date[when.astimezone(UTC).date().isoformat()] = cash
    return by_date, skipped


def backfill_snapshot_cash(
    sqlite_store: SQLiteStore,
    *,
    since: date,
    seed: float,
    account_id: str = "default",
    broker_cash: float | None = None,
    parquet_store: ParquetStore | None = None,
) -> BackfillSnapshotCashResult:
    """Write running cash onto equity_snapshots for dates >= since.

    Carries the last known balance across non-trading days. Snapshot rows with no
    prior trade yet keep the seed; rows before any trade in range receive the seed
    when their date >= since.
    """
    aid = account_id.strip() or "default"
    execs = sqlite_store.get_executions_chronological(
        since_timestamp=datetime(since.year, since.month, since.day, tzinfo=UTC).isoformat(),
        account_id=aid,
        submitted_only=True,
    )
    trade_cash, skipped = cash_by_date_from_executions(
        execs,
        since=since,
        seed=seed,
        parquet_store=parquet_store,
    )
    snaps = sqlite_store.get_equity_snapshots(
        start=since.isoformat(),
        account_id=aid,
    )
    running = float(seed)
    updated = 0
    for snap in snaps:
        d = str(snap["date"])
        if d in trade_cash:
            running = trade_cash[d]
        sqlite_store.update_equity_snapshot_cash(d, running, account_id=aid)
        updated += 1

    delta: float | None = None
    if broker_cash is not None:
        delta = running - float(broker_cash)
        logger.info(
            "backfill-snapshot-cash self-check: final_cash=%.2f broker_cash=%.2f delta=%.2f skipped=%s",
            running,
            float(broker_cash),
            delta,
            skipped,
        )
        if abs(delta) > 5.0:
            logger.warning(
                "backfill-snapshot-cash: non-trivial delta vs broker (%.2f). "
                "Cash may have moved outside trade_executions (dividends/fees), "
                "or fill prices were unresolved (skipped=%s).",
                delta,
                skipped,
            )

    return BackfillSnapshotCashResult(
        since=since,
        seed=float(seed),
        rows_updated=updated,
        final_cash=running,
        broker_cash=float(broker_cash) if broker_cash is not None else None,
        delta_vs_broker=delta,
        skipped_fills=skipped,
    )
