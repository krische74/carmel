"""Backfill ``trade_executions.filled_qty`` from broker fill quotes.

NULL ``filled_qty`` is replayed as the requested ``qty`` by the lot rebuild.
A known-unknown (broker no longer has the order) stays NULL. Never default to
the requested size, and never rewrite ``qty`` / ``side`` / ``symbol`` / ``submitted``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

_QTY_EQ_TOL = 1e-6


class BrokerOrderMissingError(Exception):
    """Broker no longer has this order. The execution row must stay NULL."""


class FillQuote(BaseModel):
    """Broker fill fields for one order. ``None`` means the broker did not say."""

    filled_qty: float | None
    filled_avg_price: float | None = None


class PartialFillRow(BaseModel):
    """A submitted row whose broker fill differs from the requested qty."""

    timestamp: str
    symbol: str
    side: str
    order_id: str
    qty: float
    filled_qty: float
    filled_avg_price: float | None = None


class BackfillFillQtyResult(BaseModel):
    """Summary of a filled_qty backfill."""

    rows_considered: int = 0
    rows_updated: int = 0
    rows_left_null: int = 0
    dry_run: bool = False
    partials: list[PartialFillRow] = Field(default_factory=list)
    missing_order_ids: list[str] = Field(default_factory=list)


def _is_partial(requested: float | None, filled: float) -> bool:
    if requested is None:
        return False
    return abs(float(filled) - float(requested)) > _QTY_EQ_TOL


def backfill_filled_qty(
    sqlite_store: SQLiteStore,
    *,
    since: datetime,
    account_id: str,
    fetch_fill: Callable[[str], FillQuote],
    dry_run: bool,
) -> BackfillFillQtyResult:
    """Write broker ``filled_qty`` / ``filled_avg_price`` onto NULL fill rows.

    Rows whose broker order is gone stay NULL. ``dry_run`` fetches and reports
    the same diff but writes nothing.
    """
    aid = account_id.strip() or "default"
    since_iso = since.isoformat()
    rows = sqlite_store.list_submitted_executions_missing_filled_qty(
        since_timestamp=since_iso,
        account_id=aid,
    )
    result = BackfillFillQtyResult(rows_considered=len(rows), dry_run=dry_run)

    for row in rows:
        oid = str(row.get("order_id") or "").strip()
        symbol = str(row.get("symbol") or "")
        side = str(row.get("side") or "")
        ts = str(row.get("timestamp") or "")
        requested_raw = row.get("qty")
        requested = float(requested_raw) if requested_raw is not None else None
        if not oid:
            logger.warning(
                "backfill-fill-qty: submitted row id=%s %s %s has no order_id; leaving filled_qty NULL",
                row.get("id"),
                symbol,
                side,
            )
            result.rows_left_null += 1
            continue

        try:
            quote = fetch_fill(oid)
        except BrokerOrderMissingError:
            logger.warning(
                "backfill-fill-qty: broker has no order %s (%s %s %s); leaving filled_qty NULL",
                oid,
                ts,
                symbol,
                side,
            )
            result.rows_left_null += 1
            result.missing_order_ids.append(oid)
            continue

        if quote.filled_qty is None:
            logger.warning(
                "backfill-fill-qty: broker order %s (%s %s) has no filled_qty; leaving NULL",
                oid,
                symbol,
                side,
            )
            result.rows_left_null += 1
            continue

        filled = float(quote.filled_qty)
        price = quote.filled_avg_price
        if _is_partial(requested, filled):
            partial = PartialFillRow(
                timestamp=ts,
                symbol=symbol,
                side=side,
                order_id=oid,
                qty=float(requested) if requested is not None else 0.0,
                filled_qty=filled,
                filled_avg_price=float(price) if price is not None else None,
            )
            result.partials.append(partial)
            logger.warning(
                "backfill-fill-qty PARTIAL %s %s %s requested=%.6f filled=%.6f price=%s order_id=%s%s",
                ts,
                symbol,
                side,
                float(requested) if requested is not None else 0.0,
                filled,
                f"{float(price):.6f}" if price is not None else "n/a",
                oid,
                " (dry-run)" if dry_run else "",
            )

        if dry_run:
            continue

        sqlite_store.update_execution_fill_fields(
            int(row["id"]),
            filled_qty=filled,
            filled_avg_price=float(price) if price is not None else None,
        )
        result.rows_updated += 1

    return result


def format_backfill_fill_qty_summary(result: BackfillFillQtyResult) -> str:
    """Operator-facing summary, including every silent partial fill."""
    lines = [
        (
            f"backfill-fill-qty dry_run={result.dry_run} "
            f"considered={result.rows_considered} "
            f"updated={result.rows_updated} "
            f"left_null={result.rows_left_null} "
            f"partials={len(result.partials)} "
            f"missing={len(result.missing_order_ids)}"
        ),
    ]
    if result.partials:
        lines.append(
            f"{'timestamp':<28} {'symbol':<6} {'side':<4} {'requested':>12} {'filled':>12} {'price':>12}  order_id"
        )
        for p in result.partials:
            px = f"{p.filled_avg_price:.6f}" if p.filled_avg_price is not None else "n/a"
            lines.append(
                f"{p.timestamp:<28} {p.symbol:<6} {p.side:<4} {p.qty:12.6f} {p.filled_qty:12.6f} {px:>12}  {p.order_id}"
            )
    else:
        lines.append("partials: none (every broker fill matched requested qty, or no rows)")
    if result.missing_order_ids:
        lines.append("left NULL (broker order missing):")
        lines.extend(f"  {oid}" for oid in result.missing_order_ids)
    return "\n".join(lines)
