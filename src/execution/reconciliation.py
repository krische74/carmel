"""Compare SQLite execution log rows to broker-reported orders."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Per-cycle broker scope: include fills shortly before/after logged cycle rows.
_DEFAULT_CYCLE_LOOKBACK = timedelta(hours=1)
_DEFAULT_CYCLE_FORWARD = timedelta(minutes=15)

ReconciliationStatus = Literal[
    "matched",
    "missing_from_broker",
    "missing_from_log",
    "field_mismatch",
    "qty_mismatch",
    "price_mismatch",
]


class ReconciliationEntry(BaseModel):
    """One order compared across internal log and broker."""

    order_id: str
    symbol: str = ""
    side: str = ""
    status: ReconciliationStatus
    local_qty: float | None = Field(None, description="Qty from internal log when present.")
    broker_qty: float | None = Field(None, description="Broker filled or order qty when present.")
    local_price: float | None = Field(None, description="Internal filled_avg_price when present.")
    broker_price: float | None = Field(None, description="Broker filled_avg_price when present.")


class ReconciliationResult(BaseModel):
    """Summary of reconciliation between internal and broker state."""

    entries: list[ReconciliationEntry]
    matched: int
    discrepancies: int


def _is_submitted(row: dict[str, Any]) -> bool:
    s = row.get("submitted")
    if s is None or s is False or s == 0 or s == "0":
        return False
    return s == 1 or s is True or s == "1"


def submitted_execution_order_ids(sqlite_rows: list[dict[str, Any]]) -> set[str]:
    """Return broker ``order_id`` values from submitted execution rows.

    Matches the rows :func:`reconcile_executions` indexes from SQLite so callers
    can restrict broker order lists to the same scope (e.g. one trading cycle).
    """
    out: set[str] = set()
    for row in sqlite_rows:
        oid = row.get("order_id")
        if not oid or not _is_submitted(row):
            continue
        out.add(str(oid))
    return out


def _parse_ts(raw: object) -> datetime | None:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    if not raw:
        return None
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def broker_orders_in_cycle_window(
    broker_orders: list[dict[str, Any]],
    sqlite_rows: list[dict[str, Any]],
    *,
    as_of: datetime | None = None,
    lookback: timedelta = _DEFAULT_CYCLE_LOOKBACK,
    forward: timedelta = _DEFAULT_CYCLE_FORWARD,
) -> list[dict[str, Any]]:
    """Scope broker orders to the cycle time window (not by known order ids alone).

    Includes every broker order whose timestamp falls in
    ``[min(cycle_ts) - lookback, max(cycle_ts) + forward]``. When cycle rows have
    no timestamps, ``as_of`` anchors the window. Orders with no timestamp are kept
    only when their ``order_id`` is already in the cycle log — so historical broker
    noise without timestamps stays out, while known cycle ids still match.
    """
    cycle_ids = submitted_execution_order_ids(sqlite_rows)
    stamps: list[datetime] = []
    for row in sqlite_rows:
        if not _is_submitted(row):
            continue
        ts = _parse_ts(row.get("timestamp"))
        if ts is not None:
            stamps.append(ts)
    if stamps:
        start = min(stamps) - lookback
        end = max(stamps) + forward
    elif as_of is not None:
        anchor = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=UTC)
        start = anchor - lookback
        end = anchor + forward
    else:
        # No temporal anchor: only orders already logged for this cycle.
        return [o for o in broker_orders if str(o.get("order_id") or "") in cycle_ids]

    out: list[dict[str, Any]] = []
    for o in broker_orders:
        oid = str(o.get("order_id") or "")
        if oid and oid in cycle_ids:
            out.append(o)
            continue
        ts = _parse_ts(o.get("timestamp"))
        if ts is None:
            continue
        if start <= ts <= end:
            out.append(o)
    return out


def reconcile_executions(
    sqlite_rows: list[dict[str, Any]],
    broker_orders: list[dict[str, Any]],
) -> ReconciliationResult:
    """Compare logged executions to broker orders by ``order_id``.

    ``sqlite_rows`` should be rows from ``SQLiteStore.get_executions`` (optionally
    filtered). ``broker_orders`` are dicts from ``BrokerInterface.list_recent_orders``.
    """
    sqlite_by_id: dict[str, dict[str, Any]] = {}
    for row in sqlite_rows:
        oid = row.get("order_id")
        if not oid or not _is_submitted(row):
            continue
        sid = str(oid)
        if sid in sqlite_by_id:
            logger.warning(
                "Duplicate order_id in SQLite execution rows: %s (keeping last row)",
                sid,
            )
        sqlite_by_id[sid] = row

    broker_by_id: dict[str, dict[str, Any]] = {}
    for o in broker_orders:
        oid = o.get("order_id")
        if not oid:
            continue
        bid = str(oid)
        if bid in broker_by_id:
            logger.warning(
                "Duplicate order_id in broker order list: %s (keeping last row)",
                bid,
            )
        broker_by_id[bid] = o

    entries: list[ReconciliationEntry] = []
    matched = 0

    for oid, row in sqlite_by_id.items():
        b = broker_by_id.get(oid)
        if b is None:
            entries.append(
                ReconciliationEntry(
                    order_id=oid,
                    symbol=str(row.get("symbol", "")),
                    side=str(row.get("side", "")),
                    status="missing_from_broker",
                ),
            )
            continue

        sym_row = str(row.get("symbol", "")).strip().upper()
        sym_b = str(b.get("symbol", "")).strip().upper()
        side_row = str(row.get("side", "")).strip().lower()
        side_b = str(b.get("side", "")).strip().lower()

        if sym_row != sym_b or side_row != side_b:
            entries.append(
                ReconciliationEntry(
                    order_id=oid,
                    symbol=sym_row,
                    side=side_row,
                    status="field_mismatch",
                ),
            )
            continue

        sqty = row.get("qty")
        bqty = b.get("filled_qty")
        if bqty is None:
            bqty = b.get("qty")
        lq = float(sqty) if sqty is not None else None
        bq = float(bqty) if bqty is not None else None

        if lq is not None and bq is not None and abs(lq - bq) > 1e-5:
            entries.append(
                ReconciliationEntry(
                    order_id=oid,
                    symbol=sym_row,
                    side=side_row,
                    status="qty_mismatch",
                    local_qty=lq,
                    broker_qty=bq,
                ),
            )
            continue

        spr = row.get("filled_avg_price")
        bpr = b.get("filled_avg_price")
        lp = float(spr) if spr is not None else None
        bp = float(bpr) if bpr is not None else None

        if lp is not None and bp is not None and abs(lp - bp) > 1e-4:
            entries.append(
                ReconciliationEntry(
                    order_id=oid,
                    symbol=sym_row,
                    side=side_row,
                    status="price_mismatch",
                    local_price=lp,
                    broker_price=bp,
                ),
            )
            continue

        entries.append(
            ReconciliationEntry(
                order_id=oid,
                symbol=sym_row,
                side=side_row,
                status="matched",
                local_qty=lq,
                broker_qty=bq,
                local_price=lp,
                broker_price=bp,
            ),
        )
        matched += 1

    for oid, b in broker_by_id.items():
        if oid not in sqlite_by_id:
            entries.append(
                ReconciliationEntry(
                    order_id=oid,
                    symbol=str(b.get("symbol", "")),
                    side=str(b.get("side", "")),
                    status="missing_from_log",
                ),
            )

    discrepancies = sum(1 for e in entries if e.status != "matched")
    return ReconciliationResult(entries=entries, matched=matched, discrepancies=discrepancies)
