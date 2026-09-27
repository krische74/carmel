"""Daily portfolio equity snapshots and historical reconstruction from lots + Parquet."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel

from src.reporting.portfolio_analytics import compute_portfolio_valuation, load_current_prices

if TYPE_CHECKING:
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore
    from src.portfolio.tax_lots import LotLedger


class DailyEquityPoint(BaseModel):
    """One stored daily portfolio valuation.

    ``total_market_value`` is positions-only. ``cash`` is broker cash when known;
    equity = MV + cash. ``cash is None`` means unknown (backfill / broker failure).
    """

    date: str
    total_market_value: float
    total_cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    cash: float | None = None
    broker_equity: float | None = None


def _as_utc_date(dt: datetime) -> date:
    if dt.tzinfo is None:
        return dt.date()
    return dt.astimezone(UTC).date()


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


def _build_lot_state(
    lot_ledger: LotLedger,
    *,
    account_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Map lot_id to open lot (if any) and closed tranches."""
    open_lots = lot_ledger.get_open_lots(account_id=account_id)
    closed = lot_ledger.get_closed_lots(account_id=account_id)
    by_id: dict[str, dict[str, Any]] = {}
    for lot in open_lots:
        by_id[lot.id] = {"open": lot, "closed_chunks": []}
    for cl in closed:
        if cl.lot_id not in by_id:
            by_id[cl.lot_id] = {
                "open": None,
                "symbol": cl.symbol,
                "opened_at": cl.opened_at,
                "cost_per_share": cl.cost_per_share,
                "closed_chunks": [],
            }
        by_id[cl.lot_id]["closed_chunks"].append((float(cl.qty), cl.closed_at))
    return by_id


def _qty_on_date(lot_id: str, as_of: date, state: dict[str, Any]) -> float:
    """Shares held from this tax lot on ``as_of`` (FIFO tranches closed on or before)."""
    ol = state.get("open")
    opened_at = ol.opened_at if ol is not None else state["opened_at"]
    if _as_utc_date(opened_at) > as_of:
        return 0.0
    remaining_open = float(ol.qty) if ol is not None else 0.0
    sold_by_d = sum(q for q, ct in state["closed_chunks"] if _as_utc_date(ct) <= as_of)
    original = remaining_open + sum(q for q, _ in state["closed_chunks"])
    return max(0.0, original - sold_by_d)


def _valuation_for_historical_date(
    lot_ledger: LotLedger,
    parquet_store: ParquetStore,
    as_of: date,
    *,
    account_id: str | None = None,
) -> DailyEquityPoint:
    """Point-in-time portfolio metrics using only prices and lot events available by end of ``as_of``."""
    states = _build_lot_state(lot_ledger, account_id=account_id)
    total_market_value = 0.0
    total_cost_basis = 0.0
    for _lot_id, st in states.items():
        q = _qty_on_date(_lot_id, as_of, st)
        if q <= 1e-12:
            continue
        ol = st.get("open")
        sym = ol.symbol if ol is not None else str(st["symbol"])
        cps = float(ol.cost_per_share) if ol is not None else float(st["cost_per_share"])
        px = _parquet_close_on_or_before(parquet_store, sym, as_of)
        total_market_value += q * px
        total_cost_basis += q * cps

    closed = lot_ledger.get_closed_lots(account_id=account_id)
    realized_pnl = sum(
        float(cl.realized_pnl) for cl in closed if _as_utc_date(cl.closed_at) <= as_of
    )
    unrealized_pnl = total_market_value - total_cost_basis
    total_pnl = unrealized_pnl + realized_pnl

    return DailyEquityPoint(
        date=as_of.isoformat(),
        total_market_value=total_market_value,
        total_cost_basis=total_cost_basis,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        total_pnl=total_pnl,
    )


def record_equity_snapshot(
    sqlite_store: SQLiteStore,
    lot_ledger: LotLedger,
    parquet_store: ParquetStore,
    *,
    as_of_date: date | None = None,
    account_id: str = "default",
    cash: float | None = None,
    broker_equity: float | None = None,
) -> DailyEquityPoint:
    """Compute current valuation from open lots + latest Parquet closes and persist one row.

    ``cash`` is broker cash at snapshot time (nullable). Pass ``None`` when unknown.
    ``broker_equity`` is the broker account equity from the same read (nullable).
    """
    d = as_of_date or datetime.now(UTC).date()
    aid = account_id.strip() or "default"
    syms = sorted({lot.symbol for lot in lot_ledger.get_open_lots(account_id=aid)})
    prices = load_current_prices(parquet_store, syms)
    v = compute_portfolio_valuation(lot_ledger, prices, account_id=aid)
    sqlite_store.write_equity_snapshot(
        d.isoformat(),
        v.total_market_value,
        v.total_cost_basis,
        v.total_unrealized_pnl,
        v.total_realized_pnl,
        v.total_pnl,
        account_id=aid,
        cash=cash,
        broker_equity=broker_equity,
    )
    return DailyEquityPoint(
        date=d.isoformat(),
        total_market_value=v.total_market_value,
        total_cost_basis=v.total_cost_basis,
        unrealized_pnl=v.total_unrealized_pnl,
        realized_pnl=v.total_realized_pnl,
        total_pnl=v.total_pnl,
        cash=cash,
        broker_equity=broker_equity,
    )


def backfill_equity_curve(
    sqlite_store: SQLiteStore,
    lot_ledger: LotLedger,
    parquet_store: ParquetStore,
    start: date,
    end: date,
    *,
    account_id: str = "default",
    preserve_cash_by_date: dict[str, float | None] | None = None,
    preserve_broker_equity_by_date: dict[str, float | None] | None = None,
) -> list[DailyEquityPoint]:
    """Write missing daily snapshots between ``start`` and ``end`` (inclusive).

    Lot-derived rows have no cash source — ``cash`` is NULL unless
    ``preserve_cash_by_date`` / ``preserve_broker_equity_by_date`` supply a
    prior broker read for that date (e.g. lot-ledger rebuild must not clobber
    Tier 46/47 columns).
    """
    aid = account_id.strip() or "default"
    cash_preserve = preserve_cash_by_date or {}
    be_preserve = preserve_broker_equity_by_date or {}
    out: list[DailyEquityPoint] = []
    existing_rows = sqlite_store.get_equity_snapshots(
        start=start.isoformat(),
        end=end.isoformat(),
        account_id=aid,
    )
    existing = {str(r["date"]) for r in existing_rows}
    d = start
    while d <= end:
        ds = d.isoformat()
        if ds not in existing:
            pt = _valuation_for_historical_date(
                lot_ledger,
                parquet_store,
                d,
                account_id=aid,
            )
            sqlite_store.write_equity_snapshot(
                pt.date,
                pt.total_market_value,
                pt.total_cost_basis,
                pt.unrealized_pnl,
                pt.realized_pnl,
                pt.total_pnl,
                account_id=aid,
                cash=cash_preserve.get(ds),
                broker_equity=be_preserve.get(ds),
            )
            out.append(
                DailyEquityPoint(
                    date=pt.date,
                    total_market_value=pt.total_market_value,
                    total_cost_basis=pt.total_cost_basis,
                    unrealized_pnl=pt.unrealized_pnl,
                    realized_pnl=pt.realized_pnl,
                    total_pnl=pt.total_pnl,
                    cash=cash_preserve.get(ds),
                    broker_equity=be_preserve.get(ds),
                ),
            )
        d += timedelta(days=1)
    return out
