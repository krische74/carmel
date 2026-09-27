"""Pattern day trader (PDT) awareness from recorded executions."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class PDTCheckResult(BaseModel):
    """Outcome of comparing projected day trades to equity-based limits."""

    allowed: bool
    warning: str | None = Field(default=None)
    day_trade_count: int = Field(
        ...,
        ge=0,
        description="Projected day-trade count used for the decision (including this sell if applicable).",
    )


def _as_utc_date(ts: dt.datetime) -> dt.date:
    if ts.tzinfo is None:
        return ts.date()
    return ts.astimezone(dt.UTC).date()


def _parse_execution_timestamp(raw: str | None) -> dt.date | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return _as_utc_date(parsed)


def _last_n_weekday_dates(end: dt.date, n: int) -> frozenset[dt.date]:
    """Last ``n`` Monday-Friday dates ending at ``end`` (inclusive when weekday)."""
    out: list[dt.date] = []
    d = end
    guard = 0
    while len(out) < n and guard < 366:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
        guard += 1
    return frozenset(out)


def _execution_counts_by_symbol_date(
    executions: list[dict[str, Any]],
    *,
    allowed_dates: frozenset[dt.date],
) -> dict[tuple[str, dt.date], tuple[int, int]]:
    """Map (symbol, date) -> (buy_count, sell_count) for filled, submitted rows."""
    buckets: dict[tuple[str, dt.date], list[int]] = {}

    for row in executions:
        if not int(row.get("submitted") or 0):
            continue
        status = str(row.get("order_status") or "filled").strip().lower()
        if status not in ("filled",):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        side = str(row.get("side") or "").strip().lower()
        if side not in ("buy", "sell"):
            continue
        d = _parse_execution_timestamp(row.get("timestamp"))
        if d is None or d not in allowed_dates:
            continue
        key = (sym, d)
        if key not in buckets:
            buckets[key] = [0, 0]
        if side == "buy":
            buckets[key][0] += 1
        else:
            buckets[key][1] += 1

    return {k: (v[0], v[1]) for k, v in buckets.items()}


def count_day_trades(
    executions: list[dict[str, Any]],
    *,
    as_of: dt.datetime,
    lookback_days: int = 5,
) -> int:
    """Count completed round-trip day trades in the rolling weekday lookback window."""
    n = max(1, min(int(lookback_days), 60))
    end = _as_utc_date(as_of)
    allowed = _last_n_weekday_dates(end, n)
    counts = _execution_counts_by_symbol_date(executions, allowed_dates=allowed)
    total = 0
    for buys, sells in counts.values():
        total += min(buys, sells)
    return int(total)


def sell_adds_day_trade(
    executions: list[dict[str, Any]],
    *,
    symbol: str,
    as_of: dt.datetime,
) -> bool:
    """True if a sell now would complete another same-day round trip for ``symbol``."""
    sym = symbol.strip().upper()
    d = _as_utc_date(as_of)
    if d.weekday() >= 5:
        return False
    buys = 0
    sells = 0
    for row in executions:
        if not int(row.get("submitted") or 0):
            continue
        status = str(row.get("order_status") or "filled").strip().lower()
        if status not in ("filled",):
            continue
        if str(row.get("symbol") or "").strip().upper() != sym:
            continue
        row_d = _parse_execution_timestamp(row.get("timestamp"))
        if row_d != d:
            continue
        side = str(row.get("side") or "").strip().lower()
        if side == "buy":
            buys += 1
        elif side == "sell":
            sells += 1
    return buys > sells


def projected_day_trades_after_sell(
    executions: list[dict[str, Any]],
    *,
    symbol: str,
    as_of: dt.datetime,
    lookback_days: int = 5,
) -> int:
    """Completed day trades in lookback plus one if this sell closes an open same-day leg."""
    base = count_day_trades(executions, as_of=as_of, lookback_days=lookback_days)
    if sell_adds_day_trade(executions, symbol=symbol, as_of=as_of):
        return base + 1
    return base


def evaluate_pdt_risk(
    day_trade_count: int,
    equity: float,
    *,
    pdt_threshold: int = 4,
    equity_floor: float = 25_000.0,
) -> PDTCheckResult:
    """Return allow / warn / block using projected day-trade count and equity."""
    count = max(0, int(day_trade_count))
    if float(equity) >= float(equity_floor):
        return PDTCheckResult(allowed=True, warning=None, day_trade_count=count)
    thr = max(1, int(pdt_threshold))
    if count >= thr:
        return PDTCheckResult(
            allowed=False,
            warning=None,
            day_trade_count=count,
        )
    if count >= thr - 1:
        return PDTCheckResult(
            allowed=True,
            warning=(
                f"Approaching PDT limit: {count} day trade(s) in the lookback window "
                f"(threshold {thr}) with equity below ${equity_floor:,.0f}."
            ),
            day_trade_count=count,
        )
    return PDTCheckResult(allowed=True, warning=None, day_trade_count=count)
