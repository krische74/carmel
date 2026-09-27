"""IRS wash-sale window detection against FIFO tax lots.

**Intra-account** — ``detect_wash_sales()`` pairs a loss sale only with same-symbol
replacements in the *same* hub ``account_id`` (open lots and other closed lots on that
account). Use this for Form 8949 / disallowed-loss style rows per account.

**Cross-account** — ``detect_cross_account_wash_sales()`` flags losses in account A when
the same symbol is repurchased in account B within ±``window_days`` (informational for
household review; does not adjust basis). ``LotLedger.detect_wash_sales()`` may merge
intra-account results across hub accounts when ``account_id`` is omitted; cross-account
patterns are *not* included in that merge — call the cross-account helper separately.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel, Field

from src.portfolio.tax_lots import ClosedLot, TaxLot  # noqa: TC001


class CrossAccountWashSale(BaseModel):
    """Loss sale in one account with same-symbol replacement in another (informational)."""

    selling_account: str
    buying_account: str
    symbol: str
    loss_amount: float = Field(..., ge=0.0, description="Absolute loss on the sale lot.")
    replacement_date: date = Field(..., description="Open date of earliest qualifying replacement lot.")
    sell_date: date = Field(..., description="Calendar date of the loss sale (UTC).")


class WashSale(BaseModel):
    """A detected wash sale: a loss that may be disallowed for tax purposes."""

    closed_lot_id: str = Field(
        ...,
        description="Stable id for the loss-producing closed lot (lot id + close time).",
    )
    symbol: str
    disallowed_loss: float = Field(
        ..., ge=0.0, description="Absolute loss amount flagged (see module docs)."
    )
    replacement_lot_id: str | None = Field(
        None,
        description="Open lot id or closed lot lot_id for the replacement purchase, if known.",
    )
    sell_date: date = Field(..., description="Calendar date of the loss sale (UTC).")
    buy_date: date = Field(..., description="Calendar date of the replacement purchase (UTC).")


def _to_utc_date(dt: datetime) -> date:
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).date()
    return dt.date()


def _closed_lot_identity(cl: ClosedLot) -> tuple[str, datetime]:
    """Value identity for a closed-lot row (not object identity)."""
    return (cl.lot_id, cl.closed_at)


def detect_wash_sales(
    closed_lots: list[ClosedLot],
    open_lots: list[TaxLot],
    *,
    window_days: int = 30,
) -> list[WashSale]:
    """Detect wash-sale patterns using a ±``window_days`` calendar-day window.

    For each closed lot with negative realized P&L, looks for the same symbol in
    open lots or other closed lots whose *purchase* (``opened_at``) falls within
    the window before or after the loss sale date.

    **Simplification:** ``disallowed_loss`` is always ``abs(realized_pnl)`` for the
    loss lot, not prorated by replacement quantity. Partial replacements (e.g. sell
    100 / buy 50) would warrant a smaller disallowed amount in a full tax engine;
    this detector is informational only.
    """
    losses = [cl for cl in closed_lots if cl.realized_pnl < 0.0]
    out: list[WashSale] = []
    win = max(0, int(window_days))

    for loss in losses:
        sell_d = _to_utc_date(loss.closed_at)
        window_start = sell_d - timedelta(days=win)
        window_end = sell_d + timedelta(days=win)
        closed_lot_id = f"{loss.lot_id}|{loss.closed_at.isoformat()}"
        loss_aid = str(getattr(loss, "account_id", None) or "default").strip() or "default"

        candidates: list[tuple[datetime, str, date]] = []
        for ol in open_lots:
            if ol.symbol != loss.symbol:
                continue
            ol_aid = str(getattr(ol, "account_id", None) or "default").strip() or "default"
            if ol_aid != loss_aid:
                continue
            od = _to_utc_date(ol.opened_at)
            if window_start <= od <= window_end:
                candidates.append((ol.opened_at, ol.id, od))

        for other in closed_lots:
            if _closed_lot_identity(other) == _closed_lot_identity(loss):
                continue
            if other.symbol != loss.symbol:
                continue
            o_aid = str(getattr(other, "account_id", None) or "default").strip() or "default"
            if o_aid != loss_aid:
                continue
            od = _to_utc_date(other.opened_at)
            if window_start <= od <= window_end:
                candidates.append((other.opened_at, other.lot_id, od))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        _, repl_id, buy_d = candidates[0]
        out.append(
            WashSale(
                closed_lot_id=closed_lot_id,
                symbol=loss.symbol,
                disallowed_loss=abs(float(loss.realized_pnl)),
                replacement_lot_id=repl_id,
                sell_date=sell_d,
                buy_date=buy_d,
            ),
        )

    out.sort(key=lambda w: w.sell_date)
    return out


def partition_lots_by_account(
    closed_lots: list[ClosedLot],
    open_lots: list[TaxLot],
) -> tuple[dict[str, list[ClosedLot]], dict[str, list[TaxLot]]]:
    """Group lots by ``account_id`` (missing id → default string ``default``)."""
    by_c: dict[str, list[ClosedLot]] = {}
    for cl in closed_lots:
        aid = str(getattr(cl, "account_id", None) or "default").strip() or "default"
        by_c.setdefault(aid, []).append(cl)
    by_o: dict[str, list[TaxLot]] = {}
    for ol in open_lots:
        aid = str(getattr(ol, "account_id", None) or "default").strip() or "default"
        by_o.setdefault(aid, []).append(ol)
    return by_c, by_o


def detect_cross_account_wash_sales(
    all_accounts_closed: dict[str, list[ClosedLot]],
    all_accounts_open: dict[str, list[TaxLot]],
    *,
    window_days: int = 30,
) -> list[CrossAccountWashSale]:
    """Flag losses in account A with same-symbol purchases in B within ±``window_days``.

    Informational only — for household review; does not adjust basis.
    """
    account_keys = {*all_accounts_closed.keys(), *all_accounts_open.keys()}
    if len(account_keys) < 2:
        return []

    out: list[CrossAccountWashSale] = []
    win = max(0, int(window_days))

    for selling_account, closed_list in all_accounts_closed.items():
        for loss in closed_list:
            if float(loss.realized_pnl) >= 0.0:
                continue
            sell_d = _to_utc_date(loss.closed_at)
            win_start = sell_d - timedelta(days=win)
            win_end = sell_d + timedelta(days=win)
            candidates: list[tuple[str, datetime, date]] = []

            for buying_account, olist in all_accounts_open.items():
                if buying_account == selling_account:
                    continue
                for ol in olist:
                    if ol.symbol != loss.symbol:
                        continue
                    od = _to_utc_date(ol.opened_at)
                    if win_start <= od <= win_end:
                        candidates.append((buying_account, ol.opened_at, od))

            for buying_account, clist in all_accounts_closed.items():
                if buying_account == selling_account:
                    continue
                for other in clist:
                    if other.symbol != loss.symbol:
                        continue
                    od = _to_utc_date(other.opened_at)
                    if win_start <= od <= win_end:
                        candidates.append((buying_account, other.opened_at, od))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[1])
            buy_acct, _, repl_d = candidates[0]
            out.append(
                CrossAccountWashSale(
                    selling_account=selling_account,
                    buying_account=buy_acct,
                    symbol=loss.symbol,
                    loss_amount=abs(float(loss.realized_pnl)),
                    replacement_date=repl_d,
                    sell_date=sell_d,
                ),
            )

    out.sort(key=lambda x: (x.sell_date, x.selling_account, x.buying_account, x.symbol))
    return out
