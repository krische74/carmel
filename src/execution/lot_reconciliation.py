"""Compare tax-lot open quantities to broker positions (Tier 46).

Execution-log reconciliation (:mod:`src.execution.reconciliation`) does not cover
the lot ledger. This module catches the class of failure where ``tax_lots_open``
drifts from the broker by an order of magnitude (e.g. after a paper-account
reset that left stale lots in SQLite).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LotReconciliationStatus = Literal["matched", "qty_mismatch"]


class LotReconciliationEntry(BaseModel):
    """Per-symbol lot vs broker quantity comparison."""

    symbol: str
    status: LotReconciliationStatus
    ledger_qty: float
    broker_qty: float


class LotReconciliationResult(BaseModel):
    """Summary of lot-ledger vs broker position reconciliation."""

    entries: list[LotReconciliationEntry]
    qty_mismatches: int
    equity_mismatch: bool
    ledger_equity: float = Field(description="Ledger market value + cash.")
    broker_equity: float


def _qty_mismatch(a: float, b: float, *, abs_tol: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) > abs_tol


def _equity_out_of_tolerance(
    ledger_equity: float,
    broker_equity: float,
    *,
    rel_tol: float = 0.03,
    abs_tol: float = 50.0,
) -> bool:
    """True when |ledger - broker| exceeds max(abs_tol, rel_tol * |broker|)."""
    diff = abs(float(ledger_equity) - float(broker_equity))
    scale = abs(float(broker_equity))
    return diff > max(float(abs_tol), float(rel_tol) * scale)


def reconcile_lot_ledger(
    *,
    ledger_qty_by_symbol: dict[str, float],
    broker_qty_by_symbol: dict[str, float],
    ledger_market_value: float,
    cash: float,
    broker_equity: float,
    rel_tol: float = 0.03,
    abs_tol: float = 50.0,
) -> LotReconciliationResult:
    """Compare open-lot aggregates to broker positions and total equity.

    Symbols present on either side are compared. Zero on both sides is omitted.
    ``ledger_equity`` is ``ledger_market_value + cash`` so it is comparable to
    broker equity (cash + positions).
    """
    symbols = sorted(
        {
            *(s.strip().upper() for s in ledger_qty_by_symbol),
            *(s.strip().upper() for s in broker_qty_by_symbol),
        },
    )
    entries: list[LotReconciliationEntry] = []
    mismatches = 0
    for sym in symbols:
        lq = float(ledger_qty_by_symbol.get(sym, 0.0))
        bq = float(broker_qty_by_symbol.get(sym, 0.0))
        if abs(lq) <= 1e-12 and abs(bq) <= 1e-12:
            continue
        if _qty_mismatch(lq, bq):
            entries.append(
                LotReconciliationEntry(
                    symbol=sym,
                    status="qty_mismatch",
                    ledger_qty=lq,
                    broker_qty=bq,
                ),
            )
            mismatches += 1
        else:
            entries.append(
                LotReconciliationEntry(
                    symbol=sym,
                    status="matched",
                    ledger_qty=lq,
                    broker_qty=bq,
                ),
            )

    ledger_equity = float(ledger_market_value) + float(cash)
    equity_bad = _equity_out_of_tolerance(
        ledger_equity,
        float(broker_equity),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    return LotReconciliationResult(
        entries=entries,
        qty_mismatches=mismatches,
        equity_mismatch=equity_bad,
        ledger_equity=ledger_equity,
        broker_equity=float(broker_equity),
    )
