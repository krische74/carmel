"""Turn reconciliation results into operator alerts."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.automation.alerts import Alert, AlertLevel

if TYPE_CHECKING:
    from src.execution.lot_reconciliation import LotReconciliationResult
    from src.execution.reconciliation import ReconciliationResult


def generate_reconciliation_alerts(
    result: ReconciliationResult,
    *,
    warning_threshold: int = 1,
    critical_threshold: int = 5,
) -> list[Alert]:
    """Build zero or one alert summarizing broker vs log discrepancies.

    When ``result.discrepancies == 0``, returns an empty list. Otherwise groups
    non-matched rows by ``status`` and sets level to CRITICAL when the count
    is at or above ``critical_threshold``, else WARNING (if above zero).
    """
    if result.discrepancies == 0:
        return []

    bad = [e for e in result.entries if e.status != "matched"]
    counts = Counter(e.status for e in bad)
    parts = [f"{status}={counts[status]}" for status in sorted(counts)]
    breakdown = ", ".join(parts)
    n = result.discrepancies
    crit_at = int(critical_threshold)
    level = AlertLevel.CRITICAL if n >= crit_at else AlertLevel.WARNING

    _ = warning_threshold  # wired from settings for forward-compatible tuning

    return [
        Alert(
            timestamp=datetime.now(UTC),
            level=level,
            category="reconciliation",
            message=f"Execution reconciliation: {n} discrepancy(ies). By status: {breakdown}.",
        ),
    ]


def generate_lot_ledger_alerts(result: LotReconciliationResult) -> list[Alert]:
    """Alerts for lot-ledger vs broker drift (Tier 46).

    Per-symbol quantity mismatches → WARNING. Total equity mismatch (ledger MV +
    cash vs broker equity beyond tolerance) → CRITICAL. Both may fire together.
    """
    now = datetime.now(UTC)
    out: list[Alert] = []
    if result.qty_mismatches > 0:
        bad = [e for e in result.entries if e.status == "qty_mismatch"]
        parts = [
            f"{e.symbol}: ledger={e.ledger_qty:.4f} broker={e.broker_qty:.4f}" for e in bad
        ]
        detail = "; ".join(parts)
        out.append(
            Alert(
                timestamp=now,
                level=AlertLevel.WARNING,
                category="lot_reconciliation",
                message=(
                    f"Lot ledger qty mismatch ({result.qty_mismatches} symbol(s)): {detail}."
                ),
            ),
        )
    if result.equity_mismatch:
        out.append(
            Alert(
                timestamp=now,
                level=AlertLevel.CRITICAL,
                category="lot_reconciliation",
                message=(
                    f"Lot ledger equity mismatch: ledger_equity=${result.ledger_equity:,.2f} "
                    f"vs broker_equity=${result.broker_equity:,.2f}."
                ),
            ),
        )
    return out

