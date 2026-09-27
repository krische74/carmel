"""Reconciliation → alert mapping (Tier 25)."""

from __future__ import annotations

from src.automation.alerts import AlertLevel
from src.automation.reconciliation_alerts import generate_reconciliation_alerts
from src.execution.reconciliation import ReconciliationEntry, ReconciliationResult


def test_zero_discrepancies_no_alerts() -> None:
    r = ReconciliationResult(
        entries=[
            ReconciliationEntry(
                order_id="1",
                symbol="SPY",
                side="buy",
                status="matched",
            ),
        ],
        matched=1,
        discrepancies=0,
    )
    assert generate_reconciliation_alerts(r) == []


def test_warning_on_single_discrepancy() -> None:
    r = ReconciliationResult(
        entries=[
            ReconciliationEntry(
                order_id="1",
                symbol="SPY",
                side="buy",
                status="missing_from_broker",
            ),
        ],
        matched=0,
        discrepancies=1,
    )
    alts = generate_reconciliation_alerts(r, critical_threshold=5)
    assert len(alts) == 1
    assert alts[0].level == AlertLevel.WARNING


def test_critical_on_many_discrepancies() -> None:
    entries = [
        ReconciliationEntry(
            order_id=str(i),
            symbol="SPY",
            side="buy",
            status="qty_mismatch",
        )
        for i in range(5)
    ]
    r = ReconciliationResult(entries=entries, matched=0, discrepancies=5)
    alts = generate_reconciliation_alerts(r, critical_threshold=5)
    assert len(alts) == 1
    assert alts[0].level == AlertLevel.CRITICAL


def test_alert_message_groups_by_status() -> None:
    r = ReconciliationResult(
        entries=[
            ReconciliationEntry(
                order_id="a",
                symbol="SPY",
                side="buy",
                status="missing_from_broker",
            ),
            ReconciliationEntry(
                order_id="b",
                symbol="QQQ",
                side="buy",
                status="missing_from_broker",
            ),
            ReconciliationEntry(
                order_id="c",
                symbol="GLD",
                side="sell",
                status="qty_mismatch",
            ),
        ],
        matched=0,
        discrepancies=3,
    )
    alts = generate_reconciliation_alerts(r, critical_threshold=10)
    msg = alts[0].message
    assert "missing_from_broker=2" in msg
    assert "qty_mismatch=1" in msg
