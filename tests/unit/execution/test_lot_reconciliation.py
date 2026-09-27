"""Lot ledger vs broker position reconciliation (Tier 46)."""

from __future__ import annotations

import pytest

from src.automation.alerts import Alert, AlertLevel
from src.automation.reconciliation_alerts import generate_lot_ledger_alerts
from src.execution.lot_reconciliation import (
    LotReconciliationEntry,
    LotReconciliationResult,
    reconcile_lot_ledger,
)


def test_lot_reconcile_matched_when_qtys_agree() -> None:
    r = reconcile_lot_ledger(
        ledger_qty_by_symbol={"VOO": 1.0, "BIL": 10.0},
        broker_qty_by_symbol={"VOO": 1.0, "BIL": 10.0},
        ledger_market_value=2_000.0,
        cash=200.0,
        broker_equity=2_200.0,
    )
    assert r.qty_mismatches == 0
    assert r.equity_mismatch is False
    assert all(e.status == "matched" for e in r.entries)


def test_lot_reconcile_qty_mismatch_per_symbol() -> None:
    r = reconcile_lot_ledger(
        ledger_qty_by_symbol={"QQQ": 2.41, "VOO": 0.82},
        broker_qty_by_symbol={"QQQ": 0.0, "VOO": 0.82},
        ledger_market_value=3_000.0,
        cash=200.0,
        broker_equity=3_200.0,
    )
    assert r.qty_mismatches == 1
    qqq = next(e for e in r.entries if e.symbol == "QQQ")
    assert qqq.status == "qty_mismatch"
    assert qqq.ledger_qty == pytest.approx(2.41)
    assert qqq.broker_qty == pytest.approx(0.0)


def test_lot_reconcile_equity_mismatch_when_order_of_magnitude_off() -> None:
    # Production failure: ledger MV roughly 10x broker equity.
    r = reconcile_lot_ledger(
        ledger_qty_by_symbol={"QQQ": 37.0},
        broker_qty_by_symbol={"QQQ": 0.0},
        ledger_market_value=37_400.0,
        cash=240.0,
        broker_equity=3_437.0,
    )
    assert r.equity_mismatch is True
    assert r.qty_mismatches >= 1


def test_lot_reconcile_small_equity_diff_within_tolerance() -> None:
    r = reconcile_lot_ledger(
        ledger_qty_by_symbol={"BIL": 10.0},
        broker_qty_by_symbol={"BIL": 10.0},
        ledger_market_value=910.0,
        cash=100.0,
        broker_equity=1_015.0,  # $5 drift
    )
    assert r.equity_mismatch is False


def test_lot_reconcile_equity_mismatch_at_three_percent() -> None:
    r = reconcile_lot_ledger(
        ledger_qty_by_symbol={"BIL": 10.0},
        broker_qty_by_symbol={"BIL": 10.0},
        ledger_market_value=3_000.0,
        cash=434.0,
        broker_equity=3_434.0,
    )
    assert r.equity_mismatch is False
    r_bad = reconcile_lot_ledger(
        ledger_qty_by_symbol={"BIL": 10.0},
        broker_qty_by_symbol={"BIL": 10.0},
        ledger_market_value=3_000.0,
        cash=434.0,
        broker_equity=3_434.0 - 120.0,
    )
    assert r_bad.equity_mismatch is True


def test_generate_lot_ledger_alerts_qty_warning_equity_critical() -> None:
    result = LotReconciliationResult(
        entries=[
            LotReconciliationEntry(
                symbol="QQQ",
                status="qty_mismatch",
                ledger_qty=2.41,
                broker_qty=0.0,
            ),
        ],
        qty_mismatches=1,
        equity_mismatch=True,
        ledger_equity=37_640.0,
        broker_equity=3_437.0,
    )
    alts = generate_lot_ledger_alerts(result)
    assert any(a.level == AlertLevel.WARNING and a.category == "lot_reconciliation" for a in alts)
    assert any(a.level == AlertLevel.CRITICAL and a.category == "lot_reconciliation" for a in alts)
    assert all(isinstance(a, Alert) for a in alts)


def test_generate_lot_ledger_alerts_empty_when_clean() -> None:
    result = LotReconciliationResult(
        entries=[
            LotReconciliationEntry(symbol="VOO", status="matched", ledger_qty=1.0, broker_qty=1.0),
        ],
        qty_mismatches=0,
        equity_mismatch=False,
        ledger_equity=1_000.0,
        broker_equity=1_000.0,
    )
    assert generate_lot_ledger_alerts(result) == []
