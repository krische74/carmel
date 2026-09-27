"""Tests for OrderExecutionResult defaults (Tier 5)."""

from datetime import UTC, datetime, timedelta

from src.models import OrderExecutionResult


def test_order_execution_result_has_timestamp_by_default() -> None:
    before = datetime.now(UTC)
    r = OrderExecutionResult(symbol="SPY", submitted=True, order_id="x")
    after = datetime.now(UTC)
    assert before - timedelta(seconds=2) <= r.timestamp <= after + timedelta(seconds=2)
    assert r.timestamp.tzinfo is not None


def test_order_execution_result_side_defaults_to_buy() -> None:
    r = OrderExecutionResult(symbol="QQQ", submitted=False, reason="no")
    assert r.side == "buy"
