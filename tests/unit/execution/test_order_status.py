"""SQLite order status columns and stale limit cancellation."""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import ExecutionRetryConfig, RiskConfig, SchedulerConfig, Settings
from src.data.storage.sqlite_store import SQLiteStore
from src.execution.order_manager import OrderManager, _bucket_order_status
from src.models import OrderExecutionResult
from src.portfolio.tax_lots import LotLedger
from src.risk.kill_switch import KillSwitch


def _settings() -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        risk=RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            daily_loss_limit_pct=0.03,
        ),
        scheduler=SchedulerConfig(
            execution_retry=ExecutionRetryConfig(
                max_attempts=2,
                backoff_base_seconds=0.1,
                use_jitter=False,
            ),
        ),
    )


def test_order_status_persisted_to_sqlite(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "os.db")
    cid = "c1"
    store.log_execution(
        cid,
        OrderExecutionResult(
            symbol="SPY",
            submitted=True,
            order_id="lim-9",
            side="buy",
            qty=1.0,
            fill_price=None,
            order_type="limit",
            limit_price=400.12,
            order_status="pending",
            filled_qty=None,
        ),
    )
    rows = store.get_executions(limit=5)
    assert len(rows) == 1
    assert rows[0]["order_type"] == "limit"
    assert float(rows[0]["limit_price"]) == pytest.approx(400.12)
    assert rows[0]["order_status"] == "pending"
    assert rows[0]["filled_qty"] is None


def test_cancel_stale_orders_cancels_old_pending(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "stale.db")
    old_ts = datetime.now(UTC) - timedelta(hours=2)
    store.log_execution(
        "c-old",
        OrderExecutionResult(
            symbol="SPY",
            submitted=True,
            order_id="pending-1",
            side="buy",
            qty=1.0,
            timestamp=old_ts,
            order_type="limit",
            limit_price=100.0,
            order_status="pending",
        ),
    )
    broker = MagicMock()
    broker.cancel_order.return_value = True
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
        sqlite_store=store,
    )
    cancelled = mgr.cancel_stale_orders(timeout_minutes=30)
    assert "pending-1" in cancelled
    broker.cancel_order.assert_called_once_with("pending-1")
    rows = store.get_executions(limit=5)
    assert rows[0]["order_status"] == "cancelled"


def test_cancel_stale_orders_refetches_status_when_cancel_returns_false(tmp_path: Path) -> None:
    """If cancel fails (e.g. fill in flight), sync SQLite from get_order_status."""
    store = SQLiteStore(tmp_path / "race.db")
    old_ts = datetime.now(UTC) - timedelta(hours=2)
    store.log_execution(
        "c-race",
        OrderExecutionResult(
            symbol="SPY",
            submitted=True,
            order_id="race-1",
            side="buy",
            qty=10.0,
            timestamp=old_ts,
            order_type="limit",
            limit_price=100.0,
            order_status="pending",
        ),
    )
    broker = MagicMock()
    broker.cancel_order.return_value = False
    broker.get_order_status.return_value = {
        "order_id": "race-1",
        "status": "filled",
        "filled_qty": 10.0,
        "filled_avg_price": 99.5,
    }
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
        sqlite_store=store,
    )
    assert mgr.cancel_stale_orders(timeout_minutes=30) == []
    broker.cancel_order.assert_called_once_with("race-1")
    broker.get_order_status.assert_called_once_with("race-1")
    rows = store.get_executions(limit=5)
    assert rows[0]["order_status"] == "filled"
    assert float(rows[0]["filled_qty"]) == pytest.approx(10.0)
    assert float(rows[0]["filled_avg_price"]) == pytest.approx(99.5)


def test_cancel_stale_orders_partial_fill_when_cancel_false(tmp_path: Path) -> None:
    """Partially filled then canceled: persist fill fields and terminal status."""
    store = SQLiteStore(tmp_path / "partial.db")
    old_ts = datetime.now(UTC) - timedelta(hours=2)
    store.log_execution(
        "c-partial",
        OrderExecutionResult(
            symbol="QQQ",
            submitted=True,
            order_id="part-1",
            side="buy",
            qty=10.0,
            timestamp=old_ts,
            order_type="limit",
            limit_price=200.0,
            order_status="pending",
        ),
    )
    broker = MagicMock()
    broker.cancel_order.return_value = False
    broker.get_order_status.return_value = {
        "status": "canceled",
        "filled_qty": 5.0,
        "filled_avg_price": 199.25,
    }
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
        sqlite_store=store,
    )
    mgr.cancel_stale_orders(timeout_minutes=30)
    rows = store.get_executions(limit=5)
    assert rows[0]["order_status"] == "cancelled"
    assert float(rows[0]["filled_qty"]) == pytest.approx(5.0)
    assert float(rows[0]["filled_avg_price"]) == pytest.approx(199.25)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("new", "pending"),
        ("accepted", "pending"),
        ("queued", "pending"),
        ("held", "pending"),
        ("pending_cancel", "pending"),
        ("pending_replace", "pending"),
        ("partially_filled", "pending"),
        ("filled", "filled"),
        ("done_for_day", "filled"),
        ("canceled", "cancelled"),
        ("cancelled", "cancelled"),
        ("expired", "cancelled"),
        ("rejected", "cancelled"),
    ],
)
def test_bucket_order_status_maps_known_alpaca_strings(raw: str, expected: str) -> None:
    assert _bucket_order_status(raw) == expected


def test_bucket_order_status_unknown_logs_and_returns_unknown(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert _bucket_order_status("not_a_real_alpaca_status_xyz") == "unknown"
    assert any("Unknown broker order status" in r.message for r in caplog.records)


def test_timeout_zero_fill_writes_through_to_lot_ledger_when_later_fill_observed(
    tmp_path: Path,
) -> None:
    """Buy wait times out at filled=0; a later stale sync that sees the fill must update lots.

    The 2026-09-08 cycle repaired trade_executions after ``Order filled while cancel in
    flight`` and left tax_lots_open at the timeout's zero.
    """
    db = tmp_path / "leak.db"
    store = SQLiteStore(db)
    ledger = LotLedger(db)
    old_ts = datetime.now(UTC) - timedelta(hours=2)
    store.log_execution(
        "c-timeout",
        OrderExecutionResult(
            symbol="BND",
            submitted=True,
            order_id="leak-buy-1",
            side="buy",
            qty=10.0,
            timestamp=old_ts,
            order_type="market",
            order_status="pending",
            filled_qty=0.0,
        ),
    )
    ledger.record_buy("BND", 0.0, 70.0, old_ts)
    assert sum(lot.qty for lot in ledger.get_open_lots(account_id="default")) == pytest.approx(0.0)

    broker = MagicMock()
    broker.cancel_order.return_value = False
    broker.get_order_status.return_value = {
        "order_id": "leak-buy-1",
        "status": "filled",
        "filled_qty": 0.0892,
        "filled_avg_price": 72.1,
    }
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
        sqlite_store=store,
        lot_ledger=ledger,
    )
    assert mgr.cancel_stale_orders(timeout_minutes=30) == []

    row = store.get_executions(limit=5)[0]
    assert float(row["qty"]) == pytest.approx(10.0)
    assert row["side"] == "buy"
    assert row["order_status"] == "filled"
    assert float(row["filled_qty"]) == pytest.approx(0.0892)
    open_qty = sum(lot.qty for lot in ledger.get_open_lots(symbol="BND", account_id="default"))
    assert open_qty == pytest.approx(0.0892)


def test_timeout_zero_sell_writes_through_to_lot_ledger_when_later_fill_observed(
    tmp_path: Path,
) -> None:
    """A sell that timed out at 0 must reduce lots once the later sync sees the fill."""
    db = tmp_path / "leak-sell.db"
    store = SQLiteStore(db)
    ledger = LotLedger(db)
    old_ts = datetime.now(UTC) - timedelta(hours=2)
    ledger.record_buy("BIL", 10.0, 91.0, old_ts - timedelta(days=1))
    store.log_execution(
        "c-timeout-sell",
        OrderExecutionResult(
            symbol="BIL",
            submitted=True,
            order_id="leak-sell-1",
            side="sell",
            qty=2.0,
            timestamp=old_ts,
            order_type="market",
            order_status="pending",
            filled_qty=0.0,
        ),
    )
    broker = MagicMock()
    broker.cancel_order.return_value = False
    broker.get_order_status.return_value = {
        "status": "filled",
        "filled_qty": 0.7058,
        "filled_avg_price": 91.4,
    }
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
        sqlite_store=store,
        lot_ledger=ledger,
    )
    mgr.cancel_stale_orders(timeout_minutes=30)
    open_qty = sum(lot.qty for lot in ledger.get_open_lots(symbol="BIL", account_id="default"))
    assert open_qty == pytest.approx(10.0 - 0.7058)
    assert float(store.get_executions(limit=5)[0]["qty"]) == pytest.approx(2.0)


def test_cancel_stale_orders_skips_filled(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ok.db")
    old_ts = datetime.now(UTC) - timedelta(hours=2)
    store.log_execution(
        "c-old",
        OrderExecutionResult(
            symbol="QQQ",
            submitted=True,
            order_id="done-1",
            side="buy",
            qty=2.0,
            timestamp=old_ts,
            order_type="market",
            order_status="filled",
            fill_price=50.0,
        ),
    )
    broker = MagicMock()
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
        sqlite_store=store,
    )
    assert mgr.cancel_stale_orders(timeout_minutes=30) == []
    broker.cancel_order.assert_not_called()
