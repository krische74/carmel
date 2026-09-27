"""Limit / stop order wiring (broker + order manager)."""

from datetime import UTC, datetime
from unittest.mock import ANY, MagicMock

import pytest

from src.config import ExecutionConfig, ExecutionRetryConfig, RiskConfig, SchedulerConfig, Settings
from src.execution.alpaca_adapter import AlpacaBrokerAdapter
from src.execution.order_manager import OrderManager, _limit_buy_price
from src.models import Signal
from src.risk.kill_switch import KillSwitch


def test_limit_order_price_calculation() -> None:
    assert _limit_buy_price(100.0, 10.0) == pytest.approx(100.1)
    # Sub-penny raw price is floored at $0.01
    assert _limit_buy_price(0.005, 0.0) == pytest.approx(0.01)


def test_execute_signals_uses_limit_order_when_configured() -> None:
    broker = MagicMock()
    broker.submit_limit_order.return_value = "lim-a"
    broker.get_order_status.return_value = {
        "order_id": "lim-a",
        "status": "filled",
        "filled_qty": 1.25,
        "filled_avg_price": 100.0,
    }
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        risk=RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            daily_loss_limit_pct=0.03,
        ),
        execution=ExecutionConfig(default_order_type="limit", limit_offset_bps=0.0),
    )
    mgr = OrderManager(broker=broker, settings=settings, kill_switch=KillSwitch(0.03))
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="Test.",
        timestamp=datetime.now(UTC),
    )
    results = mgr.execute_signals(
        [sig],
        last_prices={"SPY": 100.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
    )
    assert len(results) == 1
    assert results[0].submitted is True
    assert results[0].order_type == "limit"
    assert results[0].order_status == "filled"
    broker.submit_limit_order.assert_called_once()
    call_kw = broker.submit_limit_order.call_args
    assert call_kw[0][0] == "SPY"
    assert call_kw[1]["limit_price"] == pytest.approx(100.0)
    broker.submit_market_order.assert_not_called()


def test_execute_signals_places_stop_loss_after_fill() -> None:
    broker = MagicMock()
    broker.submit_limit_order.return_value = "lim-b"
    broker.get_order_status.return_value = {
        "order_id": "lim-b",
        "status": "filled",
        "filled_qty": 1.0,
        "filled_avg_price": 100.0,
    }
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        risk=RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            daily_loss_limit_pct=0.03,
        ),
        execution=ExecutionConfig(
            default_order_type="limit",
            limit_offset_bps=0.0,
            stop_loss_enabled=True,
            stop_loss_pct=5.0,
        ),
    )
    mgr = OrderManager(broker=broker, settings=settings, kill_switch=KillSwitch(0.03))
    sig = Signal(
        symbol="VOO",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="Test.",
        timestamp=datetime.now(UTC),
    )
    mgr.execute_signals(
        [sig],
        last_prices={"VOO": 100.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"VOO": 0.0},
        daily_pnl_pct=0.0,
    )
    broker.submit_stop_order.assert_called_once()
    args, kwargs = broker.submit_stop_order.call_args
    assert args[2] == "sell"
    assert kwargs["stop_price"] == pytest.approx(95.0)


def test_close_position_always_uses_market() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "m-sell"
    broker.get_order_fill_price.return_value = 50.0
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        execution=ExecutionConfig(default_order_type="limit"),
    )
    mgr = OrderManager(broker=broker, settings=settings, kill_switch=KillSwitch(0.03))
    r = mgr.close_position(
        "SPY",
        3.0,
        price=51.0,
        equity=10_000.0,
        daily_pnl_pct=0.0,
    )
    assert r.submitted is True
    broker.submit_market_order.assert_called_once_with(
        "SPY",
        3.0,
        "sell",
        client_order_id=ANY,
    )
    broker.submit_limit_order.assert_not_called()


def test_limit_order_with_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MagicMock()
    broker.submit_limit_order.side_effect = [OSError("transient"), "lim-ok"]
    broker.get_order_status.return_value = {
        "order_id": "lim-ok",
        "status": "filled",
        "filled_qty": 5.0,
        "filled_avg_price": 10.0,
    }
    monkeypatch.setattr("src.execution.order_manager.time.sleep", lambda _s: None)
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        risk=RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            daily_loss_limit_pct=0.03,
        ),
        execution=ExecutionConfig(default_order_type="limit", limit_offset_bps=0.0),
        scheduler=SchedulerConfig(
            execution_retry=ExecutionRetryConfig(
                max_attempts=2,
                backoff_base_seconds=0.1,
                use_jitter=False,
            ),
        ),
    )
    mgr = OrderManager(broker=broker, settings=settings, kill_switch=KillSwitch(0.03))
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.5,
        confidence=1.0,
        rationale="Test.",
        timestamp=datetime.now(UTC),
        strategy_name="DCAStrategy",
    )
    results = mgr.execute_signals(
        [sig],
        last_prices={"SPY": 10.0},
        equity=10_000.0,
        cash=10_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
        dca_budget=100.0,
    )
    assert broker.submit_limit_order.call_count == 2
    assert len(results) == 1
    assert results[0].submitted is True


def test_alpaca_submit_limit_order_uses_client() -> None:
    client = MagicMock()
    client.submit_order.return_value = MagicMock(id="alp-lim-1")
    adapter = AlpacaBrokerAdapter(client)
    oid = adapter.submit_limit_order("spy", 2.0, "buy", limit_price=401.25)
    assert oid == "alp-lim-1"
    client.submit_order.assert_called_once()
    req = client.submit_order.call_args[0][0]
    assert float(req.limit_price) == pytest.approx(401.25)
    assert str(req.client_order_id).startswith("cml-")


def test_alpaca_submit_stop_order_uses_client() -> None:
    client = MagicMock()
    client.submit_order.return_value = MagicMock(id="alp-st-1")
    adapter = AlpacaBrokerAdapter(client)
    oid = adapter.submit_stop_order("QQQ", 1.0, "sell", stop_price=90.5)
    assert oid == "alp-st-1"
    req = client.submit_order.call_args[0][0]
    assert float(req.stop_price) == pytest.approx(90.5)
    assert str(req.client_order_id).startswith("cml-")


def test_alpaca_cancel_order_calls_client() -> None:
    client = MagicMock()
    adapter = AlpacaBrokerAdapter(client)
    assert adapter.cancel_order("oid-x") is True
    client.cancel_order_by_id.assert_called_once_with("oid-x")


def test_alpaca_get_order_status_returns_dict() -> None:
    client = MagicMock()
    o = MagicMock()
    o.id = "o1"
    o.status = "filled"
    o.filled_qty = 3.0
    o.filled_avg_price = 101.5
    client.get_order_by_id.return_value = o
    adapter = AlpacaBrokerAdapter(client)
    d = adapter.get_order_status("o1")
    assert d["order_id"] == "o1"
    assert d["filled_qty"] == pytest.approx(3.0)
    assert d["filled_avg_price"] == pytest.approx(101.5)
