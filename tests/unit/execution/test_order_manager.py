"""Tests for OrderManager (broker mocked)."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest

from src.config import ExecutionRetryConfig, RiskConfig, SchedulerConfig, Settings
from src.data.storage.sqlite_store import SQLiteStore
from src.execution.order_manager import OrderManager
from src.models import OrderExecutionResult, Signal
from src.risk.kill_switch import KillSwitch


def _settings(risk: RiskConfig | None = None) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        risk=risk
        or RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            daily_loss_limit_pct=0.03,
        ),
    )


def _retry_settings(risk: RiskConfig | None = None) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        risk=risk
        or RiskConfig(
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


def test_order_manager_submits_when_checks_pass() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-1"
    broker.get_order_fill_price.return_value = 400.0
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
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
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
    )
    assert len(results) == 1
    assert results[0].submitted is True
    assert results[0].order_id == "ord-1"
    assert results[0].qty == pytest.approx(500.0 / 400.0)
    assert results[0].fill_price == pytest.approx(400.0)
    broker.submit_market_order.assert_called_once()
    broker.get_order_fill_price.assert_called_once_with("ord-1")


def test_order_manager_skips_flat_signal() -> None:
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    sig = Signal(
        symbol="SPY",
        direction="flat",
        weight=0.0,
        confidence=1.0,
        rationale="No trade.",
        timestamp=datetime.now(UTC),
    )
    results = mgr.execute_signals(
        [sig],
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={},
        daily_pnl_pct=0.0,
    )
    assert results == []
    broker.submit_market_order.assert_not_called()


def test_order_manager_dca_budget_uses_weight_times_budget() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-dca"
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
    sig = Signal(
        symbol="VOO",
        direction="long",
        weight=1.0 / 3.0,
        confidence=1.0,
        rationale="DCA slice.",
        timestamp=datetime.now(UTC),
        strategy_name="DCAStrategy",
    )
    mgr.execute_signals(
        [sig],
        last_prices={"VOO": 100.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"VOO": 0.0},
        daily_pnl_pct=0.0,
        dca_budget=300.0,
    )
    # 300 * (1/3) = $100 notional → 1 share at $100
    broker.submit_market_order.assert_called_once_with(
        "VOO",
        1.0,
        "buy",
        client_order_id=ANY,
    )


def test_order_manager_missing_price_records_failure() -> None:
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
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
        last_prices={},
        equity=10_000.0,
        cash=5_000.0,
        positions={},
        daily_pnl_pct=0.0,
    )
    assert len(results) == 1
    assert results[0].submitted is False
    assert "price" in (results[0].reason or "").lower()
    broker.submit_market_order.assert_not_called()


def test_order_manager_cash_direction_submits_buy() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-shv"
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
    sig = Signal(
        symbol="SHV",
        direction="cash",
        weight=1.0,
        confidence=0.85,
        rationale="Defensive cash ETF.",
        timestamp=datetime.now(UTC),
        strategy_name="MomentumRotationStrategy",
    )
    results = mgr.execute_signals(
        [sig],
        last_prices={"SHV": 50.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SHV": 0.0},
        daily_pnl_pct=0.0,
    )
    assert len(results) == 1
    assert results[0].submitted is True
    broker.submit_market_order.assert_called_once_with(
        "SHV",
        50.0,
        "buy",
        client_order_id=ANY,
    )


def test_order_retry_reuses_same_client_order_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retries must reuse one idempotency key so Alpaca does not create duplicate orders."""
    broker = MagicMock()
    ids: list[str | None] = []

    def _side_effect(
        *_a: object,
        client_order_id: str | None = None,
        **_k: object,
    ) -> str:
        ids.append(client_order_id)
        if len(ids) == 1:
            raise OSError("transient")
        return "ord-ok"

    broker.submit_market_order.side_effect = _side_effect
    broker.get_order_fill_price.return_value = 400.0
    monkeypatch.setattr("src.execution.order_manager.time.sleep", lambda _s: None)
    mgr = OrderManager(
        broker=broker,
        settings=_retry_settings(),
        kill_switch=KillSwitch(0.03),
    )
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="Test.",
        timestamp=datetime.now(UTC),
    )
    mgr.execute_signals(
        [sig],
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
    )
    assert broker.submit_market_order.call_count == 2
    assert ids[0] is not None and ids[0] == ids[1]


def test_order_retry_succeeds_on_second_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MagicMock()
    broker.submit_market_order.side_effect = [OSError("transient"), "ord-ok"]
    broker.get_order_fill_price.return_value = 400.0
    monkeypatch.setattr("src.execution.order_manager.time.sleep", lambda _s: None)
    mgr = OrderManager(
        broker=broker,
        settings=_retry_settings(),
        kill_switch=KillSwitch(0.03),
    )
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
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
    )
    assert broker.submit_market_order.call_count == 2
    assert len(results) == 1
    assert results[0].submitted is True


def test_order_retry_exhausted_oserror_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = MagicMock()
    broker.submit_market_order.side_effect = OSError("always fails")
    monkeypatch.setattr("src.execution.order_manager.time.sleep", lambda _s: None)
    mgr = OrderManager(
        broker=broker,
        settings=_retry_settings(),
        kill_switch=KillSwitch(0.03),
    )
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
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
    )
    assert broker.submit_market_order.call_count == 2
    assert len(results) == 1
    assert results[0].submitted is False
    assert "Broker error" in (results[0].reason or "")


def test_close_position_retries_on_transient_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = MagicMock()
    broker.submit_market_order.side_effect = [OSError("x"), "sell-ok"]
    broker.get_order_fill_price.return_value = 99.0
    monkeypatch.setattr("src.execution.order_manager.time.sleep", lambda _s: None)
    mgr = OrderManager(
        broker=broker,
        settings=_retry_settings(),
        kill_switch=KillSwitch(0.03),
    )
    res = mgr.close_position(
        "SPY",
        5.0,
        price=100.0,
        equity=10_000.0,
        daily_pnl_pct=0.0,
    )
    assert broker.submit_market_order.call_count == 2
    assert res.submitted is True


def test_order_manager_broker_exception_returns_failure() -> None:
    broker = MagicMock()
    broker.submit_market_order.side_effect = RuntimeError("timeout")
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
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
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
    )
    assert len(results) == 1
    assert results[0].submitted is False
    assert "broker" in (results[0].reason or "").lower()
    assert "timeout" in (results[0].reason or "").lower()


def test_order_manager_second_signal_uses_decremented_cash() -> None:
    """After a fill, remaining cash must constrain the next pre-trade check."""
    broker = MagicMock()
    broker.submit_market_order.return_value = "ok"
    mgr = OrderManager(
        broker=broker,
        settings=_settings(
            RiskConfig(
                max_position_pct=0.99,
                min_cash_reserve_pct=0.05,
                daily_loss_limit_pct=0.03,
            )
        ),
        kill_switch=KillSwitch(0.03),
    )
    sigs = [
        Signal(
            symbol="A",
            direction="long",
            weight=0.5,
            confidence=1.0,
            rationale="First.",
            timestamp=datetime.now(UTC),
            strategy_name="DCAStrategy",
        ),
        Signal(
            symbol="B",
            direction="long",
            weight=0.5,
            confidence=1.0,
            rationale="Second.",
            timestamp=datetime.now(UTC),
            strategy_name="DCAStrategy",
        ),
    ]
    # Each leg: 0.5 * dca_budget=600 → $300. First: $1000→$700 (≥$500 reserve). Second: $700→$400 (<$500).
    results = mgr.execute_signals(
        sigs,
        last_prices={"A": 10.0, "B": 10.0},
        equity=10_000.0,
        cash=1_000.0,
        positions={"A": 0.0, "B": 0.0},
        daily_pnl_pct=0.0,
        dca_budget=600.0,
    )
    assert len(results) == 2
    assert results[0].submitted is True
    assert results[1].submitted is False
    assert results[1].reason is not None


def test_order_manager_skips_when_kill_switch() -> None:
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
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
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=-0.05,
    )
    assert len(results) == 1
    assert results[0].submitted is False
    assert "kill" in (results[0].reason or "").lower()
    broker.submit_market_order.assert_not_called()


def test_close_position_submits_sell_order() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "sell-1"
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
    r = mgr.close_position(
        "SPY",
        5.0,
        price=400.0,
        equity=10_000.0,
        daily_pnl_pct=0.0,
    )
    assert r.submitted is True
    assert r.side == "sell"
    assert r.order_id == "sell-1"
    assert r.qty == pytest.approx(5.0)
    broker.submit_market_order.assert_called_once_with(
        "SPY",
        5.0,
        "sell",
        client_order_id=ANY,
    )


def test_close_position_blocked_by_kill_switch() -> None:
    broker = MagicMock()
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
    r = mgr.close_position(
        "SPY",
        5.0,
        price=400.0,
        equity=10_000.0,
        daily_pnl_pct=-0.05,
    )
    assert r.submitted is False
    assert r.side == "sell"
    assert "kill" in (r.reason or "").lower()
    broker.submit_market_order.assert_not_called()


def test_close_position_handles_broker_exception() -> None:
    broker = MagicMock()
    broker.submit_market_order.side_effect = RuntimeError("down")
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
    r = mgr.close_position(
        "QQQ",
        2.0,
        price=300.0,
        equity=10_000.0,
        daily_pnl_pct=0.0,
    )
    assert r.submitted is False
    assert r.side == "sell"
    assert "broker" in (r.reason or "").lower()


def test_order_execution_result_qty_defaults_to_none() -> None:
    r = OrderExecutionResult(symbol="X", submitted=False, reason="x")
    assert r.qty is None


def test_execute_signals_updates_positions_within_batch() -> None:
    """Second buy for same symbol sees updated qty for concentration check.

    Uses DCA signals: post-Tier 43 the fixed-weight path is delta-sized (a second
    identical momentum signal deltas to 0), whereas DCA contributions still stack
    full notional and remain gated by the position cap.
    """
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord"
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
    ts = datetime.now(UTC)
    sigs = [
        Signal(
            symbol="SPY",
            direction="long",
            weight=0.2,
            confidence=1.0,
            rationale="First slice.",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
        Signal(
            symbol="SPY",
            direction="long",
            weight=0.2,
            confidence=1.0,
            rationale="Second slice.",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
    ]
    # Each leg: 0.2 * dca_budget=10_000 → $2_000 (20 sh @ $100). Two legs → 40 sh =
    # 40% of equity > 25% cap, so the second is rejected on concentration.
    results = mgr.execute_signals(
        sigs,
        last_prices={"SPY": 100.0},
        equity=10_000.0,
        cash=10_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
        dca_budget=10_000.0,
    )
    assert len(results) == 2
    assert results[0].submitted is True
    assert results[0].qty == pytest.approx(20.0)
    assert results[1].submitted is False
    assert results[1].reason is not None
    assert "maximum position" in (results[1].reason or "").lower()
    assert broker.submit_market_order.call_count == 1


def test_execute_signals_does_not_mutate_caller_positions() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord"
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    pos = {"SPY": 0.0}
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="Buy.",
        timestamp=datetime.now(UTC),
    )
    mgr.execute_signals(
        [sig],
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions=pos,
        daily_pnl_pct=0.0,
    )
    assert pos["SPY"] == 0.0


def test_close_position_rejects_zero_qty() -> None:
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    r = mgr.close_position(
        "SPY",
        0.0,
        price=400.0,
        equity=10_000.0,
        daily_pnl_pct=0.0,
    )
    assert r.submitted is False
    assert "Invalid" in (r.reason or "")
    broker.submit_market_order.assert_not_called()


def test_close_position_rejects_zero_price() -> None:
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    r = mgr.close_position(
        "SPY",
        5.0,
        price=0.0,
        equity=10_000.0,
        daily_pnl_pct=0.0,
    )
    assert r.submitted is False
    assert "Invalid" in (r.reason or "")
    broker.submit_market_order.assert_not_called()


def test_order_manager_fill_price_none_on_lookup_failure() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-x"
    broker.get_order_fill_price.side_effect = RuntimeError("timeout")
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
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
        last_prices={"SPY": 400.0},
        equity=10_000.0,
        cash=5_000.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
    )
    assert len(results) == 1
    assert results[0].submitted is True
    assert results[0].fill_price is None


def test_order_manager_uses_atr_sizing_when_configured() -> None:
    """ATR mode yields different qty than fixed-weight for the same signal weight."""
    risk = RiskConfig(
        max_position_pct=0.25,
        min_cash_reserve_pct=0.05,
        daily_loss_limit_pct=0.03,
        sizing_method="atr_risk_parity",
        atr_risk_pct=0.01,
        atr_period=14,
    )
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-atr"
    broker.get_order_fill_price.return_value = 100.0
    mgr = OrderManager(broker=broker, settings=_settings(risk=risk), kill_switch=KillSwitch(0.03))
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="Test.",
        timestamp=datetime.now(UTC),
        strategy_name="MomentumRotationStrategy",
    )
    results = mgr.execute_signals(
        [sig],
        last_prices={"SPY": 100.0},
        equity=10_000.0,
        cash=10_000.0,
        positions={},
        daily_pnl_pct=0.0,
        atr_values={"SPY": 2.0},
    )
    assert len(results) == 1
    assert results[0].submitted is True
    # Fixed-weight qty would be 5.0; ATR caps notional at 2500 → qty 25.0
    assert results[0].qty == pytest.approx(25.0)


def test_order_manager_dca_ignores_atr_values() -> None:
    risk = RiskConfig(
        max_position_pct=0.25,
        min_cash_reserve_pct=0.05,
        daily_loss_limit_pct=0.03,
        sizing_method="atr_risk_parity",
        atr_risk_pct=0.01,
        atr_period=14,
    )
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-dca"
    broker.get_order_fill_price.return_value = 100.0
    mgr = OrderManager(broker=broker, settings=_settings(risk=risk), kill_switch=KillSwitch(0.03))
    sig = Signal(
        symbol="VOO",
        direction="long",
        weight=1.0 / 3.0,
        confidence=1.0,
        rationale="DCA.",
        timestamp=datetime.now(UTC),
        strategy_name="DCAStrategy",
    )
    results = mgr.execute_signals(
        [sig],
        last_prices={"VOO": 100.0},
        equity=10_000.0,
        cash=10_000.0,
        positions={},
        daily_pnl_pct=0.0,
        dca_budget=300.0,
        atr_values={"VOO": 500.0},
    )
    assert results[0].qty == pytest.approx(100.0 / 100.0)


def test_execute_signals_blocks_third_same_symbol_when_cumulative_notional_exceeds_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = MagicMock()
    broker.submit_market_order.side_effect = ["ord-1", "ord-2"]
    broker.get_order_fill_price.return_value = 100.0
    broker.list_recent_orders.return_value = []
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    ts = datetime.now(UTC)
    # DCA legs stack full notional (delta sizing does not apply), so cumulative
    # exposure still trips the concentration cap on the third slice.
    signals = [
        Signal(
            symbol="QQQ",
            direction="long",
            weight=0.10,
            confidence=1.0,
            rationale="slice 1",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
        Signal(
            symbol="QQQ",
            direction="long",
            weight=0.10,
            confidence=1.0,
            rationale="slice 2",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
        Signal(
            symbol="QQQ",
            direction="long",
            weight=0.10,
            confidence=1.0,
            rationale="slice 3",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
    ]

    with caplog.at_level("WARNING"):
        results = mgr.execute_signals(
            signals,
            last_prices={"QQQ": 100.0},
            equity=100_000.0,
            cash=100_000.0,
            positions={"QQQ": 0.0},
            daily_pnl_pct=0.0,
            dca_budget=100_000.0,
        )

    assert [r.submitted for r in results] == [True, True, False]
    assert results[2].reason is not None
    assert "maximum position" in results[2].reason.lower()
    assert "pre_trade_reject" in caplog.text
    assert broker.submit_market_order.call_count == 2


def test_execute_signals_does_not_double_count_filled_in_cycle_submissions() -> None:
    """Filled in-cycle buys update ``positions``; tally excludes them so cap uses post-fill qty."""
    broker = MagicMock()
    broker.submit_market_order.side_effect = ["ord-1", "ord-2", "ord-3", "ord-4"]
    broker.get_order_fill_price.return_value = 100.0
    broker.list_recent_orders.return_value = []
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    ts = datetime.now(UTC)
    # DCA legs stack full notional: 5k + 5k + 5k = 15k (150 sh); fourth 15k reaches
    # 30k > 25% of 100k. (Fixed-weight would delta-size; DCA does not.)
    signals = [
        Signal(
            symbol="QQQ",
            direction="long",
            weight=0.05,
            confidence=1.0,
            rationale="5k 1",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
        Signal(
            symbol="QQQ",
            direction="long",
            weight=0.05,
            confidence=1.0,
            rationale="5k 2",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
        Signal(
            symbol="QQQ",
            direction="long",
            weight=0.05,
            confidence=1.0,
            rationale="5k 3",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
        Signal(
            symbol="QQQ",
            direction="long",
            weight=0.15,
            confidence=1.0,
            rationale="15k over cap",
            timestamp=ts,
            strategy_name="DCAStrategy",
        ),
    ]
    results = mgr.execute_signals(
        signals,
        last_prices={"QQQ": 100.0},
        equity=100_000.0,
        cash=100_000.0,
        positions={"QQQ": 0.0},
        daily_pnl_pct=0.0,
        dca_budget=100_000.0,
    )
    assert [r.submitted for r in results] == [True, True, True, False]
    assert results[3].reason is not None
    assert "maximum position" in results[3].reason.lower()
    assert broker.submit_market_order.call_count == 3


def test_execute_signals_blocks_new_order_when_pending_order_already_near_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-should-not-submit"
    broker.get_order_fill_price.return_value = 100.0
    broker.list_recent_orders.return_value = [
        {
            "symbol": "QQQ",
            "side": "buy",
            "status": "pending_new",
            "qty": 200.0,
            "filled_qty": 0.0,
            "filled_avg_price": None,
            "limit_price": 100.0,
            "notional": None,
        },
    ]
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    # DCA leg stacks full notional on top of the pending buy exposure, tripping the
    # cap. (A fixed-weight leg would delta against pending exposure and skip quietly.)
    signal = Signal(
        symbol="QQQ",
        direction="long",
        weight=0.10,
        confidence=1.0,
        rationale="new buy",
        timestamp=datetime.now(UTC),
        strategy_name="DCAStrategy",
    )

    with caplog.at_level("WARNING"):
        results = mgr.execute_signals(
            [signal],
            last_prices={"QQQ": 100.0},
            equity=100_000.0,
            cash=100_000.0,
            positions={"QQQ": 0.0},
            daily_pnl_pct=0.0,
            dca_budget=100_000.0,
        )

    assert len(results) == 1
    assert results[0].submitted is False
    assert results[0].reason is not None
    assert "maximum position" in results[0].reason.lower()
    assert "pre_trade_reject" in caplog.text
    broker.submit_market_order.assert_not_called()


def _momentum_signal(symbol: str = "SPY", weight: float = 1.0) -> Signal:
    return Signal(
        symbol=symbol,
        direction="long",
        weight=weight,
        confidence=1.0,
        rationale="Momentum target.",
        timestamp=datetime.now(UTC),
        strategy_name="MomentumRotationStrategy",
    )


def test_execute_signals_submits_delta_notional_not_full_target() -> None:
    """A position below target is topped up by the delta, not a fresh full target."""
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-delta"
    broker.get_order_fill_price.return_value = 1.0
    broker.list_recent_orders.return_value = []
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    # weight 1.0, equity 10k, cap 25% → target 2500. Held 2000 @ $1 → delta 500.
    results = mgr.execute_signals(
        [_momentum_signal()],
        last_prices={"SPY": 1.0},
        equity=10_000.0,
        cash=10_000.0,
        positions={"SPY": 2_000.0},
        daily_pnl_pct=0.0,
    )
    assert len(results) == 1
    assert results[0].submitted is True
    assert results[0].qty == pytest.approx(500.0)  # delta 500 @ $1, not full 2500
    broker.submit_market_order.assert_called_once_with("SPY", 500.0, "buy", client_order_id=ANY)


def test_execute_signals_skips_order_when_already_at_target_no_warning_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Position at target → delta 0 → no broker order and no pre_trade_reject WARNING."""
    broker = MagicMock()
    broker.list_recent_orders.return_value = []
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    # Held 2500 @ $1 == target 2500 → delta 0.
    with caplog.at_level("WARNING"):
        results = mgr.execute_signals(
            [_momentum_signal()],
            last_prices={"SPY": 1.0},
            equity=10_000.0,
            cash=10_000.0,
            positions={"SPY": 2_500.0},
            daily_pnl_pct=0.0,
        )
    assert len(results) == 1
    assert results[0].submitted is False
    assert "pre_trade_reject" not in caplog.text
    broker.submit_market_order.assert_not_called()


def test_execute_signals_skips_order_below_min_notional_floor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Delta above 0 but below the $5 floor → skipped at DEBUG, submitted=False."""
    broker = MagicMock()
    broker.list_recent_orders.return_value = []
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    # Held 2497 @ $1 → delta 3 (< $5 floor).
    with caplog.at_level(logging.DEBUG, logger="src.execution.order_manager"):
        results = mgr.execute_signals(
            [_momentum_signal()],
            last_prices={"SPY": 1.0},
            equity=10_000.0,
            cash=10_000.0,
            positions={"SPY": 2_497.0},
            daily_pnl_pct=0.0,
        )
    assert len(results) == 1
    assert results[0].submitted is False
    assert results[0].reason is not None
    assert "minimum order size" in results[0].reason.lower()
    assert "order_skipped_below_floor" in caplog.text
    assert "pre_trade_reject" not in caplog.text
    broker.submit_market_order.assert_not_called()


def test_execute_signals_submits_order_at_or_above_min_notional_floor() -> None:
    """Delta exactly at the $5 floor is submitted (boundary is >=, not >)."""
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-floor"
    broker.get_order_fill_price.return_value = 1.0
    broker.list_recent_orders.return_value = []
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    # Held 2495 @ $1 → delta exactly 5.0.
    results = mgr.execute_signals(
        [_momentum_signal()],
        last_prices={"SPY": 1.0},
        equity=10_000.0,
        cash=10_000.0,
        positions={"SPY": 2_495.0},
        daily_pnl_pct=0.0,
    )
    assert len(results) == 1
    assert results[0].submitted is True
    assert results[0].qty == pytest.approx(5.0)
    broker.submit_market_order.assert_called_once_with("SPY", 5.0, "buy", client_order_id=ANY)


def test_sweep_buy_submits_and_bypasses_max_position() -> None:
    """Sweep buy skips the position cap: an over-cap BIL notional still submits."""
    broker = MagicMock()
    broker.submit_market_order.return_value = "sweep-b"
    broker.get_order_fill_price.return_value = 100.0
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    # equity 10k, cap 25% = 2500; notional 5000 (50%) far exceeds cap but must still submit.
    res = mgr.sweep_buy(
        "BIL",
        5_000.0,
        price=100.0,
        equity=10_000.0,
        cash=10_000.0,
        daily_pnl_pct=0.0,
    )
    assert res.submitted is True
    assert res.side == "buy"
    assert res.strategy_name == "CashSweep"
    assert res.qty == pytest.approx(50.0)
    broker.submit_market_order.assert_called_once_with("BIL", 50.0, "buy", client_order_id=ANY)


def test_sweep_buy_blocked_by_kill_switch() -> None:
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    res = mgr.sweep_buy(
        "BIL",
        1_000.0,
        price=100.0,
        equity=10_000.0,
        cash=10_000.0,
        daily_pnl_pct=-0.05,
    )
    assert res.submitted is False
    assert "kill" in (res.reason or "").lower()
    assert res.strategy_name == "CashSweep"
    broker.submit_market_order.assert_not_called()


def test_sweep_buy_enforces_cash_reserve() -> None:
    """Sweep buy respects the min cash reserve (unlike the position cap)."""
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    # equity 10k, reserve 5% = 500; cash 1000; notional 800 → post-trade 200 < 500 → block.
    res = mgr.sweep_buy(
        "BIL",
        800.0,
        price=100.0,
        equity=10_000.0,
        cash=1_000.0,
        daily_pnl_pct=0.0,
    )
    assert res.submitted is False
    assert "reserve" in (res.reason or "").lower()
    broker.submit_market_order.assert_not_called()


def test_sweep_buy_skips_below_min_order_floor(caplog: pytest.LogCaptureFixture) -> None:
    """Sub-$5 sweep buy is skipped at DEBUG (Tier 43 floor applies to the sweep too)."""
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    with caplog.at_level(logging.DEBUG, logger="src.execution.order_manager"):
        res = mgr.sweep_buy(
            "BIL",
            3.0,
            price=100.0,
            equity=10_000.0,
            cash=10_000.0,
            daily_pnl_pct=0.0,
        )
    assert res.submitted is False
    assert "minimum order size" in (res.reason or "").lower()
    assert "order_skipped_below_floor" in caplog.text
    broker.submit_market_order.assert_not_called()


def test_sweep_sell_submits_market_sell() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "sweep-s"
    broker.get_order_fill_price.return_value = 100.0
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    res = mgr.sweep_sell("BIL", 4.0, price=100.0, equity=10_000.0, daily_pnl_pct=0.0)
    assert res.submitted is True
    assert res.side == "sell"
    assert res.strategy_name == "CashSweep"
    assert res.qty == pytest.approx(4.0)
    broker.submit_market_order.assert_called_once_with("BIL", 4.0, "sell", client_order_id=ANY)


def test_sweep_sell_blocked_by_kill_switch() -> None:
    broker = MagicMock()
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    res = mgr.sweep_sell("BIL", 4.0, price=100.0, equity=10_000.0, daily_pnl_pct=-0.05)
    assert res.submitted is False
    assert "kill" in (res.reason or "").lower()
    assert res.strategy_name == "CashSweep"
    broker.submit_market_order.assert_not_called()


def test_close_position_blocked_by_pdt(tmp_path: Path) -> None:
    """Fourth projected day trade with equity under $25k blocks the sell."""
    db = tmp_path / "pdt.db"
    store = SQLiteStore(db)
    risk = RiskConfig(
        max_position_pct=0.25,
        min_cash_reserve_pct=0.05,
        daily_loss_limit_pct=0.03,
        pdt_protection=True,
    )

    def pair(sym: str, day: int) -> None:
        tsb = datetime(2026, 4, day, 10, 0, tzinfo=UTC)
        tss = datetime(2026, 4, day, 11, 0, tzinfo=UTC)
        store.log_execution(
            "cx",
            OrderExecutionResult(
                symbol=sym,
                submitted=True,
                order_id=f"{sym}-b",
                side="buy",
                qty=1.0,
                fill_price=100.0,
                timestamp=tsb,
                order_status="filled",
            ),
        )
        store.log_execution(
            "cx",
            OrderExecutionResult(
                symbol=sym,
                submitted=True,
                order_id=f"{sym}-s",
                side="sell",
                qty=1.0,
                fill_price=100.0,
                timestamp=tss,
                order_status="filled",
            ),
        )

    # Mon 6, Tue 7, Wed 8 — three day trades (different symbols).
    pair("AAA", 6)
    pair("BBB", 7)
    pair("CCC", 8)
    store.log_execution(
        "cx",
        OrderExecutionResult(
            symbol="SPY",
            submitted=True,
            order_id="spy-b",
            side="buy",
            qty=5.0,
            fill_price=400.0,
            timestamp=datetime(2026, 4, 9, 10, 0, tzinfo=UTC),
            order_status="filled",
        ),
    )

    broker = MagicMock()
    mgr = OrderManager(
        broker=broker,
        settings=_settings(risk=risk),
        kill_switch=KillSwitch(0.03),
        sqlite_store=store,
    )
    as_of = datetime(2026, 4, 9, 15, 0, tzinfo=UTC)
    res = mgr.close_position(
        "SPY",
        5.0,
        price=400.0,
        equity=10_000.0,
        daily_pnl_pct=0.0,
        as_of=as_of,
    )
    assert res.submitted is False
    assert res.reason is not None
    assert "PDT" in res.reason
    broker.submit_market_order.assert_not_called()


def test_intended_buy_notionals_matches_execute_signals_submitted_notional() -> None:
    """Unsweep sizing and live execution must share one path (Tier 47B-1)."""
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-eq"
    broker.get_order_fill_price.return_value = 400.0
    broker.list_recent_orders.return_value = []
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="Equality path.",
        timestamp=datetime.now(UTC),
        strategy_name="MomentumRotationStrategy",
    )
    kwargs = {
        "last_prices": {"SPY": 400.0},
        "equity": 10_000.0,
        "positions": {"SPY": 0.0},
        "dca_budget": None,
        "atr_values": None,
        "regime_multiplier": 1.0,
    }
    intended = mgr.intended_buy_notionals([sig], **kwargs)
    results = mgr.execute_signals(
        [sig],
        cash=5_000.0,
        daily_pnl_pct=0.0,
        **kwargs,
    )
    assert results[0].submitted is True
    submitted_notional = float(results[0].qty) * 400.0
    assert intended["SPY"] == pytest.approx(submitted_notional)
    assert intended["SPY"] == pytest.approx(500.0)


def test_execute_signals_partial_unsweep_uses_distinct_reason() -> None:
    broker = MagicMock()
    broker.list_recent_orders.return_value = []
    mgr = OrderManager(
        broker=broker,
        settings=_settings(),
        kill_switch=KillSwitch(0.03),
    )
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="Unfunded after partial unsweep.",
        timestamp=datetime.now(UTC),
        strategy_name="MomentumRotationStrategy",
    )
    results = mgr.execute_signals(
        [sig],
        last_prices={"SPY": 100.0},
        equity=3_433.79,
        cash=200.0,
        positions={"SPY": 0.0},
        daily_pnl_pct=0.0,
        unsweep_attempted=True,
    )
    assert len(results) == 1
    assert results[0].submitted is False
    assert results[0].reason == "insufficient cash after partial unsweep"
    broker.submit_market_order.assert_not_called()
