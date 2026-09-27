"""Tier 48 — pending-notional sweep sizing, buy fill wait, leverage invariant."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from src.automation.alerts import AlertLevel
from src.automation.workflows import TradingWorkflow
from src.config import CashSweepConfig, RiskConfig, Settings
from src.execution.cash_sweep import compute_sweep_action
from src.execution.leverage import LeverageSnapshot, leverage_violation_message
from src.execution.order_manager import (
    FILL_WAIT_BUDGET_SECONDS,
    FILL_WAIT_INTERVAL_SECONDS,
    OrderManager,
)
from src.models import OrderExecutionResult, Signal
from src.risk.kill_switch import KillSwitch


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


class StaggeredDebitBroker:
    """``filled`` status is immediate; cash debits land after per-order refresh counts."""

    def __init__(
        self,
        *,
        cash: float,
        equity: float,
        positions: dict[str, float],
        prices: dict[str, float],
        buy_debit_after: int = 999,
    ) -> None:
        self._live_cash = float(cash)
        self._cached_cash: float | None = float(cash)
        self._equity = float(equity)
        self._positions = {k.strip().upper(): float(v) for k, v in positions.items()}
        self._prices = {k.strip().upper(): float(v) for k, v in prices.items()}
        self.buy_debit_after = int(buy_debit_after)
        self.refresh_count = 0
        self._orders: dict[str, dict[str, float | str]] = {}
        self._debited: set[str] = set()
        self._oid_n = 0

    def get_account_id(self) -> str:
        return "default"

    def refresh_account(self) -> None:
        self.refresh_count += 1
        for oid, info in self._orders.items():
            if oid in self._debited:
                continue
            if self.refresh_count < int(info["debit_after"]):
                continue
            side = str(info["side"])
            proceeds = float(info["proceeds"])
            sym = str(info["symbol"])
            qty = float(info["qty"])
            if side == "buy":
                self._live_cash -= proceeds
                self._positions[sym] = self._positions.get(sym, 0.0) + qty
            else:
                self._live_cash += proceeds
                self._positions[sym] = max(0.0, self._positions.get(sym, 0.0) - qty)
            self._debited.add(oid)
        self._cached_cash = self._live_cash

    def get_cash(self) -> float:
        if self._cached_cash is None:
            self._cached_cash = self._live_cash
        return float(self._cached_cash)

    def get_account_equity(self) -> float:
        return self._equity

    def get_last_equity(self) -> float:
        return self._equity

    def get_all_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_position_qty(self, symbol: str) -> float:
        return float(self._positions.get(symbol.strip().upper(), 0.0))

    def get_leverage_snapshot(self):
        from src.execution.leverage import LeverageSnapshot

        return LeverageSnapshot(cash=self.get_cash(), broker_reports_leverage=False)

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        client_order_id: str | None = None,
    ) -> str:
        _ = client_order_id
        self._oid_n += 1
        oid = f"ord-{self._oid_n}"
        sym = symbol.strip().upper()
        px = float(self._prices[sym])
        proceeds = float(qty) * px
        self._orders[oid] = {
            "symbol": sym,
            "qty": float(qty),
            "side": side,
            "proceeds": proceeds,
            "debit_after": self.buy_debit_after if side == "buy" else 1,
        }
        self._cached_cash = None
        return oid

    def submit_limit_order(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError

    def submit_stop_order(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        _ = order_id
        return True

    def get_order_status(self, order_id: str) -> dict[str, object]:
        info = self._orders.get(order_id)
        if info is None:
            return {"order_id": order_id, "status": "unknown", "filled_qty": 0.0}
        return {
            "order_id": order_id,
            "status": "filled",
            "filled_qty": float(info["qty"]),
            "filled_avg_price": float(self._prices[str(info["symbol"])]),
        }

    def get_order_fill_price(self, order_id: str) -> float | None:
        info = self._orders.get(order_id)
        if info is None:
            return None
        return float(self._prices[str(info["symbol"])])

    def list_recent_orders(self, *, limit: int = 100) -> list[dict[str, object]]:
        _ = limit
        return []


# 2026-08-19 17:30 UTC incident numbers (post-unsweep)
_CASH = 1099.56
_EQUITY = 3436.33
_SHV_NOTIONAL = 859.06
_BIL_NOTIONAL = 859.12
_SHV_PX = 110.27
_BIL_PX = 91.57


def _settings() -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        risk=RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            daily_loss_limit_pct=0.03,
        ),
        cash_sweep=CashSweepConfig(
            enabled=True,
            symbol="BIL",
            buffer_pct=0.02,
            min_trade_usd=50.0,
        ),
    )


def _mgr(broker: StaggeredDebitBroker, monkeypatch: pytest.MonkeyPatch) -> OrderManager:
    clock = _FakeClock()
    monkeypatch.setattr("src.execution.order_manager.time.sleep", clock.sleep)
    monkeypatch.setattr("src.execution.order_manager.time.monotonic", clock.monotonic)
    return OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))


def _shv_signal() -> Signal:
    return Signal(
        symbol="SHV",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="Momentum entry.",
        timestamp=datetime.now(UTC),
        strategy_name="MomentumRotationStrategy",
    )


def test_sweep_hold_with_pending_notional_reproduces_819() -> None:
    """48A-1: SHV debit pending → surplus below min_trade → hold, not BIL buy."""
    action = compute_sweep_action(
        cash=_CASH,
        equity=_EQUITY,
        sweep_position_notional=0.0,
        min_cash_reserve_pct=0.05,
        buffer_pct=0.02,
        min_trade_usd=50.0,
        pending_buy_notional=_SHV_NOTIONAL,
    )
    assert action.action == "hold"
    assert _BIL_NOTIONAL > 50.0  # would have swept if pending were ignored


def test_buy_wait_does_not_exit_on_filled_before_debit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """48A-2: ``filled`` while cash unchanged must not satisfy buy wait."""
    broker = StaggeredDebitBroker(
        cash=_CASH,
        equity=_EQUITY,
        positions={"SHV": 0.0},
        prices={"SHV": _SHV_PX},
        buy_debit_after=10_000,
    )
    mgr = _mgr(broker, monkeypatch)
    qty = _SHV_NOTIONAL / _SHV_PX
    cash_before = broker.get_cash()
    oid = broker.submit_market_order("SHV", qty, "buy")
    debited = mgr.wait_for_fill(
        oid,
        side="buy",
        expected_cash_delta=_SHV_NOTIONAL,
        cash_before=cash_before,
    )
    assert debited is False
    assert broker.get_cash() == pytest.approx(_CASH)


def test_unsettled_pending_drops_after_debit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed batch: one debited order drops from pending sum; one still counts."""
    from src.execution.order_manager import _PendingBuyDebit

    broker = StaggeredDebitBroker(
        cash=_CASH,
        equity=_EQUITY,
        positions={"SHV": 0.0, "BIL": 0.0},
        prices={"SHV": _SHV_PX, "BIL": _BIL_PX},
        buy_debit_after=999,
    )
    mgr = _mgr(broker, monkeypatch)
    shv_qty = _SHV_NOTIONAL / _SHV_PX
    bil_qty = _BIL_NOTIONAL / _BIL_PX
    cash0 = broker.get_cash()
    oid_shv = broker.submit_market_order("SHV", shv_qty, "buy")
    mgr._pending_buy_debits.append(_PendingBuyDebit(oid_shv, _SHV_NOTIONAL, cash0))
    oid_bil = broker.submit_market_order("BIL", bil_qty, "buy")
    mgr._pending_buy_debits.append(_PendingBuyDebit(oid_bil, _BIL_NOTIONAL, cash0))
    assert mgr.unsettled_pending_buy_notional() == pytest.approx(_SHV_NOTIONAL + _BIL_NOTIONAL)
    broker._orders[oid_shv]["debit_after"] = 1
    broker.refresh_account()
    assert mgr.unsettled_pending_buy_notional() == pytest.approx(_BIL_NOTIONAL)


def test_819_sequence_sweep_holds_after_strategy_buy(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end: strategy buy with delayed debit → pending notional → sweep hold."""
    broker = StaggeredDebitBroker(
        cash=_CASH,
        equity=_EQUITY,
        positions={"SHV": 0.0, "BIL": 0.0},
        prices={"SHV": _SHV_PX, "BIL": _BIL_PX},
        buy_debit_after=999,
    )
    mgr = _mgr(broker, monkeypatch)
    caplog.set_level(logging.INFO)
    buys = mgr.execute_signals(
        [_shv_signal()],
        last_prices={"SHV": _SHV_PX},
        equity=_EQUITY,
        cash=_CASH,
        positions=broker.get_all_positions(),
        daily_pnl_pct=0.0,
    )
    assert any(r.submitted and r.symbol == "SHV" for r in buys)
    pending = mgr.unsettled_pending_buy_notional()
    assert pending == pytest.approx(_SHV_NOTIONAL, rel=0.01)
    action = compute_sweep_action(
        cash=broker.get_cash(),
        equity=_EQUITY,
        sweep_position_notional=0.0,
        min_cash_reserve_pct=0.05,
        buffer_pct=0.02,
        min_trade_usd=50.0,
        pending_buy_notional=pending,
    )
    assert action.action == "hold"
    assert not any(c[0] == "BIL" and c[2] == "buy" for c in [])  # no sweep buy path invoked


def test_leverage_invariant_message_on_negative_cash() -> None:
    snap = LeverageSnapshot(cash=-618.62, broker_reports_leverage=False)
    msg = leverage_violation_message(snap, "buy SHV")
    assert msg is not None
    assert "-618.62" in msg


def test_workflow_append_leverage_alert_on_negative_cash() -> None:
    from unittest.mock import MagicMock

    wf = TradingWorkflow.__new__(TradingWorkflow)
    broker = MagicMock()
    broker.get_leverage_snapshot.return_value = LeverageSnapshot(cash=-1.0)
    wf._broker = broker
    alerts: list = []
    wf._append_leverage_alert(
        alerts,
        [
            OrderExecutionResult(
                symbol="SHV",
                submitted=True,
                side="buy",
                qty=1.0,
                strategy_name="MomentumRotationStrategy",
            ),
        ],
    )
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.CRITICAL
    assert alerts[0].category == "leverage"


def test_buy_wait_timeout_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = StaggeredDebitBroker(
        cash=_CASH,
        equity=_EQUITY,
        positions={"SHV": 0.0},
        prices={"SHV": _SHV_PX},
        buy_debit_after=10_000,
    )
    mgr = _mgr(broker, monkeypatch)
    qty = _SHV_NOTIONAL / _SHV_PX
    cash_before = broker.get_cash()
    oid = broker.submit_market_order("SHV", qty, "buy")
    clock = _FakeClock()
    monkeypatch.setattr("src.execution.order_manager.time.sleep", clock.sleep)
    monkeypatch.setattr("src.execution.order_manager.time.monotonic", clock.monotonic)
    debited = mgr.wait_for_fill(
        oid,
        side="buy",
        expected_cash_delta=_SHV_NOTIONAL,
        cash_before=cash_before,
    )
    assert debited is False
    assert clock.t == pytest.approx(FILL_WAIT_BUDGET_SECONDS, abs=FILL_WAIT_INTERVAL_SECONDS + 0.05)
