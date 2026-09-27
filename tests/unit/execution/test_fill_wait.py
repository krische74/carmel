"""Fill wait + 8/18 stale-cash regression (Tier 47A / 47B-2).

The fake broker's sell proceeds arrive only after N ``refresh_account`` calls.
A synchronous broker does not exercise this tier.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from src.config import RiskConfig, Settings
from src.execution.order_manager import (
    FILL_WAIT_BUDGET_SECONDS,
    FILL_WAIT_INTERVAL_SECONDS,
    OrderManager,
)
from src.models import Signal
from src.risk.kill_switch import KillSwitch


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


class AsyncCashBroker:
    """Sell proceeds land only after ``fills_after_refreshes`` account refreshes."""

    def __init__(
        self,
        *,
        cash: float,
        equity: float,
        positions: dict[str, float],
        prices: dict[str, float],
        fills_after_refreshes: int,
        last_equity: float | None = None,
    ) -> None:
        self._live_cash = float(cash)
        self._cached_cash: float | None = float(cash)
        self._equity = float(equity)
        self._last_equity = float(equity if last_equity is None else last_equity)
        self._positions = {k.strip().upper(): float(v) for k, v in positions.items()}
        self._prices = {k.strip().upper(): float(v) for k, v in prices.items()}
        self.fills_after_refreshes = int(fills_after_refreshes)
        self.refresh_count = 0
        self._pending: dict[str, dict[str, float | str]] = {}
        self._filled: set[str] = set()
        self.submit_calls: list[tuple[str, float, str, str]] = []
        self._oid_n = 0

    def get_account_id(self) -> str:
        return "default"

    def refresh_account(self) -> None:
        self.refresh_count += 1
        self._settle_ready_orders()
        self._cached_cash = self._live_cash

    def _settle_ready_orders(self) -> None:
        if self.refresh_count < self.fills_after_refreshes:
            return
        for oid, info in self._pending.items():
            if oid in self._filled:
                continue
            side = str(info["side"])
            qty = float(info["qty"])
            proceeds = float(info["proceeds"])
            sym = str(info["symbol"])
            if side == "sell":
                self._live_cash += proceeds
                self._positions[sym] = max(0.0, self._positions.get(sym, 0.0) - qty)
            else:
                self._live_cash -= proceeds
                self._positions[sym] = self._positions.get(sym, 0.0) + qty
            self._filled.add(oid)

    def _ensure_cache(self) -> None:
        if self._cached_cash is None:
            self._cached_cash = self._live_cash

    def get_cash(self) -> float:
        self._ensure_cache()
        return float(self._cached_cash)

    def get_account_equity(self) -> float:
        return self._equity

    def get_last_equity(self) -> float:
        return self._last_equity

    def get_all_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_leverage_snapshot(self):
        from src.execution.leverage import LeverageSnapshot

        return LeverageSnapshot(cash=self.get_cash(), broker_reports_leverage=False)

    def get_position_qty(self, symbol: str) -> float:
        return float(self._positions.get(symbol.strip().upper(), 0.0))

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
        self._pending[oid] = {
            "symbol": sym,
            "qty": float(qty),
            "side": side,
            "proceeds": proceeds,
        }
        self._cached_cash = None
        self.submit_calls.append((sym, float(qty), side, oid))
        return oid

    def submit_limit_order(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError

    def submit_stop_order(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        _ = order_id
        return True

    def get_order_status(self, order_id: str) -> dict[str, object]:
        info = self._pending.get(order_id)
        qty = float(info["qty"]) if info is not None else 0.0
        if order_id in self._filled:
            return {
                "order_id": order_id,
                "status": "filled",
                "filled_qty": qty,
                "filled_avg_price": float(self._prices[str(info["symbol"])]) if info else None,
            }
        return {
            "order_id": order_id,
            "status": "new",
            "filled_qty": 0.0,
            "filled_avg_price": None,
        }

    def get_order_fill_price(self, order_id: str) -> float | None:
        info = self._pending.get(order_id)
        if order_id in self._filled and info is not None:
            return float(self._prices[str(info["symbol"])])
        return None

    def list_recent_orders(self, *, limit: int = 100) -> list[dict[str, object]]:
        _ = limit
        return []


# 2026-08-18 17:30 UTC incident numbers
_CASH_BEFORE = 193.96
_EQUITY = 3433.79
_SHV_QTY = 7.791065502
_SHV_PX = 110.249
_SPY_PX = 500.0


def _settings() -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        risk=RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            daily_loss_limit_pct=0.03,
        ),
    )


def _mgr(broker: AsyncCashBroker, monkeypatch: pytest.MonkeyPatch) -> OrderManager:
    clock = _FakeClock()
    monkeypatch.setattr("src.execution.order_manager.time.sleep", clock.sleep)
    monkeypatch.setattr("src.execution.order_manager.time.monotonic", clock.monotonic)
    return OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))


def _spy_signal() -> Signal:
    return Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="Rotate to SPY.",
        timestamp=datetime.now(UTC),
        strategy_name="MomentumRotationStrategy",
    )


@pytest.mark.parametrize(
    ("fills_after_refreshes", "expect_buy"),
    [
        (1, True),
        (999, False),
    ],
)
def test_rotation_buy_waits_for_async_sale_proceeds(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fills_after_refreshes: int,
    expect_buy: bool,
) -> None:
    """delay=0 (fill on first refresh) → buy; delay > 5s budget → skip + INFO."""
    broker = AsyncCashBroker(
        cash=_CASH_BEFORE,
        equity=_EQUITY,
        positions={"SHV": _SHV_QTY, "SPY": 0.0},
        prices={"SHV": _SHV_PX, "SPY": _SPY_PX},
        fills_after_refreshes=fills_after_refreshes,
    )
    mgr = _mgr(broker, monkeypatch)
    caplog.set_level(logging.INFO)
    sell = mgr.close_position(
        "SHV",
        _SHV_QTY,
        price=_SHV_PX,
        equity=_EQUITY,
        daily_pnl_pct=0.0,
        wait_for_fill=True,
    )
    assert sell.submitted is True

    skip_buy = sell.order_status != "filled"
    if skip_buy:
        buys = []
    else:
        buys = mgr.execute_signals(
            [_spy_signal()],
            last_prices={"SPY": _SPY_PX, "SHV": _SHV_PX},
            equity=_EQUITY,
            cash=broker.get_cash(),
            positions=broker.get_all_positions(),
            daily_pnl_pct=0.0,
        )

    buy_submitted = any(r.submitted and r.side == "buy" and r.symbol == "SPY" for r in buys)
    assert buy_submitted is expect_buy
    if expect_buy:
        assert broker.get_cash() + 1e-6 >= 0.05 * _EQUITY
        assert any(c[0] == "SPY" and c[2] == "buy" for c in broker.submit_calls)
    else:
        assert not any(c[0] == "SPY" and c[2] == "buy" for c in broker.submit_calls)
        assert sell.order_status == "pending"
        assert any("rotation fill timeout" in r.message for r in caplog.records)
        assert any("recoverable" in r.message.lower() for r in caplog.records)


def test_fill_wait_observes_mid_loop_cash_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deadlock guard: poll must refresh_account or the cache stays frozen."""
    broker = AsyncCashBroker(
        cash=_CASH_BEFORE,
        equity=_EQUITY,
        positions={"SHV": _SHV_QTY},
        prices={"SHV": _SHV_PX},
        fills_after_refreshes=4,
    )
    mgr = _mgr(broker, monkeypatch)
    oid = broker.submit_market_order("SHV", _SHV_QTY, "sell")
    cash_before = broker.get_cash()
    assert cash_before == pytest.approx(_CASH_BEFORE)
    filled = mgr.wait_for_fill(
        oid,
        expected_cash_increase=_SHV_QTY * _SHV_PX,
        cash_before=cash_before,
    )
    assert filled is True
    assert broker.refresh_count >= 4
    assert broker.get_cash() > _CASH_BEFORE


def test_fill_wait_timeout_does_not_exceed_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = AsyncCashBroker(
        cash=_CASH_BEFORE,
        equity=_EQUITY,
        positions={"SHV": _SHV_QTY},
        prices={"SHV": _SHV_PX},
        fills_after_refreshes=10_000,
    )
    mgr = _mgr(broker, monkeypatch)
    oid = broker.submit_market_order("SHV", _SHV_QTY, "sell")
    clock = _FakeClock()
    monkeypatch.setattr("src.execution.order_manager.time.sleep", clock.sleep)
    monkeypatch.setattr("src.execution.order_manager.time.monotonic", clock.monotonic)
    filled = mgr.wait_for_fill(oid)
    assert filled is False
    assert clock.t == pytest.approx(
        FILL_WAIT_BUDGET_SECONDS, abs=FILL_WAIT_INTERVAL_SECONDS + 0.05
    )
    assert broker.get_cash() == pytest.approx(_CASH_BEFORE)
