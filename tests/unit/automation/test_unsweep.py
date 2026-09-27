"""Unsweep-before-entry (Tier 47B-3)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd

from src.config import Settings
from src.data.pipeline import IngestResult
from src.data.storage.sqlite_store import SQLiteStore
from src.execution.order_manager import OrderManager
from src.models import Signal
from src.risk.kill_switch import KillSwitch
from src.strategy.base import Strategy
from tests.unit.automation.test_workflows import ScriptedMomentum
from tests.unit.execution.test_fill_wait import AsyncCashBroker, _FakeClock

if TYPE_CHECKING:
    import pytest

_CASH = 193.96
_EQUITY = 3433.79
_PX = 91.50


def _ohlcv(close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-18")]),
    )


def _spy_signal() -> Signal:
    return Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="Enter SPY from flat.",
        timestamp=datetime(2026, 8, 18, 17, 30, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )


def _workflow(
    settings: Settings,
    broker: AsyncCashBroker,
    *,
    sqlite: SQLiteStore | None = None,
    kill_switch: KillSwitch | None = None,
):
    from src.automation.workflows import TradingWorkflow

    assert isinstance(settings, Settings)

    settings.regime.enabled = False
    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    pq = MagicMock()
    pq.read_ohlcv.return_value = _ohlcv(_PX)
    dca = MagicMock(spec=Strategy)
    dca.get_universe.return_value = []
    dca.generate_signals.return_value = []
    ks = kill_switch or KillSwitch(0.03)
    mgr = OrderManager(broker=broker, settings=settings, kill_switch=ks)
    return TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[ScriptedMomentum(settings, [_spy_signal()]), dca],
        order_manager=mgr,
        broker=broker,
        sqlite_store=sqlite,
        kill_switch=ks,
    )


def test_unsweep_funds_entry_from_flat_then_buy_submits(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("src.execution.order_manager.time.sleep", clock.sleep)
    monkeypatch.setattr("src.execution.order_manager.time.monotonic", clock.monotonic)
    broker = AsyncCashBroker(
        cash=_CASH,
        equity=_EQUITY,
        positions={"BIL": 20.0, "SPY": 0.0, "SHV": 0.0},
        prices={"BIL": _PX, "SPY": _PX, "SHV": _PX},
        fills_after_refreshes=1,
    )
    wf = _workflow(settings, broker)
    caplog.set_level(logging.INFO)
    result = wf.run_cycle(as_of=datetime(2026, 8, 18, 17, 30, tzinfo=UTC))
    assert any(
        "cash_sweep unsweep" in r.message and "funding SPY entry" in r.message
        for r in caplog.records
    )
    sells = [c for c in broker.submit_calls if c[0] == "BIL" and c[2] == "sell"]
    buys = [c for c in broker.submit_calls if c[0] == "SPY" and c[2] == "buy"]
    assert sells, "expected unsweep sell of BIL"
    assert buys, "expected SPY buy after unsweep fill"
    assert sells[0][1] > 5.0
    assert sells[0][3]  # order id
    # unsweep before buy
    sell_idx = broker.submit_calls.index(sells[0])
    buy_idx = broker.submit_calls.index(buys[0])
    assert sell_idx < buy_idx
    assert any(
        r.submitted and r.symbol == "SPY" and r.side == "buy" for r in result.execution_results
    )


def test_partial_unsweep_attempts_buy_with_distinct_reason(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("src.execution.order_manager.time.sleep", clock.sleep)
    monkeypatch.setattr("src.execution.order_manager.time.monotonic", clock.monotonic)
    broker = AsyncCashBroker(
        cash=_CASH,
        equity=_EQUITY,
        positions={"BIL": 1.0, "SPY": 0.0, "SHV": 0.0},
        prices={"BIL": _PX, "SPY": _PX, "SHV": _PX},
        fills_after_refreshes=1,
    )
    store = SQLiteStore(tmp_path / "u.db")
    wf = _workflow(settings, broker, sqlite=store)
    result = wf.run_cycle(as_of=datetime(2026, 8, 18, 17, 30, tzinfo=UTC))
    spy = [r for r in result.execution_results if r.symbol == "SPY" and r.side == "buy"]
    assert spy
    assert spy[0].submitted is False
    assert spy[0].reason == "insufficient cash after partial unsweep"
    alerts = store.get_alerts()
    assert any(
        a.get("category") == "unsweep_failed" and a.get("level") == "warning" for a in alerts
    )


def test_kill_switch_blocks_unsweep(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("src.execution.order_manager.time.sleep", clock.sleep)
    monkeypatch.setattr("src.execution.order_manager.time.monotonic", clock.monotonic)
    broker = AsyncCashBroker(
        cash=_CASH,
        equity=_EQUITY,
        last_equity=_EQUITY / 0.9,
        positions={"BIL": 20.0, "SPY": 0.0, "SHV": 0.0},
        prices={"BIL": _PX, "SPY": _PX, "SHV": _PX},
        fills_after_refreshes=1,
    )
    # daily pnl from last_equity: (3433.79 / (3433.79/0.9) - 1) = -10%
    wf = _workflow(settings, broker, kill_switch=KillSwitch(0.03))
    wf.run_cycle(as_of=datetime(2026, 8, 18, 17, 30, tzinfo=UTC))
    assert not any(c[0] == "BIL" and c[2] == "sell" for c in broker.submit_calls)
    assert not any(c[0] == "SPY" and c[2] == "buy" for c in broker.submit_calls)
