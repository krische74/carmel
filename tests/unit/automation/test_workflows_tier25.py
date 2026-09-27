"""Tier 25 workflow behaviors: ML staleness, reconciliation alerts."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.automation import workflows as wf_mod
from src.automation.workflows import TradingWorkflow
from src.config import MLConfig, Settings
from src.data.pipeline import IngestResult
from src.models import OrderExecutionResult
from src.risk.kill_switch import KillSwitch
from src.strategy.dca import DCAStrategy


def test_stale_ml_model_warning_and_alert(
    settings: Settings,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
    p = tmp_path / "data" / "ml" / "hub.joblib"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    old_ts = datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    os.utime(p, (old_ts, old_ts))

    ml_d = settings.ml.model_dump()
    ml_d.update(
        {
            "affect_orders": True,
            "model_path": "data/ml/hub.joblib",
            "max_model_age_days": 30,
            "enabled": False,
        },
    )
    s2 = settings.model_copy(update={"ml": MLConfig.model_validate(ml_d)})

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    ohlcv = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    pq = MagicMock()
    pq.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    sqlite = MagicMock()
    sqlite.get_latest_ml_scores_batch.return_value = []

    wf = TradingWorkflow(
        settings=s2,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[DCAStrategy(s2)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
        kill_switch=KillSwitch(s2.risk.daily_loss_limit_pct),
    )
    caplog.set_level("WARNING")
    wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
    assert "ML model is" in caplog.text and "days old" in caplog.text
    recon_cats = [c.args[0].category for c in sqlite.log_alert.call_args_list if c.args]
    assert "ml_model_stale" in recon_cats


def test_workflow_generates_reconciliation_alert(settings: Settings) -> None:
    holder: dict[str, str | None] = {"cid": None}
    orig_u4 = uuid.uuid4

    def _uuid4():
        u = orig_u4()
        holder["cid"] = u.hex[:12]
        return u

    wf_mod.uuid.uuid4 = _uuid4
    try:
        pipeline = MagicMock()
        pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
        ohlcv = pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "volume": [1_000_000.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
        )
        pq = MagicMock()
        pq.read_ohlcv.return_value = ohlcv
        broker = MagicMock()
        broker.get_account_equity.return_value = 10_000.0
        broker.get_last_equity.return_value = 10_000.0
        broker.get_cash.return_value = 5_000.0
        broker.get_position_qty.return_value = 0.0
        broker.list_recent_orders.return_value = []
        order_mgr = MagicMock()
        order_mgr.execute_signals.return_value = [
            OrderExecutionResult(symbol="VOO", submitted=True, order_id="a", side="buy"),
        ]
        sqlite = MagicMock()

        def _gx(*_a, **_kw):
            cid = holder["cid"]
            if not cid:
                return []
            return [
                {
                    "cycle_id": cid,
                    "order_id": "a",
                    "submitted": 1,
                    "symbol": "VOO",
                    "side": "buy",
                    "qty": 1.0,
                    "filled_avg_price": 100.0,
                },
            ]

        sqlite.get_executions.side_effect = _gx

        wf = TradingWorkflow(
            settings=settings,
            data_pipeline=pipeline,
            parquet_store=pq,
            strategies=[DCAStrategy(settings)],
            order_manager=order_mgr,
            broker=broker,
            sqlite_store=sqlite,
            kill_switch=KillSwitch(settings.risk.daily_loss_limit_pct),
        )
        wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
        cats = [c.args[0].category for c in sqlite.log_alert.call_args_list if c.args]
        assert "reconciliation" in cats
    finally:
        wf_mod.uuid.uuid4 = orig_u4


def test_workflow_reconciliation_ignores_broker_orders_outside_cycle(settings: Settings) -> None:
    """Broker returns many historical orders; per-cycle reconcile only compares cycle ids."""
    holder: dict[str, str | None] = {"cid": None}
    orig_u4 = uuid.uuid4

    def _uuid4():
        u = orig_u4()
        holder["cid"] = u.hex[:12]
        return u

    wf_mod.uuid.uuid4 = _uuid4
    try:
        pipeline = MagicMock()
        pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
        ohlcv = pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "volume": [1_000_000.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
        )
        pq = MagicMock()
        pq.read_ohlcv.return_value = ohlcv
        broker = MagicMock()
        broker.get_account_equity.return_value = 10_000.0
        broker.get_last_equity.return_value = 10_000.0
        broker.get_cash.return_value = 5_000.0
        broker.get_position_qty.return_value = 0.0
        hist = [
            {
                "order_id": f"h{i}",
                "symbol": "SPY",
                "side": "buy",
                "qty": 1.0,
                "filled_qty": 1.0,
                "filled_avg_price": 100.0,
                "status": "filled",
            }
            for i in range(9)
        ]
        broker.list_recent_orders.return_value = [
            *hist,
            {
                "order_id": "a",
                "symbol": "VOO",
                "side": "buy",
                "qty": 1.0,
                "filled_qty": 1.0,
                "filled_avg_price": 100.0,
                "status": "filled",
            },
        ]
        order_mgr = MagicMock()
        order_mgr.execute_signals.return_value = [
            OrderExecutionResult(symbol="VOO", submitted=True, order_id="a", side="buy"),
        ]
        sqlite = MagicMock()

        def _gx(*_a, **_kw):
            cid = holder["cid"]
            if not cid:
                return []
            return [
                {
                    "cycle_id": cid,
                    "order_id": "a",
                    "submitted": 1,
                    "symbol": "VOO",
                    "side": "buy",
                    "qty": 1.0,
                    "filled_avg_price": 100.0,
                },
            ]

        sqlite.get_executions.side_effect = _gx

        wf = TradingWorkflow(
            settings=settings,
            data_pipeline=pipeline,
            parquet_store=pq,
            strategies=[DCAStrategy(settings)],
            order_manager=order_mgr,
            broker=broker,
            sqlite_store=sqlite,
            kill_switch=KillSwitch(settings.risk.daily_loss_limit_pct),
        )
        wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
        cats = [c.args[0].category for c in sqlite.log_alert.call_args_list if c.args]
        assert "reconciliation" not in cats
    finally:
        wf_mod.uuid.uuid4 = orig_u4


def test_workflow_reconciliation_flags_broker_order_missing_from_log(
    settings: Settings,
) -> None:
    """Finding C: a broker fill in the cycle window with no SQLite row is missing_from_log."""
    holder: dict[str, str | None] = {"cid": None}
    orig_u4 = uuid.uuid4

    def _uuid4():
        u = orig_u4()
        holder["cid"] = u.hex[:12]
        return u

    wf_mod.uuid.uuid4 = _uuid4
    try:
        pipeline = MagicMock()
        pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
        ohlcv = pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "volume": [1_000_000.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
        )
        pq = MagicMock()
        pq.read_ohlcv.return_value = ohlcv
        as_of = datetime(2026, 6, 9, 17, 30, 13, tzinfo=UTC)
        broker = MagicMock()
        broker.get_account_equity.return_value = 10_000.0
        broker.get_last_equity.return_value = 10_000.0
        broker.get_cash.return_value = 5_000.0
        broker.get_position_qty.return_value = 0.0
        broker.list_recent_orders.return_value = [
            {
                "order_id": "logged-bnd",
                "symbol": "BND",
                "side": "buy",
                "qty": 0.02,
                "filled_qty": 0.02,
                "filled_avg_price": 72.0,
                "status": "filled",
                "timestamp": as_of.isoformat(),
                "client_order_id": "cml-bnd",
            },
            {
                "order_id": "orphan-voo",
                "symbol": "VOO",
                "side": "buy",
                "qty": 0.011,
                "filled_qty": 0.011,
                "filled_avg_price": 700.0,
                "status": "filled",
                "timestamp": as_of.isoformat(),
                "client_order_id": "cml-voo-orphan",
            },
        ]
        order_mgr = MagicMock()
        order_mgr.execute_signals.return_value = [
            OrderExecutionResult(
                symbol="BND",
                submitted=True,
                order_id="logged-bnd",
                side="buy",
            ),
        ]
        sqlite = MagicMock()

        def _gx(*_a, **_kw):
            cid = holder["cid"]
            if not cid:
                return []
            return [
                {
                    "cycle_id": cid,
                    "order_id": "logged-bnd",
                    "submitted": 1,
                    "symbol": "BND",
                    "side": "buy",
                    "qty": 0.02,
                    "filled_avg_price": 72.0,
                    "timestamp": as_of.isoformat(),
                },
            ]

        sqlite.get_executions.side_effect = _gx

        wf = TradingWorkflow(
            settings=settings,
            data_pipeline=pipeline,
            parquet_store=pq,
            strategies=[DCAStrategy(settings)],
            order_manager=order_mgr,
            broker=broker,
            sqlite_store=sqlite,
            kill_switch=KillSwitch(settings.risk.daily_loss_limit_pct),
        )
        wf.run_cycle(as_of=as_of)
        msgs = [
            c.args[0].message
            for c in sqlite.log_alert.call_args_list
            if c.args and getattr(c.args[0], "category", None) == "reconciliation"
        ]
        assert msgs, "expected a reconciliation alert for missing_from_log"
        assert any("missing_from_log" in m for m in msgs)
    finally:
        wf_mod.uuid.uuid4 = orig_u4
