"""Tier 32: digest summary aggregation for scheduled notifications."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.automation.alerts import Alert, AlertLevel
from src.automation.digest import generate_digest_summary
from src.data.regime import MarketRegime, OverallRegime, VolatilityRegime, YieldCurveRegime
from src.data.storage.sqlite_store import SQLiteStore
from src.models import OrderExecutionResult
from src.portfolio.tax_lots import LotLedger


def test_digest_summary_empty_when_no_activity(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "d0.sqlite")
    d = generate_digest_summary(store, lookback_hours=168)
    assert d.trade_count == 0
    assert d.alert_count == 0
    assert d.open_positions_count == 0
    assert d.top_movers == []


def test_digest_summary_aggregates_recent_data(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "d1.sqlite")
    now = datetime.now(UTC)
    for i in range(5):
        store.log_execution(
            "c1",
            OrderExecutionResult(
                symbol="SPY",
                submitted=True,
                qty=1.0,
                fill_price=100.0 + i,
                filled_qty=1.0,
                timestamp=now - timedelta(hours=i),
            ),
            account_id="default",
        )
    for j in range(3):
        store.log_alert(
            Alert(
                timestamp=now - timedelta(minutes=j),
                level=AlertLevel.INFO,
                category="test",
                message=f"a{j}",
            ),
        )
    store.write_equity_snapshot(
        (now - timedelta(days=3)).date().isoformat(),
        10_000.0,
        9_000.0,
        1_000.0,
        0.0,
        1_000.0,
        account_id="default",
    )
    store.write_equity_snapshot(
        now.date().isoformat(),
        10_500.0,
        9_000.0,
        1_500.0,
        0.0,
        1_500.0,
        account_id="default",
    )
    reg = MarketRegime(
        timestamp=now,
        vix_close=18.5,
        yield_spread=None,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.RISK_ON,
        sizing_multiplier=1.0,
    )
    store.write_regime_snapshot(reg)
    led = LotLedger(tmp_path / "d1.sqlite")
    led.record_buy("QQQ", 2.0, 50.0, now, account_id="default")

    d = generate_digest_summary(store, lookback_hours=168)
    assert d.trade_count == 5
    assert d.alert_count == 3
    assert d.net_pnl == pytest.approx(500.0)
    assert d.open_positions_count == 1
    assert d.regime_summary
    assert any(m[0] == "SPY" for m in d.top_movers)


def test_digest_summary_per_account_filter(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "d2.sqlite")
    now = datetime.now(UTC)
    store.log_execution(
        "c",
        OrderExecutionResult(
            symbol="A",
            submitted=True,
            qty=1.0,
            fill_price=10.0,
            timestamp=now,
        ),
        account_id="Main",
    )
    store.log_execution(
        "c",
        OrderExecutionResult(
            symbol="B",
            submitted=True,
            qty=1.0,
            fill_price=20.0,
            timestamp=now,
        ),
        account_id="Other",
    )
    d_main = generate_digest_summary(store, lookback_hours=24, account_id="Main")
    d_all = generate_digest_summary(store, lookback_hours=24, account_id=None)
    assert d_main.trade_count == 1
    assert d_all.trade_count == 2
