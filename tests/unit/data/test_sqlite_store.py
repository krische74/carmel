"""Unit tests for SQLite metadata store."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.automation.alerts import Alert, AlertLevel
from src.data.storage.sqlite_store import SQLiteStore
from src.models import OrderExecutionResult, Signal


def test_sqlite_store_registers_symbol_and_last_fetch(tmp_path: Path) -> None:
    db = tmp_path / "meta.db"
    store = SQLiteStore(db_path=db)
    store.register_symbol("SPY", source="yfinance")
    store.record_last_fetch("SPY", "2024-01-15T00:00:00Z")
    assert store.get_last_fetch("SPY") == "2024-01-15T00:00:00Z"
    assert store.list_symbols() == ["SPY"]


def test_sqlite_store_data_source_registry(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=tmp_path / "m.db")
    store.register_data_source("yfinance", "Yahoo Finance via yfinance")
    store.register_data_source("fred", "Federal Reserve Economic Data")
    sources = store.list_data_sources()
    assert any(s["id"] == "yfinance" for s in sources)
    assert any(s["id"] == "fred" for s in sources)


def test_log_signal_persists_explanation(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "exp.db")
    cid = "c1"
    store.log_signal(
        cid,
        Signal(
            symbol="QQQ",
            direction="long",
            weight=1.0,
            confidence=0.8,
            rationale="short rationale",
            timestamp=datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
            strategy_name="MomentumRotationStrategy",
        ),
        explanation="Long template explanation with indicators.",
    )
    rows = store.get_signals(limit=5)
    assert len(rows) == 1
    assert rows[0]["explanation"] == "Long template explanation with indicators."
    assert rows[0]["rationale"] == "short rationale"


def test_log_signal_and_get_signals_round_trip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "t.db")
    cid = "abc123def456"
    for i in range(3):
        store.log_signal(
            cid,
            Signal(
                symbol="SPY",
                direction="long",
                weight=0.1 * (i + 1),
                confidence=0.9,
                rationale=f"R{i}",
                timestamp=datetime(2026, 1, i + 1, tzinfo=UTC),
                strategy_name="TestStrat",
            ),
        )
    rows = store.get_signals(limit=10)
    assert len(rows) == 3
    assert rows[0]["symbol"] == "SPY"
    assert rows[0]["cycle_id"] == cid
    assert rows[0]["rationale"] == "R2"
    assert rows[-1]["rationale"] == "R0"


def test_get_signals_with_offset_and_count_signals(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "sig_off.db")
    cid = "c-off"
    for i in range(3):
        store.log_signal(
            cid,
            Signal(
                symbol="SPY",
                direction="long",
                weight=0.1,
                confidence=0.9,
                rationale=f"R{i}",
                timestamp=datetime(2026, 1, i + 1, tzinfo=UTC),
                strategy_name="S",
            ),
        )
    assert store.count_signals() == 3
    page0 = store.get_signals(limit=2, offset=0)
    page1 = store.get_signals(limit=2, offset=2)
    assert len(page0) == 2
    assert len(page1) == 1
    assert page0[0]["rationale"] == "R2"
    assert page1[0]["rationale"] == "R0"
    assert store.get_signals(limit=10, offset=99) == []


def test_get_signals_offset_beyond_total_returns_empty(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "sig_beyond.db")
    store.log_signal(
        "c",
        Signal(
            symbol="X",
            direction="long",
            weight=1.0,
            confidence=1.0,
            rationale="one",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            strategy_name="S",
        ),
    )
    assert store.get_signals(limit=5, offset=10) == []


def test_log_execution_and_get_executions_round_trip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "e.db")
    cid = "cycle001"
    eid = store.log_execution(
        cid,
        OrderExecutionResult(
            symbol="QQQ",
            submitted=True,
            order_id="o1",
            side="buy",
        ),
    )
    assert eid is not None and eid > 0
    store.log_execution(
        cid,
        OrderExecutionResult(
            symbol="SPY",
            submitted=False,
            reason="blocked",
            side="sell",
        ),
    )
    rows = store.get_executions(limit=10)
    assert len(rows) == 2
    assert rows[0]["side"] == "sell"
    assert rows[0]["submitted"] == 0
    assert rows[1]["side"] == "buy"


def test_execution_quality_round_trip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "q.db")
    eid = store.log_execution(
        "c1",
        OrderExecutionResult(
            symbol="SPY",
            submitted=True,
            order_id="o9",
            side="buy",
            qty=1.0,
            fill_price=100.0,
        ),
    )
    assert eid is not None
    store.write_execution_quality(
        {
            "execution_id": eid,
            "cycle_id": "c1",
            "execution_ts": "2026-01-01T16:00:00+00:00",
            "symbol": "SPY",
            "side": "buy",
            "qty": 1.0,
            "fill_price": 100.0,
            "reference_price": 99.5,
            "slippage_bps": 50.25,
            "note": "test",
        },
    )
    rows = store.get_execution_quality_rows(limit=10)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SPY"
    assert float(rows[0]["slippage_bps"]) == pytest.approx(50.25)


def test_sqlite_schema_has_cycle_id_indexes(tmp_path: Path) -> None:
    db = tmp_path / "meta.db"
    SQLiteStore(db_path=db)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' ORDER BY name",
        ).fetchall()
    names = {str(r[0]) for r in rows}
    assert "idx_signals_cycle" in names
    assert "idx_executions_cycle" in names


def test_write_and_read_equity_snapshot_round_trip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "e.db")
    store.write_equity_snapshot(
        "2026-01-10",
        total_market_value=1000.0,
        total_cost_basis=900.0,
        unrealized_pnl=100.0,
        realized_pnl=50.0,
        total_pnl=150.0,
        cash=250.0,
    )
    rows = store.get_equity_snapshots()
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-01-10"
    assert float(rows[0]["total_market_value"]) == pytest.approx(1000.0)
    assert float(rows[0]["total_pnl"]) == pytest.approx(150.0)
    assert float(rows[0]["cash"]) == pytest.approx(250.0)


def test_write_equity_snapshot_cash_none_round_trips(tmp_path: Path) -> None:
    """NULL cash means unknown — must not coerce to 0.0."""
    store = SQLiteStore(tmp_path / "e.db")
    store.write_equity_snapshot(
        "2026-01-10",
        total_market_value=1000.0,
        total_cost_basis=900.0,
        unrealized_pnl=100.0,
        realized_pnl=0.0,
        total_pnl=100.0,
        cash=None,
    )
    rows = store.get_equity_snapshots()
    assert rows[0]["cash"] is None


def test_write_equity_snapshot_broker_equity_none_round_trips(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "e.db")
    store.write_equity_snapshot(
        "2026-01-10",
        total_market_value=1000.0,
        total_cost_basis=900.0,
        unrealized_pnl=100.0,
        realized_pnl=0.0,
        total_pnl=100.0,
        cash=250.0,
        broker_equity=None,
    )
    rows = store.get_equity_snapshots()
    assert rows[0]["broker_equity"] is None
    store.write_equity_snapshot(
        "2026-01-11",
        total_market_value=1000.0,
        total_cost_basis=900.0,
        unrealized_pnl=100.0,
        realized_pnl=0.0,
        total_pnl=100.0,
        cash=250.0,
        broker_equity=3_434.0,
    )
    rows = store.get_equity_snapshots()
    assert rows[1]["broker_equity"] == pytest.approx(3_434.0)


def test_migrate_equity_snapshots_add_cash_idempotent(tmp_path: Path) -> None:
    """Legacy DB without cash column gains it; running migration twice is safe."""
    import sqlite3

    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE equity_snapshots (
                account_id TEXT NOT NULL DEFAULT 'default',
                date TEXT NOT NULL,
                total_market_value REAL NOT NULL,
                total_cost_basis REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                PRIMARY KEY (account_id, date)
            );
            INSERT INTO equity_snapshots VALUES
                ('default', '2026-05-16', 846.51, 846.51, 0.0, 0.0, 0.0);
            """,
        )
    store = SQLiteStore(db)
    store._migrate_equity_snapshots_add_cash()
    store._migrate_equity_snapshots_add_cash()
    with sqlite3.connect(db) as conn:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(equity_snapshots)")}
    assert "cash" in cols
    rows = store.get_equity_snapshots()
    assert len(rows) == 1
    assert rows[0]["cash"] is None
    assert float(rows[0]["total_market_value"]) == pytest.approx(846.51)


def test_equity_snapshot_upserts_on_same_date(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "e.db")
    store.write_equity_snapshot(
        "2026-01-10",
        total_market_value=1000.0,
        total_cost_basis=900.0,
        unrealized_pnl=100.0,
        realized_pnl=0.0,
        total_pnl=100.0,
    )
    store.write_equity_snapshot(
        "2026-01-10",
        total_market_value=2000.0,
        total_cost_basis=900.0,
        unrealized_pnl=1100.0,
        realized_pnl=0.0,
        total_pnl=1100.0,
    )
    rows = store.get_equity_snapshots()
    assert len(rows) == 1
    assert float(rows[0]["total_market_value"]) == pytest.approx(2000.0)


def test_get_equity_snapshots_filters_by_date_range(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "e.db")
    for d, mv in [("2026-01-01", 1.0), ("2026-01-02", 2.0), ("2026-01-03", 3.0)]:
        store.write_equity_snapshot(
            d,
            total_market_value=mv,
            total_cost_basis=mv,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=mv,
        )
    mid = store.get_equity_snapshots(start="2026-01-02", end="2026-01-02")
    assert len(mid) == 1
    assert mid[0]["date"] == "2026-01-02"


def test_get_equity_snapshots_limit_offset_and_count(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "eq_lim.db")
    for d, mv in [("2026-01-01", 1.0), ("2026-01-02", 2.0), ("2026-01-03", 3.0)]:
        store.write_equity_snapshot(
            d,
            total_market_value=mv,
            total_cost_basis=mv,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=mv,
        )
    assert store.count_equity_snapshots() == 3
    head = store.get_equity_snapshots(limit=2, offset=0)
    assert len(head) == 2
    assert head[0]["date"] == "2026-01-01"
    tail = store.get_equity_snapshots(limit=2, offset=2)
    assert len(tail) == 1
    assert tail[0]["date"] == "2026-01-03"


def test_log_alert_and_get_alerts_round_trip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "alerts.db")
    store.log_alert(
        Alert(
            timestamp=datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
            level=AlertLevel.WARNING,
            category="ingest_failure",
            message="failed: SPY",
        ),
    )
    rows = store.get_alerts(limit=10)
    assert len(rows) == 1
    assert rows[0]["level"] == "warning"
    assert rows[0]["category"] == "ingest_failure"
    assert "SPY" in rows[0]["message"]


def test_save_and_retrieve_backtest_run(tmp_path: Path) -> None:
    from src.backtesting.engine import BacktestConfig, BacktestResult
    from src.reporting.returns import ReturnMetrics

    store = SQLiteStore(tmp_path / "bt.db")
    cfg = BacktestConfig(slippage_bps=3.0, rebalance_frequency="weekly")
    m = ReturnMetrics(
        total_return_pct=5.0,
        cagr_pct=4.0,
        sharpe_ratio=1.1,
        sortino_ratio=1.2,
        max_drawdown_pct=2.0,
        max_drawdown_duration_days=3,
        calmar_ratio=2.0,
        annual_volatility_pct=10.0,
        best_day_pct=1.0,
        worst_day_pct=-1.0,
        trading_days=50,
    )
    result = BacktestResult(
        equity_curve=[
            {"date": "2024-01-02", "equity": 10000.0},
            {"date": "2024-01-03", "equity": 10050.0},
        ],
        trades=[],
        return_metrics=m,
        initial_capital=10_000.0,
        final_equity=10_050.0,
    )
    rid = store.save_backtest_run(
        "TestStrat", cfg, result, start_date="2024-01-01", end_date="2024-06-01"
    )
    assert rid >= 1
    row = store.get_backtest_run(rid)
    assert row is not None
    assert row["strategy_name"] == "TestStrat"
    assert float(row["total_return_pct"]) == pytest.approx(5.0)
    assert "equity" in str(row["equity_curve_json"])


def test_save_backtest_run_persists_trades_json(tmp_path: Path) -> None:
    from src.backtesting.engine import BacktestConfig, BacktestResult, BacktestTrade
    from src.reporting.returns import ReturnMetrics

    store = SQLiteStore(tmp_path / "bt_trades.db")
    cfg = BacktestConfig()
    m = ReturnMetrics(
        total_return_pct=2.0,
        cagr_pct=1.0,
        sharpe_ratio=0.5,
        sortino_ratio=0.6,
        max_drawdown_pct=1.0,
        max_drawdown_duration_days=1,
        calmar_ratio=1.0,
        annual_volatility_pct=5.0,
        best_day_pct=0.5,
        worst_day_pct=-0.5,
        trading_days=10,
    )
    trades = [
        BacktestTrade(
            date="2024-01-15",
            symbol="SPY",
            side="buy",
            qty=1.0,
            price=400.0,
            slippage_cost=0.2,
        ),
        BacktestTrade(
            date="2024-02-01",
            symbol="SPY",
            side="sell",
            qty=1.0,
            price=410.0,
            slippage_cost=0.21,
        ),
    ]
    result = BacktestResult(
        equity_curve=[{"date": "2024-01-02", "equity": 10_000.0}],
        trades=trades,
        return_metrics=m,
        initial_capital=10_000.0,
        final_equity=10_100.0,
    )
    rid = store.save_backtest_run(
        "momentum", cfg, result, start_date="2024-01-01", end_date="2024-06-01"
    )
    row = store.get_backtest_run(rid)
    assert row is not None
    raw = row.get("trades_json")
    assert raw is not None
    parsed = json.loads(str(raw))
    assert len(parsed) == 2
    assert parsed[0]["symbol"] == "SPY"
    assert parsed[0]["side"] == "buy"
    assert parsed[0]["qty"] == 1.0
    assert parsed[1]["side"] == "sell"
    assert parsed[1]["slippage_cost"] == pytest.approx(0.21)


def test_backtest_trades_json_migration_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "bt_mig.db"
    store = SQLiteStore(db_path=db)
    store._migrate_backtest_trades_column()
    store._migrate_backtest_trades_column()


def test_save_backtest_run_persists_tier50_metadata(tmp_path: Path) -> None:
    from src.backtesting.engine import BacktestConfig, BacktestResult
    from src.reporting.returns import ReturnMetrics

    store = SQLiteStore(tmp_path / "bt_t50.db")
    cfg = BacktestConfig(apply_risk_layer=True, cash_yield_annual_pct=4.0)
    m = ReturnMetrics(
        total_return_pct=3.0,
        cagr_pct=2.0,
        sharpe_ratio=0.8,
        sortino_ratio=0.9,
        max_drawdown_pct=1.0,
        max_drawdown_duration_days=2,
        calmar_ratio=2.0,
        annual_volatility_pct=5.0,
        best_day_pct=0.5,
        worst_day_pct=-0.5,
        trading_days=20,
    )
    bench = ReturnMetrics(
        total_return_pct=5.0,
        cagr_pct=4.0,
        sharpe_ratio=1.0,
        sortino_ratio=1.1,
        max_drawdown_pct=2.0,
        max_drawdown_duration_days=3,
        calmar_ratio=2.0,
        annual_volatility_pct=6.0,
        best_day_pct=0.6,
        worst_day_pct=-0.6,
        trading_days=20,
    )
    result = BacktestResult(
        equity_curve=[{"date": "2024-01-02", "equity": 10_000.0}],
        trades=[],
        return_metrics=m,
        initial_capital=10_000.0,
        final_equity=10_300.0,
        sizing_mode="constrained",
        cash_yield_annual_pct=4.0,
    )
    rid = store.save_backtest_run(
        "TestStrat",
        cfg,
        result,
        start_date="2024-01-01",
        end_date="2024-06-01",
        benchmark_symbol="SPY",
        benchmark_metrics=bench,
    )
    row = store.get_backtest_run(rid)
    assert row is not None
    assert row["sizing_mode"] == "constrained"
    assert float(row["cash_yield_annual_pct"]) == pytest.approx(4.0)
    assert row["benchmark_symbol"] == "SPY"
    assert row["benchmark_metrics_json"] is not None


def test_get_backtest_runs_newest_first(tmp_path: Path) -> None:
    from src.backtesting.engine import BacktestConfig, BacktestResult
    from src.reporting.returns import ReturnMetrics

    store = SQLiteStore(tmp_path / "bt2.db")
    cfg = BacktestConfig()
    m = ReturnMetrics(
        total_return_pct=1.0,
        cagr_pct=1.0,
        sharpe_ratio=None,
        sortino_ratio=None,
        max_drawdown_pct=0.0,
        max_drawdown_duration_days=0,
        calmar_ratio=None,
        annual_volatility_pct=0.0,
        best_day_pct=0.0,
        worst_day_pct=0.0,
        trading_days=1,
    )
    r = BacktestResult(
        equity_curve=[{"date": "2024-01-01", "equity": 1.0}],
        trades=[],
        return_metrics=m,
        initial_capital=1.0,
        final_equity=1.0,
    )
    store.save_backtest_run("A", cfg, r, start_date="2024-01-01", end_date="2024-01-02")
    store.save_backtest_run("B", cfg, r, start_date="2024-01-01", end_date="2024-01-02")
    rows = store.get_backtest_runs(limit=10)
    assert [row["strategy_name"] for row in rows[:2]] == ["B", "A"]


def test_write_and_read_macro_indicator(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "macro.db")
    store.write_macro_indicator("2024-01-01", "DGS10", 4.5)
    store.write_macro_indicator("2024-01-02", "DGS10", 4.6)
    store.write_macro_indicator("2024-01-03", "DGS10", 4.4)
    rows = store.get_macro_indicator("DGS10")
    assert len(rows) == 3
    assert [r["date"] for r in rows] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert rows[1]["value"] == pytest.approx(4.6)


def test_get_latest_macro_returns_most_recent(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "macro2.db")
    store.write_macro_indicator("2024-01-01", "DGS2", 1.0)
    store.write_macro_indicator("2024-01-10", "DGS2", 1.2)
    store.write_macro_indicator("2024-01-05", "DGS2", 1.1)
    latest = store.get_latest_macro("DGS2")
    assert latest is not None
    assert latest["date"] == "2024-01-10"
    assert float(latest["value"]) == pytest.approx(1.2)


def test_get_latest_macro_returns_none_when_empty(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "macro3.db")
    assert store.get_latest_macro("DGS10") is None


def test_write_and_read_regime_snapshot(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from src.data.regime import (
        MarketRegime,
        OverallRegime,
        VolatilityRegime,
        YieldCurveRegime,
    )

    store = SQLiteStore(tmp_path / "reg.db")
    mr = MarketRegime(
        timestamp=datetime(2026, 4, 1, 16, 0, tzinfo=UTC),
        vix_close=18.0,
        yield_spread=0.4,
        yield_curve=YieldCurveRegime.FLAT,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.CAUTIOUS,
        sizing_multiplier=0.75,
    )
    store.write_regime_snapshot(mr)
    hist = store.get_regime_history(limit=5)
    assert len(hist) == 1
    assert hist[0]["overall"] == "cautious"
    assert float(hist[0]["sizing_multiplier"]) == pytest.approx(0.75)


def test_get_latest_regime_returns_newest(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from src.data.regime import (
        MarketRegime,
        OverallRegime,
        VolatilityRegime,
        YieldCurveRegime,
    )

    store = SQLiteStore(tmp_path / "reg2.db")
    for d in (1, 2, 3):
        store.write_regime_snapshot(
            MarketRegime(
                timestamp=datetime(2026, 4, d, 16, 0, tzinfo=UTC),
                vix_close=float(15 + d),
                yield_spread=1.0,
                yield_curve=YieldCurveRegime.NORMAL,
                volatility=VolatilityRegime.NORMAL,
                overall=OverallRegime.RISK_ON,
                sizing_multiplier=1.0,
            ),
        )
    latest = store.get_latest_regime()
    assert latest is not None
    assert latest["date"] == "2026-04-03"


def test_write_and_read_fundamental_score(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "fs.db")
    store.write_fundamental_score(
        "AAPL",
        "2025-12-31",
        7,
        {"profitability": 4, "leverage": 2, "efficiency": 1},
    )
    rows = store.get_fundamental_scores(symbol="AAPL")
    assert len(rows) == 1
    assert rows[0]["score"] == 7
    assert rows[0]["data"]["profitability"] == 4


def test_get_latest_fundamental_score(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "fs2.db")
    store.write_fundamental_score("MSFT", "2024-12-31", 5, {})
    store.write_fundamental_score("MSFT", "2025-12-31", 8, {})
    latest = store.get_latest_fundamental_score("MSFT")
    assert latest is not None
    assert latest["period"] == "2025-12-31"
    assert latest["score"] == 8


def test_macro_indicator_upsert_updates_value(tmp_path: Path) -> None:
    """M19: Writing the same (date, series_id) updates the value, not duplicates."""
    store = SQLiteStore(tmp_path / "upsert.db")
    store.write_macro_indicator("2024-06-01", "DGS10", 4.5)
    store.write_macro_indicator("2024-06-01", "DGS10", 4.8)
    rows = store.get_macro_indicator("DGS10")
    assert len(rows) == 1
    assert rows[0]["value"] == pytest.approx(4.8)


def test_regime_snapshot_upsert_updates_value(tmp_path: Path) -> None:
    """M19: Writing two regime snapshots on the same date updates, not duplicates."""
    from src.data.regime import (
        MarketRegime,
        OverallRegime,
        VolatilityRegime,
        YieldCurveRegime,
    )

    store = SQLiteStore(tmp_path / "upsert2.db")
    ts = datetime(2026, 4, 1, 16, 0, tzinfo=UTC)
    store.write_regime_snapshot(
        MarketRegime(
            timestamp=ts,
            vix_close=18.0,
            yield_spread=1.0,
            yield_curve=YieldCurveRegime.NORMAL,
            volatility=VolatilityRegime.NORMAL,
            overall=OverallRegime.RISK_ON,
            sizing_multiplier=1.0,
        ),
    )
    store.write_regime_snapshot(
        MarketRegime(
            timestamp=ts,
            vix_close=35.0,
            yield_spread=-1.0,
            yield_curve=YieldCurveRegime.INVERTED,
            volatility=VolatilityRegime.CRISIS,
            overall=OverallRegime.CRISIS,
            sizing_multiplier=0.25,
        ),
    )
    history = store.get_regime_history()
    assert len(history) == 1
    assert history[0]["overall"] == "crisis"
    assert history[0]["sizing_multiplier"] == pytest.approx(0.25)


def test_fundamental_scores_no_data_json_key_leak(tmp_path: Path) -> None:
    """M9: Returned dicts should have 'data' key, not 'data_json'."""
    store = SQLiteStore(tmp_path / "leak.db")
    store.write_fundamental_score("AAPL", "2025-12-31", 6, {"k": "v"})
    rows = store.get_fundamental_scores(symbol="AAPL")
    assert len(rows) == 1
    assert "data" in rows[0]
    assert "data_json" not in rows[0]
    latest = store.get_latest_fundamental_score("AAPL")
    assert latest is not None
    assert "data" in latest
    assert "data_json" not in latest


def test_write_ml_score_and_get_ml_scores(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ml.db")
    store.write_ml_score(
        cycle_id="c1",
        symbol="spy",
        score=0.62,
        model_version="1",
        computed_at="2026-04-01T12:00:00+00:00",
        top_features_json="SHAP: ret_1: +0.01",
    )
    rows = store.get_ml_scores(limit=10)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SPY"
    assert rows[0]["score"] == pytest.approx(0.62)
    assert rows[0]["top_features_json"] == "SHAP: ret_1: +0.01"


def test_get_latest_ml_scores_batch_empty_table(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ml_empty.db")
    assert store.get_latest_ml_scores_batch() == []


def test_get_latest_ml_scores_batch_uses_newest_cycle(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ml2.db")
    store.write_ml_score(
        cycle_id="old",
        symbol="SPY",
        score=0.1,
        model_version="1",
        computed_at="2026-03-01T00:00:00+00:00",
    )
    store.write_ml_score(
        cycle_id="new",
        symbol="GLD",
        score=0.2,
        model_version="1",
        computed_at="2026-04-02T00:00:00+00:00",
    )
    store.write_ml_score(
        cycle_id="new",
        symbol="QQQ",
        score=0.3,
        model_version="1",
        computed_at="2026-04-02T00:00:01+00:00",
    )
    batch = store.get_latest_ml_scores_batch()
    syms = {r["symbol"] for r in batch}
    assert syms == {"GLD", "QQQ"}
    assert all(r["cycle_id"] == "new" for r in batch)
    assert batch[0]["symbol"] == "GLD"
    assert batch[1]["symbol"] == "QQQ"


def test_migration_adds_account_id_with_default(tmp_path: Path) -> None:
    """Pre-tier-30 DBs without ``account_id`` get the column and existing rows default."""
    db = tmp_path / "legacy_hub.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE trade_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            weight REAL NOT NULL,
            confidence REAL NOT NULL,
            strategy_name TEXT,
            rationale TEXT NOT NULL
        );
        """,
    )
    conn.execute(
        """
        INSERT INTO trade_signals(
            cycle_id, timestamp, symbol, direction, weight, confidence,
            strategy_name, rationale
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "c-legacy",
            "2026-01-01T12:00:00+00:00",
            "SPY",
            "long",
            1.0,
            0.9,
            "S",
            "legacy row",
        ),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path=db)
    with sqlite3.connect(str(db)) as c2:
        cols = {str(r[1]) for r in c2.execute("PRAGMA table_info(trade_signals)").fetchall()}
        assert "account_id" in cols
        row = c2.execute("SELECT account_id FROM trade_signals LIMIT 1").fetchone()
        assert row is not None
        assert str(row[0]) == "default"

    rows = store.get_signals(limit=5, account_id="default")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SPY"


def test_sqlite_signals_filtered_by_account(tmp_path: Path) -> None:
    """Signals logged under different ``account_id`` values partition in queries."""
    store = SQLiteStore(tmp_path / "acct_sig.db")
    ts = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    base = dict(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=0.9,
        rationale="r",
        timestamp=ts,
        strategy_name="S",
    )
    store.log_signal("c1", Signal(**base), account_id="acct_a")
    store.log_signal("c1", Signal(**base), account_id="acct_b")
    assert store.count_signals() == 2
    assert store.count_signals(account_id="acct_a") == 1
    assert store.count_signals(account_id="acct_b") == 1
    ra = store.get_signals(limit=10, account_id="acct_a")
    rb = store.get_signals(limit=10, account_id="acct_b")
    assert len(ra) == 1 and len(rb) == 1
    assert ra[0]["account_id"] == "acct_a"
    assert rb[0]["account_id"] == "acct_b"
