"""Tests for end-to-end trading workflow (dependencies mocked)."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.config import (
    DataConfig,
    MeanReversionConfig,
    MomentumConfig,
    RegimeConfig,
    Settings,
    StrategyConfig,
    TaxConfig,
)
from src.data.pipeline import IngestResult
from src.data.regime import (
    MarketRegime,
    OverallRegime,
    VolatilityRegime,
    YieldCurveRegime,
)
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.data.symbol_resolver import resolve_required_symbols
from src.execution.order_manager import OrderManager
from src.models import OrderExecutionResult, Signal
from src.portfolio.tax_lots import LotLedger
from src.risk.kill_switch import KillSwitch
from src.strategy.base import Strategy
from src.strategy.dca import DCAStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumRotationStrategy


def test_run_trading_cycle_ingests_universe_runs_dca_and_executes(
    settings: Settings,
) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.strategy.dca.percent_of_equity = None  # exercise the fixed-amount budget path
    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)

    close_price = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    strategies = [DCAStrategy(settings)]
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=strategies,
        order_manager=order_mgr,
        broker=broker,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    result = wf.run_cycle(as_of=monday)

    expected = set(resolve_required_symbols(settings))
    vix = settings.regime.vix_symbol.strip().upper()
    if settings.regime.enabled and vix and vix not in expected:
        expected.add(vix)
    assert pipeline.ingest_ohlcv.call_count == len(expected)
    for sym in expected:
        pipeline.ingest_ohlcv.assert_any_call(sym)

    assert len(result.signals) >= 1
    assert all(s.strategy_name == "DCAStrategy" for s in result.signals)

    order_mgr.execute_signals.assert_called_once()
    call_kw = order_mgr.execute_signals.call_args.kwargs
    assert call_kw["equity"] == 10_000.0
    assert call_kw["cash"] == 5_000.0
    assert call_kw["dca_budget"] == pytest.approx(float(settings.strategy.dca.amount))
    dca_syms = [t.symbol.strip().upper() for t in settings.data.dca_targets]
    for sym in dca_syms:
        assert call_kw["last_prices"][sym] == close_price


def test_run_cycle_uses_percent_of_equity_for_dca_budget(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.strategy.dca.percent_of_equity = 0.005

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv

    broker = MagicMock()
    broker.get_account_equity.return_value = 100_000.0
    broker.get_last_equity.return_value = 100_000.0
    broker.get_cash.return_value = 50_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    wf.run_cycle(as_of=monday)

    call_kw = order_mgr.execute_signals.call_args.kwargs
    assert call_kw["dca_budget"] == pytest.approx(500.0)


def test_dca_percent_of_equity_with_cautious_regime_no_double_scaling(
    settings: Settings,
) -> None:
    """Budget is equity*pct (no regime); strategy weights apply regime once -> VOO = 0.6*500*0.75."""
    from unittest.mock import patch

    from src.automation.workflows import TradingWorkflow
    from src.data.regime import MarketRegime, OverallRegime, VolatilityRegime, YieldCurveRegime
    from src.risk.position_sizing import compute_order_notional

    s = settings.model_copy(
        update={
            "regime": settings.regime.model_copy(update={"enabled": True}),
            "strategy": settings.strategy.model_copy(
                update={
                    "dca": settings.strategy.dca.model_copy(
                        update={"percent_of_equity": 0.005},
                    ),
                },
            ),
        },
    )

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)

    store = MagicMock()
    ohlcv = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1e6],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store.read_ohlcv.side_effect = lambda _sym: ohlcv

    broker = MagicMock()
    broker.get_account_equity.return_value = 100_000.0
    broker.get_last_equity.return_value = 100_000.0
    broker.get_cash.return_value = 50_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    sqlite = MagicMock()
    mr = MarketRegime(
        timestamp=datetime(2026, 3, 30, 16, 0, tzinfo=UTC),
        vix_close=20.0,
        yield_spread=1.0,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.CAUTIOUS,
        sizing_multiplier=0.75,
    )
    with patch("src.data.regime.RegimeDetector") as mrd:
        mrd.return_value.detect.return_value = mr
        wf = TradingWorkflow(
            settings=s,
            data_pipeline=pipeline,
            parquet_store=store,
            strategies=[DCAStrategy(s)],
            order_manager=order_mgr,
            broker=broker,
            sqlite_store=sqlite,
            kill_switch=MagicMock(),
        )
        wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))

    args, call_kw = order_mgr.execute_signals.call_args
    signals = args[0]
    assert call_kw["dca_budget"] == pytest.approx(500.0)

    voo = next(
        x
        for x in signals
        if x.strategy_name == "DCAStrategy" and x.symbol.strip().upper() == "VOO"
    )
    assert voo.weight == pytest.approx(0.45)
    n = compute_order_notional(
        signal_weight=voo.weight,
        equity=100_000.0,
        max_position_pct=float(s.risk.max_position_pct),
        dca_budget=float(call_kw["dca_budget"]),
        use_dca_budget=True,
    )
    assert n == pytest.approx(225.0)


def test_run_cycle_calls_cancel_stale_orders_when_sqlite_wired(
    settings: Settings,
    tmp_path: Path,
) -> None:
    from src.automation.workflows import TradingWorkflow

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    store.read_ohlcv.return_value = pd.DataFrame()

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.cancel_stale_orders.return_value = []
    order_mgr.execute_signals.return_value = []

    sqlite = SQLiteStore(tmp_path / "hub.db")
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    wf.run_cycle(as_of=monday)
    order_mgr.cancel_stale_orders.assert_called_once_with(
        timeout_minutes=int(settings.execution.unfilled_timeout_minutes),
    )


def test_run_trading_cycle_skips_dca_when_not_contribution_day(
    settings: Settings,
) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.strategy.dca.frequency = "weekly"  # weekly DCA so a Wednesday is skipped
    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    store.read_ohlcv.return_value = pd.DataFrame()

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
    )
    # Wednesday — weekly DCA only on Monday
    wed = datetime(2026, 4, 1, 16, 0, tzinfo=UTC)
    result = wf.run_cycle(as_of=wed)

    assert result.signals == []
    order_mgr.execute_signals.assert_called_once()
    args, kwargs = order_mgr.execute_signals.call_args
    assert args[0] == []
    assert kwargs["last_prices"] == {}
    assert kwargs["dca_budget"] is None
    assert kwargs["equity"] == 10_000.0


def test_run_ingest_only_calls_pipeline_for_each_universe_symbol(
    settings: Settings,
) -> None:
    """Data cron ingests the canonical resolved symbol set plus VIX when regime needs it."""
    from src.automation.workflows import TradingWorkflow

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    broker = MagicMock()
    broker.get_last_equity.return_value = 0.0
    order_mgr = MagicMock()

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
    )
    wf.run_ingest_only()

    expected = set(resolve_required_symbols(settings))
    vix = settings.regime.vix_symbol.strip().upper()
    if settings.regime.enabled and vix and vix not in expected:
        expected.add(vix)
    assert pipeline.ingest_ohlcv.call_count == len(expected)
    for sym in expected:
        pipeline.ingest_ohlcv.assert_any_call(sym)
    store.read_ohlcv.assert_not_called()
    order_mgr.execute_signals.assert_not_called()


def test_workflow_ingests_momentum_cash_symbol_during_cycle(tmp_path: Path) -> None:
    """Resolver includes momentum cash (e.g. SHV) even when strategies list is DCA-only."""
    from src.automation.workflows import TradingWorkflow

    yml = tmp_path / "missing.yml"
    assert not yml.exists()
    s = Settings(
        _yaml_path=yml,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "c"),
            universe=["SPY"],
            dca_targets=[],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol="SHV"),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
        regime=RegimeConfig(enabled=False),
    )

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    broker = MagicMock()
    order_mgr = MagicMock()

    wf = TradingWorkflow(
        settings=s,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(s)],
        order_manager=order_mgr,
        broker=broker,
    )
    wf.run_ingest_only()
    syms = {c[0][0] for c in pipeline.ingest_ohlcv.call_args_list}
    assert "SHV" in syms
    assert "SPY" in syms


def test_workflow_ingests_mean_reversion_universe_during_cycle(tmp_path: Path) -> None:
    """Mean reversion tickers are ingested even when strategies list omits mean reversion."""
    from src.automation.workflows import TradingWorkflow

    yml = tmp_path / "missing2.yml"
    s = Settings(
        _yaml_path=yml,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "c"),
            universe=["SPY"],
            dca_targets=[],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=["ZZZ", "AAA"]),
        ),
        tax=TaxConfig(replacement_map={}),
        regime=RegimeConfig(enabled=False),
    )

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    broker = MagicMock()
    order_mgr = MagicMock()

    wf = TradingWorkflow(
        settings=s,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(s)],
        order_manager=order_mgr,
        broker=broker,
    )
    wf.run_ingest_only()
    syms = {c[0][0] for c in pipeline.ingest_ohlcv.call_args_list}
    assert "ZZZ" in syms
    assert "AAA" in syms


def test_run_cycle_uses_stale_parquet_when_one_ingest_fails(
    settings: Settings,
) -> None:
    """Failed fetch for one symbol does not block signals if Parquet still has bars."""
    from src.automation.workflows import TradingWorkflow

    def ingest_side_effect(sym: str) -> IngestResult:
        # Universe is sorted: BND, VOO, VXUS — fail first symbol only.
        if sym.strip().upper() == "BND":
            return IngestResult(success=False, error="upstream timeout")
        return IngestResult(success=True)

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.side_effect = ingest_side_effect

    close_price = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    result = wf.run_cycle(as_of=monday)

    assert any(not ir.success for _, ir in result.ingest_results)
    assert len(result.signals) >= 1
    order_mgr.execute_signals.assert_called_once()


class ScriptedMomentum(MomentumRotationStrategy):
    """Test double: fixed momentum signals with real ``get_universe`` / rotation hooks."""

    def __init__(self, settings: Settings, fixed_signals: list[Signal]) -> None:
        super().__init__(settings)
        self._fixed_signals = fixed_signals

    def generate_signals(self, data, *, as_of=None, **kwargs):
        return list(self._fixed_signals)


class ScriptedMR(MeanReversionStrategy):
    """Test double: fixed mean-reversion signals."""

    def __init__(self, settings: Settings, fixed_signals: list[Signal]) -> None:
        super().__init__(settings)
        self._fixed_signals = fixed_signals

    def generate_signals(self, data, *, as_of=None, **kwargs):
        return list(self._fixed_signals)


def test_workflow_logs_cycle_to_sqlite(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
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
    order_mgr.execute_signals.return_value = [
        OrderExecutionResult(symbol="VOO", submitted=True, order_id="a", side="buy"),
        OrderExecutionResult(symbol="VXUS", submitted=True, order_id="b", side="buy"),
        OrderExecutionResult(symbol="BND", submitted=True, order_id="c", side="buy"),
    ]
    sqlite = MagicMock()

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    result = wf.run_cycle(as_of=monday)

    assert sqlite.log_signal.call_count == len(result.signals)
    assert sqlite.log_execution.call_count == len(result.execution_results)
    sig_cycle_ids = {c.args[0] for c in sqlite.log_signal.call_args_list}
    ex_cycle_ids = {c.args[0] for c in sqlite.log_execution.call_args_list}
    assert len(sig_cycle_ids) == 1
    assert sig_cycle_ids == ex_cycle_ids


def test_workflow_sends_cycle_summary_via_notifier(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow
    from src.risk.kill_switch import KillSwitch

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    pq = MagicMock()
    pq.read_ohlcv.return_value = pd.DataFrame()

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    notifier = MagicMock()
    ks = KillSwitch(settings.risk.daily_loss_limit_pct)
    sqlite = MagicMock()

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
        kill_switch=ks,
        notifier=notifier,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    wf.run_cycle(as_of=monday)

    messages = [c.args[0].message for c in notifier.send.call_args_list if c.args]
    assert any("Cycle summary" in m for m in messages)


def test_workflow_sells_old_momentum_position_before_buying_new(
    settings: Settings,
) -> None:
    from src.automation.workflows import TradingWorkflow

    sig = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="Rotate to QQQ.",
        timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    mom = ScriptedMomentum(settings, [sig])
    dca = MagicMock(spec=Strategy)
    dca.get_universe.return_value = []
    dca.generate_signals.return_value = []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 300.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
    )
    pq = MagicMock()
    pq.read_ohlcv.return_value = ohlcv

    def pos(sym: str) -> float:
        return {"SPY": 10.0, "QQQ": 0.0, "SHV": 0.0}.get(str(sym).strip().upper(), 0.0)

    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.side_effect = pos

    order_mgr = MagicMock()
    order_mgr.close_position.return_value = OrderExecutionResult(
        symbol="SPY",
        submitted=True,
        order_id="sell-1",
        side="sell",
        qty=10.0,
    )
    order_mgr.execute_signals.return_value = [
        OrderExecutionResult(symbol="QQQ", submitted=True, order_id="buy-1", side="buy", qty=1.0),
    ]

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[mom, dca],
        order_manager=order_mgr,
        broker=broker,
    )
    wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))

    order_mgr.close_position.assert_called_once()
    assert order_mgr.close_position.call_args[0][0] == "SPY"
    assert order_mgr.close_position.call_args[0][1] == 10.0
    order_mgr.execute_signals.assert_called_once()


def test_workflow_does_not_sell_dca_positions_during_rotation(
    settings: Settings,
) -> None:
    from src.automation.workflows import TradingWorkflow

    sig = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="Rotate to QQQ.",
        timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    mom = ScriptedMomentum(settings, [sig])
    dca = MagicMock(spec=Strategy)
    dca.get_universe.return_value = ["VOO"]
    dca.generate_signals.return_value = []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    px = 400.0
    ohlcv = pd.DataFrame(
        {
            "open": [px],
            "high": [px],
            "low": [px],
            "close": [px],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
    )
    pq = MagicMock()
    pq.read_ohlcv.return_value = ohlcv

    def pos(sym: str) -> float:
        u = str(sym).strip().upper()
        if u == "SPY":
            return 10.0
        if u == "VOO":
            return 5.0
        return 0.0

    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.side_effect = pos

    order_mgr = MagicMock()
    order_mgr.close_position.return_value = OrderExecutionResult(
        symbol="SPY",
        submitted=True,
        order_id="s",
        side="sell",
        qty=10.0,
    )
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[mom, dca],
        order_manager=order_mgr,
        broker=broker,
    )
    when = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
    wf.run_cycle(as_of=when)

    order_mgr.close_position.assert_called_once_with(
        "SPY",
        10.0,
        price=px,
        equity=50_000.0,
        daily_pnl_pct=0.0,
        as_of=when,
        wait_for_fill=True,
    )


def test_workflow_records_buy_lot_after_execution(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.cash_sweep.enabled = False  # not a sweep test (Tier 44)

    one_sig = Signal(
        symbol="VOO",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="One lot test.",
        timestamp=datetime(2026, 3, 30, 16, 0, tzinfo=UTC),
        strategy_name="TestStrat",
    )
    strat = MagicMock(spec=Strategy)
    strat.get_universe.return_value = ["VOO"]
    strat.generate_signals.return_value = [one_sig]

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = [
        OrderExecutionResult(
            symbol="VOO",
            submitted=True,
            order_id="a",
            side="buy",
            qty=2.5,
        ),
    ]
    ledger = MagicMock(spec=LotLedger)
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[strat],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    wf.run_cycle(as_of=monday)
    ledger.record_buy.assert_called_once()
    args, _kwargs = ledger.record_buy.call_args
    assert args[0] == "VOO"
    assert args[1] == pytest.approx(2.5)
    assert args[2] == pytest.approx(close_price)


def test_workflow_records_sell_lot_after_rotation(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.cash_sweep.enabled = False  # not a sweep test (Tier 44)

    sig = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="Rotate to QQQ.",
        timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    mom = ScriptedMomentum(settings, [sig])
    dca = MagicMock(spec=Strategy)
    dca.get_universe.return_value = []
    dca.generate_signals.return_value = []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 300.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
    )
    pq = MagicMock()
    pq.read_ohlcv.return_value = ohlcv

    def pos(sym: str) -> float:
        return {"SPY": 10.0, "QQQ": 0.0, "SHV": 0.0}.get(str(sym).strip().upper(), 0.0)

    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.side_effect = pos

    order_mgr = MagicMock()
    order_mgr.close_position.return_value = OrderExecutionResult(
        symbol="SPY",
        submitted=True,
        order_id="sell-1",
        side="sell",
        qty=10.0,
    )
    order_mgr.execute_signals.return_value = []
    ledger = MagicMock(spec=LotLedger)
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[mom, dca],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))
    ledger.record_sell.assert_called_once()
    args, _kwargs = ledger.record_sell.call_args
    assert args[0] == "SPY"
    assert args[1] == pytest.approx(10.0)
    assert args[2] == pytest.approx(close_price)


def test_workflow_uses_fill_price_for_lot_buy(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.cash_sweep.enabled = False  # not a sweep test (Tier 44)

    one_sig = Signal(
        symbol="VOO",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="Fill price test.",
        timestamp=datetime(2026, 3, 30, 16, 0, tzinfo=UTC),
        strategy_name="TestStrat",
    )
    strat = MagicMock(spec=Strategy)
    strat.get_universe.return_value = ["VOO"]
    strat.generate_signals.return_value = [one_sig]
    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = [
        OrderExecutionResult(
            symbol="VOO",
            submitted=True,
            order_id="a",
            side="buy",
            qty=2.5,
            fill_price=250.25,
        ),
    ]
    ledger = MagicMock(spec=LotLedger)
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[strat],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
    ledger.record_buy.assert_called_once()
    assert ledger.record_buy.call_args[0][2] == pytest.approx(250.25)


def test_workflow_falls_back_to_parquet_close_when_no_fill_price(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.cash_sweep.enabled = False  # not a sweep test (Tier 44)

    one_sig = Signal(
        symbol="VOO",
        direction="long",
        weight=0.05,
        confidence=1.0,
        rationale="Fallback test.",
        timestamp=datetime(2026, 3, 30, 16, 0, tzinfo=UTC),
        strategy_name="TestStrat",
    )
    strat = MagicMock(spec=Strategy)
    strat.get_universe.return_value = ["VOO"]
    strat.generate_signals.return_value = [one_sig]
    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 88.5
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = [
        OrderExecutionResult(
            symbol="VOO",
            submitted=True,
            order_id="a",
            side="buy",
            qty=2.5,
            fill_price=None,
        ),
    ]
    ledger = MagicMock(spec=LotLedger)
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[strat],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
    ledger.record_buy.assert_called_once()
    assert ledger.record_buy.call_args[0][2] == pytest.approx(close_price)


def test_workflow_warns_when_multiple_momentum_signals(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from src.automation.workflows import TradingWorkflow

    s1 = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="A.",
        timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    s2 = Signal(
        symbol="IWM",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="B.",
        timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    mom = ScriptedMomentum(settings, [s1, s2])
    dca = MagicMock(spec=Strategy)
    dca.get_universe.return_value = []
    dca.generate_signals.return_value = []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    px = 300.0
    ohlcv = pd.DataFrame(
        {
            "open": [px],
            "high": [px],
            "low": [px],
            "close": [px],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
    )
    pq = MagicMock()
    pq.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[mom, dca],
        order_manager=order_mgr,
        broker=broker,
    )
    with caplog.at_level(logging.WARNING):
        wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))
    assert "Multiple momentum" in caplog.text


def test_workflow_executes_harvest_sells_when_enabled(settings: Settings) -> None:
    from unittest.mock import patch

    from src.automation.workflows import TradingWorkflow

    settings.cash_sweep.enabled = False  # not a sweep test (Tier 44)
    from src.strategy.tax_loss_harvest import HarvestCandidate

    settings.tax.harvest_enabled = True
    settings.tax.harvest_threshold_pct = 0.01
    settings.tax.harvest_min_loss_dollars = 1.0

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    px = 80.0
    ohlcv = pd.DataFrame(
        {
            "open": [px],
            "high": [px],
            "low": [px],
            "close": [px],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    order_mgr.close_position.return_value = OrderExecutionResult(
        symbol="SPY",
        submitted=True,
        order_id="o1",
        side="sell",
        qty=1.0,
        fill_price=80.0,
    )
    ledger = MagicMock()
    cand = HarvestCandidate(
        lot_id="lid1",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        current_price=80.0,
        unrealized_loss=-20.0,
        unrealized_loss_pct=-0.2,
        replacement_symbol=None,
        reason="test",
    )
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    with patch(
        "src.strategy.tax_loss_harvest.TaxLossHarvester.find_candidates",
        return_value=[cand],
    ):
        wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))
    order_mgr.close_position.assert_called_once()
    ledger.record_sell_lot.assert_called_once()


def test_workflow_skips_harvest_when_disabled(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.tax.harvest_enabled = False
    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    store.read_ohlcv.return_value = pd.DataFrame()
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    ledger = MagicMock()
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))
    ledger.record_sell_lot.assert_not_called()


def test_workflow_generates_replacement_buy_after_harvest(settings: Settings) -> None:
    from unittest.mock import patch

    from src.automation.workflows import TradingWorkflow

    settings.cash_sweep.enabled = False  # not a sweep test (Tier 44)
    from src.strategy.tax_loss_harvest import HarvestCandidate

    settings.tax.harvest_enabled = True
    settings.tax.harvest_threshold_pct = 0.01
    settings.tax.harvest_min_loss_dollars = 1.0

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)

    def _read(sym: str) -> pd.DataFrame:
        px = 80.0 if sym == "SPY" else 150.0
        return pd.DataFrame(
            {
                "open": [px],
                "high": [px],
                "low": [px],
                "close": [px],
                "volume": [1_000_000.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
        )

    store = MagicMock()
    store.read_ohlcv.side_effect = _read
    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    order_mgr.close_position.return_value = OrderExecutionResult(
        symbol="SPY",
        submitted=True,
        order_id="o1",
        side="sell",
        qty=1.0,
        fill_price=80.0,
    )
    ledger = MagicMock()
    cand = HarvestCandidate(
        lot_id="lid1",
        symbol="SPY",
        qty=1.0,
        cost_per_share=100.0,
        current_price=80.0,
        unrealized_loss=-20.0,
        unrealized_loss_pct=-0.2,
        replacement_symbol="VOO",
        reason="test",
    )
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    with patch(
        "src.strategy.tax_loss_harvest.TaxLossHarvester.find_candidates",
        return_value=[cand],
    ):
        wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))
    call_sigs = order_mgr.execute_signals.call_args[0][0]
    assert any(s.symbol == "VOO" and s.strategy_name == "TaxLossHarvest" for s in call_sigs)


def test_workflow_passes_lot_selection_method_to_record_sell(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.cash_sweep.enabled = False  # not a sweep test (Tier 44)
    settings.tax.lot_selection_method = "hifo"
    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    px = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [px],
            "high": [px],
            "low": [px],
            "close": [px],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.side_effect = lambda s: 5.0 if str(s).upper() == "QQQ" else 0.0
    order_mgr = MagicMock()
    order_mgr.close_position.return_value = OrderExecutionResult(
        symbol="QQQ",
        submitted=True,
        order_id="x",
        side="sell",
        qty=5.0,
        fill_price=px,
    )
    order_mgr.execute_signals.return_value = []
    ledger = MagicMock()
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="m",
        timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    mom = ScriptedMomentum(settings, [sig])

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[mom],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))
    ledger.record_sell.assert_called_once()
    assert ledger.record_sell.call_args.kwargs.get("method") == "hifo"


def test_workflow_ingests_fred_series_when_regime_enabled(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.regime.enabled = True
    settings.regime.fred_series = ["DGS10", "DGS2"]

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    fred_pl = MagicMock()
    fred_pl.ingest_macro.return_value = IngestResult(success=True)

    store = MagicMock()
    store.read_ohlcv.return_value = pd.DataFrame()

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=MagicMock(),
        kill_switch=MagicMock(),
        fred_pipeline=fred_pl,
    )
    wf.run_ingest_only()
    assert fred_pl.ingest_macro.call_count == 2
    fred_pl.ingest_macro.assert_any_call("DGS10")
    fred_pl.ingest_macro.assert_any_call("DGS2")


def test_workflow_skips_fred_when_no_fred_pipeline(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.regime.enabled = True

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    store.read_ohlcv.return_value = pd.DataFrame()

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        fred_pipeline=None,
    )
    wf.run_ingest_only()
    assert pipeline.ingest_ohlcv.called


def test_workflow_detects_regime_and_passes_multiplier(settings: Settings) -> None:
    from unittest.mock import patch

    from src.automation.workflows import TradingWorkflow
    from src.data.regime import MarketRegime, OverallRegime, VolatilityRegime, YieldCurveRegime

    settings.regime.enabled = True

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)

    store = MagicMock()
    ohlcv = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1e6],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store.read_ohlcv.side_effect = lambda _s: ohlcv

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    sqlite = MagicMock()
    mr = MarketRegime(
        timestamp=datetime(2026, 3, 30, 16, 0, tzinfo=UTC),
        vix_close=20.0,
        yield_spread=1.0,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.CAUTIOUS,
        sizing_multiplier=0.75,
    )
    with patch("src.data.regime.RegimeDetector") as mrd:
        mrd.return_value.detect.return_value = mr
        wf = TradingWorkflow(
            settings=settings,
            data_pipeline=pipeline,
            parquet_store=store,
            strategies=[DCAStrategy(settings)],
            order_manager=order_mgr,
            broker=broker,
            sqlite_store=sqlite,
            kill_switch=MagicMock(),
        )
        wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))

    order_mgr.execute_signals.assert_called_once()
    assert order_mgr.execute_signals.call_args.kwargs["regime_multiplier"] == pytest.approx(0.75)
    sqlite.write_regime_snapshot.assert_called_once()


def test_workflow_uses_multiplier_1_when_regime_disabled(settings: Settings) -> None:
    from src.automation.workflows import TradingWorkflow

    settings.regime.enabled = False

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    ohlcv = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1e6],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store.read_ohlcv.side_effect = lambda _s: ohlcv

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=MagicMock(),
        kill_switch=MagicMock(),
    )
    wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
    assert order_mgr.execute_signals.call_args.kwargs["regime_multiplier"] == pytest.approx(1.0)


def test_workflow_tlh_excludes_protected_momentum_symbol(settings: Settings) -> None:
    """M20: A symbol with an active momentum long signal is protected from TLH."""
    from unittest.mock import patch

    settings.cash_sweep.enabled = False  # not a sweep test (Tier 44)

    from src.automation.workflows import TradingWorkflow
    from src.strategy.tax_loss_harvest import TaxLossHarvester

    settings.tax.harvest_enabled = True
    settings.tax.harvest_threshold_pct = 0.01
    settings.tax.harvest_min_loss_dollars = 1.0
    settings.regime.enabled = False

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    px = 80.0
    ohlcv = pd.DataFrame(
        {
            "open": [px],
            "high": [px],
            "low": [px],
            "close": [px],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    ledger = MagicMock()

    mom_signal = Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="momentum top pick",
        timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    strat_mock = MagicMock()
    strat_mock.get_universe.return_value = ["SPY"]
    strat_mock.generate_signals.return_value = [mom_signal]

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[strat_mock],
        order_manager=order_mgr,
        broker=broker,
        lot_ledger=ledger,
    )
    with patch.object(
        TaxLossHarvester,
        "find_candidates",
        return_value=[],
    ) as mock_find:
        wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))
        assert mock_find.called
        protected_arg = mock_find.call_args[0][1]
        assert "SPY" in protected_arg


def test_workflow_uses_llm_explanation_when_available(settings: Settings) -> None:
    """When llm_client returns a result, it is used instead of the template."""
    from unittest.mock import patch

    from src.automation.workflows import TradingWorkflow

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    ohlcv = pd.DataFrame(
        {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1e6]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    sqlite = MagicMock()
    llm = MagicMock()
    llm.generate.return_value = None

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
        llm_client=llm,
    )
    with patch(
        "src.ai.llm_explainer.generate_llm_explanation",
        return_value="LLM said buy",
    ) as mock_llm_expl:
        wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
        assert mock_llm_expl.called
    logged_expl = sqlite.log_signal.call_args_list[0].kwargs.get("explanation", "")
    assert logged_expl == "LLM said buy"


def test_workflow_falls_back_to_template_when_llm_returns_none(settings: Settings) -> None:
    """When llm_client returns None, the template explanation is used."""
    from unittest.mock import patch

    from src.automation.workflows import TradingWorkflow

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    ohlcv = pd.DataFrame(
        {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1e6]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    sqlite = MagicMock()
    llm = MagicMock()

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
        llm_client=llm,
    )
    with patch(
        "src.ai.llm_explainer.generate_llm_explanation",
        return_value=None,
    ):
        wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
    logged_expl = sqlite.log_signal.call_args_list[0].kwargs.get("explanation", "")
    assert len(logged_expl) > 0
    assert "LLM" not in logged_expl


def test_workflow_falls_back_to_template_when_no_llm_client(settings: Settings) -> None:
    """When no llm_client is provided, template explanations are used."""
    from src.automation.workflows import TradingWorkflow

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    ohlcv = pd.DataFrame(
        {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1e6]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    sqlite = MagicMock()

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
    )
    wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
    logged_expl = sqlite.log_signal.call_args_list[0].kwargs.get("explanation", "")
    assert len(logged_expl) > 0


def test_workflow_passes_market_regime_to_strategies_when_regime_enabled(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RegimeDetector path supplies the same ``market_regime`` to every strategy."""
    from src.automation.workflows import TradingWorkflow

    settings = settings.model_copy(
        update={"regime": settings.regime.model_copy(update={"enabled": True})},
    )

    captured: dict[str, object] = {}

    class RecordingDCA(DCAStrategy):
        def generate_signals(self, data, *, as_of=None, market_regime=None):
            captured["dca"] = market_regime
            return []

    class RecordingMR(MeanReversionStrategy):
        def generate_signals(self, data, *, as_of=None, market_regime=None):
            captured["mr"] = market_regime
            return []

    monday = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    mr = MarketRegime(
        timestamp=monday,
        vix_close=22.0,
        yield_spread=None,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.RISK_ON,
        sizing_multiplier=1.0,
    )

    class FakeDetector:
        def __init__(self, _settings: Settings) -> None:
            pass

        def detect(self, **_kw: object) -> MarketRegime:
            return mr

    monkeypatch.setattr("src.data.regime.RegimeDetector", FakeDetector)

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    sqlite = MagicMock()

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[RecordingDCA(settings), RecordingMR(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
    )
    wf.run_cycle(as_of=monday)

    assert captured["dca"] is mr
    assert captured["mr"] is mr
    assert mr.vix_close == pytest.approx(22.0)
    assert mr.overall is OverallRegime.RISK_ON
    assert mr.volatility is VolatilityRegime.NORMAL
    sqlite.write_regime_snapshot.assert_called_once_with(mr)


def test_workflow_passes_none_market_regime_when_regime_disabled(
    settings: Settings,
) -> None:
    """When ``regime.enabled`` is false, strategies receive ``market_regime=None``."""
    from src.automation.workflows import TradingWorkflow

    s = settings.model_copy(
        update={"regime": settings.regime.model_copy(update={"enabled": False})},
    )
    captured: list[object] = []

    class RecordingDCA(DCAStrategy):
        def generate_signals(self, data, *, as_of=None, market_regime=None):
            captured.append(market_regime)
            return []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    close_price = 100.0
    ohlcv = pd.DataFrame(
        {
            "open": [close_price],
            "high": [close_price],
            "low": [close_price],
            "close": [close_price],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=s,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[RecordingDCA(s)],
        order_manager=order_mgr,
        broker=broker,
    )
    wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))

    assert captured == [None]


def test_workflow_calls_ensemble_merge_when_enabled(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.automation.workflows import TradingWorkflow

    calls: list[int] = []

    def _capture(sigs: list[Signal], *, ensemble_config, timestamp) -> list[Signal]:
        calls.append(len(sigs))
        return list(sigs)

    monkeypatch.setattr("src.strategy.ensemble.merge_signals", _capture)

    s = settings.model_copy(
        update={
            "strategy": settings.strategy.model_copy(
                update={
                    "ensemble": settings.strategy.ensemble.model_copy(update={"enabled": True}),
                },
            ),
        },
    )

    ts = datetime(2026, 4, 1, 16, 0, tzinfo=UTC)
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=0.5,
        rationale="momentum",
        timestamp=ts,
        strategy_name="MomentumRotationStrategy",
    )
    mom = ScriptedMomentum(s, [sig])

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    ohlcv = pd.DataFrame(
        {"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0], "volume": [1e6]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-01")]),
    )
    store = MagicMock()
    store.read_ohlcv.return_value = ohlcv
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=s,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[mom],
        order_manager=order_mgr,
        broker=broker,
    )
    wf.run_cycle(as_of=ts)
    assert calls == [1]


def test_workflow_momentum_rotation_uses_raw_signals_when_ensemble_enabled(
    settings: Settings,
) -> None:
    """Merged tactical signals are ``Ensemble``; rotation target still from pre-merge momentum."""
    from src.automation.workflows import TradingWorkflow

    s = settings.model_copy(
        update={
            "strategy": settings.strategy.model_copy(
                update={
                    "ensemble": settings.strategy.ensemble.model_copy(update={"enabled": True}),
                },
            ),
        },
    )
    ts = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
    sig_mom = Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="Rotate to SPY.",
        timestamp=ts,
        strategy_name="MomentumRotationStrategy",
    )
    sig_mr = Signal(
        symbol="SPY",
        direction="long",
        weight=0.5,
        confidence=0.7,
        rationale="MR SPY",
        timestamp=ts,
        strategy_name="MeanReversionStrategy",
    )
    mom = ScriptedMomentum(s, [sig_mom])
    mr = ScriptedMR(s, [sig_mr])
    dca = MagicMock(spec=Strategy)
    dca.get_universe.return_value = []
    dca.generate_signals.return_value = []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    px = 300.0
    ohlcv = pd.DataFrame(
        {
            "open": [px],
            "high": [px],
            "low": [px],
            "close": [px],
            "volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-01")]),
    )
    pq = MagicMock()
    pq.read_ohlcv.return_value = ohlcv

    def pos(sym: str) -> float:
        return {"SPY": 0.0, "QQQ": 10.0, "SHV": 0.0}.get(str(sym).strip().upper(), 0.0)

    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.side_effect = pos

    order_mgr = MagicMock()
    order_mgr.close_position.return_value = OrderExecutionResult(
        symbol="QQQ",
        submitted=True,
        order_id="sell-1",
        side="sell",
        qty=10.0,
    )
    order_mgr.execute_signals.return_value = [
        OrderExecutionResult(symbol="SPY", submitted=True, order_id="buy-1", side="buy", qty=1.0),
    ]

    wf = TradingWorkflow(
        settings=s,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[mom, mr, dca],
        order_manager=order_mgr,
        broker=broker,
    )
    wf.run_cycle(as_of=ts)

    order_mgr.close_position.assert_called_once()
    assert order_mgr.close_position.call_args[0][0] == "QQQ"
    assert order_mgr.close_position.call_args[0][1] == 10.0
    exec_sigs = order_mgr.execute_signals.call_args[0][0]
    assert any(x.strategy_name == "Ensemble" for x in exec_sigs)


def test_trading_workflow_resolves_hub_partition_from_broker_get_account_id(
    settings: Settings,
) -> None:
    """Hub ``account_id`` follows ``broker.get_account_id()`` for SQLite / lot wiring."""
    from src.automation.workflows import TradingWorkflow

    broker = MagicMock()
    broker.get_account_id.return_value = "Joint"

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    store = MagicMock()
    store.read_ohlcv.return_value = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=store,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
    )
    assert wf._account_id == "Joint"


def _flat_ohlcv(price: float, day: str = "2026-06-01") -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [price], "high": [price], "low": [price], "close": [price], "volume": [1e6]},
        index=pd.DatetimeIndex([pd.Timestamp(day)]),
    )


def test_workflow_rotation_never_sells_sweep_symbol(settings: Settings) -> None:
    """Tier 44B pin: rotation sells non-target momentum positions but never the sweep symbol."""
    from src.automation.workflows import TradingWorkflow

    # Sweep vehicle is outside the momentum universe, so rotation must ignore a BIL holding.
    assert settings.cash_sweep.symbol == "BIL"
    sig = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.9,
        rationale="Rotate to QQQ.",
        timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    mom = ScriptedMomentum(settings, [sig])
    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    pq = MagicMock()
    pq.read_ohlcv.return_value = _flat_ohlcv(100.0)

    def pos(sym: str) -> float:
        return {"SPY": 10.0, "BIL": 7.0, "QQQ": 0.0}.get(str(sym).strip().upper(), 0.0)

    broker = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    broker.get_last_equity.return_value = 50_000.0
    broker.get_cash.return_value = 3_500.0  # inside the sweep hold band → sweep holds
    broker.get_position_qty.side_effect = pos

    order_mgr = MagicMock()
    order_mgr.close_position.return_value = OrderExecutionResult(
        symbol="SPY",
        submitted=True,
        order_id="sell-1",
        side="sell",
        qty=10.0,
    )
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[mom],
        order_manager=order_mgr,
        broker=broker,
    )
    wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))

    sold_symbols = [c.args[0] for c in order_mgr.close_position.call_args_list]
    assert "SPY" in sold_symbols  # non-target momentum position is rotated out
    assert "BIL" not in sold_symbols  # sweep symbol is never touched by rotation


def test_workflow_sweep_buy_executes_after_buys_and_logs_cashsweep(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """Sweep runs after strategy buys and persists a CashSweep-attributed execution row."""
    from src.automation.workflows import TradingWorkflow
    from src.execution.account_factory import resolve_sqlite_account_id

    strat = MagicMock(spec=Strategy)
    strat.get_universe.return_value = []
    strat.generate_signals.return_value = []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    pq = MagicMock()
    pq.read_ohlcv.return_value = _flat_ohlcv(100.0)

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    broker.list_recent_orders.return_value = []

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []
    order_mgr.unsettled_pending_buy_notional.return_value = 0.0
    order_mgr.sweep_buy.return_value = OrderExecutionResult(
        symbol="BIL",
        submitted=True,
        order_id="sweep-1",
        side="buy",
        qty=43.0,
        fill_price=100.0,
        strategy_name="CashSweep",
    )

    sqlite = SQLiteStore(tmp_path / "hub.db")
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[strat],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
    )
    wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))

    # target_cash = (0.05 + 0.02) * 10_000 = 700 → sweep buys 5000 - 700 = 4300.
    order_mgr.sweep_buy.assert_called_once()
    assert order_mgr.sweep_buy.call_args.args[1] == pytest.approx(4_300.0)

    # Sweep runs strictly after strategy-buy execution.
    called = [c[0] for c in order_mgr.mock_calls]
    assert called.index("execute_signals") < called.index("sweep_buy")

    account_id = resolve_sqlite_account_id(broker)
    rows = sqlite.get_executions(limit=100, account_id=account_id)
    sweep_rows = [r for r in rows if r.get("strategy_name") == "CashSweep"]
    assert len(sweep_rows) == 1
    assert sweep_rows[0]["symbol"] == "BIL"


def test_workflow_sweep_skips_when_symbol_unpriced(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No price for the sweep symbol this cycle → WARNING skip, never trade unpriced."""
    from src.automation.workflows import TradingWorkflow

    strat = MagicMock(spec=Strategy)
    strat.get_universe.return_value = []
    strat.generate_signals.return_value = []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)

    def read(sym: str) -> pd.DataFrame:
        return pd.DataFrame() if str(sym).strip().upper() == "BIL" else _flat_ohlcv(100.0)

    pq = MagicMock()
    pq.read_ohlcv.side_effect = read

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0

    order_mgr = MagicMock()
    order_mgr.execute_signals.return_value = []

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[strat],
        order_manager=order_mgr,
        broker=broker,
    )
    with caplog.at_level(logging.WARNING, logger="src.automation.workflows"):
        wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))

    assert "cash_sweep skipped" in caplog.text
    order_mgr.sweep_buy.assert_not_called()
    order_mgr.sweep_sell.assert_not_called()


def test_workflow_sweep_no_activity_when_kill_switch_halted(settings: Settings) -> None:
    """Halted kill switch → the sweep submits no broker order (halted means halted)."""
    from src.automation.workflows import TradingWorkflow
    from src.execution.order_manager import OrderManager
    from src.risk.kill_switch import KillSwitch

    strat = MagicMock(spec=Strategy)
    strat.get_universe.return_value = []
    strat.generate_signals.return_value = []

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    pq = MagicMock()
    pq.read_ohlcv.return_value = _flat_ohlcv(100.0)

    broker = MagicMock()
    broker.get_account_equity.return_value = 9_600.0  # -4% vs last → below the 3% limit
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    broker.list_recent_orders.return_value = []

    order_mgr = OrderManager(broker=broker, settings=settings, kill_switch=KillSwitch(0.03))

    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[strat],
        order_manager=order_mgr,
        broker=broker,
    )
    result = wf.run_cycle(as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))

    broker.submit_market_order.assert_not_called()  # no sweep (or any) order submitted
    sweep = [r for r in result.execution_results if r.strategy_name == "CashSweep"]
    assert sweep and all(not r.submitted for r in sweep)
    assert any("kill" in (r.reason or "").lower() for r in sweep)


def test_run_cycle_writes_broker_cash_onto_equity_snapshot(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """Tier 46: broker cash is snapshotted before record_equity_snapshot."""
    from src.automation.workflows import TradingWorkflow

    settings.cash_sweep.enabled = False
    settings.regime.enabled = False
    db = tmp_path / "hub.sqlite"
    sqlite = SQLiteStore(db)
    ledger = LotLedger(db)
    pq = ParquetStore(tmp_path / "parquet")
    ohlcv = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1e6],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-30")]),
    )
    for sym in ("VOO", "SPY", "QQQ", "TLT", "GLD", "SHV", "BIL"):
        pq.write_ohlcv(sym, ohlcv)

    pipeline = MagicMock()
    pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    pipeline.ingest_macro.return_value = IngestResult(success=True)

    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 2_490.0
    broker.get_position_qty.return_value = 0.0
    broker.list_recent_orders.return_value = []
    broker.submit_market_order.return_value = "ord-cash-1"
    broker.get_order_fill_price.return_value = 100.0

    order_mgr = OrderManager(
        broker=broker,
        settings=settings,
        kill_switch=KillSwitch(0.05),
        sqlite_store=sqlite,
    )
    wf = TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=pq,
        strategies=[DCAStrategy(settings)],
        order_manager=order_mgr,
        broker=broker,
        sqlite_store=sqlite,
        lot_ledger=ledger,
        kill_switch=KillSwitch(0.05),
    )
    wf.run_cycle(as_of=datetime(2026, 3, 30, 16, 0, tzinfo=UTC))
    snaps = sqlite.get_equity_snapshots()
    assert snaps
    assert float(snaps[-1]["cash"]) == pytest.approx(2_490.0)
