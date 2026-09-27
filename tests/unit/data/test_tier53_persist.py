"""Tier 53B: persist full backtest configuration on every analysis run."""

from __future__ import annotations

import json

from src.backtesting.engine import BacktestConfig, BacktestResult
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.returns import ReturnMetrics


def test_save_backtest_run_persists_ablation_config_json(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "bt_t53.db")
    cfg = BacktestConfig(
        apply_risk_layer=False,
        rebalance_frequency="weekly",
        rebalance_weekday=1,
        disable_adx=True,
        top_n=2,
        band_k=0.5,
        pin_regime_multiplier=1.0,
    )
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
    result = BacktestResult(
        equity_curve=[{"date": "2024-01-02", "equity": 10_000.0}],
        trades=[],
        return_metrics=m,
        initial_capital=10_000.0,
        final_equity=10_300.0,
        sizing_mode="raw_signal",
        turnover=1.25,
    )
    rid = store.save_backtest_run(
        "MomentumRotationStrategy",
        cfg,
        result,
        start_date="2007-01-01",
        end_date="2021-12-31",
        experiment_label="B-ADX",
    )
    row = store.get_backtest_run(rid)
    assert row is not None
    assert row["experiment_label"] == "B-ADX"
    assert float(row["turnover"]) == 1.25
    raw = row["config_json"]
    assert raw is not None
    parsed = json.loads(str(raw))
    assert parsed["disable_adx"] is True
    assert parsed["top_n"] == 2
    assert parsed["band_k"] == 0.5
    assert parsed["pin_regime_multiplier"] == 1.0
    assert parsed["rebalance_weekday"] == 1
    assert parsed["rebalance_frequency"] == "weekly"
