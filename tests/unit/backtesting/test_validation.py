"""Tests for walk-forward and Monte Carlo validation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.backtesting import validation as validation_mod
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.validation import (
    MonteCarloConfig,
    WalkForwardConfig,
    run_monte_carlo,
    run_walk_forward,
)
from src.config import DataConfig, MeanReversionConfig, Settings, StrategyConfig
from src.strategy.mean_reversion import MeanReversionStrategy


def _panel(n: int, start: date) -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range(start, periods=n, freq="B")
    out = {}
    for sym, drift in [("SPY", 0.0002), ("QQQ", 0.0001)]:
        close = 100.0 * np.cumprod(1.0 + np.full(n, drift))
        out[sym] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1e6,
            },
            index=idx,
        )
    return out


def test_walk_forward_produces_folds() -> None:
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=["SPY", "QQQ"]),
        strategy=StrategyConfig(
            mean_reversion=MeanReversionConfig(
                bb_period=5,
                rsi_period=5,
                sma_trend_period=0,
                universe=["SPY", "QQQ"],
            ),
        ),
    )
    data = _panel(504, date(2023, 1, 3))
    engine = BacktestEngine()
    strat = MeanReversionStrategy(settings)
    wf = run_walk_forward(
        engine,
        strat,
        data,
        start=date(2023, 1, 3),
        end=date(2024, 12, 31),
        backtest_config=BacktestConfig(rebalance_frequency="monthly"),
        wf_config=WalkForwardConfig(train_days=252, test_days=63, step_days=63),
    )
    assert len(wf.folds) >= 2


def test_walk_forward_aggregate_trading_days_matches_concatenated_returns() -> None:
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=["SPY", "QQQ"]),
        strategy=StrategyConfig(
            mean_reversion=MeanReversionConfig(
                bb_period=5,
                rsi_period=5,
                sma_trend_period=0,
                universe=["SPY", "QQQ"],
            ),
        ),
    )
    data = _panel(504, date(2023, 1, 3))
    engine = BacktestEngine()
    strat = MeanReversionStrategy(settings)
    wf = run_walk_forward(
        engine,
        strat,
        data,
        start=date(2023, 1, 3),
        end=date(2024, 12, 31),
        backtest_config=BacktestConfig(rebalance_frequency="monthly"),
        wf_config=WalkForwardConfig(train_days=252, test_days=63, step_days=63),
    )
    expected_n = sum(
        len(validation_mod._daily_returns_from_equity_curve(f.test_result.equity_curve))
        for f in wf.folds
    )
    assert wf.aggregate_metrics.trading_days == expected_n


def test_monte_carlo_median_near_expected() -> None:
    daily = pd.Series(np.full(500, 0.001))
    mc = run_monte_carlo(
        daily,
        initial_capital=10_000.0,
        config=MonteCarloConfig(n_simulations=200, n_days=50, random_seed=123),
    )
    assert mc.median_final_equity > 10_000.0


def test_monte_carlo_same_seed_same_median() -> None:
    rng = np.random.default_rng(42)
    daily = pd.Series(rng.normal(0.0, 0.01, 300))
    cfg = MonteCarloConfig(n_simulations=100, n_days=50, random_seed=4242)
    a = run_monte_carlo(daily, initial_capital=1.0, config=cfg)
    b = run_monte_carlo(daily, initial_capital=1.0, config=cfg)
    assert a.median_final_equity == b.median_final_equity


def test_monte_carlo_percentiles_ordered() -> None:
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0.0, 0.01, 300))
    mc = run_monte_carlo(
        daily,
        initial_capital=1.0,
        config=MonteCarloConfig(n_simulations=500, n_days=100, random_seed=7),
    )
    p = mc.percentiles
    assert p["p5"] < p["p25"] < p["p50"] < p["p75"] < p["p95"]
