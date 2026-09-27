"""Walk-forward and Monte Carlo validation helpers for backtests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    trading_days_in_range,
)
from src.reporting.returns import ReturnMetrics, compute_return_metrics

if TYPE_CHECKING:
    from datetime import date

    from src.strategy.base import Strategy


class WalkForwardConfig(BaseModel):
    """Window sizes for walk-forward evaluation (trading days)."""

    train_days: int = 252
    test_days: int = 63
    step_days: int = 63


class WalkForwardFold(BaseModel):
    """One train/test split and the out-of-sample backtest result."""

    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_result: BacktestResult


class WalkForwardResult(BaseModel):
    """All folds plus metrics on concatenated test-period returns."""

    folds: list[WalkForwardFold]
    aggregate_metrics: ReturnMetrics


class MonteCarloConfig(BaseModel):
    """Bootstrap simulation over historical daily returns."""

    n_simulations: int = 1000
    n_days: int = 252
    random_seed: int | None = None


class MonteCarloResult(BaseModel):
    """Summary of simulated terminal wealth and drawdowns."""

    percentiles: dict[str, float]
    median_final_equity: float
    median_max_drawdown_pct: float


def _daily_returns_from_equity_curve(curve: list[dict[str, object]]) -> list[float]:
    if len(curve) < 2:
        return []
    eqs = [float(p["equity"]) for p in curve]
    out: list[float] = []
    for i in range(1, len(eqs)):
        prev = eqs[i - 1]
        if prev <= 0.0:
            continue
        out.append((eqs[i] - prev) / prev)
    return out


def run_walk_forward(
    engine: BacktestEngine,
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    *,
    start: date,
    end: date,
    backtest_config: BacktestConfig,
    wf_config: WalkForwardConfig,
) -> WalkForwardResult:
    """Run sequential out-of-sample backtests along a rolling calendar."""
    trading_days = trading_days_in_range(data, start, end)
    n = len(trading_days)
    step = max(1, wf_config.step_days)
    train = max(1, wf_config.train_days)
    test = max(1, wf_config.test_days)

    folds: list[WalkForwardFold] = []
    all_test_returns: list[float] = []
    k = 0
    while True:
        test_start_idx = train + k * step
        test_end_idx = test_start_idx + test - 1
        if test_end_idx >= n:
            break
        train_start = trading_days[test_start_idx - train]
        train_end = trading_days[test_start_idx - 1]
        test_start = trading_days[test_start_idx]
        test_end = trading_days[test_end_idx]

        test_result = engine.run(
            strategy,
            data,
            start=test_start,
            end=test_end,
            config=backtest_config,
        )
        folds.append(
            WalkForwardFold(
                fold_index=k,
                train_start=train_start.isoformat(),
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
                test_result=test_result,
            ),
        )
        all_test_returns.extend(_daily_returns_from_equity_curve(test_result.equity_curve))
        k += 1

    if not all_test_returns:
        aggregate = compute_return_metrics(pd.Series(dtype=float))
    else:
        aggregate = compute_return_metrics(pd.Series(all_test_returns))

    return WalkForwardResult(folds=folds, aggregate_metrics=aggregate)


def _max_drawdown_pct_from_equity(equity_path: np.ndarray) -> float:
    if equity_path.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_path)
    dd = equity_path / np.maximum(peak, 1e-18) - 1.0
    return float(abs(np.min(dd)) * 100.0)


def run_monte_carlo(
    daily_returns: pd.Series,
    *,
    initial_capital: float,
    config: MonteCarloConfig,
) -> MonteCarloResult:
    """Bootstrap resample daily returns into many synthetic paths."""
    r = daily_returns.astype(float).dropna()
    rng = np.random.default_rng(config.random_seed)
    if r.empty or config.n_simulations <= 0 or config.n_days <= 0:
        return MonteCarloResult(
            percentiles={"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0},
            median_final_equity=float(initial_capital),
            median_max_drawdown_pct=0.0,
        )
    arr = r.to_numpy(dtype=float)
    finals: list[float] = []
    dds: list[float] = []
    for _ in range(config.n_simulations):
        draws = rng.choice(arr, size=config.n_days, replace=True)
        equity = float(initial_capital) * np.cumprod(1.0 + draws)
        finals.append(float(equity[-1]))
        dds.append(_max_drawdown_pct_from_equity(equity))
    finals_a = np.asarray(finals, dtype=float)
    dds_a = np.asarray(dds, dtype=float)
    pct = {
        "p5": float(np.percentile(finals_a, 5)),
        "p25": float(np.percentile(finals_a, 25)),
        "p50": float(np.percentile(finals_a, 50)),
        "p75": float(np.percentile(finals_a, 75)),
        "p95": float(np.percentile(finals_a, 95)),
    }
    return MonteCarloResult(
        percentiles=pct,
        median_final_equity=float(np.median(finals_a)),
        median_max_drawdown_pct=float(np.median(dds_a)),
    )
