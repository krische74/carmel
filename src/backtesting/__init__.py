"""Strategy backtesting (walk-forward simulation, no lookahead)."""

from src.backtesting.engine import (
    BacktestConfig,
    BacktestDataMissingError,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    trading_days_in_range,
)
from src.backtesting.validation import (
    MonteCarloConfig,
    MonteCarloResult,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardResult,
    run_monte_carlo,
    run_walk_forward,
)

__all__ = [
    "BacktestConfig",
    "BacktestDataMissingError",
    "BacktestEngine",
    "BacktestResult",
    "BacktestTrade",
    "MonteCarloConfig",
    "MonteCarloResult",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardResult",
    "run_monte_carlo",
    "run_walk_forward",
    "trading_days_in_range",
]
