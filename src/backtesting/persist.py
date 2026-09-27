"""Persist a backtest run with full ablation config (shared by CLI and analysis scripts)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import hub_sqlite_path
from src.data.storage.sqlite_store import SQLiteStore

if TYPE_CHECKING:
    from datetime import date

    from src.backtesting.engine import BacktestConfig, BacktestResult
    from src.config import Settings
    from src.reporting.returns import ReturnMetrics


def persist_backtest_run(
    settings: Settings,
    *,
    strategy_name: str,
    config: BacktestConfig,
    result: BacktestResult,
    start: date,
    end: date,
    benchmark_symbol: str | None = None,
    benchmark_metrics: ReturnMetrics | None = None,
    experiment_label: str | None = None,
) -> int:
    """Write one run to ``backtest_runs`` including ``config_json`` (Tier 53B)."""
    store = SQLiteStore(hub_sqlite_path(settings))
    return store.save_backtest_run(
        strategy_name,
        config,
        result,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        benchmark_symbol=benchmark_symbol,
        benchmark_metrics=benchmark_metrics,
        experiment_label=experiment_label,
    )
