"""Measure the post-54A empty-target boundary against the pre-54A behavior."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.tier52_analysis import _cash_map, _load_data
from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.persist import persist_backtest_run
from src.config import get_settings
from src.reporting.episodes import strategy_usable_start_dates

START = date(2011, 1, 28)
END = date(2026, 8, 19)
DRIFT_BAND_PCT = 0.05


def _run_cell(
    *,
    settings,
    data,
    cash_map,
    constrained: bool,
    legacy_empty_target_skip: bool,
) -> dict:
    """Run one raw or constrained empty-target-boundary cell."""
    mode = "constrained" if constrained else "raw"
    legacy_label = "legacy_skip_on" if legacy_empty_target_skip else "legacy_skip_off"
    label = f"54-step0-empty-target-{legacy_label}-{mode}"
    config = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=constrained,
        include_dca=True,
        include_cash_sweep=True,
        cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
        cash_yield_by_date=cash_map,
        min_coverage_ratio=0.8,
        drift_band_pct=DRIFT_BAND_PCT,
        legacy_empty_target_skip=legacy_empty_target_skip,
        experiment_label=label,
    )
    strategy = build_strategy_for_backtest(settings, "momentum")
    result = BacktestEngine().run(
        strategy,
        data,
        start=START,
        end=END,
        config=config,
        settings=settings,
    )
    persist_backtest_run(
        settings,
        strategy_name=type(strategy).__name__,
        config=config,
        result=result,
        start=START,
        end=END,
        benchmark_symbol="SPY",
        benchmark_metrics=None,
        experiment_label=label,
    )
    metrics = result.return_metrics
    row = {
        "legacy_empty_target_skip": legacy_empty_target_skip,
        "sizing_mode": result.sizing_mode,
        "trades": len(result.trades),
        "final_equity": round(result.final_equity, 2),
        "cagr_pct": round(metrics.cagr_pct, 4),
        "sharpe": None if metrics.sharpe_ratio is None else round(metrics.sharpe_ratio, 4),
        "max_dd_pct": round(metrics.max_drawdown_pct, 2),
        "empty_target_periods": result.empty_target_periods,
        "empty_target_liquidation_trades": result.empty_target_liquidation_trades,
        "config": config.ablation_config_dict(),
    }
    print(
        f"{label} trades={row['trades']} final={row['final_equity']} "
        f"CAGR={row['cagr_pct']} empty_periods={row['empty_target_periods']} "
        f"empty_liquidations={row['empty_target_liquidation_trades']}",
        flush=True,
    )
    return row


def main() -> None:
    """Run and persist the four empty-target probe cells."""
    settings = get_settings()
    data = _load_data(settings)
    start = max(START, strategy_usable_start_dates(data)["dca"])
    cash_map = _cash_map(settings)
    cells = [
        _run_cell(
            settings=settings,
            data=data,
            cash_map=cash_map,
            constrained=constrained,
            legacy_empty_target_skip=legacy,
        )
        for legacy in (False, True)
        for constrained in (True, False)
    ]
    payload = {
        "window": {"start": start.isoformat(), "end": END.isoformat()},
        "drift_band_pct": DRIFT_BAND_PCT,
        "cells": cells,
    }
    out = Path("data/cache/tier54_empty_target_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
