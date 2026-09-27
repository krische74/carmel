"""Tier 51 full-window backtests and episode table (run from repo root)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.config import get_settings, parquet_dir
from src.data.storage.parquet_store import ParquetStore
from src.reporting.episodes import (
    BULL_EPISODES,
    DRAWDOWN_EPISODES,
    EpisodeSlice,
    buy_and_hold_equity_curve,
    episode_slice_metrics,
    static_blend_equity_curve,
    strategy_usable_start_dates,
)


def _series_rows(label: str, curve: list[dict], start: date, end: date) -> list[dict]:
    full = EpisodeSlice(name="full", start=start, end=end, label="full")
    rows: list[dict] = []
    for ep in (full, *DRAWDOWN_EPISODES, *BULL_EPISODES):
        m = episode_slice_metrics(curve, ep)
        rows.append(
            {
                "series": label,
                "episode": ep.name,
                "label": ep.label,
                "ret": round(m.total_return_pct, 2),
                "cagr": round(m.cagr_pct, 2),
                "dd": round(m.max_drawdown_pct, 2),
                "calmar": None if m.calmar_ratio is None else round(m.calmar_ratio, 2),
                "recover": m.recovery_trading_days,
                "tdays": m.trading_days,
            }
        )
    return rows


def main() -> None:
    settings = get_settings()
    pq = ParquetStore(parquet_dir(settings))
    strat = build_strategy_for_backtest(settings, "momentum")
    data: dict[str, pd.DataFrame] = {}
    for sym in [*strat.get_universe(), settings.regime.vix_symbol]:
        df = pq.read_ohlcv(sym)
        if not df.empty:
            data[sym.strip().upper()] = df
    for extra in ("VOO", "VXUS", "BND", "BIL"):
        df = pq.read_ohlcv(extra)
        if not df.empty:
            data[extra] = df

    starts = {k: v.isoformat() for k, v in strategy_usable_start_dates(data).items()}
    start = strategy_usable_start_dates(data)["momentum"]
    end = date(2026, 8, 19)
    rf = float(settings.backtest.cash_yield_annual_pct)
    cfg_c = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=True,
        cash_yield_annual_pct=rf,
        min_coverage_ratio=0.8,
    )
    cfg_r = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=False,
        cash_yield_annual_pct=rf,
        min_coverage_ratio=0.8,
    )
    eng = BacktestEngine()
    rc = eng.run(strat, data, start=start, end=end, config=cfg_c, settings=settings)
    rr = eng.run(strat, data, start=start, end=end, config=cfg_r, settings=settings)

    spy = data["SPY"].copy()
    spy.index = pd.DatetimeIndex(pd.to_datetime(spy.index).tz_localize(None)).normalize()
    spy_w = spy[(spy.index.date >= start) & (spy.index.date <= end)]

    spy_curve = buy_and_hold_equity_curve(spy_w)
    blend_curve = static_blend_equity_curve(spy_w, cash_yield_annual_pct=rf)

    rows = (
        _series_rows("constrained", rc.equity_curve, start, end)
        + _series_rows("raw_signal", rr.equity_curve, start, end)
        + _series_rows("spy", spy_curve, start, end)
        + _series_rows("blend_75_25", blend_curve, start, end)
    )
    payload = {
        "usable_starts": starts,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "constrained": {
            "trades": len(rc.trades),
            "final_equity": round(rc.final_equity, 2),
            "total_return_pct": round(rc.return_metrics.total_return_pct, 2),
            "cagr_pct": round(rc.return_metrics.cagr_pct, 2),
            "sharpe": rc.return_metrics.sharpe_ratio,
            "max_dd_pct": round(rc.return_metrics.max_drawdown_pct, 2),
            "vol_pct": round(rc.return_metrics.annual_volatility_pct, 2),
            "calmar": rc.return_metrics.calmar_ratio,
            "peak_alloc": round(rc.peak_single_symbol_allocation_pct, 4),
        },
        "raw_signal": {
            "trades": len(rr.trades),
            "final_equity": round(rr.final_equity, 2),
            "total_return_pct": round(rr.return_metrics.total_return_pct, 2),
            "cagr_pct": round(rr.return_metrics.cagr_pct, 2),
            "sharpe": rr.return_metrics.sharpe_ratio,
            "max_dd_pct": round(rr.return_metrics.max_drawdown_pct, 2),
            "vol_pct": round(rr.return_metrics.annual_volatility_pct, 2),
            "calmar": rr.return_metrics.calmar_ratio,
            "peak_alloc": round(rr.peak_single_symbol_allocation_pct, 4),
        },
        "episodes": rows,
    }
    out = Path("data/cache/tier51_episode_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({"usable_starts": starts, "constrained": payload["constrained"], "raw_signal": payload["raw_signal"]}, indent=2))


if __name__ == "__main__":
    main()
