"""Tier 55C — full-cycle satellite hurdle (four pre-registered arms).

Measurement only. Momentum-only raw, no DCA, no sweep, DTB3 historical cash yield.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.cash_yield import series_from_macro_rows
from src.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
)
from src.backtesting.persist import persist_backtest_run
from src.config import get_settings, hub_sqlite_path, parquet_dir
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.episodes import strategy_usable_start_dates
from src.reporting.returns import compute_return_metrics

logger = logging.getLogger(__name__)

WINDOW_START = date(2007, 1, 11)
WINDOW_END = date(2026, 8, 19)
ARMS = (
    ("B", False, "weekly"),
    ("B", False, "monthly"),
    ("B-ADX", True, "weekly"),
    ("B-ADX", True, "monthly"),
)
PRIMARY_SPEC = "55C-B-monthly"
CAGR_HURDLE_PP = 1.0
MAX_DD_FRACTION = 0.6


def _cash_map(settings) -> dict[str, float] | None:
    sid = str(settings.backtest.cash_yield_series_id or "").strip()
    if not sid:
        return None
    ser = series_from_macro_rows(SQLiteStore(hub_sqlite_path(settings)).get_macro_indicator(sid))
    if ser.empty:
        return None
    return {ts.date().isoformat(): float(v) for ts, v in ser.items()}


def _load_data(settings) -> dict:
    pq = ParquetStore(parquet_dir(settings))
    strat = build_strategy_for_backtest(settings, "momentum")
    data: dict = {}
    for sym in [*strat.get_universe(), settings.regime.vix_symbol, "SPY"]:
        df = pq.read_ohlcv(sym)
        if not df.empty:
            data[sym.strip().upper()] = df
    return data


def _window_start(data: dict) -> date:
    starts = strategy_usable_start_dates(data)
    if "momentum" in starts:
        return max(WINDOW_START, starts["momentum"])
    return WINDOW_START


def _pack(result, *, label: str, run_id: int | None = None) -> dict:
    m = result.return_metrics
    row = {
        "experiment_label": label,
        "sizing_mode": result.sizing_mode,
        "trades": len(result.trades),
        "cagr_pct": round(m.cagr_pct, 4),
        "sharpe": None if m.sharpe_ratio is None else round(m.sharpe_ratio, 4),
        "max_dd_pct": round(m.max_drawdown_pct, 2),
        "turnover": round(result.turnover, 4),
        "final_equity": round(result.final_equity, 2),
        "avg_exposure_pct": round(result.avg_exposure_pct, 4),
    }
    if run_id is not None:
        row["run_id"] = run_id
    return row


def _run_arm(
    eng: BacktestEngine,
    settings,
    data: dict,
    cash_map: dict[str, float] | None,
    *,
    spec: str,
    disable_adx: bool,
    cadence: str,
    start: date,
    end: date,
) -> dict:
    label = f"55C-{spec}-{cadence}"
    strat = build_strategy_for_backtest(settings, "momentum")
    cfg = BacktestConfig(
        rebalance_frequency=cadence,
        apply_risk_layer=False,
        include_dca=False,
        include_cash_sweep=False,
        cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
        cash_yield_by_date=cash_map,
        min_coverage_ratio=0.8,
        drift_band_pct=1.0,
        disable_adx=disable_adx,
        experiment_label=label,
    )
    result = eng.run(strat, data, start=start, end=end, config=cfg, settings=settings)
    rid = persist_backtest_run(
        settings,
        strategy_name=type(strat).__name__,
        config=cfg,
        result=result,
        start=start,
        end=end,
        benchmark_symbol="SPY",
        benchmark_metrics=None,
        experiment_label=label,
    )
    row = _pack(result, label=label, run_id=rid)
    print(
        f"{label} CAGR={row['cagr_pct']} Sharpe={row['sharpe']} "
        f"maxDD={row['max_dd_pct']} trades={row['trades']} id={rid}",
        flush=True,
    )
    return row


def _spy_benchmark(
    settings,
    cash_map: dict[str, float] | None,
    fallback: float,
    *,
    start: date,
    end: date,
) -> dict:
    """Buy-and-hold SPY on its own adjusted close calendar (same store as backtests)."""
    store = ParquetStore(parquet_dir(settings))
    frame = store.read_ohlcv("SPY")
    if frame.empty:
        return {"error": "SPY benchmark unavailable"}
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    closes = frame.loc[
        (frame.index.date >= start) & (frame.index.date <= end),
        "close",
    ].astype(float)
    closes = closes[~closes.index.duplicated()].sort_index()
    if len(closes) < 2:
        return {"error": "SPY benchmark unavailable"}
    rf_daily = None
    if cash_map:
        sid = str(settings.backtest.cash_yield_series_id or "").strip()
        if sid:
            ser = series_from_macro_rows(
                SQLiteStore(hub_sqlite_path(settings)).get_macro_indicator(sid)
            )
            if not ser.empty:
                ser.index = pd.DatetimeIndex(ser.index).tz_localize(None).normalize()
                rf_daily = (ser.astype(float) / 100.0) / 252.0
    returns = closes.pct_change().dropna()
    metrics = compute_return_metrics(
        returns,
        risk_free_daily=rf_daily,
        risk_free_rate_annual=fallback if rf_daily is None else 0.0,
    )
    years = (closes.index[-1] - closes.index[0]).days / 365.25
    cagr = ((float(closes.iloc[-1]) / float(closes.iloc[0])) ** (1.0 / years) - 1.0) * 100.0
    return {
        "symbol": "SPY",
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "cagr_pct": round(cagr, 4),
        "sharpe": None if metrics.sharpe_ratio is None else round(metrics.sharpe_ratio, 4),
        "max_dd_pct": round(float(metrics.max_drawdown_pct), 2),
        "total_return_pct": round(float((closes.iloc[-1] / closes.iloc[0] - 1.0) * 100.0), 2),
        "trading_days": len(returns),
    }


def _verdict(primary: dict, spy: dict) -> dict:
    if "error" in spy or "error" in primary:
        return {"verdict": "error", "reason": "missing primary or SPY row"}
    cagr_floor = float(spy["cagr_pct"]) - CAGR_HURDLE_PP
    dd_ceiling = MAX_DD_FRACTION * float(spy["max_dd_pct"])
    cagr_ok = float(primary["cagr_pct"]) >= cagr_floor
    dd_ok = float(primary["max_dd_pct"]) <= dd_ceiling
    retained = cagr_ok and dd_ok
    return {
        "primary_spec": PRIMARY_SPEC,
        "cagr_hurdle_pct": round(cagr_floor, 4),
        "cagr_actual_pct": primary["cagr_pct"],
        "cagr_pass": cagr_ok,
        "max_dd_ceiling_pct": round(dd_ceiling, 2),
        "max_dd_actual_pct": primary["max_dd_pct"],
        "max_dd_pass": dd_ok,
        "satellite_retained": retained,
        "recommendation": "satellite retained" if retained else "satellite retirement recommended",
        "b_adx_can_win": False,
        "note": "B-ADX rows reported but cannot win per 54D refutation; weekly rows price cadence only.",
    }


def run(*, persist: bool = True) -> dict:
    settings = get_settings()
    data = _load_data(settings)
    start = _window_start(data)
    end = WINDOW_END
    cash_map = _cash_map(settings)
    fallback = float(settings.backtest.cash_yield_annual_pct)
    eng = BacktestEngine()

    spy = _spy_benchmark(settings, cash_map, fallback, start=start, end=end)
    print(
        f"SPY {start}..{end} CAGR={spy.get('cagr_pct')} maxDD={spy.get('max_dd_pct')}",
        flush=True,
    )

    arms: list[dict] = []
    for spec, disable_adx, cadence in ARMS:
        arms.append(
            _run_arm(
                eng,
                settings,
                data,
                cash_map,
                spec=spec,
                disable_adx=disable_adx,
                cadence=cadence,
                start=start,
                end=end,
            )
        )

    by_label = {a["experiment_label"]: a for a in arms}
    primary = by_label.get(PRIMARY_SPEC)
    verdict = _verdict(primary or {"error": "missing"}, spy) if primary else {"verdict": "error"}

    cmd = (
        "$env:PYTHONIOENCODING='utf-8'; "
        ".venv\\Scripts\\python scripts/tier55c_full_cycle.py"
    )
    payload = {
        "tier": "55C",
        "contract": ".agents/FABLE_STRATEGY_REVIEW.md §7",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "config": (
            "momentum-only raw: apply_risk_layer=False, include_dca=False, "
            "include_cash_sweep=False; adjusted parquet; DTB3 historical cash yield"
        ),
        "command": cmd,
        "spy_benchmark": spy,
        "arms": arms,
        "verdict": verdict,
        "thresholds": {
            "primary_spec": PRIMARY_SPEC,
            "cagr_min_pct": f"SPY - {CAGR_HURDLE_PP} pp",
            "max_dd_max_pct": f"{MAX_DD_FRACTION} x SPY maxDD (both negative)",
        },
    }
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    payload = run(persist=True)
    out = Path("data/cache/tier55c_full_cycle.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    v = payload["verdict"]
    print(
        f"VERDICT primary={v.get('primary_spec')} "
        f"cagr_pass={v.get('cagr_pass')} dd_pass={v.get('max_dd_pass')} "
        f"=> {v.get('recommendation')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
