"""Tier 53 ablation matrix, weekday banding falsification, and regime latch audit.

Measurement only. ASCII output. band_k provenance: Google Doc export stripped the
recommended k (401 on fetch 2026-08-21); starting k=0.5 chosen as a half-sigma
relative hurdle (conservative vs 1.0) — result conditional on that choice.
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
    compute_benchmark_metrics,
    trading_days_in_range,
)
from src.backtesting.persist import persist_backtest_run
from src.config import get_settings, hub_sqlite_path, parquet_dir
from src.data.regime import build_market_regime_snapshot
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.episodes import strategy_usable_start_dates
from src.strategy.indicators import sma

logger = logging.getLogger(__name__)

# Unrecoverable from Doc export; half-sigma relative switch hurdle.
BAND_K = 0.5
BAND_K_PROVENANCE = (
    "Doc https://docs.google.com/document/d/1j4JgSUir4krKGHo0ZLyqLawHVA-XXM7N2pc4w5Hcvpw "
    "returned 401; export in docs/research/banding-design-DR-synthesis.md has blank k. "
    "Chose k=0.5 (half cross-sectional sigma) as a conservative starting hurdle; "
    "results are conditional on this choice."
)

WINDOW_START = date(2007, 1, 11)  # SHV-bound after warm data
WINDOW_END = date(2021, 12, 31)
WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def _cash_map(settings) -> dict[str, float] | None:
    sid = str(settings.backtest.cash_yield_series_id or "").strip()
    if not sid:
        return None
    ser = series_from_macro_rows(SQLiteStore(hub_sqlite_path(settings)).get_macro_indicator(sid))
    if ser.empty:
        return None
    return {ts.date().isoformat(): float(v) for ts, v in ser.items()}


def _load_data(settings) -> dict[str, pd.DataFrame]:
    pq = ParquetStore(parquet_dir(settings))
    strat = build_strategy_for_backtest(settings, "momentum")
    data: dict[str, pd.DataFrame] = {}
    for sym in [*strat.get_universe(), settings.regime.vix_symbol, "SPY"]:
        df = pq.read_ohlcv(sym)
        if not df.empty:
            data[sym.strip().upper()] = df
    return data


def _pack(result, *, label: str) -> dict:
    m = result.return_metrics
    return {
        "label": label,
        "sizing_mode": result.sizing_mode,
        "trades": len(result.trades),
        "cagr_pct": round(m.cagr_pct, 4),
        "total_return_pct": round(m.total_return_pct, 2),
        "sharpe": None if m.sharpe_ratio is None else round(m.sharpe_ratio, 4),
        "sortino": None if m.sortino_ratio is None else round(m.sortino_ratio, 4),
        "max_dd_pct": round(m.max_drawdown_pct, 2),
        "calmar": None if m.calmar_ratio is None else round(m.calmar_ratio, 4),
        "vol_pct": round(m.annual_volatility_pct, 2),
        "avg_exposure_pct": round(result.avg_exposure_pct, 4),
        "max_exposure_pct": round(result.max_exposure_pct, 4),
        "turnover": round(result.turnover, 4),
        "final_equity": round(result.final_equity, 2),
        "cash_yield_mode": result.cash_yield_mode,
        "realized_cash_yield_annual_pct": round(result.realized_cash_yield_annual_pct, 3),
    }


def _cfg(
    *,
    constrained: bool,
    cash_map: dict[str, float] | None,
    fallback: float,
    label: str,
    weekday: int | None = None,
    disable_adx: bool = False,
    top_n: int = 1,
    band_k: float | None = None,
    pin_regime_multiplier: float | None = None,
) -> BacktestConfig:
    return BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=constrained,
        cash_yield_annual_pct=fallback,
        cash_yield_by_date=cash_map,
        min_coverage_ratio=0.8,
        rebalance_weekday=weekday,
        disable_adx=disable_adx,
        top_n=top_n,
        band_k=band_k,
        pin_regime_multiplier=pin_regime_multiplier,
        experiment_label=label,
    )


def _run_one(
    eng: BacktestEngine,
    strat,
    data,
    settings,
    cfg: BacktestConfig,
    *,
    start: date,
    end: date,
) -> dict:
    result = eng.run(strat, data, start=start, end=end, config=cfg, settings=settings)
    days = trading_days_in_range(data, start, end)
    rf_daily = None
    if cfg.cash_yield_by_date:
        rf_daily = pd.Series(
            {k: (float(v) / 100.0) / 252.0 for k, v in cfg.cash_yield_by_date.items()},
            dtype=float,
        )
    bench = compute_benchmark_metrics(
        data,
        "SPY",
        days,
        risk_free_rate_annual=cfg.cash_yield_annual_pct,
        risk_free_daily=rf_daily,
    )
    rid = persist_backtest_run(
        settings,
        strategy_name=type(strat).__name__,
        config=cfg,
        result=result,
        start=start,
        end=end,
        benchmark_symbol="SPY",
        benchmark_metrics=bench,
        experiment_label=cfg.experiment_label,
    )
    row = _pack(result, label=str(cfg.experiment_label))
    row["run_id"] = rid
    if bench is not None:
        row["spy_cagr_pct"] = round(bench.cagr_pct, 4)
        row["spy_max_dd_pct"] = round(bench.max_drawdown_pct, 2)
        row["spy_sharpe"] = None if bench.sharpe_ratio is None else round(bench.sharpe_ratio, 4)
    print(
        f"{row['label']} mode={row['sizing_mode']} CAGR={row['cagr_pct']} "
        f"DD={row['max_dd_pct']} Sharpe={row['sharpe']} TO={row['turnover']} id={rid}",
        flush=True,
    )
    return row


def _ablation_specs() -> list[dict]:
    """One-factor then synergistic pairs. Do not stack arbitrarily."""
    return [
        {"label": "B", "disable_adx": False, "top_n": 1, "band_k": None, "pin_regime_multiplier": None},
        {"label": "B-ADX", "disable_adx": True, "top_n": 1, "band_k": None, "pin_regime_multiplier": None},
        {"label": "B+Top2", "disable_adx": False, "top_n": 2, "band_k": None, "pin_regime_multiplier": None},
        {"label": "B+Band", "disable_adx": False, "top_n": 1, "band_k": BAND_K, "pin_regime_multiplier": None},
        {"label": "B-Regime", "disable_adx": False, "top_n": 1, "band_k": None, "pin_regime_multiplier": 1.0},
        {"label": "B+Band+Top2", "disable_adx": False, "top_n": 2, "band_k": BAND_K, "pin_regime_multiplier": None},
        {"label": "B+Band-ADX", "disable_adx": True, "top_n": 1, "band_k": BAND_K, "pin_regime_multiplier": None},
    ]


def _weekday_spread(rows: list[dict]) -> float:
    cagrs = [float(r["cagr_pct"]) for r in rows]
    return max(cagrs) - min(cagrs) if cagrs else 0.0


def _latch_audit(settings, data: dict[str, pd.DataFrame], start: date, end: date) -> dict:
    """Count cautious-or-lower stretches >60 days while SPY above 200-day SMA."""
    spy = data.get("SPY")
    vix = data.get(settings.regime.vix_symbol.strip().upper())
    if spy is None or spy.empty or vix is None or vix.empty:
        return {"error": "missing SPY or VIX", "periods": []}
    spy_n = spy.copy()
    spy_n.index = pd.DatetimeIndex(pd.to_datetime(spy_n.index).tz_localize(None)).normalize()
    closes = spy_n["close"].astype(float)
    trend = sma(closes, period=200)
    days = trading_days_in_range(data, start, end)
    periods: list[dict] = []
    run_start: date | None = None
    run_mults: list[float] = []
    run_len = 0

    def _flush(end_d: date) -> None:
        nonlocal run_start, run_mults, run_len
        if run_start is not None and run_len > 60:
            periods.append(
                {
                    "start": run_start.isoformat(),
                    "end": end_d.isoformat(),
                    "trading_days": run_len,
                    "sizing_multiplier": round(sum(run_mults) / len(run_mults), 4),
                }
            )
        run_start = None
        run_mults = []
        run_len = 0

    from datetime import UTC, datetime, time

    vix_n = vix.copy()
    vix_n.index = pd.DatetimeIndex(pd.to_datetime(vix_n.index).tz_localize(None)).normalize()
    for d in days:
        ts = pd.Timestamp(d)
        if ts not in closes.index or ts not in trend.index:
            continue
        px = float(closes.loc[ts])
        sm = float(trend.loc[ts])
        if not (px == px and sm == sm) or sm <= 0:
            continue
        above = px > sm
        as_of = datetime.combine(d, time.min, tzinfo=UTC)
        vx = None
        sub = vix_n.loc[vix_n.index <= ts]
        if not sub.empty:
            vx = float(sub["close"].iloc[-1])
        regime = build_market_regime_snapshot(settings, as_of=as_of, vix_close=vx, yield_spread=None)
        defensive = regime.overall.value in ("cautious", "defensive", "crisis")
        if above and defensive:
            if run_start is None:
                run_start = d
            run_len += 1
            run_mults.append(float(regime.sizing_multiplier))
        else:
            if run_start is not None:
                _flush(d)
    if run_start is not None:
        _flush(days[-1])
    return {"count": len(periods), "periods": periods}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    fallback = float(settings.backtest.cash_yield_annual_pct)
    cash_map = _cash_map(settings)
    data = _load_data(settings)
    starts = {k: v.isoformat() for k, v in strategy_usable_start_dates(data).items()}
    start = max(WINDOW_START, strategy_usable_start_dates(data)["momentum"])
    end = WINDOW_END
    eng = BacktestEngine()
    strat = build_strategy_for_backtest(settings, "momentum")

    matrix: list[dict] = []
    # 53C one-factor first (order in _ablation_specs: B..B-Regime), then pairs last
    one_factor = [s for s in _ablation_specs() if s["label"] not in ("B+Band+Top2", "B+Band-ADX")]
    pairs = [s for s in _ablation_specs() if s["label"] in ("B+Band+Top2", "B+Band-ADX")]

    for spec in one_factor:
        for constrained in (True, False):
            mode = "constrained" if constrained else "raw"
            label = f"{spec['label']}/{mode}"
            cfg = _cfg(
                constrained=constrained,
                cash_map=cash_map,
                fallback=fallback,
                label=label,
                disable_adx=spec["disable_adx"],
                top_n=spec["top_n"],
                band_k=spec["band_k"],
                pin_regime_multiplier=spec["pin_regime_multiplier"],
            )
            # Fresh strategy so incumbent state does not leak across configs
            strat = build_strategy_for_backtest(settings, "momentum")
            matrix.append(_run_one(eng, strat, data, settings, cfg, start=start, end=end))

    # 53D weekday sweep: unbanded vs banded, raw-signal only
    unbanded_wd: list[dict] = []
    banded_wd: list[dict] = []
    for wd in range(5):
        for band_k, bucket, tag in (
            (None, unbanded_wd, "unbanded"),
            (BAND_K, banded_wd, "banded"),
        ):
            label = f"52D-{tag}-{WEEKDAY_NAMES[wd]}/raw"
            cfg = _cfg(
                constrained=False,
                cash_map=cash_map,
                fallback=fallback,
                label=label,
                weekday=wd,
                band_k=band_k,
            )
            strat = build_strategy_for_backtest(settings, "momentum")
            row = _run_one(eng, strat, data, settings, cfg, start=start, end=end)
            row["weekday"] = wd
            row["weekday_name"] = WEEKDAY_NAMES[wd]
            bucket.append(row)

    # pairs after banding implemented
    for spec in pairs:
        for constrained in (True, False):
            mode = "constrained" if constrained else "raw"
            label = f"{spec['label']}/{mode}"
            cfg = _cfg(
                constrained=constrained,
                cash_map=cash_map,
                fallback=fallback,
                label=label,
                disable_adx=spec["disable_adx"],
                top_n=spec["top_n"],
                band_k=spec["band_k"],
                pin_regime_multiplier=spec["pin_regime_multiplier"],
            )
            strat = build_strategy_for_backtest(settings, "momentum")
            matrix.append(_run_one(eng, strat, data, settings, cfg, start=start, end=end))

    ub_spread = _weekday_spread(unbanded_wd)
    bd_spread = _weekday_spread(banded_wd)
    ub_to = sum(r["turnover"] for r in unbanded_wd) / max(len(unbanded_wd), 1)
    bd_to = sum(r["turnover"] for r in banded_wd) / max(len(banded_wd), 1)
    turnover_fell = bd_to < ub_to * 0.9  # material = at least 10% drop
    if turnover_fell and bd_spread < 2.0:
        verdict = "VALIDATED: turnover fell materially and weekday CAGR spread < 2 pp."
    elif turnover_fell and bd_spread >= 2.0:
        verdict = (
            "FALSIFIED: turnover fell but weekday CAGR spread stayed >= 2 pp. "
            "Dispersion is not noise-driven turnover; banding treats the wrong thing."
        )
    else:
        verdict = (
            f"INCONCLUSIVE on pre-stated rule: turnover_fell={turnover_fell} "
            f"banded_spread={bd_spread:.3f} (need turnover drop AND spread < 2)."
        )

    latch = _latch_audit(settings, data, start, end)
    b_row = next(r for r in matrix if r["label"] == "B/constrained")
    br_row = next(r for r in matrix if r["label"] == "B-Regime/constrained")
    regime_cost = {
        "cagr_delta_pp": round(br_row["cagr_pct"] - b_row["cagr_pct"], 4),
        "avg_exposure_delta": round(br_row["avg_exposure_pct"] - b_row["avg_exposure_pct"], 4),
        "B_cagr": b_row["cagr_pct"],
        "B_Regime_cagr": br_row["cagr_pct"],
    }

    payload = {
        "data_adjustment": "dividend_adjusted_total_return_reingest_2026-08-21",
        "band_k": BAND_K,
        "band_k_provenance": BAND_K_PROVENANCE,
        "usable_starts": starts,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "calmar_note": "Do not rank different-exposure books on Calmar alone.",
        "guardrail": "Configs judged only against pre-stated thresholds; highest-CAGR cell is not the result.",
        "matrix": matrix,
        "weekday_unbanded": unbanded_wd,
        "weekday_banded": banded_wd,
        "falsification": {
            "unbanded_cagr_spread_pp": round(ub_spread, 3),
            "banded_cagr_spread_pp": round(bd_spread, 3),
            "unbanded_mean_turnover": round(ub_to, 4),
            "banded_mean_turnover": round(bd_to, 4),
            "verdict": verdict,
        },
        "latch_audit": latch,
        "regime_cost_B_vs_B_Regime_constrained": regime_cost,
    }
    out = Path("data/cache/tier53_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print("FALSIFICATION: " + verdict, flush=True)
    print(json.dumps({"latch_count": latch.get("count"), "regime_cost": regime_cost}, indent=2), flush=True)


if __name__ == "__main__":
    main()
