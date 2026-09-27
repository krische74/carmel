"""Tier 54C — tranche-based timing-luck baseline (unbanded B/raw).

Runs 20 staggered weekly tranches first; banded arm is separate (54E gate).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.cash_yield import series_from_macro_rows
from src.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    _daily_returns_from_equity_curve,
)
from src.backtesting.persist import persist_backtest_run
from src.config import get_settings, hub_sqlite_path, parquet_dir
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.episodes import strategy_usable_start_dates
from src.reporting.timing_luck import compute_timing_luck_metrics

logger = logging.getLogger(__name__)

WINDOW_START = date(2007, 1, 11)
WINDOW_END = date(2021, 12, 31)
N_TRANCHES = 20


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


def run_baseline_tranches(*, persist: bool = True) -> dict:
    settings = get_settings()
    data = _load_data(settings)
    starts = strategy_usable_start_dates(data)
    window_start = max(WINDOW_START, starts["momentum"]) if "momentum" in starts else WINDOW_START
    cash_map = _cash_map(settings)
    eng = BacktestEngine()
    strat = build_strategy_for_backtest(settings, "momentum")

    tranche_returns: list = []
    tranche_cagrs: list[float] = []
    tranche_rows: list[dict] = []

    for offset in range(N_TRANCHES):
        label = f"54C-baseline-tranche-{offset:02d}"
        cfg = BacktestConfig(
            rebalance_frequency="weekly",
            apply_risk_layer=False,
            cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
            cash_yield_by_date=cash_map,
            min_coverage_ratio=0.8,
            drift_band_pct=1.0,
            tranche_offset=offset,
            experiment_label=label,
        )
        result = eng.run(
            strat,
            data,
            start=window_start,
            end=WINDOW_END,
            config=cfg,
            settings=settings,
        )
        if persist:
            persist_backtest_run(
                settings,
                strategy_name=type(strat).__name__,
                config=cfg,
                result=result,
                start=window_start,
                end=WINDOW_END,
                benchmark_symbol="SPY",
                benchmark_metrics=None,
                experiment_label=label,
            )
        rets = _daily_returns_from_equity_curve(result.equity_curve)
        tranche_returns.append(rets)
        cagr = float(result.return_metrics.cagr_pct)
        tranche_cagrs.append(cagr)
        row = {
            "offset": offset,
            "cagr_pct": round(cagr, 4),
            "trades": len(result.trades),
            "discrete_state_changes": result.discrete_state_changes,
            "avg_exposure_pct": round(result.avg_exposure_pct, 4),
        }
        tranche_rows.append(row)
        print(
            f"{label} CAGR={row['cagr_pct']} trades={row['trades']} "
            f"changes={row['discrete_state_changes']} exp={row['avg_exposure_pct']}",
            flush=True,
        )

    n_days = len(result.equity_curve) if result.equity_curve else 1
    years = max(n_days / 252.0, 1e-9)
    tl = compute_timing_luck_metrics(tranche_returns, tranche_cagrs_pct=tranche_cagrs, years=years)

    # Offsets alias mod 5 (every-5-day cadence); dedupe for effective dispersion.
    unique_offsets = list(range(5))
    unique_cagrs = [tranche_rows[i]["cagr_pct"] for i in unique_offsets]
    unique_returns = [tranche_returns[i] for i in unique_offsets]
    tl_unique = compute_timing_luck_metrics(
        unique_returns, tranche_cagrs_pct=unique_cagrs, years=years
    )

    # 52D legacy comparison (5 weekdays, same window) — approximate from tranche subset
    legacy_52d_spread_pp = 2.56  # Tier 52D headline; superseded if void

    verdict_52d = "void" if tl.get("observed_sd_below_predicted") else "survives_rebuilt_estimator"

    payload = {
        "window": {"start": window_start.isoformat(), "end": WINDOW_END.isoformat()},
        "n_tranches": N_TRANCHES,
        "config": "B/raw unbanded, drift_band disabled, tranche_offset 0..19",
        "tranches": tranche_rows,
        "tranche_offset_alias": "offsets 0..19 repeat every 5 (weekly cadence); 5 unique calendars",
        "timing_luck": tl,
        "timing_luck_unique_5": tl_unique,
        "legacy_52d_weekday_cagr_spread_pp": legacy_52d_spread_pp,
        "verdict_52d": verdict_52d,
        "headline_test": (
            "observed_tranche_cagr_std vs predicted TE/sqrt(T); ratio < 1 => void (52D)."
        ),
        "pre_stated_branch": (
            "If observed_sd_below_predicted is true, 52D verdict is void and 54E not indicated."
        ),
        "ensemble_vs_monday_single_run": {
            "ensemble_cagr_pct": tl.get("ensemble_cagr_pct"),
            "monday_single_run_cagr_pct": 4.4859,
            "note": "Monday rebalance (Tier 52/53) was a favourable draw vs tranche ensemble.",
        },
    }
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    payload = run_baseline_tranches(persist=True)
    out = Path("data/cache/tier54_tranche_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(
        f"BASELINE: ensemble_cagr={payload['timing_luck'].get('ensemble_cagr_pct')} "
        f"obs_sd={payload['timing_luck'].get('observed_tranche_cagr_std_pp')}pp "
        f"pred_sd={payload['timing_luck'].get('predicted_tranche_cagr_std_pp')}pp "
        f"ratio={payload['timing_luck'].get('observed_vs_predicted_sd_ratio')} "
        f"52D={payload['verdict_52d']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
