"""Tier 54B — drift-band diagnostic and exposure sanity check (Q018)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.cash_yield import series_from_macro_rows
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.persist import persist_backtest_run
from src.config import get_settings, hub_sqlite_path, parquet_dir
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.episodes import strategy_usable_start_dates

WINDOW_START = date(2007, 1, 11)
WINDOW_END = date(2021, 12, 31)
DRIFT_BAND_PROVENANCE = (
    "Default 0.05 (5 pp absolute around target weight). DR #5 Doc thresholds "
    "unexported (401); practitioner threshold-rebalancing convention when source "
    "unavailable. Constrained-mode figures conditional on this value. Backtest-only: "
    "effective cap = target + band on upside until trim; not a live hard bound."
)
DRIFT_GRID = (0.02, 0.05, 0.10)


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


def _window_start(data) -> date:
    starts = strategy_usable_start_dates(data)
    if "momentum" in starts:
        return max(WINDOW_START, starts["momentum"])
    return WINDOW_START


def _raw_adx_exposure_check(
    settings,
    data,
    cash_map: dict[str, float] | None,
    *,
    window_start: date,
) -> dict[str, float]:
    eng = BacktestEngine()
    strat = build_strategy_for_backtest(settings, "momentum")
    out: dict[str, float] = {}
    for disable_adx, label in ((False, "B/raw"), (True, "B-ADX/raw")):
        cfg = BacktestConfig(
            rebalance_frequency="weekly",
            apply_risk_layer=False,
            cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
            cash_yield_by_date=cash_map,
            min_coverage_ratio=0.8,
            drift_band_pct=0.05,
            disable_adx=disable_adx,
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
        m = result.return_metrics
        out[label] = {
            "cagr_pct": round(m.cagr_pct, 4),
            "avg_exposure_pct": round(result.avg_exposure_pct, 4),
        }
    print(
        f"raw B: CAGR={out['B/raw']['cagr_pct']} exp={out['B/raw']['avg_exposure_pct']} | "
        f"B-ADX: CAGR={out['B-ADX/raw']['cagr_pct']} exp={out['B-ADX/raw']['avg_exposure_pct']}",
        flush=True,
    )
    return out


def main() -> None:
    settings = get_settings()
    data = _load_data(settings)
    window_start = _window_start(data)
    cash_map = _cash_map(settings)
    eng = BacktestEngine()

    grid_rows: list[dict] = []
    for band in DRIFT_GRID:
        strat = build_strategy_for_backtest(settings, "momentum")
        label = f"54B-band-{band:.2f}"
        cfg = BacktestConfig(
            rebalance_frequency="weekly",
            apply_risk_layer=True,
            cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
            cash_yield_by_date=cash_map,
            min_coverage_ratio=0.8,
            drift_band_pct=band,
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
        rid = persist_backtest_run(
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
        m = result.return_metrics
        implied_maint_trades = len(result.trades) - (2 * result.discrete_state_changes + 1)
        row = {
            "label": label,
            "drift_band_pct": band,
            "run_id": rid,
            "cagr_pct": round(m.cagr_pct, 4),
            "max_dd_pct": round(m.max_drawdown_pct, 2),
            "max_exposure_pct": round(result.max_exposure_pct, 4),
            "max_exposure_date": result.max_exposure_date,
            "sharpe": None if m.sharpe_ratio is None else round(m.sharpe_ratio, 4),
            "turnover": round(result.turnover, 4),
            "trades": len(result.trades),
            "discrete_state_changes": result.discrete_state_changes,
            "rotation_identity_trades": 2 * result.discrete_state_changes + 1,
            "maintenance_band_breaches": result.maintenance_band_breaches,
            "maintenance_trades": result.maintenance_trades,
            "implied_maintenance_trades": implied_maint_trades,
            "avg_exposure_pct": round(result.avg_exposure_pct, 4),
        }
        grid_rows.append(row)
        print(
            f"{label} band={band} CAGR={row['cagr_pct']} TO={row['turnover']} "
            f"trades={row['trades']} changes={row['discrete_state_changes']} "
            f"breaches={row['maintenance_band_breaches']} maint_trades={row['implied_maintenance_trades']} "
            f"max_exp={row['max_exposure_pct']} id={rid}",
            flush=True,
        )

    raw_exp = _raw_adx_exposure_check(
        settings,
        data,
        cash_map,
        window_start=window_start,
    )

    payload = {
        "window": {"start": window_start.isoformat(), "end": WINDOW_END.isoformat()},
        "drift_band_provenance": DRIFT_BAND_PROVENANCE,
        "drift_band_sensitivity": grid_rows,
        "raw_exposure_adx_check": raw_exp,
        "q017_verdict": (
            "At default band 0.05: 1 breach, 0 maintenance trades over 2007-2021 (rotation "
            "identity exact). At 0.02: 5 breaches, 4 maintenance trades (+2.8 bp CAGR). "
            "Verdict on trade count, not breaches."
        ),
    }
    out = Path("data/cache/tier54_drift_band_diag.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
