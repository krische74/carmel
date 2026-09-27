"""Tier 52: weekday dispersion, historical cash yield, full-system restatement.

Run from repo root. ASCII only in printed output (Windows cp1252).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.cash_yield import series_from_macro_rows
from src.backtesting.engine import BacktestConfig, BacktestEngine, _daily_returns_from_equity_curve
from src.config import get_settings, hub_sqlite_path, parquet_dir
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.episodes import (
    BULL_EPISODES,
    DRAWDOWN_EPISODES,
    EpisodeSlice,
    buy_and_hold_equity_curve,
    episode_slice_metrics,
    static_blend_equity_curve,
    strategy_usable_start_dates,
)
from src.reporting.returns import compute_return_metrics, illustrative_vol_scaled_cagr

logger = logging.getLogger(__name__)
WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def _load_data(settings) -> dict[str, pd.DataFrame]:
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
    return data


def _cash_map(settings) -> dict[str, float] | None:
    sid = str(settings.backtest.cash_yield_series_id or "").strip()
    if not sid:
        logger.warning("cash_yield_series_id empty; using fallback_flat")
        return None
    store = SQLiteStore(hub_sqlite_path(settings))
    ser = series_from_macro_rows(store.get_macro_indicator(sid))
    if ser.empty:
        logger.warning(
            "cash_yield_series_id=%s missing; using fallback_flat %.2f%%",
            sid,
            settings.backtest.cash_yield_annual_pct,
        )
        return None
    return {ts.date().isoformat(): float(v) for ts, v in ser.items()}


def _ingest_dtb3(settings) -> None:
    sid = str(settings.backtest.cash_yield_series_id or "DTB3").strip()
    from src.data.adapters.fred_adapter import FredAdapter
    from src.data.adapters.yfinance_adapter import YFinanceAdapter
    from src.data.pipeline import DataPipeline

    fred_key = (settings.fred_api_key or "").strip()
    if not fred_key:
        logger.warning("No FRED API key; cannot ingest %s", sid)
        return
    pipeline = DataPipeline(
        adapter=YFinanceAdapter(),
        parquet_store=ParquetStore(parquet_dir(settings)),
        sqlite_store=SQLiteStore(hub_sqlite_path(settings)),
        fred_adapter=FredAdapter(fred_key),
    )
    res = pipeline.ingest_macro(sid)
    logger.info("ingest_macro %s success=%s error=%s", sid, res.success, res.error)


def _pack(result, *, spy_vol: float | None = None) -> dict:
    m = result.return_metrics
    rets = _daily_returns_from_equity_curve(result.equity_curve)
    vol_scaled = None
    if spy_vol is not None and spy_vol > 0:
        vol_scaled = illustrative_vol_scaled_cagr(rets, target_vol_pct=spy_vol)
    return {
        "run_kind": result.run_kind,
        "sizing_mode": result.sizing_mode,
        "trades": len(result.trades),
        "final_equity": round(result.final_equity, 2),
        "total_return_pct": round(m.total_return_pct, 2),
        "cagr_pct": round(m.cagr_pct, 2),
        "sharpe": None if m.sharpe_ratio is None else round(m.sharpe_ratio, 4),
        "max_dd_pct": round(m.max_drawdown_pct, 2),
        "vol_pct": round(m.annual_volatility_pct, 2),
        "calmar": None if m.calmar_ratio is None else round(m.calmar_ratio, 4),
        "peak_alloc": round(result.peak_single_symbol_allocation_pct, 4),
        "avg_exposure_pct": round(result.avg_exposure_pct, 4),
        "max_exposure_pct": round(result.max_exposure_pct, 4),
        "cash_yield_mode": result.cash_yield_mode,
        "realized_cash_yield_annual_pct": round(result.realized_cash_yield_annual_pct, 3),
        "illustrative_vol_scaled_cagr_pct": (
            None if vol_scaled is None else round(vol_scaled, 2)
        ),
    }


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


def _cfg(
    *,
    constrained: bool,
    cash_map: dict[str, float] | None,
    fallback: float,
    weekday: int | None = None,
    month_phase: int = 0,
    frequency: str = "weekly",
    include_dca: bool = False,
    include_cash_sweep: bool = False,
) -> BacktestConfig:
    return BacktestConfig(
        rebalance_frequency=frequency,
        apply_risk_layer=constrained,
        cash_yield_annual_pct=fallback,
        cash_yield_by_date=cash_map,
        min_coverage_ratio=0.8,
        rebalance_weekday=weekday,
        rebalance_month_phase=month_phase,
        include_dca=include_dca,
        include_cash_sweep=include_cash_sweep,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    fallback = float(settings.backtest.cash_yield_annual_pct)
    _ingest_dtb3(settings)
    cash_map = _cash_map(settings)
    data = _load_data(settings)
    starts = {k: v.isoformat() for k, v in strategy_usable_start_dates(data).items()}
    mom_start = strategy_usable_start_dates(data)["momentum"]
    dca_start = strategy_usable_start_dates(data)["dca"]
    end = date(2026, 8, 19)
    strat = build_strategy_for_backtest(settings, "momentum")
    eng = BacktestEngine()

    # --- 52D weekday dispersion (constrained + raw), historical cash if available ---
    weekday_rows: list[dict] = []
    for wd in range(5):
        for constrained in (True, False):
            r = eng.run(
                strat,
                data,
                start=mom_start,
                end=end,
                config=_cfg(
                    constrained=constrained,
                    cash_map=cash_map,
                    fallback=fallback,
                    weekday=wd,
                ),
                settings=settings,
            )
            row = _pack(r)
            row["weekday"] = wd
            row["weekday_name"] = WEEKDAY_NAMES[wd]
            weekday_rows.append(row)
            print(
                f"52D weekday={WEEKDAY_NAMES[wd]} constrained={constrained} "
                f"CAGR={row['cagr_pct']} DD={row['max_dd_pct']} Sharpe={row['sharpe']}",
                flush=True,
            )

    monthly_rows: list[dict] = []
    for phase in (0, 5, 10):
        r = eng.run(
            strat,
            data,
            start=mom_start,
            end=end,
            config=_cfg(
                constrained=True,
                cash_map=cash_map,
                fallback=fallback,
                frequency="monthly",
                month_phase=phase,
            ),
            settings=settings,
        )
        row = _pack(r)
        row["month_phase"] = phase
        monthly_rows.append(row)
        print(f"52D monthly phase={phase} CAGR={row['cagr_pct']} DD={row['max_dd_pct']}", flush=True)

    cagr_c = [x["cagr_pct"] for x in weekday_rows if x["sizing_mode"] == "constrained"]
    sharpe_c = [x["sharpe"] or 0.0 for x in weekday_rows if x["sizing_mode"] == "constrained"]
    cagr_r = [x["cagr_pct"] for x in weekday_rows if x["sizing_mode"] == "raw_signal"]
    sharpe_r = [x["sharpe"] or 0.0 for x in weekday_rows if x["sizing_mode"] == "raw_signal"]
    cagr_range = max(cagr_c) - min(cagr_c)
    sharpe_floor = min(sharpe_c)
    raw_cagr_range = max(cagr_r) - min(cagr_r) if cagr_r else 0.0
    raw_sharpe_floor = min(sharpe_r) if sharpe_r else 0.0
    parts: list[str] = []
    if cagr_range > 2.0 or sharpe_floor < 0.1:
        parts.append(
            "CONSTRAINED WIDE: do not rest an allocation decision on a single weekday."
        )
    elif cagr_range < 1.0 and sharpe_floor > 0.15:
        parts.append(
            "CONSTRAINED TIGHT on CAGR: 25% cap damps weekday luck in wealth terms."
        )
    else:
        parts.append("CONSTRAINED MODERATE weekday dispersion.")
    if raw_cagr_range > 2.0 or raw_sharpe_floor < 0.1:
        parts.append(
            "RAW WIDE: unconstrained N=1 CAGR span exceeds 2pp; the signal path is timing-sensitive."
        )
    interpretation = " ".join(parts)

    # --- 52E restatement: default first-ISO-week (weekday=None), historical cash ---
    cfg_c = _cfg(constrained=True, cash_map=cash_map, fallback=fallback)
    cfg_r = _cfg(constrained=False, cash_map=cash_map, fallback=fallback)
    rc = eng.run(strat, data, start=mom_start, end=end, config=cfg_c, settings=settings)
    rr = eng.run(strat, data, start=mom_start, end=end, config=cfg_r, settings=settings)

    spy = data["SPY"].copy()
    spy.index = pd.DatetimeIndex(pd.to_datetime(spy.index).tz_localize(None)).normalize()
    spy_w = spy[(spy.index.date >= mom_start) & (spy.index.date <= end)]
    spy_curve = buy_and_hold_equity_curve(spy_w)
    blend_curve = static_blend_equity_curve(
        spy_w,
        cash_yield_annual_pct=fallback,
        cash_yield_by_date=cash_map,
    )
    spy_m = compute_return_metrics(_daily_returns_from_equity_curve(spy_curve))
    spy_vol = spy_m.annual_volatility_pct

    cfg_full_c = _cfg(
        constrained=True,
        cash_map=cash_map,
        fallback=fallback,
        include_dca=True,
        include_cash_sweep=True,
    )
    cfg_full_r = _cfg(
        constrained=False,
        cash_map=cash_map,
        fallback=fallback,
        include_dca=True,
        include_cash_sweep=True,
    )
    print(f"52E full-system window {dca_start} -> {end}", flush=True)
    fc = eng.run(strat, data, start=dca_start, end=end, config=cfg_full_c, settings=settings)
    fr = eng.run(strat, data, start=dca_start, end=end, config=cfg_full_r, settings=settings)

    spy_dca = spy[(spy.index.date >= dca_start) & (spy.index.date <= end)]
    spy_dca_curve = buy_and_hold_equity_curve(spy_dca)
    blend_dca_curve = static_blend_equity_curve(
        spy_dca,
        cash_yield_annual_pct=fallback,
        cash_yield_by_date=cash_map,
    )

    payload = {
        "usable_starts": starts,
        "cash_yield_series_id": settings.backtest.cash_yield_series_id,
        "fallback_cash_yield_annual_pct": fallback,
        "calmar_note": (
            "Calmar must not be used to rank portfolios with materially different "
            "equity exposure; it rewards not investing."
        ),
        "vol_scaled_note": (
            "illustrative_vol_scaled_cagr_pct scales daily returns to SPY volatility; "
            "illustrative only (no borrowing costs or leverage constraints)."
        ),
        "weekday_dispersion": {
            "window": {"start": mom_start.isoformat(), "end": end.isoformat(), "kind": "momentum_only"},
            "rows": weekday_rows,
            "constrained_cagr_range_pp": round(cagr_range, 3),
            "constrained_sharpe_min": round(sharpe_floor, 4),
            "raw_cagr_range_pp": round(raw_cagr_range, 3),
            "raw_sharpe_min": round(raw_sharpe_floor, 4),
            "interpretation": interpretation,
        },
        "monthly_phase": {
            "window": {"start": mom_start.isoformat(), "end": end.isoformat(), "kind": "momentum_only"},
            "rows": monthly_rows,
        },
        "momentum_only_2007": {
            "window": {"start": mom_start.isoformat(), "end": end.isoformat()},
            "constrained": _pack(rc, spy_vol=spy_vol),
            "raw_signal": _pack(rr, spy_vol=spy_vol),
            "spy": {
                "cagr_pct": round(spy_m.cagr_pct, 2),
                "total_return_pct": round(spy_m.total_return_pct, 2),
                "max_dd_pct": round(spy_m.max_drawdown_pct, 2),
                "vol_pct": round(spy_m.annual_volatility_pct, 2),
                "sharpe": None if spy_m.sharpe_ratio is None else round(spy_m.sharpe_ratio, 4),
                "avg_exposure_pct": 1.0,
                "max_exposure_pct": 1.0,
            },
            "blend_75_25": _pack_curve(blend_curve, spy_vol),
            "episodes": (
                _series_rows("constrained_momentum", rc.equity_curve, mom_start, end)
                + _series_rows("raw_signal_momentum", rr.equity_curve, mom_start, end)
                + _series_rows("spy", spy_curve, mom_start, end)
                + _series_rows("blend_75_25", blend_curve, mom_start, end)
            ),
        },
        "full_system_2011": {
            "window": {"start": dca_start.isoformat(), "end": end.isoformat()},
            "constrained": _pack(fc, spy_vol=spy_vol),
            "raw_signal": _pack(fr, spy_vol=spy_vol),
            "spy": _pack_curve(spy_dca_curve, spy_vol),
            "blend_75_25": _pack_curve(blend_dca_curve, spy_vol),
            "episodes": (
                _series_rows("constrained_full_system", fc.equity_curve, dca_start, end)
                + _series_rows("raw_signal_full_system", fr.equity_curve, dca_start, end)
                + _series_rows("spy", spy_dca_curve, dca_start, end)
                + _series_rows("blend_75_25", blend_dca_curve, dca_start, end)
            ),
        },
    }
    out = Path("data/cache/tier52_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print("INTERPRETATION: " + interpretation, flush=True)
    print(
        json.dumps(
            {
                "weekday_cagr_range_pp": payload["weekday_dispersion"]["constrained_cagr_range_pp"],
                "momentum_only_2007": payload["momentum_only_2007"]["constrained"],
                "full_system_2011": payload["full_system_2011"]["constrained"],
            },
            indent=2,
        ),
        flush=True,
    )


def _pack_curve(curve: list[dict], spy_vol: float) -> dict:
    m = compute_return_metrics(_daily_returns_from_equity_curve(curve))
    rets = _daily_returns_from_equity_curve(curve)
    scaled = illustrative_vol_scaled_cagr(rets, target_vol_pct=spy_vol) if spy_vol else None
    return {
        "total_return_pct": round(m.total_return_pct, 2),
        "cagr_pct": round(m.cagr_pct, 2),
        "sharpe": None if m.sharpe_ratio is None else round(m.sharpe_ratio, 4),
        "max_dd_pct": round(m.max_drawdown_pct, 2),
        "vol_pct": round(m.annual_volatility_pct, 2),
        "calmar": None if m.calmar_ratio is None else round(m.calmar_ratio, 4),
        "illustrative_vol_scaled_cagr_pct": None if scaled is None else round(scaled, 2),
    }


if __name__ == "__main__":
    main()
